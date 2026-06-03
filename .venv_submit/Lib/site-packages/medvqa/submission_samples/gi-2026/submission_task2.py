from sklearn.preprocessing import normalize
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import sqrtm
from scipy.spatial.distance import pdist
from PIL import Image
import numpy as np
import os
from diffusers import DiffusionPipeline
from datasets import load_dataset
from transformers import AutoProcessor, AutoModel
import torch
import json
import time
from tqdm import tqdm
import subprocess
import platform
import sys
import requests

jsons__ = requests.get(
    "https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-test/resolve/main/imagen-test").json()
test_prompts = [c for qa in jsons__.values()
                for pair in qa.values() for c in pair]
gpu_name = torch.cuda.get_device_name(
    0) if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_mem(): return torch.cuda.memory_allocated(device) / \
    (1024 ** 2) if torch.cuda.is_available() else 0


initial_mem = get_mem()

# ✏️✏️--------EDIT SECTION 1: SUBMISISON DETAILS and MODEL LOADING --------✏️✏️#

SUBMISSION_INFO = {
    # 🔹 TODO: PARTICIPANTS MUST ADD PROPER SUBMISSION INFO FOR THE SUBMISSION 🔹
    # This will be visible to the organizers
    # DONT change the keys, only add your info
    "Participant_Names": "Sushant Gautam, Steven Hicks and Vajita Thambawita",
    "Affiliations": "SimulaMet",
    "Contact_emails": ["sushant@simula.no", "steven@simula.no"],
    # But, the first email only will be used for correspondance
    "Team_Name": "SimulaMetmedVQA Rangers",
    "Country": "Norway",
    "Notes_to_organizers": '''
        eg, We have finetund XXX model
        This is optional . .
        Used data augmentations . .
        Custom info about the model . .
        Any insights. .
        + Any informal things you like to share about this submission.
        '''
}
# 🔹 TODO: PARTICIPANTS MUST LOAD THEIR MODEL HERE, EDIT AS NECESSARY FOR YOUR MODEL 🔹
# can add necessary library imports here

hf_pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
hf_pipe.load_lora_weights("waitwhoami/sd-kvasir-imagen-demo")
hf_pipe.safety_checker = lambda images, clip_input: (images, None)
hf_pipe.set_progress_bar_config(disable=True)

print("🔍 Model loaded successfully. Proceeding to image generation...")

# 🏁----------------END  SUBMISISON DETAILS and MODEL LOADING -----------------🏁#

start_time, post_model_mem = time.time(), get_mem()
total_time, final_mem = round(
    time.time() - start_time, 4), round(get_mem() - post_model_mem, 2)
model_mem_used = round(post_model_mem - initial_mem, 2)
num_per_prompt = 10
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_folder = f"generated_images_{timestamp}"
# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)
# print full path of output folder
print(f"🔍 Output folder: {os.path.abspath(output_folder)}")

# ✏️✏️___________EDIT SECTION 2: IMAGE GENERATION___________✏️✏️#
# 🔹 TODO: PARTICIPANTS SHOULD MODIFY THIS STEP  🔹
# you have access to 'test_prompts'  with all the prompts needed to be generated

batch_size = 2  # Adjust based on your GPU memory, number of prompts to generate in one go
print(
    f"🔍 We have {len(test_prompts)} prompts and we are generating for {batch_size} prompts at once. ")
for i in tqdm(range(0, len(test_prompts), batch_size), desc="🌀 Generating images"):
    batch = test_prompts[i:i + batch_size]
    batched_prompts = [p for p in batch for _ in range(num_per_prompt)]
    images = hf_pipe(batched_prompts).images
    for j, img in enumerate(images):
        p_idx = i + j // num_per_prompt + 1
        i_idx = j % num_per_prompt + 1
        img.save(f"{output_folder}/prompt{p_idx:04d}_img{i_idx:04d}.png")
print("🔍 Image generation completed. Proceeding to feature extraction...")
# make sure 'output_folder'  with generated images is available with proper filenames

# 🏁________________ END IMAGE GENERATION ________________🏁#

# ⛔ DO NOT EDIT any lines below from here, can edit only upto decoding step above as required. ⛔
# Ensures answer is a string

saved_files = [f for f in os.listdir(output_folder) if f.endswith('.png')]
expected_count = len(test_prompts) * num_per_prompt

assert len(
    saved_files) == expected_count, f"Expected {expected_count} images, but found {len(saved_files)}."

total_time, final_mem = round(
    time.time() - start_time, 4), round(get_mem() - post_model_mem, 2)
model_mem_used = round(post_model_mem - initial_mem, 2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelx = AutoModel.from_pretrained(
    "ikim-uk-essen/BiomedCLIP_ViT_patch16_224", trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained(
    "ikim-uk-essen/BiomedCLIP_ViT_patch16_224", trust_remote_code=True)
modelx.eval()


def extract_features(batch):
    inputs = processor(images=batch['image'], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = modelx(**inputs).last_hidden_state[:, 0, :]
    return {'features': feats.cpu().numpy()}


def extract_features_from_paths(image_paths, batch_size=32):
    imgs = [Image.open(p).convert('RGB') for p in image_paths]
    dataset = Dataset.from_dict({'image': imgs})
    dataset = dataset.map(extract_features, batched=True,
                          batch_size=batch_size)
    return np.vstack(dataset['features'])


def fid_score(feat1, feat2):
    mu1, mu2 = feat1.mean(0), feat2.mean(0)
    sigma1, sigma2 = np.cov(feat1, rowvar=False), np.cov(feat2, rowvar=False)
    covmean = sqrtm(sigma1 @ sigma2).real
    return ((mu1 - mu2)**2).sum() + np.trace(sigma1 + sigma2 - 2 * covmean)


def diversity_score(features):
    features = normalize(features, axis=1)
    return pdist(features).mean()


def mean_cosine_sim(feat1, feat2):
    feat1 = normalize(feat1, axis=1)
    feat2 = normalize(feat2, axis=1)
    return cosine_similarity(feat1, feat2).mean()


# --- Organize generated images ---
generated_files = sorted(
    [f for f in os.listdir(output_folder) if f.endswith(".png")])
prompt_to_images = {}
for f in generated_files:
    prompt_idx = int(f.split("_")[0].replace("prompt", ""))
    prompt_to_images.setdefault(prompt_idx, []).append(
        os.path.join(output_folder, f))

print("🔍 Extracting features for generated images...")
all_features = {}
for prompt_idx, paths in tqdm(prompt_to_images.items(), desc="Extracting generated image's features"):
    all_features[prompt_idx] = extract_features_from_paths(paths)
print("🔍 Feature extraction completed. Proceeding to scoring...")

val_dataset = load_dataset("SimulaMet/Kvasir-VQA-test", split="validation")
prompt_to_real = requests.get(
    "https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-test/resolve/main/real_mapping").json()

print("Now, extracting real image's features...")
seen = set()
real_features_cache_ = val_dataset.filter(lambda x: x["img_id"] not in seen and not seen.add(x["img_id"])).map(
    extract_features,
    batched=True,
    batch_size=128
)
real_features_cache = {
    image_id: feature
    for image_id, feature in zip(real_features_cache_["img_id"], real_features_cache_["features"])
}


# --- Pair prompts: (0,1), (2,3), ...
sorted_prompts = sorted(all_features.keys())
objectives = []
for i in range(0, len(sorted_prompts)//2, 2):
    idx_A = sorted_prompts[i]
    idx_B = sorted_prompts[i + 1]
    A = all_features[idx_A]
    B = all_features[idx_B]
    objectives.append((idx_A, idx_B, A, B))

# --- Per-objective Metrics ---
fids, agreements, diversities = [], [], []
all_generated, all_real = [], []
per_prompt_data = []

print("🔍 Calculating metrics and preparing output data...")
for idx_A, idx_B, A, B in tqdm(objectives, desc="Scoring"):
    sim_ab = mean_cosine_sim(A, B)
    fid_ab = fid_score(A, B)
    div_A = diversity_score(A)
    div_B = diversity_score(B)

    # Shared real reference for both prompts
    # same as prompt_to_real[str(idx_B)]
    real_keys = prompt_to_real[str(idx_A)]
    #  flag by SUSHANT, just to debug ;)
    # real_keys = random.sample(val_dataset['img_id'], len(real_keys))
    real_feats = np.array([real_features_cache[key] for key in real_keys])
    fid_A_real = fid_score(A, real_feats)
    fid_B_real = fid_score(B, real_feats)

    # Collect for global metrics
    all_generated.extend([*A, *B])
    all_real.extend(real_feats)

    fids.append((fid_A_real + fid_B_real) / 2)
    agreements.append(sim_ab)
    diversities.extend([div_A, div_B])

    per_prompt_data.append({
        "Prompt A": idx_A,
        "Prompt B": idx_B,
        "FID(A,B)": fid_ab,
        "Agreement": sim_ab,
        "Diversity A": div_A,
        "Diversity B": div_B,
        "FID A vs Real": fid_A_real,
        "FID B vs Real": fid_B_real,
        "Real Ref": real_feats
    })

# --- Global FID ---
all_generated = np.array(all_generated)
all_real = np.array(all_real)
global_fid = fid_score(all_generated, all_real)

# --- Global Scores ---
fidelity_norm = np.mean(1000 / (1 + np.array(fids)))
agreement_norm = np.mean(agreements)
diversity_norm = np.mean(diversities)
# final_score = 0.5 * fidelity_norm + 0.3 * agreement_norm + 0.2 * diversity_norm #lets not use this for now

# --- Output ---
public_scores = {
    "fidelity": round(float(fidelity_norm), 2),
    "agreement": round(float(agreement_norm), 2),
    "diversity": round(float(diversity_norm), 2),
    "FBD": round(float(global_fid), 2)
}


# end calculating metrics
print(
    f"🔍 Metrics calculated. Fidelity: {fidelity_norm}, Agreement: {agreement_norm}, Diversity: {diversity_norm}")
print("🔍 Saving results to 'predictions_2.json'...")

output_data = {"submission_info": SUBMISSION_INFO, "public_scores": public_scores, "total_time": total_time, "time_per_item": total_time / len(val_dataset),
               "memory_used_mb": final_mem, "model_memory_mb": model_mem_used, "gpu_name": gpu_name, "predictions": json.dumps({k: v.tolist() for k, v in all_features.items()}), "debug": {
                   "packages": json.loads(subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"])),
                   "system": {
                       "python": platform.python_version(),
                       "os": platform.system(),
                       "platform": platform.platform(),
                       "arch": platform.machine()
                   }}}


with open("predictions_2.json", "w") as f:
    json.dump(output_data, f, indent=4)
print("✅ Results saved successfully. Script execution completed.")
print(f"Time: {total_time}s | Mem: {final_mem}MB | Model Load Mem: {model_mem_used}MB | GPU: {gpu_name}")
print("✅ Scripts Looks Good! Generation process completed successfully. Results saved to 'predictions_2.json'.")
print("Next Step:\n 1) Upload this submission_task2.py script file to HuggingFace model repository.")
print('''\n 2) Make a submission to the competition:\n Run:: medvqa validate_and_submit --competition=gi-2025 --task=2 --repo_id=...''')
