import os
from datasets import load_dataset, Image
import torch
import json
import time
from tqdm import tqdm
import subprocess
import platform
import sys
import tempfile
from evaluate import load
from swift.infer_engine import TransformersEngine, RequestConfig, InferRequest

bleu = load("bleu")
rouge = load("rouge")
meteor = load("meteor")

val_dataset = (
    load_dataset("SimulaMet/Kvasir-VQA-x1", split="test")
    .shuffle(seed=42)
    .select(range(1500))
    .cast_column("image", Image(decode=False))  # prevent PIL decoding
)

predictions = []  # List to store predictions

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
        eg, We have finetuned XXX model
        This is optional . .
        Used data augmentations . .
        Custom info about the model . .
        Any insights. .
        + Any informal things you like to share about this submission.
        '''
}

# 🔹 TODO: PARTICIPANTS MUST LOAD THEIR MODEL HERE, EDIT AS NECESSARY FOR YOUR MODEL 🔹
# If you encounter issues with your model loading or generation below, please contact us:
# https://github.com/simula/MediaEval-Medico-2026#organizers

# 👉 You may add required library imports for your model below.
hf_model_base = "google/paligemma-3b-pt-224"
hf_model_adapters = ['SushantGautam/Kvasir-VQA-x1-pali3b-lora']  # <------- finetuned LoRA adapter if any

# 👉 You can further configure preprocessing or generation settings to control how your model should behave
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MAX_PIXELS'] = '1003520'
os.environ['FPS_MAX_FRAMES'] = '12'


MAX_BATCH_SIZE = 16  # 👉 Adjust based on VRAM; engine auto-batches up to this

engine = TransformersEngine(
    hf_model_base,
    adapters=hf_model_adapters,
    max_batch_size=MAX_BATCH_SIZE,
    use_hf=True

)
request_config = RequestConfig(max_tokens=128, temperature=0) ## < 👉  can customize as per your model's generation needs
# 🏁----------------END  SUBMISISON DETAILS and MODEL LOADING -----------------🏁#

start_time, post_model_mem = time.time(), get_mem()
total_time, final_mem = round(time.time() - start_time, 4), round(get_mem() - post_model_mem, 2)
model_mem_used = round(post_model_mem - initial_mem, 2)

# ✏️✏️___________EDIT SECTION 2: ANSWER GENERATION___________✏️✏️#

# Build all InferRequests (Swift will auto-batch internally)
infer_requests = [
    InferRequest(
        messages=[ #  👉 customize as per your model's generation needs
        # {"role":"system","content":"You are a medical assistant. Answer in one short sentence."},
        {"role": "user", "content": f"<image> {ex['question']}"}
        ],
        images=[ex["image"]["path"]],
    )
    for ex in tqdm(val_dataset, desc="Preparing", unit="samples")
]

#   ___________END SECTION 2: ANSWER GENERATION___________    #

# ⛔ DO NOT EDIT any lines below from here, can edit only upto request construction above as required. ⛔

# 👉 Swift handles batching automatically here (up to MAX_BATCH_SIZE)
responses = engine.infer(infer_requests, request_config)

for idx, (ex, resp) in enumerate(zip(val_dataset, responses)):
    answer = resp.choices[0].message.content.strip()
    assert isinstance(answer, str), f"Generated answer at index {idx} is not a string"
    predictions.append(
        {"index": idx, "img_id": ex["img_id"], "question": ex["question"], "answer": answer}
    )


# Ensure all predictions match dataset length
assert len(predictions) == len(val_dataset), "Mismatch between predictions and dataset length"

total_time, final_mem = round(time.time() - start_time, 4), round(get_mem() - post_model_mem, 2)
model_mem_used = round(post_model_mem - initial_mem, 2)

# caulcualtes metrics
references = [[e] for e in val_dataset['answer']]
preds = [pred['answer'] for pred in predictions]

bleu_result = bleu.compute(predictions=preds, references=references)
rouge_result = rouge.compute(predictions=preds, references=references)
meteor_result = meteor.compute(predictions=preds, references=references)
bleu_score = round(bleu_result['bleu'], 4)
rouge1_score = round(float(rouge_result['rouge1']), 4)
rouge2_score = round(float(rouge_result['rouge2']), 4)
rougeL_score = round(float(rouge_result['rougeL']), 4)
meteor_score = round(float(meteor_result['meteor']), 4)

public_scores = {
    'bleu': bleu_score,
    'rouge1': rouge1_score,
    'rouge2': rouge2_score,
    'rougeL': rougeL_score,
    'meteor': meteor_score
}
print("✨Public scores: ", public_scores)

# Saves predictions to a JSON file
output_data = {"submission_info": SUBMISSION_INFO, "public_scores": public_scores,
               "predictions": predictions, "total_time": total_time, "time_per_item": total_time / len(val_dataset),
               "memory_used_mb": final_mem, "model_memory_mb": model_mem_used, "gpu_name": gpu_name,
               "debug": {
                   "packages": json.loads(subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"])),
                   "system": {
                       "python": platform.python_version(),
                       "os": platform.system(),
                       "platform": platform.platform(),
                       "arch": platform.machine()
                   }}}

with open("predictions_1.json", "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Time: {total_time}s | Mem: {final_mem}MB | Model Load Mem: {model_mem_used}MB | GPU: {gpu_name}")
print("✅ Scripts Looks Good! Generation process completed successfully. Results saved to 'predictions_1.json'.")
print("Next Step:\n 1) Upload this submission_task1.py script file to HuggingFace model repository.")
print('''\n 2) Make a submission to the competition:\n Run:: medvqa validate_and_submit --competition=medico-2026 --task=1 --repo_id=...''')