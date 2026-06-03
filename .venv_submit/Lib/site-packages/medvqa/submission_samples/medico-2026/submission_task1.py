from transformers import AutoModelForCausalLM
from datasets import load_dataset, Image as HfImage
from transformers import AutoProcessor
from itertools import batched
import torch
import json
import time
from tqdm import tqdm
import subprocess
import platform
import sys

from evaluate import load

bleu = load("bleu")
rouge = load("rouge")
meteor = load("meteor")


ds = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"]
ds_shuffled = ds.shuffle(seed=42) # Shuffle with fixed seed for reproducibility
val_dataset = ds_shuffled.select(range(1500)) # Select first 1500 after shuffle
val_dataset = val_dataset.cast_column("image", HfImage())
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
# 👉 You may add required library imports for your model below.

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model_hf = Qwen3VLForConditionalGeneration.from_pretrained(
    "sosoai/qwen3_vl_2b_lora_v1", trust_remote_code=True
).to(device) # 👉 Load your vision-language (or multimodal) model here.

processor = AutoProcessor.from_pretrained(
    "sosoai/qwen3_vl_2b_lora_v1", trust_remote_code=True
) # 👉 Load the matching processor / tokenizer for your model.

# 👉 You can further configure preprocessing or generation settings to control how your model should behave.
processor.image_processor.size = {"shortest_edge": 300, "longest_edge": 600}
processor.tokenizer.padding_side = "left"

BATCH_SIZE = 32  # 👉 Adjust batch size based on your available GPU memory

# If you encounter issues with your model loading or generation below, please contact us:
# https://github.com/simula/MediaEval-Medico-2026#organizers
model_hf.eval()  # Ensure model is in evaluation mode
# 🏁----------------END  SUBMISISON DETAILS and MODEL LOADING -----------------🏁#

start_time, post_model_mem = time.time(), get_mem()
total_time, final_mem = round(
    time.time() - start_time, 4), round(get_mem() - post_model_mem, 2)
model_mem_used = round(post_model_mem - initial_mem, 2)

with tqdm(total=len(val_dataset), desc="Validating", unit="samples") as pbar:
  for batch in batched(enumerate(val_dataset), BATCH_SIZE):
    pbar.update(len(batch))
    idxs, exs = zip(*batch)
    
    # ✏️✏️___________EDIT SECTION 2: ANSWER GENERATION___________✏️✏️#

    # 👉 Participants: Build your model prompt + generation logic ONLY inside this section.
    # Do NOT modify batching logic or anything outside EDIT blocks.

    prompts = [
        processor.apply_chat_template(  # You should change how the prompt is constructed
            [{"role":"system","content":[{"type":"text","text":"You are a medical assistant. Answer in one short sentence."}]},
            {"role":"user","content":[{"type":"image","image":e["image"]}, {"type":"text","text":e["question"]}]}],
            tokenize=False, add_generation_prompt=True) for e in exs]

    # 👉 You may replace processor/model calls below with your own model inference logic
    inputs = processor(text=prompts, images=[e["image"] for e in exs], return_tensors="pt", padding=True)
    inputs = {k: v.to(model_hf.device) for k, v in inputs.items()}

    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        # 👉 Replace model_hf.generate(...) with your custom generation method if needed
        out = model_hf.generate(**inputs, max_new_tokens=128, do_sample=False)

    # 👉 Ensure outputs from this batch are decoded into a list[str] named `answers`
    answers = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    # for e, a in zip(exs, answers): print("Q:", e["question"], "\nA:", a, "\n") ## print to debug

    #   ___________END SECTION 2: ANSWER GENERATION___________    #

# ⛔ DO NOT EDIT any lines below from here, can edit only upto decoding step above as required. ⛔

    assert all(isinstance(a, str) for a in answers), next(f"Non-string answer at index {i}" 
         for i, a in zip(idxs, answers)
         if not isinstance(a, str)) # safe test

    predictions.extend(
        {"index": i, "img_id": e["img_id"], "question": e["question"], "answer": a.strip()}
        for i, e, a in zip(idxs, exs, answers)
    )

# Ensure all predictions match dataset length
assert len(predictions) == len(
    val_dataset), "Mismatch between predictions and dataset length"

total_time, final_mem = round(
    time.time() - start_time, 4), round(get_mem() - post_model_mem, 2)
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
