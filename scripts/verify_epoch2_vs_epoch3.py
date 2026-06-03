import sys
import gc
import torch
from pathlib import Path
from datasets import load_dataset
from datasets import Image as HfImage
from evaluate import load as load_metric
from PIL import Image

workspace_dir = Path("c:/Users/nguye/Project/medico_vqa_2026")
sys.path.insert(0, str(workspace_dir))

print("Loading test dataset and selecting first 100 shuffled samples...")
ds = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"]
val_dataset = ds.shuffle(seed=42).select(range(100)).cast_column("image", HfImage())
samples = [sample for sample in val_dataset]
references = [[sample["answer"]] for sample in samples]

def eval_checkpoint(ckpt_path, name):
    print(f"\n==========================================")
    print(f"EVALUATING CHECKPOINT: {name}")
    print(f"Path: {ckpt_path}")
    print(f"==========================================")
    
    # We use task1_cata_minimal_norm's submission_task1.py as our entry point
    import importlib.util
    script_path = workspace_dir / "hf_submission" / "task1_cata_minimal_norm" / "submission_task1.py"
    spec = importlib.util.spec_from_file_location("submission_test", str(script_path))
    module = importlib.util.module_from_spec(spec)
    
    # Clear sys.modules
    for key in list(sys.modules.keys()):
        if key.startswith("src.") or key == "src" or key.startswith("submission_"):
            sys.modules.pop(key)
            
    repo_dir = workspace_dir / "hf_submission" / "task1_cata_minimal_norm"
    sys.path.insert(0, str(repo_dir))
    
    spec.loader.exec_module(module)
    
    # Override checkpoint loading to load our target checkpoint
    original_find = module._find_checkpoint
    module._find_checkpoint = lambda: Path(ckpt_path)
    
    # Run prediction
    preds = []
    for idx, sample in enumerate(samples):
        question = sample["question"]
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        try:
            pred = module.predict_one(image, question).strip()
        except Exception as e:
            print(f"Error at sample #{idx}: {e}")
            pred = ""
        preds.append(pred)
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/100 samples")
            
    # Compute metrics
    bleu = load_metric("bleu")
    rouge = load_metric("rouge")
    meteor = load_metric("meteor")
    
    bleu_result = bleu.compute(predictions=preds, references=references)
    rouge_result = rouge.compute(predictions=preds, references=references)
    meteor_result = meteor.compute(predictions=preds, references=references)
    
    scores = {
        "bleu": round(float(bleu_result["bleu"]), 4),
        "rouge1": round(float(rouge_result["rouge1"]), 4),
        "rouge2": round(float(rouge_result["rouge2"]), 4),
        "rougeL": round(float(rouge_result["rougeL"]), 4),
        "meteor": round(float(meteor_result["meteor"]), 4),
    }
    
    print(f"Scores for {name}: {scores}")
    
    # Clean up model and memory to avoid OOM
    if hasattr(module, "_MODEL"):
        module._MODEL = None
    if hasattr(module, "_STRUCTURAL_EXTRACTORS"):
        module._STRUCTURAL_EXTRACTORS = None
    
    # Remove from sys.path
    if str(repo_dir) in sys.path:
        sys.path.remove(str(repo_dir))
        
    gc.collect()
    torch.cuda.empty_cache()
    
    return scores

results = {}
try:
    results["Epoch 2 (Original CATA)"] = eval_checkpoint(
        "outputs/qwen3b_curriculum_topo_adapter_full_30k/checkpoints/epoch_2.pt",
        "Epoch 2 (Original CATA)"
    )
except Exception as e:
    print(f"Epoch 2 failed: {e}")

try:
    results["Epoch 3 (Overfitted Continuation)"] = eval_checkpoint(
        "outputs/qwen3b_curriculum_topo_adapter_full_full_continue/checkpoints/epoch_3.pt",
        "Epoch 3 (Overfitted Continuation)"
    )
except Exception as e:
    print(f"Epoch 3 failed: {e}")

# Print final comparison table
print("\n\n========================================================")
print("FINAL SCORES COMPARISON (100 SAMPLE SLICE)")
print("========================================================")
print(f"{'Checkpoint':<35} | {'BLEU':<6} | {'ROUGE-1':<7} | {'ROUGE-2':<7} | {'ROUGE-L':<7} | {'METEOR':<6}")
print("-" * 85)
for name, score in results.items():
    print(f"{name:<35} | {score['bleu']:<6.4f} | {score['rouge1']:<7.4f} | {score['rouge2']:<7.4f} | {score['rougeL']:<7.4f} | {score['meteor']:<6.4f}")
