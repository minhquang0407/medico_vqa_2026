import sys
from pathlib import Path
from datasets import load_dataset
from PIL import Image

# Add the submission directories to sys.path and load their predict_one functions under different names
workspace_dir = Path("c:/Users/nguye/Project/medico_vqa_2026")
sys.path.insert(0, str(workspace_dir))

print("Loading test dataset slice...")
from datasets import Image as HfImage
ds = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"].cast_column("image", HfImage())
# Take a deterministic subset of 10 samples
samples = [ds[i] for i in range(10)]

def test_variant(variant_name, folder_name):
    print(f"\n==========================================")
    print(f"Testing variant: {variant_name} in {folder_name}")
    print(f"==========================================")
    
    # We must load the submission_task1.py from that directory
    # To avoid sys.path conflicts, we can load it dynamically or import it by setting sys.path
    import importlib.util
    
    script_path = workspace_dir / "hf_submission" / folder_name / "submission_task1.py"
    if not script_path.exists():
        print(f"Error: {script_path} does not exist!")
        return []
        
    spec = importlib.util.spec_from_file_location(f"submission_{folder_name}", str(script_path))
    module = importlib.util.module_from_spec(spec)
    
    # We need to clear cached modules in sys.modules to avoid importing the wrong src
    for key in list(sys.modules.keys()):
        if key.startswith("src.") or key == "src" or key.startswith("submission_"):
            sys.modules.pop(key)
            
    # Also set REPO_DIR in sys.path
    repo_dir = workspace_dir / "hf_submission" / folder_name
    sys.path.insert(0, str(repo_dir))
    
    spec.loader.exec_module(module)
    
    # Get predictions
    predictions = []
    for idx, sample in enumerate(samples):
        question = sample["question"]
        image = sample["image"]
        # Convert to PIL if it's not already
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        try:
            pred = module.predict_one(image, question)
            predictions.append((question, sample["answer"], pred))
            print(f"Sample #{idx+1} | Q: {question} | GT: {sample['answer']} | Pred: {pred}")
        except Exception as e:
            print(f"Sample #{idx+1} failed: {e}")
            predictions.append((question, sample["answer"], f"FAILED: {e}"))
            
    # Remove from sys.path
    if str(repo_dir) in sys.path:
        sys.path.remove(str(repo_dir))
        
    return predictions

# Run each variant
results = {}
results["RunB (task1)"] = test_variant("RunB", "task1")
results["CATA (task1_topo_recipe)"] = test_variant("CATA", "task1_topo_recipe")
results["TopoFusion (task1_topo_norecipe)"] = test_variant("TopoFusion", "task1_topo_norecipe")

# Print side-by-side comparison
print("\n\n========================================================")
print("FINAL SIDE-BY-SIDE COMPARISON OF FIRST 10 TEST SAMPLES")
print("========================================================")
for idx in range(10):
    q, gt, _ = results["RunB (task1)"][idx]
    print(f"\n#{idx+1} Question: {q}")
    print(f"   Ground Truth: {gt}")
    print(f"   RunB:       {results['RunB (task1)'][idx][2]}")
    print(f"   CATA:       {results['CATA (task1_topo_recipe)'][idx][2]}")
    print(f"   TopoFusion: {results['TopoFusion (task1_topo_norecipe)'][idx][2]}")
