from gradio_client import Client, handle_file
import sys
from huggingface_hub import snapshot_download, login, whoami
import argparse
import os
import subprocess as sp
import time
from datetime import datetime, timezone
import shutil
import json
from huggingface_hub import HfApi, grant_access
import re

HF_GATE_ACCESSLIST = ["SushantGautam",
                     "stevenah", "vlbthambawita"]

MEDVQA_SUBMIT = True if os.environ.get(
    '_MEDVQA_SUBMIT_FLAG_', 'FALSE') == 'TRUE' else False
parser = argparse.ArgumentParser(
    description='Run MEDVQA-GI 2026 Task 2 (Multimodal Explainability and Safety-Aware Reasoning)')
parser.add_argument('--repo_id', type=str, required=True,
                    help='Path to the HF submission repository')
args, _ = parser.parse_known_args()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
submission_file = "submission_task2.py"
file_from_validation = "predictions_2.json"

min_library = ["datasets>=3.4.1", "transformers", "evaluate", "scipy", "scikit-learn", "diffusers", "peft",
               "rouge_score", "tqdm", "gradio_client>=1.8.0"]

print("🌟 ImageCLEFmed-MEDVQA-GI-2026 🌟",
      "https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2026")
print("💬 Subtask 2: Multimodal Explainability and Safety-Aware Reasoning")
print(f"👀 Analyzing submission repository: {args.repo_id} 👀")

try:
    print(f"Logged in to HuggingFace as: {whoami()['name']}")
except Exception:
    print("⚠️⚠️ Not logged in to HuggingFace! Please get your login token from https://huggingface.co/settings/tokens 🌐")
    login()

client = Client("SimulaMet/MedVQA-GI-2026")
print("💓 Communicating with the Submission Server: Ping!")
result = client.predict(
    api_name="/refresh_page"
)
print(result)


hf_username = whoami()['name']
assert len(hf_username) > 0, "🚫 HuggingFace login failed for some reason"
current_timestamp = int(time.time())

snap_dir = snapshot_download(
    repo_id=args.repo_id, allow_patterns=[submission_file, "requirements.txt"])

if not os.path.isfile(os.path.join(snap_dir, submission_file)):
    raise FileNotFoundError(
        f"Submission file '{submission_file}' not found in the repository!")

if os.path.isfile(os.path.join(snap_dir, file_from_validation)):
    os.remove(os.path.join(snap_dir, file_from_validation))

print("📦 Making sure of the minimum requirements to run the script 📦")
sp.run(["python", "-m", "pip", "install", "-q"] + min_library, check=True)

if os.path.isfile(os.path.join(snap_dir, "requirements.txt")):
    print(
        f"📦 Installing requirements from the submission repo: {args.repo_id}/requirements.txt")
    sp.run(["python", "-m", "pip", "install", "-q", "-r",
            f"{snap_dir}/requirements.txt"], cwd=snap_dir, check=True)

if os.environ.get("_MEDVQA_CHALLENGE_EVALUATE_FLAG_", "FALSE") == "TRUE":
    # Patch submission file for challenge evaluation
    challenge_file = submission_file.replace(".py", "_challenge.py")
    submission_path = os.path.join(snap_dir, submission_file)
    challenge_path = os.path.join(snap_dir, challenge_file)
    with open(submission_path, "r", encoding="utf-8") as f:
        code = f.read()
    # Replace only the dataset string
    if "SimulaMet/Kvasir-VQA-test" in code:
        code = code.replace("SimulaMet/Kvasir-VQA-test",
                            "SimulaMet/Kvasir-VQA-private")
        code = code.replace('"validation"', '"test"')
        # Comment out specific lines
        lines = code.splitlines()
        for i, line in enumerate(lines):
            if ("huggingface.co/datasets" in line or
                re.search(r'^\s*prompt_to_real\s*=', line) or
                    re.search(r'^\s*jsons__\s*=', line)):
                if not line.lstrip().startswith("#"):
                    leading_ws = len(line) - len(line.lstrip())
                    lines[i] = line[:leading_ws] + "# " + line[leading_ws:]
        # Insert new code block after 'import requests'
        for i, line in enumerate(lines):
            if "import requests" in line:
                insert_idx = i + 1
                break
        else:
            insert_idx = None
        new_block = [
            'from huggingface_hub import hf_hub_download',
            'from pathlib import Path',
            'prompt_to_real = json.loads(Path(hf_hub_download("SimulaMet/Kvasir-VQA-private", "real_mapping", repo_type="dataset")).read_text(encoding="utf-8"))',
            'jsons__ = json.loads(Path(hf_hub_download("SimulaMet/Kvasir-VQA-private", "imagen-test", repo_type="dataset")).read_text(encoding="utf-8"))',
        ]
        if insert_idx is not None:
            lines[insert_idx:insert_idx] = new_block
        else:
            sys.exit("Expected `import requests` was not found in submission file for challenge patching.")
        code = "\n".join(lines)
        with open(challenge_path, "w", encoding="utf-8") as f:
            f.write(code)
        submission_file = challenge_file
        print(f"🔄 Challenge file created at: {challenge_path}")
    else:
        print(
            "⚠️ Challenge patch not applied: expected string not found in submission file.")
        sys.exit(
            "Please check the submission file for compatibility with challenge evaluation.")

print("🔍 Starting your script and loading submission details...")
sp.run(["python", f"{snap_dir}/{submission_file}"],
       cwd=snap_dir, check=True)
print(
    f"🎉 The submission script ran successfully, the intermediate files are at {snap_dir}")

if not MEDVQA_SUBMIT:
    print("\n You can now run medvqa validate_and_submit .... command to submit the task.")
else:
    print("🚀 Preparing for submission 🚀")
    file_path_to_upload = os.path.join(
        snap_dir, f"{hf_username}-_-_-{current_timestamp}-_-_-task2.json")
    shutil.copy(os.path.join(snap_dir, file_from_validation),
                file_path_to_upload)
    # add repo_id to the submission file
    with open(file_path_to_upload, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['repo_id'] = args.repo_id
    with open(file_path_to_upload, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    api = HfApi()
    api.update_repo_settings(args.repo_id, private=False)  # Make public
    api.update_repo_settings(
        args.repo_id, gated='manual')  # Enable gated access
    for user in HF_GATE_ACCESSLIST:
        try:
            grant_access(args.repo_id, user)  # Grant access
        except Exception as e:
            print(user, ":", e)
    print(
        f'''✅ {args.repo_id} model is now made public, but gated, and is shared with organizers.
        You should not make the model private or remove/update it until the competition results are announced.
        Feel free to re-submit the task if you change the model on the repository.
        We will notify you if there are any issues with the submission.
        ''')

    result = client.predict(
        file=handle_file(file_path_to_upload),
        api_name="/add_submission"
    )
    print({"User": hf_username, "Task": "task2",
           "Submitted_time": str(datetime.fromtimestamp(int(current_timestamp), tz=timezone.utc)) + " UTC"
           })
    print(result)
    print("Visit this URL to see the entry: 👇")
    Client("SimulaMet/MedVQA-GI-2026")

if os.environ.get("_MEDVQA_CHALLENGE_EVALUATE_FLAG_", "FALSE") == "TRUE":
    src_json = os.path.join(snap_dir, "predictions_2.json")
    if os.path.isfile(src_json):
        with open(src_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Remove 'debug' key if present
        data.pop("debug", None)
        # Rename 'public_scores' to 'challenge_scores' if present
        if "public_scores" in data:
            data["challenge_scores"] = data.pop("public_scores")
        # Get Team_Name from submission_info
        team_name = data.get("submission_info", {}).get(
            "Team_Name", "unknown_team")
        team_name_safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', team_name)
        out_json = os.path.join(os.getcwd(), f"task2_{team_name_safe}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Copied and processed predictions to: {out_json}")
    else:
        print("❌ predictions_2.json not found in snapshot directory!")
