from gradio_client import Client, handle_file
from huggingface_hub import snapshot_download, login, whoami
import argparse
import os
import subprocess as sp
import time
from datetime import datetime, timezone
import shutil  # Add this import
import json
from huggingface_hub import HfApi, grant_access
import re

HF_GATE_ACESSLIST = ["SushantGautam",
                     "stevenah", "vlbthambawita"]

MEDVQA_SUBMIT = True if os.environ.get(
    '_MEDVQA_SUBMIT_FLAG_', 'FALSE') == 'TRUE' else False
parser = argparse.ArgumentParser(description='Run GI-1015 Task 1 (VQA)')
parser.add_argument('--repo_id', type=str, required=True,
                    help='Path to the HF submission repository')
args, _ = parser.parse_known_args()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
submission_file = "submission_task1.py"
file_from_validation = "predictions_1.json"

min_library = ["datasets>=3.4.1", "transformers", "evaluate",
               "rouge_score", 'tqdm', "gradio_client>=1.8.0"]

print("🌟 ImageCLEFmed-MEDVQA-GI-2025 🌟",
      "https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2025")
print("🔍 Subtask 1: Algorithm Development for Question Interpretation and Response")
print(f"👀 Analyzing submission repository: {args.repo_id} 👀")

try:
    print(f"Logged in to HuggingFace as: {whoami()['name']}")
except Exception:
    print("⚠️⚠️ Not logged in to HuggingFace! Please get your login token from https://huggingface.co/settings/tokens 🌐")
    login()

client = Client("SimulaMet/medvqa")
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
    # Use regex to match the line, ignoring whitespace
    pattern = r'val_dataset\s*=\s*load_dataset\(\s*["\']SimulaMet/Kvasir-VQA-test["\']\s*,\s*split\s*=\s*["\']validation["\']\s*\)'
    new_line = 'val_dataset = load_dataset("SimulaMet/Kvasir-VQA-private", split="test")'
    if re.search(pattern, code):
        code = re.sub(pattern, new_line, code)
        with open(challenge_path, "w", encoding="utf-8") as f:
            f.write(code)
        submission_file = challenge_file
        print(f"🔄 Challenge file created at: {challenge_path}")
    else:
        print("⚠️ Challenge patch not applied: expected line not found in submission file.")
        os.exit(
            "Please check the submission file for compatibility with challenge evaluation.")


sp.run(["python", f"{snap_dir}/{submission_file}"],
       cwd=snap_dir, check=True)
print(
    f"🎉 The submission script ran successfully, the intermediate files are at {snap_dir}")

if not MEDVQA_SUBMIT:
    print("\n You can now run medvqa validate_and_submit .... command to submit the task.")
else:
    print("🚀 Preparing for submission 🚀")
    file_path_to_upload = os.path.join(
        snap_dir, f"{hf_username}-_-_-{current_timestamp}-_-_-task1.json")
    shutil.copy(os.path.join(snap_dir, file_from_validation),
                file_path_to_upload)  # Use shutil.copy here
    # add repo_id to the submission file
    with open(file_path_to_upload, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data['repo_id'] = args.repo_id
        with open(file_path_to_upload, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    api = HfApi()
    api.update_repo_visibility(args.repo_id, private=False)  # Make public
    api.update_repo_settings(
        args.repo_id, gated='manual')  # Enable gated access
    for user in HF_GATE_ACESSLIST:
        try:
            grant_access(args.repo_id, user)  # Grant access
        except Exception as e:
            print(user, ":", e)
    print(
        f'''✅ {args.repo_id} model is now made public, but gated, and is shared with organizers.
        You should not make the model private or remove/update it until the competition results are announced.
        Feel feel to re-submit the task if you change the model on the repository.
        We will notify you if there are any issues with the submission.
        ''')

    result = client.predict(
        file=handle_file(file_path_to_upload),
        api_name="/add_submission"
    )
    print({"User": hf_username, "Task": "task1",
           "Submitted_time": str(datetime.fromtimestamp(int(current_timestamp), tz=timezone.utc)) + " UTC"
           })
    print(result)
    print("Visit this URL to see the entry: 👇")
    Client("SimulaMet/medvqa")


if os.environ.get("_MEDVQA_CHALLENGE_EVALUATE_FLAG_", "FALSE") == "TRUE":
    src_json = os.path.join(snap_dir, "predictions_1.json")
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
        out_json = os.path.join(os.getcwd(), f"task1_{team_name_safe}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Copied and processed predictions to: {out_json}")
    else:
        print("❌ predictions_1.json not found in snapshot directory!")
    # === End: Post-processing predictions_1.json ===
