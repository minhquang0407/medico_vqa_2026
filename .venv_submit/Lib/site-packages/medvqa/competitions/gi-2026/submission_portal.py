import sys
import subprocess
import gradio as gr
import json
from datetime import datetime, timezone
from huggingface_hub import upload_file, snapshot_download
import shutil
import os
import glob
from pathlib import Path
from tempfile import gettempdir
from huggingface_hub import whoami
import platform

print(subprocess.check_output(
    [sys.executable, "-m", "pip", "list"]).decode("utf-8"))
print({
    "python": platform.python_version(),
    "os": platform.system(),
    "platform": platform.platform(),
    "arch": platform.machine()
})
print("Account token used to connect to HuggingFace: ", whoami()['name'])


SUBMISSION_REPO = "SimulaMet/medvqa-gi-2026-submissions"
REFRESH_INTERVAL_SECONDS = 3600
hub_path = None
submissions = None
last_submission_update_time = datetime.now(timezone.utc)


def refresh_submissions():
    global hub_path, submissions, last_submission_update_time
    if hub_path and Path(hub_path).exists():
        shutil.rmtree(hub_path, ignore_errors=True)
        print("Deleted existing submissions")

    hub_path = snapshot_download(
        repo_type="dataset", repo_id=SUBMISSION_REPO, allow_patterns=['**/*.json'])
    print("Downloaded submissions to:", hub_path)
    if not os.path.exists(hub_path):
        os.makedirs(hub_path)

    all_jsons = glob.glob(hub_path + "/**/*.json", recursive=True)
    print("json_files count:", len(all_jsons))

    submissions = []
    for file in all_jsons:
        file_ = file.split("/")[-1]
        username, sub_timestamp, task = file_.replace(
            ".json", "").split("-_-_-")
        json_data = json.load(open(file))
        public_score = json.dumps(json_data.get("public_scores", {}))
        submissions.append({"user": username, "task": task, "public_score": public_score,
                           "submitted_time": sub_timestamp})

    last_submission_update_time = datetime.now(timezone.utc)
    return hub_path


hub_path = refresh_submissions()
hub_dir = hub_path.split("snapshot")[0] + "snapshot"


def time_ago(submitted_time):
    return str(datetime.fromtimestamp(int(submitted_time), tz=timezone.utc)) + " UTC"


def filter_submissions(task_type, search_query):
    if search_query == "":
        filtered = [s for s in submissions if task_type ==
                    "all" or s["task"] == task_type]
    else:
        filtered = [s for s in submissions if (
            task_type == "all" or s["task"] == task_type) and search_query.lower() in s["user"].lower()]
    return [{"user": s["user"], "task": s["task"], "public_score": s["public_score"], "submitted_time": time_ago(s["submitted_time"])} for s in filtered]


def display_submissions(task_type="all", search_query=""):
    if submissions is None or ((datetime.now(timezone.utc) - last_submission_update_time).total_seconds() > REFRESH_INTERVAL_SECONDS):
        refresh_submissions()
    filtered_submissions = filter_submissions(task_type, search_query)
    return [[s["user"], s["task"], s["submitted_time"], s["public_score"]] for s in filtered_submissions]


def add_submission(file):
    global submissions
    try:
        uploaded_path = Path(file).resolve()
        tmp_root = Path(gettempdir()).resolve()

        if not uploaded_path.is_file():
            raise FileNotFoundError("Uploaded file was not found")
        if uploaded_path.suffix.lower() != ".json":
            raise ValueError("Only .json files are accepted")
        if tmp_root not in uploaded_path.parents:
            raise ValueError("Invalid upload path")

        with uploaded_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        filename = uploaded_path.name
        username, sub_timestamp, task = filename.replace(
            ".json", "").split("-_-_-")
        submission_time = datetime.fromtimestamp(
            int(sub_timestamp), tz=timezone.utc)

        assert task in ["task1", "task2"], "Invalid task type"
        assert len(username) > 0, "Invalid username"
        assert submission_time < datetime.now(
            timezone.utc), "Invalid submission time"

        upload_file(
            repo_type="dataset",
            path_or_fileobj=str(uploaded_path),
            path_in_repo=task + "/" + filename,
            repo_id=SUBMISSION_REPO
        )
        refresh_submissions()
        return "💪🏆🎉 Submissions registered successfully to the system!"
    except Exception as e:
        return f"❌ Error adding submission: {e}"


def refresh_page():
    return "Pong! Submission server is alive! 😊"


# Define Gradio Interface
with gr.Blocks(title="🌟 ImageCLEFmed-MEDVQA-GI 2026 Submissions 🌟") as demo:
    gr.Markdown("""
# 🌟 Welcome to the official submission portal for the **[ImageCLEFmed-MEDVQA-GI 2026](https://www.imageclef.org/2026/medical/medvqa-gi)** challenge! 🏥🔍
### 📋 [**GitHub Repository**](https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2026) | 📝 [**ImageCLEF Registration**](https://www.imageclef.org/2026#registration)
### 📦 [**Available Datasets**](https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2026#data) | 🧠 [**Task Details**](https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2026#task-descriptions) | 📮 [**Schedule**](https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2026#preliminary-schedule)
---
""")

    with gr.Tab("View Submissions"):
        gr.Markdown("### Filter and Search Submissions")

        with gr.Row():
            with gr.Column(scale=1):
                task_type_dropdown = gr.Dropdown(
                    choices=["all", "task1", "task2"],
                    value="all",
                    label="Task Type"
                )
                search_box = gr.Textbox(
                    label="Search by Username",
                    placeholder="Enter username..."
                )

            with gr.Column(scale=6):
                output_table = gr.Dataframe(
                    headers=["User", "Task", "Submitted Time", "Public Score"],
                    interactive=False,
                    wrap=True,
                    column_widths=["100px", "50px", "80px", "200px"],
                    label="Submissions"
                )

        task_type_dropdown.change(
            fn=display_submissions,
            inputs=[task_type_dropdown, search_box],
            outputs=output_table
        )
        search_box.change(
            fn=display_submissions,
            inputs=[task_type_dropdown, search_box],
            outputs=output_table
        )

        gr.Markdown(
            f'''
            🔄 Last refreshed: {last_submission_update_time.strftime('%Y-%m-%d %H:%M:%S')} UTC |  📊 Total Submissions: {len(submissions)}

            💬 For any questions or issues, [contact the organizers](https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2026#organizers) or check the documentation in the [GitHub repo](https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2026).  Good luck and thank you for contributing to medical AI research! 💪🤖🌍
            ''')

    with gr.Tab("Upload Submission", visible=False):
        file_input = gr.File(label="Upload JSON", file_types=[".json"])
        upload_output = gr.Textbox(label="Upload Result")
        file_input.upload(fn=add_submission,
                          inputs=file_input, outputs=upload_output)

    with gr.Tab("Refresh API", visible=False):
        refresh_button = gr.Button("Refresh")
        status_output = gr.Textbox(label="Status")
        refresh_button.click(fn=refresh_page, inputs=[], outputs=status_output)

    demo.load(lambda: display_submissions("all", ""),
              inputs=[], outputs=output_table)

demo.launch()
