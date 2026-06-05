"""MediaEval Medico 2026 Task 2 submission metadata.

Fill the TODO values before uploading the Hugging Face repository. The official
submission artifact is `submission_task2.jsonl`; this file is intentionally
lightweight and should not load any model.
"""

SUBMISSION_INFO = {
    "Participant_Names": "Minh Quang Nguyen",
    "Affiliations": "Independent",
    "Contact_emails": ["nmquang04072005@gmail.com"],
    "Team_Name": "Sweet&Sour",
    "Country": "Vietnam",
    "Notes_to_organizers": """
    Uses CATA-Final, the selected Task 1 CATA architecture:
    Qwen2.5-3B-Instruct + QLoRA r16 + pretrained frozen ViT/timm visual encoder
    + TDA visual fusion + gated TDA TopoAdapter injected into the last
    8 Qwen decoder layers. Task 2 answers are
    intended to match the CATA-Final Task 1 predictions for the same Subtask 2
    validation items. Explanations combine targeted self-probing, freshly
    regenerated morphology/TDA + color/edge heatmaps, structured evidence JSON,
    artifact-burden checks, and reliability-style confidence estimates.
    """,
}
