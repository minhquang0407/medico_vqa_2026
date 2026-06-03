"""MediaEval Medico 2026 Task 2 submission metadata.

Fill the TODO values before uploading the Hugging Face repository. The official
submission artifact is `submission_task2.jsonl`; this file is intentionally
lightweight and should not load any model.
"""

SUBMISSION_INFO = {
    "Participant_Names": "Nguyễn Minh Quang",
    "Affiliations": "Independent",
    "Contact_emails": ["nmquang04072005@gmail.com"],
    "Team_Name": "Sweet&Sour",
    "Country": "Vietnam",
    "Notes_to_organizers": """
    Uses CATA-Final, the selected Task 1 CATA model initialized from
    qwen3b_curriculum_topo_adapter_full_full_continue and fine-tuned for one
    additional epoch on the released test split under the organizers' permitted
    setting. Task 2 answers are intended to match the CATA-Final Task 1
    predictions for the same Subtask 2 validation items. Explanations combine
    targeted self-probing, freshly regenerated lesion-prior and morphology/TDA
    heatmaps, structured evidence JSON, and reliability-style confidence
    estimates. Test-set fine-tuning is disclosed separately from the clean
    training-only CATA checkpoint.
    """,
}
