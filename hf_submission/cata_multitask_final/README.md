# CATA Multitask Final Submission

This folder is the combined Hugging Face submission package for **MediaEval
Medico VQA 2026**.

It contains:

- **Subtask 1:** generative GI visual question answering,
- **Subtask 2:** answer-aligned visual/textual explanations.

The submitted model is **CATA-Final**:

```text
Qwen2.5-3B-Instruct + QLoRA r=16 alpha=32
+ frozen pretrained ViT/timm image encoder
+ Visual TDA fusion
+ gated TDA Adapter in the last 8 Qwen decoder layers
+ topo_mode=tda_only
+ topo_dim=36
+ vision_pretrained=True
```

---

## Team Information

| Field | Value |
|---|---|
| Team | Sweet&Sour |
| Member | Minh Quang Nguyen |
| Email | 23110203@student.hcmus.edu.vn |
| Institution | University of Science, VNU-HCM |
| Country | Vietnam |

---

## Folder Layout

```text
cata_multitask_final/
├── README.md
├── requirements.txt
├── submission_task1.py
├── submission_task2.py
├── generate_task2_cata_final.py
├── generate_task2_from_predictions.py
├── validate_task2_submission.py
├── submission_task2_cata_final.jsonl
├── visuals/
├── checkpoints/
│   └── last.pt
└── src/
```

### Main Files

| File | Purpose |
|---|---|
| `submission_task1.py` | Loads the checkpoint and runs Task 1 inference/evaluation. |
| `submission_task2.py` | Lightweight metadata entrypoint for Task 2. |
| `generate_task2_cata_final.py` | Fresh Task 2 generation with checkpoint loading, self-probes, heatmaps, evidence JSON, explanations, and confidence. |
| `generate_task2_from_predictions.py` | Fast Task 2 generation from cached full-test Task 1 predictions. |
| `validate_task2_submission.py` | Validates Task 2 JSONL schema and visual/evidence paths. |
| `submission_task2_cata_final.jsonl` | Packaged Task 2 output. |
| `visuals/` | Heatmap PNGs and evidence JSON files used by Task 2. |
| `checkpoints/last.pt` | Active CATA-Final checkpoint. |

---

## Installation

From this folder:

```bash
pip install -r requirements.txt
```

Recommended environment:

- Python 3.10 or newer,
- CUDA GPU for Task 1 and fresh Task 2 generation,
- CPU is acceptable for validation and the fast Task 2 path, but heatmap creation
  can still take time.

---

## Model Configuration

The submission model uses TDA-only conditioning:

```text
topo_mode=tda_only
topo_dim=36
use_patch_topo_loss=True
vision_pretrained=True
```

The public description of the model is:

```text
CATA-Final = frozen ViT/timm image encoder
           + Qwen2.5-3B-Instruct with QLoRA
           + Visual TDA fusion
           + gated TDA Adapter
           + patch-level TDA descriptors
```

---

## Subtask 1: VQA Inference

Run Task 1 from this folder:

```bash
python submission_task1.py
```

The script loads:

```text
checkpoints/last.pt
```

and writes:

```text
predictions_1.json
```

Expected diagnostic lines include:

```text
topo_mode=tda_only topo_dim=36
Installed 8 TopoAdapters / 36 decoder layers
```

---

## Subtask 2 Dataset Definition

Task 2 uses the organizer-defined public subset:

```python
from datasets import Image as HfImage, load_dataset

ds = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"]
val_set_task2 = (
    ds.filter(lambda x: x["complexity"] == 1)
      .shuffle(seed=42)
      .select(range(1500))
      .add_column("val_id", list(range(1500)))
      .remove_columns(["complexity", "answer", "original", "question_class"])
      .cast_column("image", HfImage())
)
```

Both Task 2 generators use this same ordering.

---

## Subtask 2: Fresh Generation Path

Use this path when you want to regenerate everything from the checkpoint.

Smoke test:

```bash
python generate_task2_cata_final.py \
  --output-jsonl debug_task2.jsonl \
  --visual-dir visuals_debug \
  --limit 2 \
  --batch-size 1
```

Full generation:

```bash
python generate_task2_cata_final.py \
  --output-jsonl submission_task2.jsonl \
  --visual-dir visuals \
  --batch-size 4 \
  --overwrite-visuals true
```

If VRAM is limited, use:

```bash
--batch-size 1
```

The fresh generator creates:

- primary CATA-Final answers,
- targeted self-probe answers,
- heatmap PNG files,
- evidence JSON files,
- clinician-oriented textual explanations,
- reliability-style confidence scores.

---

## Subtask 2: Fast Path from Cached Predictions

Use this path when full-test Task 1 predictions are already available. It avoids
loading Qwen/CATA and is much faster.

Example:

```bash
python generate_task2_from_predictions.py \
  --predictions ../../outputs/eval/CATA_Epoch5_TestAdapt/eval/predictions.jsonl \
  --output-jsonl submission_task2.jsonl \
  --visual-dir visuals \
  --overwrite-visuals true
```

The fast path:

- loads the same Task 2 subset and order,
- maps each Task 2 example to a cached full-test prediction,
- regenerates heatmap PNGs and evidence JSON files,
- writes deterministic textual explanations,
- writes reliability-style confidence scores.

Important limitation: the fast path does **not** rerun Qwen/CATA inference and
does **not** run targeted self-probing. Use the fresh path when self-probe answers
must be regenerated from the model.

---

## Validate Task 2 Output

Validate the generated JSONL:

```bash
python validate_task2_submission.py --submission submission_task2.jsonl
```

For a local-only schema/path check without loading the Hugging Face dataset:

```bash
python validate_task2_submission.py \
  --submission submission_task2.jsonl \
  --skip-dataset-check
```

If Task 1 predictions are available, check consistency between Task 1 and Task 2
answers:

```bash
python validate_task2_submission.py \
  --submission submission_task2.jsonl \
  --task1-predictions predictions_1.json
```

---

## Task 2 JSONL Format

Each row follows this structure:

```json
{
  "val_id": "0",
  "img_id": "...",
  "question": "...",
  "answer": "CATA-Final answer",
  "textual_explanation": "Clinician-oriented explanation text.",
  "visual_explanation": [
    {
      "type": "heatmap",
      "data": "visuals/0000_heatmap.png",
      "description": "CATA-Final heatmap overlay."
    },
    {
      "type": "evidence_json",
      "data": "visuals/0000_evidence.json",
      "description": "Structured evidence statistics."
    }
  ],
  "confidence_score": 0.67
}
```

The confidence score is a reliability-style estimate. It is not a calibrated
clinical probability.

---

## Packaged Output

The packaged Task 2 file is:

```text
submission_task2_cata_final.jsonl
```

If a generated file is named `submission_task2.jsonl`, it can be copied or renamed
according to the required submission format.

---

## Reporting Names

Use these names in the paper and report:

- `TDA Adapter only`
- `Visual TDA + TDA Adapter`
- `CATA Epoch 4`
- `CATA Epoch 5`
- `CATA Epoch 5 + Test Adaptation`

---

## Clinical Disclaimer

This package is for research evaluation. Generated answers, explanations,
heatmaps, evidence JSON files, and confidence scores are intended to support
human review only and must not be used as standalone clinical decisions.
