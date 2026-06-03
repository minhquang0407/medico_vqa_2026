# MediaEval Medico 2026 - CATA Multitask Final

## Team Info

- **Team name:** Sweet&Sour
- **Member:** Nguyễn Minh Quang
- **Email:** nmquang04072005@gmail.com
- **Country:** Vietnam

## Overview

This Hugging Face repository is the combined final submission package for:

- **Subtask 1:** GI visual question answering.
- **Subtask 2:** clinician-oriented multimodal explanations.

Both tasks use the same CATA model family:

```text
CATA-Final
Qwen2.5-3B-Instruct + LoRA r16/alpha32
+ pretrained frozen ViT/timm image encoder
+ prior-guided OT prefix fusion
+ lesion prior mask
+ global morphology/topology features
+ patch-level TDA features
+ TopoAdapter in the last 8 Qwen decoder layers
+ topo_mode=all
+ vision_pretrained=True
```

## Checkpoints

The repository expects the active checkpoint at:

```text
checkpoints/last.pt
```

Recommended naming in the report:

| Name | Use | Notes |
|---|---|---|
| **CATA-Clean** | clean reportable model | trained only on official training data |
| **CATA-Final** | final leaderboard model | initialized from CATA-Clean and fine-tuned for one additional epoch on the released test split under the organizers' permitted setting |

Current known scores on the official-template 1,500-sample public evaluation:

| Checkpoint | Test fine-tune | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR |
|---|---:|---:|---:|---:|---:|---:|
| CATA-Clean | No | 0.4537 | 0.6975 | 0.5105 | 0.6711 | 0.6748 |
| CATA-Final | Yes | **0.4728** | **0.7185** | **0.5340** | **0.6913** | **0.6895** |

> **Note:** If `checkpoints/last.pt` is the test-set fine-tuned checkpoint, report
> it as CATA-Final and do not describe it as held-out validation performance.

## File Layout

```text
cata_multitask_final/
├── README.md
├── requirements.txt
├── submission_task1.py              # Task 1 inference/validation script
├── submission_task2.py              # Task 2 metadata
├── generate_task2_cata_final.py     # Generates fresh Task 2 JSONL + visuals
├── validate_task2_submission.py     # Validates Task 2 JSONL/schema/paths
├── submission_task2.jsonl           # Final Task 2 output after generation
├── visuals/                         # Task 2 heatmaps and evidence JSON
├── checkpoints/
│   └── last.pt
└── src/                             # Model/topology/alignment source code
```

## Task 1 Usage

From this folder:

```bash
python submission_task1.py
```

This writes:

```text
predictions_1.json
```

The Task 1 script loads:

```text
checkpoints/last.pt
```

and should print diagnostics similar to:

```text
Runtime config: topo_mode=all topo_dim=47 vision_pretrained=True use_patch_topo_loss=True
Installed 8 TopoAdapters / 36 decoder layers | hidden=2048 topo_dim=47
Loaded checkpoint successfully. Status: OK
```

## Task 2 Generation

Task 2 must use the organizer-defined validation subset:

```python
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

Generate a small smoke run first:

```bash
python generate_task2_cata_final.py \
  --output-jsonl debug_task2.jsonl \
  --visual-dir visuals_debug \
  --limit 2 \
  --batch-size 1
```

Generate the full Task 2 submission:

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

The generator recreates:

- CATA-Final primary answers,
- CATA-Final targeted self-probe answers,
- heatmap PNGs,
- evidence JSON files,
- clinician-oriented textual explanations,
- reliability-style confidence scores.

## Task 2 Validation

Validate the generated file:

```bash
python validate_task2_submission.py --submission submission_task2.jsonl
```

For a local-only structural/path check without loading the HF dataset:

```bash
python validate_task2_submission.py --submission submission_task2.jsonl --skip-dataset-check
```

If `predictions_1.json` is available and covers the same `img_id`/`question`
items, answer consistency can also be checked:

```bash
python validate_task2_submission.py \
  --submission submission_task2.jsonl \
  --task1-predictions predictions_1.json
```

## Task 2 Output Format

Each JSONL row contains:

```json
{
  "val_id": "0",
  "img_id": "...",
  "question": "...",
  "answer": "CATA-Final prediction",
  "textual_explanation": "Clinician-oriented explanation",
  "visual_explanation": [{
    "type": "heatmap",
    "data": "visuals/0000_heatmap.png",
    "description": "Fresh CATA-Final heatmap..."
  }],
  "confidence_score": 0.67
}
```

The confidence score is a reliability estimate, not a calibrated clinical
probability. Explanations are intended to support clinician review, not replace
clinical judgment.
