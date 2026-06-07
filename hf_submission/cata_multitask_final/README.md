# CATA-Final for MediaEval Medico VQA 2026

This repository contains the final Hugging Face submission package for **MediaEval Medico VQA 2026**.

- **Team:** Sweet&Sour
- **Participant:** Minh Quang Nguyen
- **Country:** Vietnam
- **Contact:** nmquang04072005@gmail.com

## Summary

**CATA** (*Clinical-Aware Topological Adaptation*) is a generative medical VQA system for gastrointestinal endoscopy images. The final submitted model uses a frozen visual encoder, a Qwen language model, and patch-level topological descriptors to improve answers that require morphology, location, count, size, and visual evidence.

The same model family is used for both tasks:

- **Task 1:** answer generation for GI visual question answering.
- **Task 2:** multimodal explanation package generation, including text explanations, targeted self-probes, heatmaps, evidence JSON files, and reliability-style confidence scores.

## Model Architecture

The submitted checkpoint is **CATA-Final / Epoch 5 + Test Adaptation**.

```text
CATA-Final
├── Visual encoder: pretrained frozen ViT/timm backbone
├── Language model: Qwen2.5-3B-Instruct
├── Parameter-efficient tuning: QLoRA, r=16, alpha=32, dropout=0.05
├── Visual TDA fusion: patch-level TDA descriptors fused into visual features
├── Decoder adaptation: gated TDA TopoAdapter in the last 8 Qwen decoder layers
└── TDA condition vector: 36 dimensions from patch-level mean/std/max statistics
```

Important implementation notes:

- `vision_pretrained=True` is enabled.
- LoRA targets are `q_proj` and `v_proj`.
- The selected CATA configuration uses **Visual TDA fusion + gated TDA Adapter**.
- OT routing, prior-guided OT fusion, prior-alignment loss, and global structural token are disabled in the final submission model.
- The lesion-prior tensor may still be produced by the common data pipeline, but it is not routed into the selected CATA-Final model path.

## Reported Scores

### Official 1,500-sample public submission

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR |
|---|---:|---:|---:|---:|---:|
| CATA-Final, Epoch 5 + Test Adaptation | **0.4776** | **0.7193** | **0.5357** | **0.6922** | **0.6976** |

### Full-test internal evaluation

The paper also reports internal full-test evaluation on 15,955 Kvasir-VQA-x1 test samples. In those experiments, all configurations use the same 90% train / 10% validation split and the same seed for fair comparison.

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | chrF++ | BERTScore-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline, ViT + Qwen | 0.354 | 0.613 | 0.410 | 0.570 | 0.586 | 0.565 | 0.942 |
| Visual TDA only | 0.373 | 0.646 | 0.442 | 0.604 | 0.620 | 0.587 | 0.947 |
| TDA Adapter only | 0.447 | 0.701 | 0.513 | 0.673 | 0.683 | 0.647 | 0.956 |
| Visual TDA + TDA Adapter | **0.451** | **0.706** | **0.518** | **0.677** | **0.687** | **0.651** | **0.956** |

For the final checkpoint sequence, Epoch 5 + Test Adaptation reaches BLEU 0.477, ROUGE-L 0.691, METEOR 0.695, chrF++ 0.661, and BERTScore-F1 0.958 on full-test.

> The leaderboard score and the full-test internal score are reported on different evaluation subsets and should not be treated as identical protocols.

## Repository Layout

```text
cata_multitask_final/
├── README.md
├── requirements.txt
├── submission_task1.py              # Task 1 inference script
├── submission_task2.py              # Task 2 metadata for organizers
├── generate_task2_cata_final.py     # Regenerates Task 2 JSONL + visual evidence
├── validate_task2_submission.py     # Validates Task 2 JSONL format and paths
├── submission_task2.jsonl           # Submitted Task 2 explanation file
├── visuals/                         # Heatmaps and evidence JSON files
├── checkpoints/
│   └── last.pt                      # CATA-Final checkpoint
└── src/                             # Model, topology, data, and runtime code
```

## Installation

A GPU environment is recommended. From this repository folder:

```bash
pip install -r requirements.txt
```

If using a clean environment, install PyTorch according to the local CUDA version before running the submission scripts.

## Task 1: Generate Answers

Run from the repository root:

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

Expected runtime diagnostics include messages similar to:

```text
Runtime config: patch_tda_dim=12 tda_condition_dim=36 vision_pretrained=True use_patch_tda=True
Installed 8 TopoAdapters / 36 decoder layers | hidden=2048 condition_dim=36
Loaded checkpoint successfully. Status: OK
```

## Task 2: Generate Multimodal Explanations

The repository already contains the submitted Task 2 file:

```text
submission_task2.jsonl
```

To regenerate Task 2 outputs, run:

```bash
python generate_task2_cata_final.py \
  --checkpoint checkpoints/last.pt \
  --output-jsonl submission_task2.jsonl \
  --visual-dir visuals \
  --batch-size 4 \
  --overwrite-visuals true
```

For a quick smoke test:

```bash
python generate_task2_cata_final.py \
  --checkpoint checkpoints/last.pt \
  --output-jsonl debug_task2.jsonl \
  --visual-dir visuals_debug \
  --limit 2 \
  --batch-size 1
```

If VRAM is limited, use `--batch-size 1`.

Task 2 uses the organizer-defined validation subset:

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

Each generated Task 2 row contains:

- `val_id`
- `img_id`
- `question`
- `answer`
- `textual_explanation`
- `visual_explanation`
- `confidence_score`

The generator also writes heatmaps and structured evidence files under `visuals/`.

## Validate Task 2 Submission

Validate the final JSONL file:

```bash
python validate_task2_submission.py --submission submission_task2.jsonl
```

For a local structural/path check without loading the Hugging Face dataset:

```bash
python validate_task2_submission.py \
  --submission submission_task2.jsonl \
  --skip-dataset-check
```

If `predictions_1.json` is available, answer consistency can also be checked:

```bash
python validate_task2_submission.py \
  --submission submission_task2.jsonl \
  --task1-predictions predictions_1.json
```

## Output Example

A Task 2 JSONL row has the following structure:

```json
{
  "val_id": "0",
  "img_id": "...",
  "question": "...",
  "answer": "CATA-Final prediction",
  "textual_explanation": "Clinician-oriented explanation based on the answer, visual evidence, and self-probes.",
  "visual_explanation": [
    {
      "type": "heatmap",
      "data": "visuals/0000_heatmap.png",
      "description": "CATA heatmap highlighting visually relevant regions."
    }
  ],
  "confidence_score": 0.67
}
```

## Intended Use and Limitations

CATA is intended for the MediaEval Medico VQA 2026 benchmark and research on endoscopy VQA/explainability. The Task 2 explanation package is designed to support review and error analysis.

Important limitations:

- The heatmap is a visual evidence map combining TDA saliency, color/texture, edge cues, and artifact checks; it is not a pure LLM attention map.
- Targeted self-probes are generated by the same model family and can inherit model bias.
- The confidence score is a reliability-style heuristic, not a calibrated clinical probability.
- Outputs are not medical advice and must not replace clinician judgment.

## Citation

If you use this repository, please cite the accompanying CATA MediaEval Medico VQA 2026 working notes once available.
