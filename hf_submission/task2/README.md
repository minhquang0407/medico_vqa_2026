# MediaEval Medico 2026 - Task 2 CATA Explanations

## Team Info

- **Team name:** Sweet&Sour
- **Member:** Nguyễn Minh Quang
- **Email:** nmquang04072005@gmail.com
- **Country:** Vietnam

## Submission Overview

This folder contains the Subtask 2 submission for **Clinician-Oriented
Multimodal Explanations in GI**. The explanation file is built to align with the
selected Task 1 model:

```text
Task 1 CATA: qwen3b_curriculum_topo_adapter_full_full_continue
```

The Task 2 answer field should match the Task 1 CATA prediction for the same
`val_id`, `img_id`, and `question` in the official Subtask 2 validation subset.

## Base VQA Model

Answers are generated with the selected **Task 1 CATA** checkpoint:

```text
Qwen2.5-3B-Instruct
+ LoRA r16 / alpha32
+ pretrained frozen ViT/timm image encoder
+ prior-guided OT prefix fusion
+ lesion prior mask
+ global morphology/topology features
+ patch-level TDA features
+ full TopoAdapter in the last 8 Qwen decoder layers
+ topo_mode=all
+ vision_pretrained=True
```

The main reportable Task 1 CATA checkpoint was trained on official training data
only:

```text
2 epochs on 30k train subset
+ 1 continuation epoch on full train split
+ no test-set fine-tuning for the main reported score
```

## Required Validation Subset

Subtask 2 uses the organizer-defined subset:

```python
from datasets import load_dataset, Image as HfImage

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

Every row in `submission_task2.jsonl` should correspond exactly to this subset.

## Output Format

Each JSONL row has the organizer-required fields:

```json
{
  "val_id": "0",
  "img_id": "UNIQUE_IMAGE_IDENTIFIER",
  "question": "Original question posed to the model.",
  "answer": "Prediction from Task 1 CATA.",
  "textual_explanation": "Clinician-oriented explanation.",
  "visual_explanation": [{
    "type": "heatmap",
    "data": "visuals/0000_heatmap.png",
    "description": "Region supporting the answer."
  }],
  "confidence_score": 0.67
}
```

## Explanation Method

For each image-question pair:

1. The selected Task 1 CATA model generates the primary answer.
2. The question is mapped to a coarse family, such as color, location, count,
   instrument, visible text, procedure, abnormality presence, or lesion attribute.
3. A small bank of targeted self-probing questions is selected for that family.
4. The same CATA model answers the probes.
5. Structural visual evidence is computed from:
   - lesion prior mask,
   - morphology/TDA patch evidence,
   - global structural features,
   - OT / structural attention diagnostics.
6. Probe agreement, structural evidence, artifact burden, and answer specificity
   are combined into a reliability-style confidence score.
7. A deterministic clinician-oriented explanation is written.

The explanation pipeline does not use ground-truth answers. Self-probing is used
to expose model-internal support and conflict, not to rewrite the primary answer.

## Visual Explanation

Each heatmap combines complementary evidence sources:

```text
lesion prior + morphology/TDA saliency + OT/structural attention + tissue support
```

Specular artifact regions reduce heatmap support to avoid over-emphasizing
reflections. Heatmap paths are relative to this folder, for example:

```text
visuals/0000_heatmap.png
```

## Confidence Score

`confidence_score` is a reliability estimate in `[0, 1]`, not a calibrated
clinical probability. It reflects signals such as:

- lesion-prior/topology agreement,
- evidence concentration,
- tissue coverage,
- specular artifact burden,
- answer specificity,
- question-answer compatibility,
- probe support,
- probe consistency or conflict.

Low confidence, diffuse evidence, artifacts, or contradictory probe answers
should trigger cautious wording in the textual explanation.

## Regenerating Task 2 from CATA-Final

To recreate all Task 2 outputs with the current CATA-Final checkpoint, run from
the repository root:

```powershell
python .\hf_submission\task2\generate_task2_cata_final.py `
  --checkpoint .\outputs\qwen3b_curriculum_topo_adapter_test_finetune_lr5e6_reset\checkpoints\last.pt `
  --output-jsonl submission_task2_cata_final.jsonl `
  --visual-dir visuals `
  --batch-size 4 `
  --overwrite-visuals true
```

For a smoke test before the full 1,500-row run:

```powershell
python .\hf_submission\task2\generate_task2_cata_final.py `
  --checkpoint .\outputs\qwen3b_curriculum_topo_adapter_test_finetune_lr5e6_reset\checkpoints\last.pt `
  --output-jsonl debug_task2.jsonl `
  --visual-dir visuals_debug `
  --limit 2 `
  --batch-size 1
```

After reviewing the generated JSONL and visuals, replace the official file:

```powershell
Copy-Item .\hf_submission\task2\submission_task2_cata_final.jsonl .\hf_submission\task2\submission_task2.jsonl -Force
```

## Debug / Validation

Before uploading or after modifying `submission_task2.jsonl`, run:

```bash
python validate_task2_submission.py
```

This checks:

- JSONL has exactly 1500 rows.
- `val_id` is sequential from 0 to 1499.
- `img_id` and `question` match the organizer-defined Subtask 2 validation set.
- Required fields are present.
- `confidence_score` is in `[0, 1]`.
- Heatmap paths referenced by `visual_explanation` exist.
- Potential explanation contradictions are reported for manual review.

If you have a Task 1 CATA predictions file, also run:

```bash
python validate_task2_submission.py --task1-predictions path/to/predictions_1.json
```

This additionally checks whether Task 2 `answer` matches Task 1 CATA predictions
for the same `img_id` and `question`.

## Files

```text
README.md                    This documentation
submission_task2.py           Team/submission metadata
submission_task2.jsonl        Official Subtask 2 JSONL output
validate_task2_submission.py  Local validation/debug checker
visuals/                      Heatmaps and evidence JSON files
```

## Safety Notes

The explanations are intended to support clinician review, not replace it.
Presence/absence questions are handled conservatively because subtle findings,
artifacts, and diffuse mucosal changes can be ambiguous. Any test-set fine-tuned
checkpoint, if used for leaderboard optimization, should be reported separately
and not described as held-out validation performance.
