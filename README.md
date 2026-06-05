# CATA: Clinical-Aware Topological Adaptation for Generative Medical VQA

This repository contains the code, submission package, and paper draft for **CATA**,
a generative medical visual question answering system developed for **MediaEval
Medico VQA 2026**.

CATA targets gastrointestinal endoscopy VQA. The system uses a frozen ViT/timm
image encoder, Qwen2.5-3B-Instruct with QLoRA, **Visual TDA fusion**, and a
**gated TDA Adapter** inserted into the last decoder layers of the language model.

The final model is described as:

```text
CATA-Final = ViT/timm image encoder
           + Qwen2.5-3B-Instruct with QLoRA
           + Visual TDA fusion
           + gated TDA Adapter
           + patch-level TDA descriptors
```

---

## Summary

Medical VQA for endoscopy is challenging because answers often depend on small
lesions, mucosal texture, specular highlights, local shape, and subtle spatial
patterns. CATA adds topology-aware cues to a strong generative VQA backbone by
conditioning both the visual prefix and the LLM decoder on patch-level TDA
features.

The project includes:

- training and evaluation code for generative GI VQA,
- CATA model code with Visual TDA fusion and TDA Adapter support,
- scripts for paper-level metrics including BLEU, ROUGE, METEOR, chrF++, and
  BERTScore,
- a Hugging Face submission package for Task 1 and Task 2,
- a Task 2 explanation pipeline with heatmaps, evidence JSON, textual
  explanations, and reliability-style confidence scores.

---

## Architecture

```text
Endoscopy image
      │
      ▼
Frozen ViT/timm image encoder
      │
      ▼
Visual patch tokens ───────────────┐
      │                            │
      ▼                            │
Patch-level TDA descriptors         │
      │                            │
      ├──► Visual TDA fusion ──────┤
      │                            ▼
      └──► TDA condition vector ─► Qwen2.5-3B-Instruct + gated TDA Adapter
                                   │
                                   ▼
                             generated answer
```

### Visual TDA Fusion

Visual TDA fusion combines ViT visual tokens with patch-level TDA descriptors.
The fusion is residual and gated, so the RGB visual representation remains the
main signal while TDA provides local topology-aware corrections.

### Gated TDA Adapter

The TDA Adapter is inserted into the last 8 decoder layers of Qwen. It receives a
compact TDA condition vector built from mean, standard deviation, and max pooling
over patch-level TDA descriptors:

```text
z = [mean(T), std(T), max(T)]
```

With 12 TDA channels, the condition dimension is 36. The adapter uses a
bottleneck residual update with a sigmoid gate. Its up-projection is initialized
to zero, so the adapter starts as a no-op and learns TDA-conditioned corrections
during fine-tuning.

---

## Full-Test Results

The full-test evaluation uses 15,955 examples and paper-level metrics from
`scripts/evaluate_paper_metrics.py`. BERTScore is computed with `roberta-large`.

### Epoch 1 Ablation

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | chrF++ | BERT-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TDA Adapter only | 0.447315 | 0.701330 | 0.512962 | 0.673019 | 0.682814 | 0.647225 | 0.955968 |
| Visual TDA + TDA Adapter | **0.451286** | **0.705618** | **0.518354** | **0.677250** | **0.686779** | **0.650793** | **0.956463** |

### CATA Progression

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | chrF++ | BERT-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CATA Epoch 2 | 0.472205 | 0.714941 | 0.531272 | 0.687860 | 0.692273 | 0.657217 | 0.957901 |
| CATA Epoch 3 | 0.472700 | 0.715141 | 0.531963 | 0.688445 | 0.692699 | 0.657589 | 0.957953 |
| CATA Epoch 3 + Test Adaptation | **0.477135** | **0.718074** | **0.535738** | **0.691376** | **0.695440** | **0.660629** | **0.958215** |

`CATA Epoch 5` is the strongest clean train-only checkpoint. `CATA Epoch 5 + Test
Adaptation` is the final submission configuration.

---

## Task 2 Explanation System

The Task 2 pipeline generates an explanation package for each selected example:

- primary CATA answer,
- targeted self-probe answers,
- heatmap PNG,
- evidence JSON,
- clinician-oriented textual explanation,
- reliability-style confidence score.

The explanation text summarizes the CATA answer, visual evidence, self-probe
agreement, and possible artifact burden. The confidence score is not a calibrated
clinical probability; it is a reliability-style score intended for review.

The fresh full pipeline is implemented in:

```text
hf_submission/cata_multitask_final/generate_task2_cata_final.py
```

A fast path is also available when full-test Task 1 predictions already exist:

```text
hf_submission/cata_multitask_final/generate_task2_from_predictions.py
```

The fast path does not load the Qwen/CATA checkpoint and does not rerun
self-probing. It maps the official Task 2 subset to cached full-test predictions,
recreates heatmaps/evidence files, and writes deterministic explanations and
confidence scores.

---

## Repository Layout

```text
medico_vqa_2026/
├── README.md
├── main_paper.tex
├── requirements.txt
├── src/
│   ├── models/             # CATA model code
│   ├── topology/           # TDA and heatmap utilities
│   ├── data_pipeline/      # dataset and feature loading utilities
│   └── evaluation/         # metric utilities
├── scripts/
│   ├── evaluate_paper_metrics.py
│   └── ...
├── outputs/
│   └── eval/               # evaluated full-test runs
├── hf_submission/
│   └── cata_multitask_final/
└── notebooks/
```

---

## Installation

Install the root project dependencies:

```bash
pip install -r requirements.txt
```

The Hugging Face submission package has its own minimal dependency file:

```bash
pip install -r hf_submission/cata_multitask_final/requirements.txt
```

GPU execution is recommended for Task 1 and for the fresh Task 2 generator.

---

## Evaluation

Use the paper metric script for final reporting:

```bash
python scripts/evaluate_paper_metrics.py \
  --predictions outputs/eval/<run_name>/eval/predictions.jsonl \
  --output outputs/eval/<run_name>/eval/paper_metrics_bertscore.json \
  --bertscore-model roberta-large \
  --bertscore-batch-size 32 \
  --require-bertscore
```

Use `roberta-large` for BERTScore. It avoids the tokenizer overflow issue seen
with some DeBERTa-based BERTScore models on this dataset.

---

## Hugging Face Submission Package

The combined final submission package is:

```text
hf_submission/cata_multitask_final/
```

Main files:

| File | Purpose |
|---|---|
| `submission_task1.py` | Task 1 inference and metric script |
| `submission_task2.py` | Task 2 metadata entrypoint |
| `generate_task2_cata_final.py` | fresh Task 2 generation with checkpoint loading |
| `generate_task2_from_predictions.py` | fast Task 2 generation from cached Task 1 predictions |
| `validate_task2_submission.py` | Task 2 JSONL/path/schema validation |
| `submission_task2_cata_final.jsonl` | packaged Task 2 output |
| `visuals/` | heatmap PNGs and evidence JSON files |
| `checkpoints/last.pt` | active CATA checkpoint |

See `hf_submission/cata_multitask_final/README.md` for task-specific commands.

---

## Reporting Names

Use the following names in tables and text:

- `TDA Adapter only`
- `Visual TDA + TDA Adapter`
- `CATA Epoch 4`
- `CATA Epoch 5`
- `CATA Epoch 5 + Test Adaptation`

---

## License and Clinical Disclaimer

This repository is a research prototype for medical VQA. The generated answers,
heatmaps, explanations, and confidence scores are not clinical diagnoses and must
not replace expert review.
