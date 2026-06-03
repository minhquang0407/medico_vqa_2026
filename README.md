# CATA — Clinical-Aware Topological Adaptation for Medical VQA

**CATA** is a topology-aware multimodal generative VQA system for gastrointestinal
endoscopy images. The project was developed for **MediaEval Medico VQA 2026** and
focuses on combining modern vision-language modeling with clinically meaningful
structural evidence.

Instead of relying only on image appearance, CATA augments a generative VQA model
with lesion priors, morphology-aware features, topological descriptors, and
lightweight decoder adapters that inject structural context into the language
model during answer generation.

---

## Why CATA?

Medical VQA is different from generic visual question answering. Endoscopy
questions often depend on small lesions, subtle mucosal texture, spatial
localization, artifacts, and anatomical context. CATA is designed around this
observation.

The system combines:

- **Pretrained visual understanding** from a frozen ViT/timm image encoder.
- **Generative medical-language reasoning** from Qwen2.5-3B-Instruct with LoRA.
- **Lesion-prior guidance** to focus on clinically relevant image regions.
- **Topology and morphology features** to describe structural patterns beyond raw
  RGB appearance.
- **Optimal-transport visual-text fusion** for prior-guided multimodal alignment.
- **TopoAdapter**, a lightweight topology-conditioned decoder adapter.
- **Clinician-oriented explanations** with heatmaps, structured evidence, and
  confidence estimates.

---

## Key Results

Evaluation on the official-template 1,500-sample test subset:

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR |
|---|---:|---:|---:|---:|---:|
| CATA-Clean | 0.4537 | 0.6975 | 0.5105 | 0.6711 | 0.6748 |
| CATA-Final | **0.4728** | **0.7185** | **0.5340** | **0.6913** | **0.6895** |

`CATA-Clean` is trained on the official training data only. `CATA-Final` is the
final submission model using the organizer-permitted final optimization setting.

---

## Architecture

```text
              Endoscopy Image                         Question
                    │                                    │
                    ▼                                    ▼
        ┌──────────────────────┐            ┌──────────────────────┐
        │ Frozen ViT/timm       │            │ Qwen2.5-3B-Instruct  │
        │ image encoder         │            │ + LoRA               │
        └──────────┬───────────┘            └──────────┬───────────┘
                   │                                   │
                   ▼                                   │
        Visual patch embeddings                        │
                   │                                   │
                   ▼                                   │
        ┌──────────────────────┐                       │
        │ Lesion prior +       │                       │
        │ topology/morphology  │                       │
        │ feature extraction   │                       │
        └──────────┬───────────┘                       │
                   │                                   │
                   ▼                                   │
        ┌──────────────────────┐                       │
        │ Prior-guided optimal │                       │
        │ transport fusion     │                       │
        └──────────┬───────────┘                       │
                   │                                   │
                   └───────────────┬───────────────────┘
                                   ▼
                  ┌────────────────────────────────────┐
                  │ Qwen decoder + TopoAdapter layers │
                  └────────────────┬───────────────────┘
                                   ▼
                            Generated Answer
```

---

## TopoAdapter

TopoAdapter is the main architectural contribution of this repository.

It is a **topology-conditioned gated bottleneck adapter** inserted into the final
Qwen decoder layers. It receives a compact structural condition vector derived
from lesion priors, local topological descriptors, and global morphology cues.

At a high level:

```text
structural condition → gating network
hidden states → bottleneck adapter → gated residual update
```

The residual projection is initialized to zero, meaning TopoAdapter starts as an
exact no-op and learns structural corrections during fine-tuning. This preserves
the pretrained language model behavior at initialization while allowing medical
image structure to influence generation after training.

Standalone implementation:

```text
src/models/topo_adapter.py
```

---

## Structural Evidence

CATA uses three complementary structural signals:

| Signal | Purpose |
|---|---|
| Lesion prior mask | highlights suspicious image regions |
| Patch-level topological descriptors | capture local structural complexity |
| Global morphology features | summarize image-wide visual structure |

These signals are used both for model conditioning and for generating visual
explanations.

---

## Task 2 Explanation System

For explanation-oriented VQA, CATA generates more than an answer. Each sample can
include:

- the primary CATA answer,
- a clinician-oriented textual explanation,
- heatmap-based visual evidence,
- structured evidence JSON,
- reliability-style confidence score,
- targeted self-probing to check answer consistency.

Example output structure:

```json
{
  "val_id": "0",
  "img_id": "...",
  "question": "What are the colors of the observed abnormalities?",
  "answer": "pink, red, and white lesions noted",
  "textual_explanation": "...",
  "visual_explanation": [
    {"type": "heatmap", "data": "visuals/0000_heatmap.png"},
    {"type": "evidence_json", "data": "visuals/0000_evidence.json"}
  ],
  "confidence_score": 0.95
}
```

The confidence score is a reliability estimate, not a calibrated clinical
probability.

---

## Repository Structure

```text
medico_vqa_2026/
├── src/
│   ├── alignment/          # optimal transport and alignment utilities
│   ├── data_pipeline/      # dataset and feature-cache loading
│   ├── evaluation/         # VQA metrics and scoring utilities
│   ├── models/             # CATA model and TopoAdapter implementation
│   └── topology/           # lesion prior, morphology, and topology extractors
│
├── scripts/                # training, evaluation, and preprocessing scripts
├── hf_submission/          # final submission packages and generation scripts
├── notebooks/              # exploratory notebooks
├── strategy/               # experiment notes and ablation planning
├── requirements.txt
└── README.md
```

---

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
pip install transformers peft timm accelerate sentencepiece protobuf scikit-learn nltk rouge-score
```

Prepare data under:

```text
data/raw/Kvasir-VQA-x1/
data/processed/structural_features/
```

Precompute structural features:

```bash
python -m scripts.run_structural_precompute_all
```

Train the CATA model:

```bash
python scripts/train_qwen3b_curriculum_topo_adapter.py --mode train
```

Evaluate a checkpoint:

```bash
python -m scripts.evaluate_structural_vqa_generative --checkpoint path/to/last.pt
```

---

## Main Components

| Component | File / folder |
|---|---|
| CATA generative VQA model | `src/models/structural_vqa_generative.py` |
| TopoAdapter | `src/models/topo_adapter.py` |
| Structural feature extraction | `src/topology/` |
| Optimal transport alignment | `src/alignment/` |
| Training pipeline | `scripts/train_qwen3b_curriculum_topo_adapter.py` |
| Evaluation utilities | `src/evaluation/` |
| Final submission package | `hf_submission/cata_multitask_final/` |

---

## Research Directions

Future extensions include:

- deep ensembles for stronger uncertainty estimation,
- snapshot ensembles across training epochs,
- stochastic consistency with MC dropout,
- calibrated confidence estimation,
- Bayesian or variational LoRA for posterior uncertainty over task-specific
  adapter weights.

---

## Suggested Paper Name

```text
CATA: Clinical-Aware Topological Adaptation for Generative Medical VQA
```

Short description:

```text
CATA augments a Qwen2.5-based medical VQA model with lesion-prior guided
optimal-transport fusion and topology-conditioned decoder adapters, enabling
endoscopy question answering with morphology-aware structural evidence.
```
