# Medico VQA 2026 — Structural Generative VQA

This repository contains a structural, topology-aware generative VQA pipeline for the
Medico VQA 2026 project. The current best-performing branch combines a frozen
vision encoder, a frozen Qwen2.5 instruction LLM, LoRA adapters, precomputed
structural/topological features, lesion priors, and prior-guided OT fusion.

> **Current best local checkpoint**
>
> `Qwen/Qwen2.5-3B-Instruct + A3 prior OT fusion + LoRA r=16 + 30k samples + 2 epochs + cosine scheduler`
>
> Validation on 2,000 test examples:
>
> - Token F1: **0.5942**
> - ROUGE-L: **0.5504**
> - BLEU-1: **0.5566**
> - BLEU-4: **0.1883**
> - Exact Match: **0.1115**
> - Yes/No answer accuracy: **0.8637**

---

## Repository layout

```text
medico_vqa_2026/
├── data/                              # Local only; ignored by git
│   ├── raw/Kvasir-VQA-x1/             # JSONL files and images
│   └── processed/structural_features/ # Precomputed structural .npz caches
│
├── notebooks/                         # Dataset/download notebooks and experiments
├── strategy/                          # Planning/ablation notebooks
│
├── scripts/
│   ├── train_structural_vqa_generative.py
│   ├── evaluate_structural_vqa_generative.py
│   ├── run_overnight_training.py
│   ├── run_full_a3_epoch_eval.py
│   ├── precompute_structural_features.py
│   └── run_structural_precompute_all.py
│
├── src/
│   ├── alignment/                     # Sinkhorn / OT utilities
│   ├── data_pipeline/                 # Dataset, JSONL parsing, structural cache loading
│   ├── evaluation/                    # Generative VQA metrics
│   ├── models/                        # Vision, text, fusion, structural VQA models
│   └── topology/                      # TDA, morphology, lesion prior extraction
│
├── requirements.txt
├── .gitignore
└── README.md
```

Generated outputs are intentionally ignored by git:

```text
data/
checkpoints/
eval/
logs/
visuals/
*.pt
*.npy
*.npz
*.png
*.jpg
```

---

## Main idea

The project studies whether structural priors help medical VQA answer generation.
The strongest variant so far is **A3 prior-guided OT fusion**:

1. Load endoscopy image and question.
2. Load precomputed structural features from `.npz` cache.
3. Encode image with a frozen pretrained vision backbone.
4. Encode/generate text with frozen Qwen2.5 + LoRA adapters.
5. Fuse visual tokens with structural/lesion priors through OT-guided prefix fusion.
6. Train only lightweight LoRA/projector/fusion parameters.

Important ablation variants:

| Variant | Meaning | Status |
|---|---|---|
| A0 | No OT, no OT fusion | baseline comparison |
| A3 | Prior OT fusion + OT loss | best family |
| A3b | Prior OT fusion, no OT loss | worse than A3 in current runs |
| A3 r16 | A3 with LoRA r=16, alpha=32 | current best |

---

## Environment setup

### Local Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install transformers peft timm accelerate sentencepiece protobuf scikit-learn nltk rouge-score
```

### Google Colab

```bash
%cd /content
!git clone https://github.com/minhquang0407/medico_vqa_2026.git
%cd /content/medico_vqa_2026

!pip install --upgrade pip
# Colab may include torchao==0.10.0, which is incompatible with recent PEFT.
!pip uninstall -y torchao
!pip install -r requirements.txt
!pip install transformers peft timm accelerate sentencepiece protobuf scikit-learn nltk rouge-score
```

After uninstalling `torchao`, restart the Colab runtime once:

```text
Runtime → Restart runtime
```

---

## Data layout

The training scripts expect this structure:

```text
data/raw/Kvasir-VQA-x1/
├── Kvasir-VQA-x1-train.jsonl
├── Kvasir-VQA-x1-test.jsonl
└── images/
    ├── *.jpg
    └── ...

data/processed/structural_features/
├── train_original_manifest.csv
├── test_original_manifest.csv
├── *.npz
└── ...
```

For Colab, unzip data into `/content/medico_vqa_2026/data`, not directly inside
Google Drive. Reading thousands of images from Drive is much slower.

Example Colab cell:

```bash
%cd /content/medico_vqa_2026

!mkdir -p data/raw data/processed
!unzip -q "/content/drive/MyDrive/medico_vqa_2026_shared/Kvasir-VQA-x1.zip" -d data/raw/
!unzip -q "/content/drive/MyDrive/medico_vqa_2026_shared/structural_features.zip" -d data/processed/

!find data -maxdepth 4 -type f | head -50
```

### Windows path portability

Some local JSONL/manifest files may contain absolute Windows paths such as:

```text
C:\Users\...\Kvasir-VQA-x1\images\image.jpg
```

The dataset loader handles these by falling back to filename basenames and by
remapping manifest cache paths to the local structural feature directory when the
original path does not exist.

---

## Structural feature precomputation

If the structural caches are missing, precompute them before training.

```powershell
python -m scripts.run_structural_precompute_all
```

Or run a single split manually:

```powershell
python -m scripts.precompute_structural_features `
  --jsonl data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-train.jsonl `
  --images-dir data/raw/Kvasir-VQA-x1/images `
  --output-dir data/processed/structural_features `
  --manifest-out data/processed/structural_features/train_original_manifest.csv
```

---

## Training

### Best current 30k recipe: A3 LoRA r=16

```powershell
python -m scripts.run_overnight_training `
  --plan a3_lora16_only `
  --stage tenk `
  --run-prefix overnight_qwen3b_a3_30k_2ep_cosine_lora16 `
  --llm-name-or-path Qwen/Qwen2.5-3B-Instruct `
  --tenk-samples 30000 `
  --tenk-epochs 2 `
  --tenk-eval-samples 2000 `
  --seed 42 `
  --lr-scheduler cosine `
  --warmup-steps 200
```

### Run A3b and A3 r16 sequentially

```powershell
python -m scripts.run_overnight_training `
  --plan option_a_b `
  --stage tenk `
  --run-prefix overnight_qwen3b_option_ab_30k_2ep_cosine `
  --llm-name-or-path Qwen/Qwen2.5-3B-Instruct `
  --tenk-samples 30000 `
  --tenk-epochs 2 `
  --tenk-eval-samples 2000 `
  --seed 42 `
  --lr-scheduler cosine `
  --warmup-steps 200
```

### Full training with per-epoch evaluation

Use this when training large/full datasets. It saves `epoch_1.pt`, `epoch_2.pt`,
and evaluates each epoch separately so `last.pt` is not blindly selected.

```powershell
python -m scripts.run_full_a3_epoch_eval `
  --run-prefix local_qwen3b_a3_full_2ep_lora16_lr5e5 `
  --llm-name-or-path Qwen/Qwen2.5-3B-Instruct `
  --epochs 2 `
  --eval-samples 2000 `
  --seed 42 `
  --lr 0.00005 `
  --lr-scheduler cosine `
  --warmup-steps 1000 `
  --batch-size 1 `
  --gradient-accumulation-steps 8 `
  --lora-r 16 `
  --lora-alpha 32
```

Colab A100 version:

```bash
%cd /content/medico_vqa_2026

!PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m scripts.run_full_a3_epoch_eval \
  --run-prefix colab_a100_qwen3b_a3_full_1ep_lora16_lr5e5_bs4 \
  --llm-name-or-path Qwen/Qwen2.5-3B-Instruct \
  --epochs 1 \
  --eval-samples 2000 \
  --seed 42 \
  --lr 0.00005 \
  --lr-scheduler cosine \
  --warmup-steps 1000 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --lora-r 16 \
  --lora-alpha 32 \
  --backup-dir "/content/drive/MyDrive/medico_vqa_2026_outputs"
```

If Colab runs out of memory, reduce batch size:

```bash
--batch-size 2 --gradient-accumulation-steps 4
```

---

## Evaluation

Evaluate a checkpoint manually:

```powershell
python -m scripts.evaluate_structural_vqa_generative `
  --checkpoint checkpoints/YOUR_RUN/last.pt `
  --jsonl data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-test.jsonl `
  --images-dir data/raw/Kvasir-VQA-x1/images `
  --structural-manifest data/processed/structural_features/test_original_manifest.csv `
  --batch-size 1 `
  --max-new-tokens 48 `
  --output-dir eval/YOUR_RUN_test2000 `
  --max-samples 2000
```

Evaluation outputs:

```text
eval/YOUR_RUN/
├── metrics.json
├── predictions.jsonl
└── qualitative.json
```

Important metrics to track:

```text
token_f1
rouge_l
bleu_1
bleu_4
exact_match
qtype_yes_no_answer_acc
qtype_multi_attribute_answer_acc
qtype_numerical_count_answer_acc
empty_answer
question_copy
out_of_domain
```

---

## Current experiment summary

| Run | Token F1 | ROUGE-L | BLEU-1 | BLEU-4 | EM | Notes |
|---|---:|---:|---:|---:|---:|---|
| A3 r8 30k 2ep cosine | 0.5841 | 0.5396 | 0.5473 | 0.1746 | 0.1060 | previous best |
| A3 r16 30k 2ep cosine | **0.5942** | **0.5504** | **0.5566** | **0.1883** | **0.1115** | current best |
| A3b 30k 2ep cosine | 0.5649 | 0.5203 | 0.5229 | 0.1531 | 0.0850 | no OT loss, worse |
| A3 r8 50k 1ep cosine | 0.5683 | 0.5239 | 0.5256 | 0.1721 | 0.0960 | better than 50k 2ep |
| A3 r8 50k 2ep cosine | 0.5478 | 0.5000 | 0.5054 | 0.1511 | 0.0790 | overtraining/drift |

Main conclusions:

1. A3 prior-guided OT fusion is stronger than no-OT variants.
2. LoRA r=16 improves over r=8.
3. More data is not automatically better without careful LR/epoch control.
4. Full training should save and evaluate each epoch separately.
5. Answer normalization/post-processing is a promising next step and does not
   require retraining.

---

## Colab notes

### Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Pull latest code

```bash
%cd /content/medico_vqa_2026
!git pull
```

### Common PEFT/torchao issue

If training fails with:

```text
ImportError: Found an incompatible version of torchao
```

fix with:

```bash
!pip uninstall -y torchao
```

Then restart runtime.

### Backup outputs

Always copy outputs to Drive before disconnecting Colab:

```bash
!mkdir -p /content/drive/MyDrive/medico_vqa_2026_outputs/checkpoints
!mkdir -p /content/drive/MyDrive/medico_vqa_2026_outputs/eval
!mkdir -p /content/drive/MyDrive/medico_vqa_2026_outputs/logs

!cp -r checkpoints/*colab* /content/drive/MyDrive/medico_vqa_2026_outputs/checkpoints/ || true
!cp -r eval/*colab* /content/drive/MyDrive/medico_vqa_2026_outputs/eval/ || true
!cp -r logs/overnight/*colab* /content/drive/MyDrive/medico_vqa_2026_outputs/logs/ || true
```

---

## Reproducibility

The training pipeline supports deterministic seeds:

```text
--seed 42
```

The seed is used for:

- Python random
- NumPy
- PyTorch
- CUDA
- Data split
- DataLoader shuffling

For fair ablation comparisons, always use the same seed, sample count, epoch
count, LR schedule, and evaluation set size.

---

## Deadline-oriented strategy

Given the 2026-06-01 deadline:

1. Keep `A3 r16 30k 2ep cosine` as the safe fallback.
2. Use available GPU time for `Qwen3B A3 r16 full/large` with per-epoch eval.
3. Do not run broad ablations unless they directly affect final submission.
4. Try Qwen7B only as an opportunistic experiment if compute remains.
5. Prioritize answer normalization and submission formatting after the final model
   candidate is selected.
