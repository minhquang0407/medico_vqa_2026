# MediaEval Medico 2026 - Task 1 CATA Submission

## Team Info

- **Team name:** Sweet&Sour
- **Member:** Nguyễn Minh Quang
- **Email:** nmquang04072005@gmail.com
- **Country:** Vietnam

## Model Overview

This Hugging Face model repository contains the selected **Task 1 CATA**
submission: a curriculum-gated, topology-aware generative VQA model based on
**Qwen2.5-3B-Instruct**.

The submitted checkpoint is:

```text
qwen3b_curriculum_topo_adapter_full_full_continue
```

Training summary:

- 2 epochs on a 30k subset of the official training split
- 1 continuation epoch on the full official training split
- no test-set fine-tuning for the main reported score
- `topo_mode=all`
- `vision_pretrained=True`
- TopoAdapter inserted into the last 8 Qwen decoder layers
- bottleneck dimension 32

Official-template validation on 1,500 shuffled test examples:

| Metric | Score |
|---|---:|
| BLEU | 0.4537 |
| ROUGE-1 | 0.6975 |
| ROUGE-2 | 0.5105 |
| ROUGE-L | 0.6711 |
| METEOR | 0.6748 |

## Architecture Summary

```text
Image
→ pretrained ViT/timm visual encoder, frozen
→ online structural extraction:
   - lesion prior mask
   - patch-level topological/TDA features
   - global morphology/topology features
→ structural visual fusion + prior-guided OT prefix fusion
→ Qwen2.5-3B-Instruct + LoRA r16/alpha32
→ curriculum-gated TopoAdapter in last 8 decoder layers
→ generated answer
```

CATA conditions the language decoder on full structural statistics derived from
lesion priors, global morphology, and patch-level topology. During training, the
curriculum first stabilizes language adaptation and then restores the full
spatial/topological alignment objectives and gate regularization. During
inference, `submission_task1.py` computes the same structural tensors online and
uses them for both visual fusion and TopoAdapter conditioning.

## Important Runtime Configuration

The submission script should print a diagnostic similar to:

```text
Runtime config: topo_mode=all topo_dim=47 vision_pretrained=True use_patch_topo_loss=True
Installed 8 TopoAdapters / 36 decoder layers | hidden=2048 topo_dim=47
Loaded checkpoint successfully. Status: OK
```

These values are important. In particular, `vision_pretrained=True` must be kept
to match the training configuration.

## Dependencies

Install the required packages with:

```bash
pip install -r requirements.txt
```

The script also loads the base LLM from Hugging Face:

```text
Qwen/Qwen2.5-3B-Instruct
```

## How to Run

From this folder, run:

```bash
python submission_task1.py
```

The system loads the checkpoint from:

```text
checkpoints/last.pt
```

It runs prediction on the Task 1 test split and writes:

```text
predictions_1.json
```

## File Layout

```text
submission_task1.py     Main inference/validation script
requirements.txt        Python dependencies for the HF model repo
checkpoints/last.pt     Trainable checkpoint weights
src/                    Model, topology, alignment, and feature extraction code
```

## Notes on Test-Set Fine-Tuning

The organizers stated that training on the full released test set is acceptable
for final competitive submissions if clearly reported. The score above is from
the cleaner checkpoint trained only on official training data. If a test-set
fine-tuned checkpoint is used for leaderboard optimization, it should be reported
separately and not described as held-out validation performance.
