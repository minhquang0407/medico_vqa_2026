# MediaEval Medico 2026 - Task 1

## Team Info

- **Team name:** Sweet&Sour
- **Member:** Nguyễn Minh Quang
- **Email:** nmquang04072005@gmail.com

## Model Overview

This submission uses **TopoFusion-Qwen3B**, a structural VQA system that combines a frozen ViT visual encoder, online lesion-prior/topological feature extraction, full structural fusion, and **Qwen2.5-3B-Instruct with QLoRA r16** plus deep TopoAdapter layers to generate textual answers.

## Architecture Summary

```text
Image
→ ViT visual encoder
→ online structural extraction:
   - lesion prior mask M_prior
   - topological feature map
   - global morphology features
→ structural visual fusion
→ Qwen2.5-3B-Instruct + QLoRA r16
→ deep TopoAdapter injection in the last 8 decoder layers
→ generated answer
```

TopoFusion-Qwen3B uses the full structural signal directly during inference. The lesion prior and topological descriptors are used both by the visual fusion module and by the deep TopoAdapter condition vector inside the Qwen decoder.

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

The system will load the checkpoint from:

```text
checkpoints/last.pt
```

It will run prediction on the Task 1 test split and write the output file:

```text
predictions_1.json
```

## Files

```text
submission_task1.py     Main inference script
requirements.txt        Python dependencies
checkpoints/last.pt     Trainable checkpoint weights
src/                    Model, topology, and post-processing code
```
