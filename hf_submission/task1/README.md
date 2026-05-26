---
license: apache-2.0
tags:
  - medical-vqa
  - gastroenterology
  - endoscopy
  - mediaeval
  - medico-2026
  - qwen2.5
  - lora
  - optimal-transport
  - topological-data-analysis
  - structural-priors
---

# Medico 2026 Task 1 — A3 Structural VQA with Prior-Guided OT and TDA Features

This repository contains our **MediaEval Medico 2026 Task 1** submission for
**Medical Visual Question Answering in GI Endoscopy**.

The core idea is simple:

> Instead of asking a large language model to answer from a generic image
> embedding alone, we explicitly inject **clinically meaningful structural
> evidence** into the vision-language bridge: lesion priors, morphology maps,
> topology-inspired patch descriptors, and prior-guided Optimal Transport
> alignment.

The final system is a **Qwen2.5-3B generative VQA model** with a custom
structural visual encoder and A3-style OT prefix fusion.

```text
Endoscopy image
  ├─ ViT/timm visual patch encoder
  ├─ lesion prior mask
  ├─ morphology + topology-inspired patch features
  ├─ global structural summary features
  ↓
Structural visual fusion
  ↓
Prior-guided Optimal Transport alignment with question tokens
  ↓
Visual/OT prefix tokens
  ↓
Frozen Qwen2.5-3B-Instruct + LoRA
  ↓
Short clinical answer
```

---

## Why This Architecture Is Different

Most VQA systems follow a standard recipe:

```text
image encoder → projected image embedding → LLM → answer
```

That works, but it asks the model to learn every clinically relevant visual
bias implicitly. GI endoscopy images are challenging because subtle findings may
be defined by:

- focal redness,
- mucosal texture change,
- lesion-like morphology,
- specular artifacts,
- tissue/non-tissue regions,
- shape concentration,
- spatial location in the endoscopic field.

Our model makes these signals explicit.

### Key architectural ideas

1. **Lesion prior mask** tells the model *where to look*.
2. **TDA/morphology patch features** tell the model *what local structure looks
   abnormal or informative*.
3. **Global topology features** summarize the whole-frame structural context.
4. **Prior-guided Optimal Transport** aligns question tokens with visual regions
   using the lesion prior as visual mass.
5. **LoRA on a frozen Qwen2.5-3B-Instruct** keeps language ability stable while
   adapting to medical VQA.

This creates a bridge between handcrafted clinical visual cues and modern
language-model reasoning.

---

## Repository Layout

```text
.
├── submission_task1.py       # official Task 1 entry point
├── requirements.txt          # runtime dependencies
├── checkpoints/
│   └── final.pt              # final model checkpoint
└── src/
    ├── alignment/            # Sinkhorn OT router
    ├── models/               # structural generative VQA model
    ├── topology/             # lesion prior + morphology/TDA extractors
    └── postprocessing/       # minimal answer cleanup utility
```

The official validator runs:

```bash
python submission_task1.py
```

and expects:

```text
predictions_1.json
```

---

## High-Level Model Design

The model is a **structural generative VQA system**.

```text
Image + Question
      │
      ├── Image branch
      │     ├── frozen timm ViT patch encoder
      │     ├── lesion prior mask
      │     ├── morphology/topology patch tensor
      │     └── global structural vector
      │
      ├── Structural fusion
      │     ├── prior-gated visual patches
      │     ├── topology-enriched patch embeddings
      │     └── global structural token
      │
      ├── A3 OT fusion
      │     └── prior-guided alignment between visual tokens and question tokens
      │
      └── Language branch
            └── frozen Qwen2.5-3B-Instruct + LoRA adapters
```

Final answer generation is deterministic and short-form.

---

## Language Backbone

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Training mode: frozen base LLM
- Adaptation: LoRA
- LoRA rank: `16`
- LoRA alpha: `32`
- Typical target modules: `q_proj,v_proj`

The model does **not** use a fixed answer classifier. It generates natural text
answers with a causal LM objective.

Why this matters:

- classifier heads are brittle when answer wording varies;
- generative decoding can express short clinical phrases;
- freezing Qwen preserves language competence;
- LoRA provides parameter-efficient domain adaptation.

---

## Vision Backbone

- Backend: `timm`
- Input size: `224 × 224`
- Patch grid: `14 × 14`
- Patch count: `196`
- Vision hidden dimension: `768`
- Final polish stage: vision backbone frozen

The vision encoder creates patch tokens:

```text
image → patch_tokens [B, 196, 768]
```

These tokens are not passed directly to the LLM. They are first enriched with
structural evidence.

---

## Structural Evidence Pipeline

The model receives three structural tensors per image.

These are deterministic image-derived features, not manual annotations and not
ground truth labels.

### 1. Lesion Prior Mask — where to look

```text
prior_mask: [B, 14, 14]
```

Extractor:

```python
LesionPriorExtractor(
    image_size=(224, 224),
    grid_size=(14, 14),
    use_morphology=False,
)
```

Final prior formula:

```text
prior = 0.45 × red_map + 0.55 × center_map + 0.00 × morpho_map
```

This prior encodes two robust assumptions for GI endoscopy:

- abnormal mucosa often has red/pink chromatic contrast;
- endoscopic framing often places relevant tissue near the central field.

Important clarification:

```text
use_morphology=False
```

only means morphology is not mixed into the **prior mask**. The model still uses
morphology/TDA features through the separate `topo_features` tensor.

### 2. Patch Morphology/TDA Features — what each patch looks like

```text
topo_features: [B, 14, 14, 12]
```

Extractor:

```python
MorphologyTopologicalExtractor(
    image_size=(224, 224),
    grid_size=(14, 14),
)
```

Patch features:

```text
morph_score
red_gray_energy
red_excess_mean
lab_redness_mean
edge_mean
edge_std
local_contrast
specular_ratio
tissue_ratio
h0_like_components
h1_like_holes
patch_entropy
```

These features capture:

- color abnormality,
- redness in RGB and Lab space,
- local texture and edge strength,
- contrast,
- specular highlight burden,
- tissue coverage,
- connected-component-like H0 evidence,
- hole/ring-like H1 evidence,
- patch entropy.

This is where the topology-inspired part enters: the model is given structured
patch descriptors that approximate local component and hole patterns, which are
useful for irregular mucosal structures and lesion-like regions.

### 3. Global Structural Features — whole-image context

```text
global_features: [B, 8]
```

Global features:

```text
global_morph_energy
global_morph_max
global_red_excess_mean
global_lab_redness_mean
global_edge_mean
global_specular_ratio
topo_entropy
topo_concentration
```

These summarize whether structural evidence is diffuse, concentrated,
artifact-heavy, or globally prominent.

---

## Structural Visual Fusion

The model fuses visual and structural information before communicating with the
LLM.

Conceptually:

```text
fused_patch = prior-gated visual patch + projected TDA/morphology features
```

Inputs:

```text
patch_tokens:   [B, 196, 768]
prior_mask:     [B, 14, 14]
topo_features:  [B, 14, 14, 12]
```

Output:

```text
fused_tokens: [B, 196, 768]
```

The 8D global feature vector is separately projected into a global token:

```text
global_features [B, 8] → global_token [B, 1, 768]
```

Final visual context:

```text
visual_context = [global_token; fused_patch_tokens]
visual_context: [B, 197, 768]
```

This means the LLM does not receive a generic image vector. It receives a
structurally enriched visual prefix.

---

## A3 Prior-Guided Optimal Transport Fusion

A central part of the architecture is the A3-style OT router.

Enabled flags:

```text
use_ot = true
use_ot_fusion = true
ot_fusion_mode = prefix
use_prior_as_ot_target = true
```

The router aligns:

```text
visual_context tokens ↔ question token embeddings
```

Standard cross-attention-style fusion can attend everywhere. Our OT fusion adds
a stronger constraint: transport mass on the visual side is guided by the lesion
prior mask.

In effect:

```text
question tokens are encouraged to align with structurally plausible visual regions
```

The OT-aligned representation becomes additional prefix tokens for Qwen.

LLM input structure:

```text
[structural visual prefix] + [OT-aligned visual prefix] + [question prompt]
```

This is the main architectural difference from a plain vision-to-LLM projector.

---

## Training Recipe

Final run:

```text
Run B fixed: best 30k r16 epoch2 → full original polish lr5e-6
Checkpoint: checkpoints/final.pt
Local full-test token_f1: 0.604086
```

Important configuration:

```text
llm_name_or_path = Qwen/Qwen2.5-3B-Instruct
vision_backend = timm
vision_pretrained = false
freeze_vision_backbone = true
freeze_llm = true
use_lora = true
lora_r = 16
lora_alpha = 32
batch_size = 1
gradient_accumulation_steps = 8
epochs = 3
lr = 5e-6
weight_decay = 1e-4
lr_scheduler = cosine
warmup_steps = 250
seed = 42
```

Structural/OT configuration:

```text
use_ot = true
use_ot_fusion = true
ot_fusion_mode = prefix
ot_fusion_dropout = 0.10
use_prior_as_ot_target = true
use_topological_loss = true
use_prior_align_loss = true
use_global_topo_loss = true
use_patch_topo_loss = false
```

---

## Training Objective

The final model is trained primarily by answer generation, with structural and
OT regularization.

```text
loss = language_modeling_loss + 0.05 × ot_cost + topological_loss
```

For the final configuration:

```text
topological_loss ≈ 0.01 × global_topo_loss
```

because:

- the prior is already used directly as the OT visual target mass;
- patch topology reconstruction loss is disabled in the final run.

### Language modeling loss

Causal LM cross-entropy is applied only to answer tokens. Visual prefixes and
question prompt tokens are ignored in the label tensor.

### OT cost

The Sinkhorn OT router computes the cost of aligning question tokens and visual
context tokens.

### Global topology loss

The projected global structural token is asked to reconstruct the 8D global
structural feature vector:

```python
global_pred = global_topo_head(global_token)
global_topo_loss = mse(global_pred, global_features)
```

This keeps global morphology/topology information available to the language
model.

---

## Inference Behavior

Inference computes structural features online, so the model can process new
images without relying on a precomputed cache.

Runtime steps:

```text
1. load checkpoints/final.pt
2. load Qwen2.5-3B-Instruct and LoRA adapters
3. compute lesion prior + morphology/TDA features online
4. build structural visual prefix
5. perform prior-guided OT fusion
6. generate answer deterministically
7. write predictions_1.json
```

Decoding:

```text
do_sample = false
num_beams = 1
repetition_penalty = 1.15
no_repeat_ngram_size = 3
```

---

## Results

Observed public validator scores:

```text
BLEU:   0.3113
ROUGE1: 0.5940
ROUGE2: 0.3788
ROUGEL: 0.5576
METEOR: 0.5617
```

Local final-run reference:

```text
Full-test token_f1: 0.604086
```

---

## Important Note on Answer Cleanup

The model output is kept as close as possible to the generated answer.

Only minimal formatting cleanup is used, such as whitespace cleanup and removing
a leading `Answer:` prefix if produced. We intentionally avoid aggressive
question-type post-processing because it can change medical answer semantics and
was observed to reduce the score.

---

## Future Directions and Ablation Ideas

The submitted checkpoint uses the architecture described above. The following
ideas are natural extensions or ablations, but they are **not** claimed as part
of the submitted Task 1 checkpoint.

### Ablations worth studying

- **No structural prior**: remove `prior_mask` to measure how much explicit
  lesion localization helps compared with visual tokens alone.
- **No morphology/TDA tensor**: remove the 12D patch structural descriptor and
  keep only image patches plus the question.
- **Uniform OT target mass**: replace prior-guided OT with uniform visual mass to
  test whether lesion-prior routing improves question-region alignment.
- **No OT prefix**: use only the structural visual prefix and disable the
  OT-aligned prefix branch.
- **Classifier vs generative answer head**: compare the current Qwen generative
  setup against a fixed-answer classifier for frequent VQA labels.

### Potential extensions

- **Learned prior weighting**: replace fixed red/center prior weights with a
  small learned or question-aware gating module.
- **Question-aware OT mass**: combine lesion prior, TDA saliency, and question
  type so that color, count, location, and instrument questions route attention
  differently.
- **Multi-scale structural grids**: compute structural features at `7×7`,
  `14×14`, and `28×28` to capture both global location and finer lesion
  boundaries.
- **Patch topology reconstruction**: enable patch-level topology auxiliary loss
  to force fused visual tokens to preserve local TDA descriptors.
- **Endoscopy-specific visual pretraining**: pretrain or adapt the vision encoder
  on GI endoscopy frames before structural VQA training.
- **Uncertainty calibration**: estimate confidence from generation entropy,
  OT cost, prior/TDA agreement, and artifact burden.
- **Faithful Task 2 explanations**: use OT transport maps and TDA saliency as
  visual evidence for clinician-oriented explanations.

These directions follow from the same design principle: make clinically relevant
visual structure explicit instead of forcing the language model to infer it only
from a generic image embedding.

---

## Limitations

- This is a research submission for GI endoscopy VQA, not a clinical decision
  system.
- Structural features are deterministic priors and may be affected by artifacts,
  lighting, or unusual framing.
- The model generates concise answers; detailed interpretability is handled in
  the separate Task 2 submission.

Task 2 repository:

```text
minhquang47/medico2026-task2-explanations
```
