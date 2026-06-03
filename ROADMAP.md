# Long-Term Research Roadmap — Medico VQA 2026 / CATA

This document records the long-term research plan after the MediaEval Medico VQA
2026 submission. The goal is to turn the current CATA system into a sequence of
publishable research works, each with a clear contribution, evaluation strategy,
and code direction.

---

## 0. Current Paper: MediaEval System Paper

### Role

Short workshop/system paper for the MediaEval Medico VQA 2026 benchmark.

### Scope

This paper should describe the submitted CATA system honestly and concisely:

- Qwen2.5-3B-Instruct + LoRA generative VQA backbone.
- Frozen pretrained ViT/timm vision encoder.
- Lesion-prior guided structural fusion.
- Patch-level topology and morphology features.
- Prior-guided Sinkhorn/OT alignment.
- TopoAdapter inserted into the final decoder layers.
- Task 2 explanation generation with heatmaps, evidence JSON, self-probing, and
  reliability-style confidence.

### Important Writing Rule

Do **not** overclaim in the MediaEval paper.

Use careful wording:

- `encourages`, `conditions`, `modulates`, `biases`, `regularizes`

Avoid hard claims:

- `forces`, `guarantees`, `completely solves`, `overrides`, `absolute symbolic reasoning`

### Target Claim

> CATA is a structure-aware medical VQA system that combines pretrained
> vision-language modeling with topology/morphology cues and explanation-oriented
> post-processing.

### Required Additions Before Submission

- [ ] Rewrite `main_paper.tex` in English.
- [ ] Use a workshop/system-paper style.
- [ ] Add Task 1 results table.
- [ ] Add Task 2 qualitative examples.
- [ ] Add at least Group A baseline if time permits.
- [ ] Correct the TopoAdapter, OT, and loss formulas to match implementation.

---

## 1. Paper 1: Structural Features for Vision Transformers

### Tentative Title

**Structural Feature Fusion for Vision Transformers in Gastrointestinal Medical VQA**

Alternative:

**Topology- and Morphology-Aware Vision Transformer Fusion for Endoscopic Understanding**

### Target Type

Conference paper, aiming for A/A* or strong medical-imaging venue.

Possible venues:

- MICCAI
- MIDL
- ISBI
- WACV
- BMVC
- ACM Multimedia workshop/main track if multimodal angle is strong

### Main Research Question

Can structural features improve pretrained ViT representations for endoscopic
medical VQA and related visual understanding tasks?

### Main Novelty

This paper focuses only on the **vision side** of the system.

The core claim:

> Lesion priors, patch-level topological descriptors, and global morphology
> features provide complementary structural signals that improve ViT-based
> endoscopic representations beyond raw RGB patch embeddings.

### Components to Emphasize

Relevant code areas:

- `src/topology/lesion_prior.py`
- `src/topology/tda_morphology.py`
- `src/topology/tda_extractor.py`
- `src/models/vision_encoder.py`
- `src/models/fusion.py`
- `scripts/precompute_structural_features.py`
- `scripts/run_pretrained_vision_ablation.py`

### Possible Model Variants

| Variant | Description |
|---|---|
| ViT only | pretrained ViT/timm features only |
| ViT + lesion prior | spatial prior used as additional visual cue |
| ViT + global morphology | image-level morphology output fusion |
| ViT + patch TDA | local topological feature fusion |
| ViT + all structural features | full structural visual representation |

### Experiments Needed

- [ ] ViT-only baseline.
- [ ] ViT + lesion prior.
- [ ] ViT + morphology.
- [ ] ViT + patch TDA.
- [ ] ViT + all structural signals.
- [ ] Visualization of lesion priors and structural feature maps.
- [ ] Question-type analysis, especially lesion/count/location questions.

### Metrics

Possible metrics:

- downstream VQA BLEU / ROUGE-L / METEOR,
- token F1,
- count-question accuracy,
- localization agreement if masks/proxies exist,
- retrieval/classification proxy metrics if additional labels are available.

### What Not to Emphasize

Do not make TopoAdapter the main contribution here. Mention the language model
only as a downstream evaluator if needed.

---

## 2. Paper 2: TDA Adapter + OT for LLM Decoding

### Tentative Title

**Topology-Conditioned Decoder Adapters for Generative Medical Visual Question Answering**

Alternative:

**TDA-Guided Optimal Transport and Decoder Adaptation for Medical VQA**

### Target Type

Conference paper, aiming for A/A* or strong AI/multimodal/NLP/medical venue.

Possible venues:

- ACM Multimedia
- AAAI
- ACL Findings / EMNLP Findings
- COLING
- MICCAI
- MIDL

### Main Research Question

Can topology-conditioned decoder adaptation improve structural reasoning in
generative medical VQA?

### Main Novelty

This paper focuses on the **LLM decoder side**:

- TopoAdapter.
- Topology-conditioned hidden-state modulation.
- Prior-guided OT alignment.
- Gate regularization.
- Curriculum training.

### Core Claim

> Injecting topology/morphology conditions into decoder layers through a
> zero-initialized gated residual adapter improves structural reasoning in
> generative medical VQA.

### Components to Emphasize

Relevant code areas:

- `src/models/topo_adapter.py`
- `src/models/structural_vqa_generative.py`
- `src/alignment/`
- `hf_submission/task1_cata/src/alignment/sinkhorn_ot.py`
- `scripts/train_qwen3b_curriculum_topo_adapter.py`
- `scripts/run_group_a_baseline.py`

### Correct TopoAdapter Formula

Use this in the paper:

```latex
\begin{align}
\mathbf{t} &= W_2 \, \mathrm{SiLU}(W_1 \mathbf{z}), \\
\mathbf{g} &= \sigma(W_g \mathbf{t}), \\
\Delta \mathbf{h}_l &= W_{\mathrm{up}}
\left(W_{\mathrm{down}}\mathbf{h}_l \odot \mathbf{g}\right), \\
\mathbf{h}'_l &= \mathbf{h}_l + \Delta \mathbf{h}_l .
\end{align}
```

Where:

- `z` is the topology condition vector.
- `g` is the gate.
- `h_l` is the hidden state at decoder layer `l`.
- The update is a residual modulation, not a hard override.

### Correct Gate Loss

Use this instead of an MSE gate-global formula unless CATA-v2 implements that:

```latex
\mathcal{L}_{gate}
=
\frac{1}{N}
\sum_{i=1}^{N}
g_i(1-g_i).
```

### Correct Total Loss

A safe paper formula:

```latex
\mathcal{L}_{total}
=
\mathcal{L}_{CE}
+
\lambda_{OT}\mathcal{L}_{OT}
+
\lambda_{struct}\mathcal{L}_{struct}
+
\lambda_{gate}\mathcal{L}_{gate}.
```

### Experiments Needed

| Variant | Purpose |
|---|---|
| Group A: ViT + Qwen LoRA | non-structural baseline |
| OT-only | tests prior-guided alignment |
| TopoAdapter-only | tests decoder structural conditioning |
| TopoAdapter patch-only | tests local TDA condition |
| TopoAdapter all | full structural condition |
| No gate loss | tests gate regularization |
| No curriculum | tests training stability |
| CATA full | final proposed method |

### Diagnostics to Add

- [ ] Gate mean and gate distribution.
- [ ] Question-type improvement.
- [ ] Count/location/lesion subset results.
- [ ] Qualitative examples where structural conditioning changes the answer.
- [ ] OT/prior alignment visualization.

### What Not to Claim Unless Implemented

Do not claim direct decoder-attention regularization unless the model actually
uses `output_attentions=True` and computes a loss over LLM attention maps.

Use:

> prior-guided visual fusion/alignment

Do not use:

> directly forces LLM attention to look at lesion coordinates

---

## 3. Paper 3: CATA-v2 Unified Q1 Journal Paper

### Tentative Title

**CATA-v2: Unified Clinical-Aware Topological Adaptation for Explainable Medical VQA**

Alternative:

**CATA-v2: Structure-Supervised Vision-Language Adaptation for Gastrointestinal VQA**

### Target Type

Q1 journal paper.

Possible journals:

- Medical Image Analysis
- IEEE Transactions on Medical Imaging
- IEEE Journal of Biomedical and Health Informatics
- Information Fusion
- Pattern Recognition
- Artificial Intelligence in Medicine
- Computer Methods and Programs in Biomedicine
- Expert Systems with Applications
- Biomedical Signal Processing and Control

### Role

This paper unifies Paper 1 and Paper 2, but must add genuinely new contributions.
It should not be only a combination of previous papers.

### Required New Contributions

CATA-v2 should add at least two of the following:

1. **Supervised Gate Alignment**
2. **Question-aware Structural Loss Weighting**
3. **Direct Decoder Attention-Prior Alignment**
4. **Deep Ensemble or Stochastic Uncertainty**
5. **Calibrated Confidence Estimation**
6. **Faithfulness Evaluation for Explanations**
7. **Multi-dataset Validation**

### 3.1 Supervised Gate Alignment

Current CATA gate loss only encourages decisive gates:

```latex
\mathcal{L}_{gate}=\frac{1}{N}\sum_i g_i(1-g_i)
```

CATA-v2 can additionally supervise the gate to encode global morphology:

```latex
\hat{\mathbf{f}}_{global}
=
W_o\,\mathrm{Pool}(\mathbf{g})
```

```latex
\mathcal{L}_{gate-align}
=
\left\|\hat{\mathbf{f}}_{global} - \mathbf{f}_{global}\right\|_2^2.
```

This requires code implementation before claiming it.

Possible file:

```text
src/models/topo_adapter_v2.py
```

### 3.2 Question-aware Structural Weighting

Some question types need stronger topology supervision:

- count,
- location,
- lesion/abnormality,
- morphology.

Other types may need weaker topology weighting:

- visible text,
- instrument,
- procedure,
- color.

Potential formula:

```latex
\mathcal{L}_{total}
=
\mathcal{L}_{CE}
+
w_q\lambda_{struct}\mathcal{L}_{struct}
+
\lambda_{OT}\mathcal{L}_{OT}
+
\lambda_{gate}\mathcal{L}_{gate}.
```

Where `w_q` is determined by question family.

### 3.3 Direct Decoder Attention-Prior Alignment

If implemented, this allows a stronger claim that decoder attention is aligned
with lesion-prior regions.

Implementation idea:

```python
outputs = llm(..., output_attentions=True)
attn = outputs.attentions[-1]
visual_token_attn = extract_attention_to_visual_prefix(attn)
loss_attn_prior = sinkhorn_ot(visual_token_attn, prior_mask_distribution)
```

Only claim this after the code is implemented and tested.

### 3.4 Uncertainty and Calibration

CATA-v2 should include stronger uncertainty modeling:

- deep ensemble,
- snapshot ensemble,
- MC dropout / stochastic consistency,
- Bayesian or variational LoRA,
- calibration layer.

Metrics:

- ECE,
- Brier score,
- reliability diagram,
- confidence/error correlation.

### 3.5 Explanation Faithfulness

For a Q1 journal paper, heatmaps should be evaluated, not only shown.

Possible metrics:

- pointing game,
- deletion/insertion score,
- heatmap-prior overlap,
- IoU if masks exist,
- qualitative clinician review if possible.

### 3.6 Multi-dataset Validation

A Q1 journal paper should ideally evaluate beyond Kvasir-VQA-x1.

Candidate datasets:

- Kvasir-VQA-x1,
- VQA-RAD,
- SLAKE,
- PathVQA,
- HyperKvasir,
- Kvasir-SEG / segmentation proxy tasks,
- EndoVis if relevant.

### CATA-v2 Ablation Table

| Model | Structural ViT | OT | TopoAdapter | Gate Supervision | Q-aware Loss | Uncertainty | Score |
|---|---|---|---|---|---|---|---:|
| Baseline | no | no | no | no | no | no | TBD |
| Paper 1 model | yes | no | no | no | no | no | TBD |
| Paper 2 model | partial | yes | yes | no | no | no | TBD |
| CATA-v2 minus gate-align | yes | yes | yes | no | yes | yes | TBD |
| CATA-v2 minus q-aware | yes | yes | yes | yes | no | yes | TBD |
| CATA-v2 full | yes | yes | yes | yes | yes | yes | TBD |

---

## Publication Separation Strategy

There will be four outputs:

| Output | Type | Main Role | Main Novelty |
|---|---|---|---|
| MediaEval paper | workshop/system | benchmark submission report | submitted CATA system |
| Paper 1 | A/A* conference | vision-side method | structural features for ViT |
| Paper 2 | A/A* conference | decoder-side method | TDA Adapter + OT for LLM |
| Paper 3 | Q1 journal | unified framework | CATA-v2 with new structural control and validation |

### Avoiding Overlap

- MediaEval paper: concise system report, no overly broad theory.
- Paper 1: do not focus on TopoAdapter.
- Paper 2: do not make structural ViT fusion the main novelty.
- Paper 3: must add new contributions beyond simply combining Paper 1 and Paper 2.

---

## Recommended Timeline

### Phase 0 — Finish MediaEval Paper

- [ ] Convert paper to English.
- [ ] Correct implementation-faithful formulas.
- [ ] Add results and qualitative examples.
- [ ] Keep claims modest.

### Phase 1 — Paper 2 First

Reason: closest to current CATA code.

- [ ] Run Group A baseline.
- [ ] Run OT-only ablation.
- [ ] Run TopoAdapter-only ablation.
- [ ] Run patch-only vs all topology ablation.
- [ ] Run no-gate-loss ablation.
- [ ] Collect gate diagnostics.
- [ ] Write formal TopoAdapter + OT method.

### Phase 2 — Paper 1

- [ ] Build clean structural ViT fusion experiments.
- [ ] Compare ViT-only vs structural features.
- [ ] Visualize structural features.
- [ ] Evaluate downstream VQA and structural/localization proxy tasks.

### Phase 3 — Implement CATA-v2

- [ ] Create `src/models/topo_adapter_v2.py`.
- [ ] Add supervised gate alignment.
- [ ] Add question-aware structural weighting.
- [ ] Add direct attention-prior alignment if feasible.
- [ ] Add uncertainty/cali­bration pipeline.

### Phase 4 — Q1 Journal

- [ ] Multi-dataset validation.
- [ ] Full ablation suite.
- [ ] Explanation faithfulness metrics.
- [ ] Calibration/uncertainty metrics.
- [ ] Compare to strong medical VLM baselines.

---

## Claim Discipline

Always keep paper claims aligned with implemented code.

### Safe Words

Use:

- encourages,
- regularizes,
- conditions,
- modulates,
- biases,
- improves in our experiments,
- reliability-style confidence.

### Dangerous Words

Avoid unless proven:

- guarantees,
- solves,
- completely eliminates,
- forces,
- overrides,
- clinical probability,
- calibrated posterior,
- symbolic reasoning engine.

---

## Current Priority

The immediate priority is **not** CATA-v2.

Current order:

1. Finish MediaEval paper.
2. Run ablations for Paper 2.
3. Prepare Paper 2 as the first serious conference submission.
4. Then build Paper 1.
5. Finally implement and evaluate CATA-v2 for Q1 journal submission.
