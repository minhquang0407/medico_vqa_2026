# Long-Term Research Roadmap — Medico VQA 2026 / CATA

This document records the long-term research plan after the MediaEval Medico VQA
2026 submission. The goal is to turn the current CATA system into a sequence of
publishable research works, each with a clear contribution, evaluation strategy,
and code direction.

---

## Strategic Update: "Tứ bộ khúc CATA" / Four-Paper Research Arc

The long-term publication strategy is now organized as a four-paper arc. The
core principle is **decomposition before unification**: isolate the empirical
system, the decoder-side mathematical reasoning, and the vision-side geometric
encoder before combining them into a unified journal framework.

> Important boundary: the current MediaEval system remains **Visual TDA Fusion +
> Gated TDA Adapter only**. OT, GraphOT, Dual-End Fusion, and Grand Unified CATA
> are roadmap targets, not current-system claims.

| Paper | Role | Target | Main focus | Relation to current code |
|---|---|---|---|---|
| Paper 1: MediaEval Working Notes | Timestamp / empirical proof-of-concept | MediaEval 2026 | Show CATA runs end-to-end on noisy Kvasir-VQA with Qwen-3B-scale compute | Current implementation |
| Paper 2: TDA + LLM Reasoning Core | NLP / mathematical reasoning core | ICLR / NeurIPS / strong NLP-AI venue | Study how Betti-style topological summaries and dynamic gates improve count/structure-sensitive decoding | Extends the current Gated TDA Adapter, but should be tested model-agnostically |
| Paper 3: Dual-End Fusion ViT | CV / geometric perception core | CVPR / ICCV / MICCAI | Analyze ViT patchification and preserve geometric continuity through structural fusion at both input and output sides | Extends current Visual TDA Fusion; LLM should be removed from the core experiments |
| Paper 4: Grand Unified CATA | Unified neuro-symbolic framework | IEEE TPAMI / TMI / MedIA | Combine Paper 2 and Paper 3 into a differentiable framework from pixels to text | Future CATA-v2/v3 journal system |

### Paper 1 Immediate Role

The MediaEval paper is the urgent timestamp paper. It should establish priority
for the CATA idea and report the submitted system transparently. If using the
provided strategic summary, the report may mention the Qwen-3B-scale hardware
profile and the relevant METEOR score, but the final number must match the
compiled result table before submission.

### Paper 2 First-Principles Hypothesis

The second paper should ask whether topological quantities such as Betti-style
features can condition LLM decoding in a way that improves count-sensitive and
structure-sensitive reasoning. The strongest version should remove ViT from the
core experiment and evaluate multiple LLM backbones to show that the mechanism is
model-agnostic.

Safe paper wording: topology-conditioned gates **encourage** or **bias** the LLM
toward count-consistent decoding. Avoid claiming that the gate "forces" symbolic
logic unless it is formally proven and experimentally validated.

### Paper 3 First-Principles Hypothesis

The third paper should isolate the vision encoder. Its central hypothesis is
that ViT patchification can weaken spatial continuity for biological structures,
and that a **Dual-End Geometry Preservation Loop** can preserve structural
information by injecting topology/morphology cues both before and after visual
encoding. This paper should not depend on the LLM as the main contribution.

### Paper 4 First-Principles Hypothesis

The fourth paper should be the long journal version: a Grand Unified CATA
framework showing how the vision-side geometry-preserving encoder and the
language-side topology-conditioned adapter interact. A possible formal direction
is to analyze differentiable gradient flow from a Sinkhorn/OT-style structural
loss through the decoder gate and back into the geometry-preserving ViT branch.
This is a future theory target and should not be claimed in the current system
paper.

---

## 0. Current Paper: MediaEval System Paper

### Role

Short workshop/system paper for the MediaEval Medico VQA 2026 benchmark.

### Scope

This paper should describe the submitted CATA system honestly and concisely:

- Qwen2.5-3B-Instruct + LoRA generative VQA backbone.
- Frozen pretrained ViT/timm vision encoder.
- Residual Visual TDA fusion with patch-level topology/morphology features.
- TopoAdapter inserted into the final decoder layers.
- Task 2 explanation generation with heatmaps, evidence JSON, self-probing, and
  reliability-style confidence.

Current architecture boundary: **Visual TDA Fusion + TDA Adapter only**. Do not
present OT, GraphOT, lesion-prior routing, token-wise gated visual fusion,
Dual-End Fusion, or Grand Unified CATA as components of the submitted system;
keep them as future ROADMAP experiments.

### Strategic Role in the Four-Paper Arc

This is **Paper 1: the empirical timestamp paper**. Its goal is not to solve the
whole theory of CATA, but to establish that the idea is real, runnable, and
useful under the MediaEval benchmark constraints. The paper should emphasize:

- end-to-end feasibility on Kvasir-VQA-x1,
- Qwen-3B-scale compute efficiency,
- practical gains from Visual TDA Fusion and the Gated TDA Adapter,
- Task 2 explanation packaging as a review-support pipeline,
- careful separation between implemented modules and future extensions.

Deadline pressure: this is the highest-priority manuscript before the 10/06
submission window.

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
- [ ] Correct the Visual TDA, TopoAdapter, and loss formulas to match implementation.

---

## 1. Paper 1: Structural Features for Vision Transformers

### Tentative Title

**Structural Feature Fusion for Vision Transformers in Gastrointestinal Medical VQA**

Alternative:

**Topology- and Morphology-Aware Vision Transformer Fusion for Endoscopic Understanding**

### Target Type

Conference paper, aiming for A/A* or strong medical-imaging venue.

Possible venues:

- CVPR
- ICCV
- MICCAI
- MIDL
- ISBI
- WACV
- BMVC
- ACM Multimedia workshop/main track if multimodal angle is strong

### Four-Paper Arc Mapping

In the updated "Tứ bộ khúc CATA" strategy, this section corresponds to
**Paper 3: Dual-End Fusion ViT**, the CV/geometric perception core. It should
remove the LLM from the main contribution and focus on the vision encoder.

First-principles framing:

> ViT patchification can fragment continuous biological structures into discrete
> tokens. A geometry-preservation loop can inject topology/morphology cues at
> both the input side and the output side of the encoder to preserve entity
> continuity.

This is a future research hypothesis. The current MediaEval paper should only
claim the implemented scalar-residual Visual TDA Fusion.

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
- `src/alignment/graph_ot.py` **(new — GraphOT)**
- `src/topology/patch_graph.py` **(new — patch graph construction)**
- `scripts/precompute_structural_features.py`
- `scripts/run_pretrained_vision_ablation.py`
- `scripts/run_graph_ot_ablation.py` **(new)**

### Key New Idea A: Token-wise Gated Visual TDA Fusion

The current implementation in `src/models/fusion.py` is deliberately stable and
uses a learnable scalar residual coefficient shared across all patches:

```
\bar{v}_i = LN_v(v_i)
\tilde{t}_i = MLP_t(LN_t(t_i))
\hat{v}_i = LN_out(\bar{v}_i + beta · \tilde{t}_i)
```

This keeps the pretrained ViT embedding as the main information source, but the
same global `beta` is applied to every patch. A stronger experimental variant is
**Token-wise Gated TDA Fusion**:

```
alpha_i = sigmoid(MLP_g([\bar{v}_i ; \tilde{t}_i]))
\hat{v}_i = LN_out(\bar{v}_i + alpha_i · \tilde{t}_i)
```

Recommended first version:

- `alpha_i ∈ [0, 1]` is a **scalar gate per patch**.
- Keep `\tilde{t}_i` at the ViT hidden dimension.
- Log `gate_mean`, `gate_std`, and a 14×14 gate heatmap for interpretation.
- Keep the current scalar-`beta` fusion as the baseline.

Later version:

- `alpha_i ∈ [0, 1]^D` channel-wise gate for higher capacity.
- Add mild gate regularization only if the gate collapses to all-zero or all-one.

#### Why this is worth testing

- It lets topology-rich lesion patches receive stronger TDA corrections.
- It lets background/noisy patches suppress unreliable TDA descriptors.
- It is still lightweight and easier to train than GraphOT.
- It produces interpretable token-level gate maps that can be compared with
  lesion priors and TDA saliency maps.

### Key New Idea B: GraphOT Visual TDA Fusion

After establishing the token-wise gated baseline, **GraphOT** can be tested as a
more structural variant. Instead of letting each patch decide independently,
GraphOT regularizes the visual--TDA interaction over a patch graph:

```
min_P  <P, C>  +  ε · H(P)  +  λ · tr(P^T L P)
```

Where `L` is the graph Laplacian over visual patches. This enforces:

- **Spatial smoothness**: neighboring patches receive similar transport.
- **Cross-patch borrowing**: a weak-TDA patch can borrow signal from a
  strong-TDA neighbor.
- **TDA-informed graph**: the adjacency can be built from TDA feature
  similarity (kNN on persistence descriptors), not just grid position.

#### Graph construction options

| Graph type | Adjacency rule |
|---|---|
| Grid-4 | 4-connected neighbors on 14×14 patch grid |
| Grid-8 | 8-connected neighbors |
| TDA-kNN | k nearest neighbors in TDA feature space |
| Hybrid | Grid-8 ∪ TDA-kNN |

#### Why these are good contributions for Paper 1

- Paper 1 focuses on the **vision side** only.
- Token-wise gated fusion is a clean upgrade over the current scalar residual
  fusion while staying easy to ablate.
- GraphOT is a structural fusion method that operates entirely within the
  visual branch, before any LLM interaction.
- Graph construction naturally connects to TDA: the graph can be built from
  persistence features, making it genuinely topology-aware.
- Endoscopy images have spatially continuous lesions → graph smoothness is a
  strong and well-motivated inductive bias.

#### Files to create / modify

| File | Action | Description |
|---|---|---|
| `src/models/fusion.py` | **MODIFY** | Add `fusion_mode={scalar,token_gate,channel_gate}` and token-wise gate diagnostics |
| `src/models/vision_encoder.py` | **MODIFY** | Wire visual fusion mode into the encoder config |
| `src/alignment/graph_ot.py` | **NEW** | `GraphSinkhornOT`: extend Sinkhorn with Laplacian penalty |
| `src/topology/patch_graph.py` | **NEW** | `build_patch_graph()`: grid / TDA-kNN / hybrid adjacency + Laplacian |
| `scripts/run_visual_tda_gate_ablation.py` | **NEW** | Ablation script for scalar vs token-wise vs channel-wise gates |
| `scripts/run_graph_ot_ablation.py` | **NEW** | Ablation script for GraphOT variants |

### Possible Model Variants

| Variant | Description |
|---|---|
| ViT only | pretrained ViT/timm features only |
| ViT + lesion prior | spatial prior used as additional visual cue |
| ViT + global morphology | image-level morphology output fusion |
| ViT + patch TDA (scalar residual) | current implementation with learnable global `beta` |
| ViT + patch TDA (token-wise scalar gate) | proposed per-patch `alpha_i` gate |
| ViT + patch TDA (channel-wise gate) | proposed per-patch, per-channel gate |
| ViT + patch TDA (GraphOT grid) | GraphOT with grid-8 adjacency |
| ViT + patch TDA (GraphOT TDA-kNN) | GraphOT with TDA feature kNN graph |
| ViT + patch TDA (GraphOT hybrid) | GraphOT with grid-8 ∪ TDA-kNN |
| ViT + all structural features | full structural visual representation |

### Experiments Needed

- [ ] ViT-only baseline.
- [ ] ViT + lesion prior.
- [ ] ViT + morphology.
- [ ] ViT + patch TDA (scalar residual, current).
- [ ] ViT + patch TDA (token-wise scalar gate).
- [ ] ViT + patch TDA (channel-wise gate, optional if scalar gate works).
- [ ] ViT + patch TDA (GraphOT grid-8).
- [ ] ViT + patch TDA (GraphOT TDA-kNN).
- [ ] ViT + patch TDA (GraphOT hybrid).
- [ ] ViT + all structural signals.
- [ ] Visualization of lesion priors, TDA saliency, and token-wise gate maps.
- [ ] GraphOT transport plan visualization (spatial smoothness).
- [ ] Question-type analysis, especially lesion/count/location questions.
- [ ] Ablation on gate type, gate regularization, λ (Laplacian weight), and k
  (kNN neighbors).

### Metrics

Possible metrics:

- downstream VQA BLEU / ROUGE-L / METEOR,
- token F1,
- count-question accuracy,
- localization agreement if masks/proxies exist,
- retrieval/classification proxy metrics if additional labels are available,
- transport plan smoothness (entropy, Laplacian energy).

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

- TopoAdapter / Gated TDA Adapter.
- Topology-conditioned hidden-state modulation.
- Dynamic gates controlled by Betti-style/TDA statistics.
- Model-agnostic evaluation across multiple LLM backbones.
- Count-sensitive and structure-sensitive decoding.
- Gate regularization.
- Curriculum training.

In the four-paper arc, this is **Paper 2: the TDA + LLM reasoning core**. The
strongest version should temporarily remove ViT from the central experiment and
use precomputed topology/structure conditions, so that the contribution is not
confounded with the visual encoder.

Possible model-agnostic setup:

| Backbone | Purpose |
|---|---|
| Qwen2.5-3B | current efficient CATA backbone |
| Llama-family small model | cross-family generalization |
| Phi/Gemma-family small model | additional decoder architecture check |
| text-only synthetic counting task | isolates topology-conditioned decoding |
| medical VQA with frozen visual features | tests transfer back to the benchmark |

Research hypothesis:

> Betti-style topological summaries, especially connected-component and loop-like
> cues, can condition dynamic gates that bias the decoder toward more stable
> count and morphology-sensitive answers.

Use safe wording (`biases`, `encourages`, `regularizes`) until a formal proof or
controlled synthetic benchmark supports stronger claims.

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

For the **current MediaEval paper**, use only the implemented loss:

```latex
\mathcal{L}_{total}
=
\mathcal{L}_{CE}
+
\lambda_{gate}\mathcal{L}_{gate}.
```

For a **future Paper 2 / Paper 4** version that actually implements OT or
structural losses, a safe extended formula is:

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

Do not include $\mathcal{L}_{OT}$ in the MediaEval system paper unless it is
re-enabled in the submitted code and ablated.

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

This paper unifies Paper 2 and Paper 3, but must add genuinely new contributions.
It should not be only a combination of previous papers.

### Four-Paper Arc Mapping

In the updated strategy, this is **Paper 4: Grand Unified CATA**. The goal is a
long Q1 journal paper that connects the decoder-side TDA reasoning core and the
vision-side geometry preservation core into a single differentiable framework.

Target venues:

- IEEE TPAMI,
- IEEE Transactions on Medical Imaging,
- Medical Image Analysis,
- Information Fusion.

First-principles target:

```latex
\mathcal{L}_{unified}
=
\mathcal{L}_{CE}
+
\lambda_{OT}\mathcal{L}_{OT}
+
\lambda_{geo}\mathcal{L}_{geo}
+
\lambda_{gate}\mathcal{L}_{gate}.
```

A possible theory direction is to analyze a chain-rule path such as:

```latex
\frac{\partial \mathcal{L}_{unified}}{\partial \theta_{ViT}}
=
\frac{\partial \mathcal{L}_{unified}}{\partial \mathbf{h}'_l}
\frac{\partial \mathbf{h}'_l}{\partial \mathbf{g}}
\frac{\partial \mathbf{g}}{\partial \mathbf{z}}
\frac{\partial \mathbf{z}}{\partial T}
\frac{\partial T}{\partial \theta_{ViT}}.
```

This should be treated as a future mathematical program: demonstrate smooth
backpropagation from Sinkhorn/OT-style structural objectives through the decoder
gate and into the vision-side geometry-preserving loop. Only claim it after the
implementation and ablations exist.

### Required New Contributions

CATA-v2 should add at least two of the following:

1. **Supervised Gate Alignment**
2. **Question-aware Structural Loss Weighting**
3. **Direct Decoder Attention-Prior Alignment**
4. **Deep Ensemble or Stochastic Uncertainty**
5. **Calibrated Confidence Estimation**
6. **Faithfulness Evaluation for Explanations**
7. **Multi-dataset Validation**
8. **GraphOT Visual TDA Fusion** (from Paper 1, integrated into full system)

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

| Model | Structural ViT | GraphOT Fusion | OT | TopoAdapter | Gate Supervision | Q-aware Loss | Uncertainty | Score |
|---|---|---|---|---|---|---|---|---:|
| Baseline | no | no | no | no | no | no | no | TBD |
| Paper 1 model (gated) | yes | no | no | no | no | no | no | TBD |
| Paper 1 model (GraphOT) | yes | yes | no | no | no | no | no | TBD |
| Paper 2 model | partial | no | yes | yes | no | no | no | TBD |
| CATA-v2 minus GraphOT | yes | no | yes | yes | yes | yes | yes | TBD |
| CATA-v2 minus gate-align | yes | yes | yes | yes | no | yes | yes | TBD |
| CATA-v2 minus q-aware | yes | yes | yes | yes | yes | no | yes | TBD |
| CATA-v2 full | yes | yes | yes | yes | yes | yes | yes | TBD |

---

## Publication Separation Strategy

There will be four papers in the **Tứ bộ khúc CATA** arc:

| Paper | Type | Main Role | Main Novelty | What to exclude |
|---|---|---|---|---|
| Paper 1: MediaEval Working Notes | workshop/system | benchmark submission report and timestamp | submitted CATA system: Visual TDA + Gated TDA Adapter | do not claim OT, GraphOT, Dual-End Fusion, or unified theory |
| Paper 2: TDA + LLM | A/A* conference | NLP/math reasoning core | Betti/TDA-conditioned dynamic gates for count/structure-sensitive decoding | do not make ViT fusion the main novelty |
| Paper 3: Dual-End Fusion ViT | A/A* CV conference | vision-side geometric core | geometry-preserving ViT with structural injection at input and output | do not focus on LLM decoding |
| Paper 4: Grand Unified CATA | Q1 journal | unified neuro-symbolic framework | differentiable pixel-to-text topology-aware adaptation | do not submit as a simple combination without new theory/experiments |

### Avoiding Overlap

- MediaEval paper: concise system report, no overly broad theory.
- Paper 2: isolate the decoder-side TDA gate; test model-agnostic behavior.
- Paper 3: isolate the vision encoder; no LLM as the main contribution.
- Paper 4: must add new contributions beyond simply combining Paper 2 and Paper 3.
- All papers must preserve claim discipline and clearly separate implemented
  modules from future hypotheses.

---

## Recommended Timeline

### Phase 0 — Finish Paper 1: MediaEval Working Notes

- [ ] Convert paper to English.
- [ ] Correct implementation-faithful formulas.
- [ ] Add results and qualitative examples.
- [ ] Keep claims modest.
- [ ] Verify final official numbers before quoting METEOR/BLEU in the text.
- [ ] Submit before the 10/06 deadline window.

### Phase 1 — Paper 2: TDA + LLM Reasoning Core

Reason: closest to the current Gated TDA Adapter code.

- [ ] Define Betti/TDA condition vector variants: $\beta_0$, $\beta_1$, persistence summaries, and current 36-D patch statistics.
- [ ] Build model-agnostic decoder experiments across at least two LLM families.
- [ ] Run Group A baseline.
- [ ] Run TopoAdapter-only / Gated TDA Adapter ablation.
- [ ] Run patch-only vs all-topology condition ablation.
- [ ] Run no-gate-loss ablation.
- [ ] Add synthetic or controlled count-reasoning tests if possible.
- [ ] Collect gate diagnostics and count/location subset results.
- [ ] Write formal TDA-conditioned decoder method.

### Phase 2 — Paper 3: Dual-End Fusion ViT

- [ ] Build clean structural ViT fusion experiments without relying on LLM novelty.
- [ ] Compare ViT-only vs scalar Visual TDA Fusion.
- [ ] Implement token-wise gated Visual TDA Fusion.
- [ ] Design Dual-End Geometry Preservation Loop: structural injection before and after visual encoding.
- [ ] Implement `src/alignment/graph_ot.py` (GraphSinkhornOT with Laplacian).
- [ ] Implement `src/topology/patch_graph.py` (grid / TDA-kNN graph builder).
- [ ] Implement `GraphOTVisualTDAFusion` in `src/models/fusion.py`.
- [ ] Run GraphOT ablation: grid-8 vs TDA-kNN vs hybrid.
- [ ] Ablation on λ (Laplacian weight) and k (kNN neighbors).
- [ ] Visualize structural features, gate maps, and GraphOT transport plans.
- [ ] Evaluate downstream VQA and structural/localization proxy tasks.

### Phase 3 — Paper 4: Grand Unified CATA

- [ ] Create `src/models/topo_adapter_v2.py`.
- [ ] Add supervised gate alignment.
- [ ] Add question-aware structural weighting.
- [ ] Add direct attention-prior alignment if feasible.
- [ ] Add uncertainty/calibration pipeline.
- [ ] Integrate Dual-End Fusion ViT with the decoder-side TDA gate.
- [ ] Define the unified differentiable objective and chain-rule analysis.

### Phase 4 — Q1 Journal Validation

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

The immediate priority is **not** Grand Unified CATA.

Current order:

1. Finish Paper 1 / MediaEval Working Notes.
2. Build Paper 2 around the decoder-side TDA gate and model-agnostic LLM tests.
3. Build Paper 3 around Dual-End Fusion ViT and geometry preservation.
4. Finally implement Paper 4 / Grand Unified CATA for Q1 journal submission.

Strategic rule: do not mix the four papers too early. Each paper must have a
single clean mathematical message before the unified framework is attempted.
