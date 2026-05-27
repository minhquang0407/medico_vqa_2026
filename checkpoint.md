# Medico VQA 2026 Checkpoint

Last updated: 2026-05-27

This checkpoint summarizes the current state of the project across Task 1 and
Task 2, including completed work, pending runs, failed ideas, and the final
experiment direction.

---

## 0. Hard Rules / Decisions

- **Never submit Plain PaliGemma** as a final solution.
  - It is only a baseline/control.
  - Reason: it is too close to common copied baseline solutions.
- Structural features are considered the main novelty.
- Textual structural hints were rejected after ablation because they degraded
  performance.
- Normalization must remain conservative.
- Visual explanation/submission should be polished; no rushed submission.

---

## 1. Task 1: Visual Question Answering

### 1.1 Existing Baselines

#### Plain PaliGemma / PaliGemma QLoRA

Status: **completed baseline**

Known score:

```text
ROUGE1 approximately 0.7007
```

Role:

- Strong generic VLM baseline.
- Used for comparison only.
- **Not for final submission** because it lacks sufficient novelty.

#### RunB Qwen3B Structural Fusion

Status: **completed baseline / existing structural model**

Role:

- Qwen3B-based structural VQA pipeline.
- Uses vision encoder, structural visual fusion, OT fusion, and Qwen LLM.
- This is the old RunB-family model.

Important conclusion:

- The old `Qwen3B Structural Adapter` wrapper was essentially this RunB-family
  pipeline, so it should **not** be rerun as a new model.

---

### 1.2 Architectures Tried and Failed / Rejected

#### PaliGemma Structural Text Hints

Status: **failed / rejected**

Implementation idea:

- Convert structural features into textual hints.
- Add those hints to the prompt before PaliGemma generation.

Observed score:

```text
BLEU:   0.4179
ROUGE1: 0.6794
ROUGE2: 0.4822
ROUGEL: 0.6484
METEOR: 0.6528
```

Conclusion:

- Worse than Plain PaliGemma baseline.
- Textual hints likely cause prompt pollution/interference.
- Structural signals should not be injected as natural-language hints.

Decision:

- Do not continue prompt-hint direction.

#### Qwen3B Full Structural Wrapper

Status: **rejected as new run**

Files created earlier:

```text
scripts/train_qwen3b_structural_adapter.py
```

Conclusion:

- This is not a true new adapter architecture.
- It is essentially RunB/Qwen structural fusion with a wrapper/config.
- No need to rerun if RunB already exists.

---

## 2. New Task 1 Architectures Implemented

### 2.1 PaliGemma Soft-Token Structural Adapter

Status: **implemented, needs substantial run**

Main file:

```text
scripts/train_paligemma_structural_adapter.py
```

Concept:

```text
structural npz -> StructuralAdapter -> soft structural tokens
-> insert into PaliGemma embedding stream
```

Features:

- Supports full structural mode:

```text
--structural-mode all
```

- Supports TDA-only mode:

```text
--structural-mode tda_only
```

Full structural input includes:

```text
prior_mask
red_map
center_map
morpho_prior_map
topo_features
global_features
```

TDA-only input keeps:

```text
topo_features
```

and zeros/removes:

```text
prior_mask
red_map
center_map
morpho_prior_map
global_features
```

Important fix:

- Added dtype-safe adapter execution for eval/generation.
- Added learnable gate initialized near zero:

```text
gate_logit = -6.0
sigmoid(-6) approximately 0.0025
```

Reason:

- Random structural soft tokens were collapsing generation into short answers
  like `yes`, `no`, `0`, `red`.
- Zero-near gate makes the adapter start close to the original PaliGemma prior.

Smoke result:

- Training/eval technically passes.
- 64-sample smoke still produces short answers and low scores, which is not
  enough to judge final performance.

Recommended next step:

- Run 3k sanity first if GPU time permits.
- Then run 30k if output quality improves.

---

### 2.2 PaliGemma Soft-Token TDA-only Adapter

Status: **implemented, needs substantial run**

Main file:

```text
scripts/train_paligemma_tda_only_adapter.py
```

Wrapper around:

```text
scripts/train_paligemma_structural_adapter.py --structural-mode tda_only
```

Purpose:

- Test whether TDA alone provides useful signal.
- Removes prior masks/global handcrafted priors from the adapter input.

Recommended run:

```text
outputs/paligemma_tda_only_30k
```

---

### 2.3 Qwen3B Deep Topological Adapter

Status: **implemented, needs smoke + 30k runs**

Main file:

```text
scripts/train_qwen3b_topo_adapter.py
```

This is the new Qwen architecture that is actually different from RunB.

Concept:

```text
Qwen hidden states inside decoder layers
-> bottleneck down projection
-> topology-conditioned gating
-> zero-init up projection
-> residual add
```

This is a true deep-fusion adapter, similar in spirit to LoRA/Houlsby adapters.

Default injection:

```text
last 8 Qwen decoder layers
```

Configurable by:

```text
--adapter-last-n-layers 8
--adapter-every-n-layers 0
--bottleneck-dim 32
```

Zero-init behavior:

- `up_proj.weight` initialized to zero.
- Adapter is an exact no-op at initialization.
- Reduces risk of damaging pretrained Qwen behavior.

#### Qwen3B Topo-Adapter TDA-only

Status: **implemented**

Run mode:

```text
--topo-mode tda_only
```

Condition uses:

```text
mean/std/max pooled topo_features
```

Strictly zeros/removes:

```text
prior_mask
global_features
prior OT target
prior align loss
global topo loss
```

Purpose:

- Test TDA as the core novel signal.
- Not equivalent to old RunB.

#### Qwen3B Topo-Adapter Full Structural

Status: **implemented**

Run mode:

```text
--topo-mode all
```

Condition uses:

```text
mean/std/max pooled topo_features
prior_mask statistics
global_features
```

Also enables:

```text
prior OT target
prior align loss
global topo loss
patch topo loss
```

Purpose:

- Test deep full structural conditioning.
- Compare against TDA-only and RunB baseline.

---

## 3. Task 1 Models That Still Need Running

### Highest Priority Runs

#### Run 1: PaliGemma Soft-Token Full Structural 30k

Output:

```text
outputs/paligemma_struct_adapter_30k
```

Purpose:

- Main PaliGemma structural challenger.

#### Run 2: PaliGemma Soft-Token TDA-only 30k

Output:

```text
outputs/paligemma_tda_only_30k
```

Purpose:

- TDA-only ablation for PaliGemma.

#### Run 3: Qwen3B Deep Topo-Adapter TDA-only 30k

Output:

```text
outputs/qwen3b_topo_adapter_tda_30k
```

Purpose:

- New Qwen deep TDA architecture.

#### Run 4: Qwen3B Deep Topo-Adapter Full Structural 30k

Output:

```text
outputs/qwen3b_topo_adapter_full_30k
```

Purpose:

- New Qwen deep full-structural architecture.

---

## 4. Suggested Task 1 Experiment Matrix

| Backbone | Adapter / Fusion Style | Structural Mode | Status |
|---|---|---|---|
| PaliGemma | Plain / QLoRA | None | Done baseline, do not submit |
| Qwen3B | RunB structural fusion | Full | Done baseline |
| PaliGemma | Text prompt hints | Full as text | Failed/rejected |
| PaliGemma | Soft-token adapter | Full | Implemented, needs run |
| PaliGemma | Soft-token adapter | TDA-only | Implemented, needs run |
| Qwen3B | Deep Topo-Adapter | TDA-only | Implemented, needs smoke/run |
| Qwen3B | Deep Topo-Adapter | Full | Implemented, needs smoke/run |

Recommended final 4 new runs:

```text
1. PaliGemma soft-token full structural 30k
2. PaliGemma soft-token TDA-only 30k
3. Qwen3B deep Topo-Adapter TDA-only 30k
4. Qwen3B deep Topo-Adapter full structural 30k
```

Do not rerun:

```text
Qwen3B old full structural wrapper
```

because it overlaps with RunB.

---

## 5. Important Task 1 Commands

### PaliGemma full structural 30k

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/train_paligemma_structural_adapter.py \
  --mode train_eval \
  --output-dir outputs/paligemma_struct_adapter_30k \
  --epochs 2 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 \
  --max-train-samples 30000 \
  --eval-samples 1500 \
  --max-new-tokens 48 \
  --max-length 512 \
  --num-workers 2 \
  --use-augmentation \
  --structural-root . \
  --train-structural-manifest data/processed/structural_features/train_original_manifest.csv \
  --eval-structural-manifest data/processed/structural_features/test_original_manifest.csv \
  --structural-mode all
```

### PaliGemma TDA-only 30k

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/train_paligemma_tda_only_adapter.py \
  --output-dir outputs/paligemma_tda_only_30k \
  --epochs 2 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 \
  --max-train-samples 30000 \
  --eval-samples 1500 \
  --num-workers 2 \
  --use-augmentation
```

### Qwen3B Topo-Adapter TDA-only smoke

```bash
PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/train_qwen3b_topo_adapter.py \
  --output-dir outputs/qwen3b_topo_adapter_tda_smoke \
  --topo-mode tda_only \
  --max-samples 64 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --lr 3e-5 \
  --num-workers 0 \
  --adapter-last-n-layers 8 \
  --bottleneck-dim 32
```

### Qwen3B Topo-Adapter full structural smoke

```bash
PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/train_qwen3b_topo_adapter.py \
  --output-dir outputs/qwen3b_topo_adapter_full_smoke \
  --topo-mode all \
  --max-samples 64 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --lr 3e-5 \
  --num-workers 0 \
  --adapter-last-n-layers 8 \
  --bottleneck-dim 32
```

### Qwen3B Topo-Adapter TDA-only 30k

```bash
PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/train_qwen3b_topo_adapter.py \
  --output-dir outputs/qwen3b_topo_adapter_tda_30k \
  --topo-mode tda_only \
  --max-samples 30000 \
  --epochs 2 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lr 3e-5 \
  --num-workers 2 \
  --adapter-last-n-layers 8 \
  --bottleneck-dim 32
```

### Qwen3B Topo-Adapter full structural 30k

```bash
PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/train_qwen3b_topo_adapter.py \
  --output-dir outputs/qwen3b_topo_adapter_full_30k \
  --topo-mode all \
  --max-samples 30000 \
  --epochs 2 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lr 3e-5 \
  --num-workers 2 \
  --adapter-last-n-layers 8 \
  --bottleneck-dim 32
```

---

## 6. Task 2 Checkpoint

Task 2 has been planned separately in the artifact:

```text
task2_plan.md
```

Known status from current checkpoint:

- Task 2 plan exists.
- No new Task 2 execution was performed in this checkpoint segment.
- Current focus shifted to Task 1 structural adapter/challenger experiments.

Need to revisit:

- Confirm current Task 2 baseline status.
- Confirm required submission format for Task 2.
- Confirm whether Task 2 uses the same structural features or a separate
  pipeline.
- Update this checkpoint after reviewing `task2_plan.md` and any prior outputs.

---

## 7. Files Added or Modified Recently

### Added

```text
scripts/train_paligemma_structural_adapter.py
scripts/train_paligemma_tda_only_adapter.py
scripts/train_qwen3b_structural_adapter.py
scripts/train_qwen3b_tda_only.py
scripts/train_qwen3b_topo_adapter.py
```

### Modified

```text
scripts/train_paligemma_qlora.py
scripts/train_structural_vqa_generative.py
```

Key changes:

- RAM caching for structural tensors/hints.
- Strict TDA-only mode for Qwen old trainer.
- PaliGemma structural adapter gate/dtype fixes.
- New Qwen3B deep Topo-Adapter implementation.

---

## 8. Git Commits of Interest

```text
02a2eb9 Add Qwen and TDA-only structural adapter experiments
0a156dc Fix structural adapter dtype during generation
f9dcfdb Gate structural adapter influence at initialization
aecbbbe Enforce strict TDA-only mode for Qwen training
88dd11b Fix Qwen wrapper PYTHONPATH for src imports
6d47cfe Add Qwen3B deep topological adapter training
```

---

## 9. Current Risks

### PaliGemma soft-token adapter may still produce short answers

Smoke with 64 samples produced short predictions such as:

```text
yes
no
0
red
top
```

This may be due to small data, but should be checked with 3k sanity before a
full 30k run if time allows.

Potential mitigation:

```text
--struct-tokens 2
```

or reduce adapter influence further.

### Qwen3B Topo-Adapter is newly implemented

Needs smoke testing before 30k.

Potential failure points:

- HF Qwen internal layer wrapping.
- Generate/cache behavior.
- LoRA wrapper path differences.
- VRAM overhead with adapters in multiple layers.

### Dependency/runtime issue on Colab

After installing newer packages, Colab may warn:

```text
You must restart the runtime in order to use newly installed versions.
```

If weird import/runtime errors appear, restart runtime and rerun setup.

---

## 10. Immediate Next Actions

Recommended order:

```text
1. Pull latest repo on Colab.
2. Smoke Qwen3B Topo-Adapter TDA-only.
3. Smoke Qwen3B Topo-Adapter full structural.
4. Run PaliGemma full structural 3k or 30k.
5. Run PaliGemma TDA-only 3k or 30k.
6. If Qwen Topo smoke passes, run Qwen Topo TDA-only 30k.
7. Run Qwen Topo full structural 30k.
8. Compare against Plain PaliGemma and RunB baselines.
```

Final candidate should be selected based on validation/test metrics and novelty,
not just raw baseline strength.
