# Medico 2026 Task 2: Bayes-Gated Self-Probing Structural Explanations

This repository contains a Subtask 2 submission for **MediaEval Medico 2026:
Clinician-Oriented Multimodal Explanations in GI**.

The method extends our FINAL Task 1 structural VQA model with a deterministic
explanation pipeline:

```text
Task 1 answer
+ targeted self-probing QA
+ lesion prior / morphology-TDA / OT visual evidence
+ Bayes-gated confidence
→ clinician-oriented textual explanation + heatmap
```

## Files

- `submission_task2.jsonl` — one explanation record per official Subtask 2
  validation item.
- `visuals/` — heatmap overlays and evidence JSON files referenced by JSONL
  rows.
- `submission_task2.py` — team/submission metadata.

## Base VQA Model

Answers are generated with the FINAL Task 1 model:

```text
Qwen2.5-3B-Instruct
+ LoRA r16
+ A3 prior-guided OT prefix fusion
+ lesion prior mask
+ morphology/topology structural features
```

The Task 2 pipeline does not rewrite the answer. Self-probing is used only to
make explanations richer and to calibrate confidence.

## Explanation Method

For each image-question pair:

1. The FINAL Task 1 model generates the primary answer.
2. The question is mapped to a coarse family such as color, location, count,
   instrument, finding presence, lesion attribute, or size.
3. A small bank of targeted follow-up questions is selected.
4. The same VQA model answers those probes.
5. Structural visual evidence is computed from:
   - lesion prior mask,
   - morphology/TDA patch features,
   - global structural features,
   - OT transport attention.
6. Probe consistency and structural evidence are combined by a Bayes-gated
   confidence model.
7. A deterministic clinician-oriented explanation is written.

## Visual Explanation

Each heatmap combines:

```text
0.35 * lesion prior
+ 0.30 * morphology/TDA saliency
+ 0.30 * OT patch attention
+ 0.05 * tissue support
```

Specular artifact regions reduce heatmap support to avoid over-emphasizing
reflections.

## Self-Probing

The self-probing stage is inspired by the Medico 2025 winning explanation
strategy, but adapted to our structural model. Examples:

- color questions probe dominant color and redness;
- location questions probe central/left/right/upper/lower evidence;
- count questions probe whether one or multiple regions are visible;
- instrument questions probe instrument presence and location;
- finding/presence questions probe visible mucosal abnormality, focality,
  texture/color support, and artifact risk.

Probe answers are not treated as ground truth. They are used to estimate whether
the main answer is internally supported by targeted follow-up reasoning.

## Confidence

The confidence score is a Bayes-gated reliability estimate, not a clinical
probability. Inputs include:

- prior/topology agreement,
- OT/prior agreement,
- evidence concentration,
- tissue coverage,
- specular burden,
- OT cost,
- answer specificity,
- question-answer compatibility,
- probe support,
- probe consistency,
- probe conflict penalty.

Low confidence, diffuse evidence, artifacts, or contradictory probe answers
trigger cautious wording.

## Safety Notes

The explanations are intended to support clinician review, not replace it.
Presence/absence questions are handled conservatively because subtle findings can
be ambiguous. The method does not use hidden ground-truth answers when generating
answers, probes, explanations, confidence, or visual evidence.
