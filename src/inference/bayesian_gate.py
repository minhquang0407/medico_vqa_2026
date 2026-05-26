"""Bayes-gated confidence utilities for Medico 2026 Task 2 explanations.

This module is intentionally lightweight and deterministic. It converts
interpretable evidence signals (structural agreement, OT agreement, artifact
burden, answer specificity, etc.) into a calibrated-ish confidence score using a
log-odds model. The score is not a medical probability; it is an explanation
reliability estimate for submission metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Dict, Iterable, List, Mapping, MutableMapping


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


@dataclass
class BayesianGateConfig:
    """Weights for the explanation reliability gate."""

    bias: float = -0.15
    prior_topo_agreement_weight: float = 1.10
    ot_prior_agreement_weight: float = 0.90
    evidence_concentration_weight: float = 0.75
    tissue_coverage_weight: float = 0.45
    answer_specificity_weight: float = 0.35
    answer_question_compatibility_weight: float = 0.25
    diffuse_evidence_penalty_weight: float = 0.55
    specular_penalty_weight: float = 0.60
    high_ot_cost_penalty_weight: float = 0.50
    generic_answer_penalty_weight: float = 0.35
    probe_support_weight: float = 0.45
    probe_consistency_weight: float = 0.55
    probe_specificity_weight: float = 0.20
    probe_conflict_penalty_weight: float = 0.75
    min_confidence: float = 0.05
    max_confidence: float = 0.95


@dataclass
class BayesianExplanationGate:
    """Compute confidence and caution labels from multimodal evidence."""

    config: BayesianGateConfig = field(default_factory=BayesianGateConfig)

    def score(self, evidence: Mapping[str, float | str | bool]) -> Dict[str, object]:
        c = self.config
        components: MutableMapping[str, float] = {}

        prior_topo = _clamp(float(evidence.get("prior_topo_agreement", 0.5)))
        ot_prior = _clamp(float(evidence.get("ot_prior_agreement", 0.5)))
        concentration = _clamp(float(evidence.get("evidence_concentration", 0.5)))
        tissue = _clamp(float(evidence.get("tissue_coverage", 0.75)))
        specificity = _clamp(float(evidence.get("answer_specificity", 0.55)))
        compatibility = _clamp(float(evidence.get("answer_question_compatibility", 0.60)))
        diffuse_penalty = _clamp(float(evidence.get("diffuse_evidence_penalty", 1.0 - concentration)))
        specular = _clamp(float(evidence.get("specular_penalty", evidence.get("specular_ratio", 0.0))))
        ot_cost_penalty = _clamp(float(evidence.get("high_ot_cost_penalty", 0.35)))
        generic_penalty = _clamp(float(evidence.get("generic_answer_penalty", 0.0)))
        probe_support = _clamp(float(evidence.get("probe_support", 0.5)))
        probe_consistency = _clamp(float(evidence.get("probe_consistency", 0.5)))
        probe_specificity = _clamp(float(evidence.get("probe_specificity", 0.5)))
        probe_conflict = _clamp(float(evidence.get("probe_conflict_penalty", 0.0)))

        components["bias"] = c.bias
        components["prior_topo_agreement"] = c.prior_topo_agreement_weight * (prior_topo - 0.5)
        components["ot_prior_agreement"] = c.ot_prior_agreement_weight * (ot_prior - 0.5)
        components["evidence_concentration"] = c.evidence_concentration_weight * (concentration - 0.5)
        components["tissue_coverage"] = c.tissue_coverage_weight * (tissue - 0.5)
        components["answer_specificity"] = c.answer_specificity_weight * (specificity - 0.5)
        components["answer_question_compatibility"] = c.answer_question_compatibility_weight * (compatibility - 0.5)
        components["diffuse_evidence_penalty"] = -c.diffuse_evidence_penalty_weight * diffuse_penalty
        components["specular_penalty"] = -c.specular_penalty_weight * specular
        components["high_ot_cost_penalty"] = -c.high_ot_cost_penalty_weight * ot_cost_penalty
        components["generic_answer_penalty"] = -c.generic_answer_penalty_weight * generic_penalty
        components["probe_support"] = c.probe_support_weight * (probe_support - 0.5)
        components["probe_consistency"] = c.probe_consistency_weight * (probe_consistency - 0.5)
        components["probe_specificity"] = c.probe_specificity_weight * (probe_specificity - 0.5)
        components["probe_conflict_penalty"] = -c.probe_conflict_penalty_weight * probe_conflict

        logit = float(sum(components.values()))
        confidence = _clamp(_sigmoid(logit), c.min_confidence, c.max_confidence)
        bucket = "high" if confidence >= 0.80 else "moderate" if confidence >= 0.55 else "low"

        caution_reasons: List[str] = []
        if confidence < 0.55:
            caution_reasons.append("overall evidence is limited")
        if concentration < 0.35:
            caution_reasons.append("visual evidence is diffuse")
        if prior_topo < 0.40:
            caution_reasons.append("structural prior and morphology evidence disagree")
        if ot_prior < 0.40:
            caution_reasons.append("model OT attention is weakly aligned with the lesion prior")
        if specular > 0.35:
            caution_reasons.append("specular highlights may affect visual evidence")
        if generic_penalty > 0.5:
            caution_reasons.append("answer is relatively generic")
        if probe_conflict > 0.45:
            caution_reasons.append("probing answers were not fully consistent")
        if probe_support < 0.40:
            caution_reasons.append("targeted self-probing provides limited support")

        return {
            "confidence_score": round(float(confidence), 4),
            "confidence_bucket": bucket,
            "logit": round(logit, 4),
            "components": {key: round(float(value), 4) for key, value in components.items()},
            "caution_reasons": caution_reasons,
        }


def score_evidence(evidence: Mapping[str, float | str | bool]) -> Dict[str, object]:
    """Convenience function using the default gate."""

    return BayesianExplanationGate().score(evidence)
