"""Post-processing helpers copied for the HF Task 1 submission."""

from .answer_normalization import infer_question_type_for_normalization, normalize_prediction

__all__ = ["normalize_prediction", "infer_question_type_for_normalization"]
