"""LLM evaluation metric aggregation.

This module provides composite scores for evaluation.

Currently supported composite metric:
- Grounded Feature Score = recall × harmonic_mean(GT-faithfulness, KG-faithfulness)

Deprecated metrics (removed):
- Comparable Quality
"""
from __future__ import annotations

from typing import Dict, Optional

from LLM_module.utils.common import clamp01


def _harmonic_mean_2(a: float, b: float) -> float:
    """Harmonic mean of two values."""
    if a <= 0 or b <= 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def grounded_feature_parts(
    feature_recall: Optional[float],
    checks: Dict[str, object],
) -> Dict[str, float]:
    """
    Compute grounded feature score components.

    Score = recall × harmonic_mean(GT-faithfulness, KG-faithfulness)

    Based on RAGAS (2023) which uses harmonic mean for composite metrics.
    - Recall measures content coverage
    - Harmonic mean of two faithfulness dimensions measures grounding quality
    - Self-refine/CoVe improve KG-faithfulness → higher grounded score
    """
    recall = float(feature_recall or 0.0)
    gt_faith = float(checks.get("faithfulness_score") or 0.0)
    kg_faith = float(checks.get("kg_faithfulness") or 0.0)

    # Harmonic mean of two faithfulness dimensions
    combined_faith = _harmonic_mean_2(gt_faith, kg_faith)

    score = recall * combined_faith

    return {
        "recall": recall,
        "gt_faithfulness": gt_faith,
        "kg_faithfulness": kg_faith,
        "combined_faithfulness": combined_faith,
        "score": clamp01(score),
    }


def grounded_feature_score(
    feature_recall: Optional[float],
    checks: Dict[str, object],
) -> float:
    """Return grounded feature score = recall × hmean(GT-faith, KG-faith)."""
    parts = grounded_feature_parts(feature_recall, checks)
    return float(parts["score"])

