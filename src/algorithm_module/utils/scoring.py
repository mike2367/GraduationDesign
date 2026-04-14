from __future__ import annotations

from typing import Mapping, Optional, Tuple, Dict


def clamp01(x: float) -> float:
	return float(min(1.0, max(0.0, x)))


def edge_score(attrs: Mapping[str, object], *, source_weight: Mapping[str, float], relation_weight: Mapping[str, float]) -> float:
	# Strict lookup:
	# 1. Source weight must be defined (or 0.0). No arbitrary 0.3 fallback.
	# 2. Relation weight must be defined (or 0.0). No arbitrary 0.2 fallback.
	s_w = source_weight.get(str(attrs.get("source") or "unknown"), 0.0)
	r_w = relation_weight.get((etype := str(attrs.get("type") or "related_to")), 0.0)
	
	base = s_w * r_w
	
	if base <= 0.0:
		return 0.0

	attr_factor = 1.0
	if etype == "STRING_association" and (s := attrs.get("score")) is not None:
		attr_factor *= clamp01(float(s))
	if etype == "DepMap_codependency" and (c := attrs.get("corr")) is not None:
		attr_factor *= clamp01(abs(float(c)))
	if etype == "targets" and (ph := attrs.get("phase")) is not None:
		attr_factor *= clamp01(float(ph) / 4.0)
	if etype == "TF_regulates":
		attr_factor *= {"A": 1.00, "B": 0.90, "C": 0.78, "D": 0.60, "E": 0.45}.get((str(attrs.get("level") or "").strip().upper() or "")[:1], 0.70)
	# For driver_in edges, "score" is a q-value (lower = stronger evidence)
	if etype == "driver_in" and (s := attrs.get("score")) is not None:
		try:
			qval = float(s)
			# Map q-value to confidence: qval < 0.01 → boost; qval > 0.25 → penalize
			attr_factor *= clamp01(1.0 - min(1.0, qval / 0.25))
		except (ValueError, TypeError):
			pass
	# For mutated_in edges, mutation count is informative (but typically sparse/stub data)
	if etype == "mutated_in" and (m := attrs.get("mutations")) is not None:
		try:
			# Higher mutation count → slightly higher confidence (capped)
			attr_factor *= clamp01(min(1.0, 0.5 + float(m) * 0.1))
		except (ValueError, TypeError):
			pass
	# Generic FDR/p-value handling for other edge types
	for k in ("fdr", "entities_fdr", "p_value", "pValue"):
		if (v := attrs.get(k)) is not None:
			attr_factor *= clamp01(1.0 - min(1.0, float(v) / 0.25))
			break
	return clamp01(base * attr_factor)


def linear_rank_score(
	features: Mapping[str, float],
	*,
	weights: Mapping[str, float],
	enabled: Optional[Mapping[str, bool]] = None,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
	mask = {k: enabled.get(k, False) for k in features} if enabled is not None else None
	used_weights = {}
	used_features = {}
	for k, v in features.items():
		if (mask is not None and not mask.get(k, True)) or k not in weights:
			continue
		used_features[k] = min(1.0, max(0.0, v))
		used_weights[k] = weights[k]
	total_w = sum(max(0.0, w) for w in used_weights.values())
	if total_w <= 0.0 or not used_weights:
		return 0.0, used_features, used_weights
	used_weights = {k: max(0.0, w) / total_w for k, w in used_weights.items()}
	return min(1.0, max(0.0, sum(w * used_features.get(k, 0.0) for k, w in used_weights.items()))), used_features, used_weights
