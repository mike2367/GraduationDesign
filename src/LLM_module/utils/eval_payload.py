from __future__ import annotations
from hashlib import sha256
from typing import Any, Dict, Mapping, Optional

from LLM_module import eval_config as ecfg
from LLM_module.utils.common import count_citations
from LLM_module.utils.explanation_scoring import feature_embedding_prf1_by_coverage, total_text_embedding_similarity
from LLM_module.utils.metric_aggregation import (
	grounded_feature_parts,
	grounded_feature_score,
)


def hash_text(text: str) -> str:
	return sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _feature_embedding_snapshot() -> Dict[str, object]:
	return {
		"model_path": ecfg.FEATURE_EMBED_MODEL_PATH,
		"device": ecfg.FEATURE_EMBED_DEVICE,
		"max_length": ecfg.FEATURE_EMBED_MAX_LENGTH,
		"batch_size": ecfg.FEATURE_EMBED_BATCH_SIZE,
		"max_candidates": ecfg.FEATURE_MAX_CANDIDATES,
		"similarity_threshold": ecfg.FEATURE_SIM_THRESHOLD,
		"similarity_threshold_adjusted": getattr(ecfg, "FEATURE_SIM_THRESHOLD_ADJUSTED", None),
		"candidate_backend": ecfg.FEATURE_CANDIDATE_BACKEND,
		"tokencls_model_path": ecfg.FEATURE_TOKENCLS_MODEL_PATH,
		"pos_model_path": getattr(ecfg, "FEATURE_POS_MODEL_PATH", ""),
		"seq2seq_model_path": ecfg.FEATURE_SEQ2SEQ_MODEL_PATH,
		"keyphrase_model_local_only": ecfg.FEATURE_KEYPHRASE_LOCAL_ONLY,
		"similarity_mode": ecfg.FEATURE_SIMILARITY_MODE,
		"expert_judge_model_name": ecfg.EXPERT_JUDGE_MODEL_NAME,
		"expert_judge_model_path": ecfg.EXPERT_JUDGE_MODEL_PATH,
		"expert_judge_device": ecfg.EXPERT_JUDGE_DEVICE,
		"expert_judge_local_only": ecfg.EXPERT_JUDGE_LOCAL_ONLY,
	}


def make_pair_payload(
	*,
	gene_a: str,
	gene_b: str,
	prompt_text: str,
	model_payload: Dict[str, object],
	ground_truth_available: bool,
	ground_truth_features: list[str],
	ground_truth_explanation: str,
	prompt_path: Optional[str] = None,
	prompt_hash: Optional[str] = None,
) -> Dict[str, object]:
	"""Unified payload template shared by batch + single-pair evaluators."""
	return {
		"prompt_path": prompt_path,
		"gene_a": gene_a,
		"gene_b": gene_b,
		"prompt_hash": prompt_hash if prompt_hash is not None else hash_text(prompt_text),
		"model": model_payload,
		"ground_truth": {
			"available": bool(ground_truth_available),
			"features": list(ground_truth_features or []),
			"explanation": ground_truth_explanation or "",
		},
		"feature_embedding": _feature_embedding_snapshot(),
		"texts": {"prompt": prompt_text},
		"metrics": {},
	}


def score_text_metrics(
	*,
	ground_truth_features: list[str],
	ground_truth_explanation: str,
	text: str,
	effective_model: object = None,
	prompt_context: str = "",
) -> Dict[str, Any]:

	feat_eval = feature_embedding_prf1_by_coverage(
		ground_truth_features=ground_truth_features,
		text=text,
	)

	from LLM_module.utils.expert_llm_judge import ExpertJudgeSettings, heuristic_checks, judge_checks_with_expert_llm

	mode = str(getattr(ecfg, "EVAL_JUDGE_BACKEND", "auto") or "auto").strip().lower()

	primary: Optional[Dict[str, object]] = None
	checks: Dict[str, object]
	if mode == "heuristic":
		checks = heuristic_checks(text, prompt_context=prompt_context or "")
		checks.setdefault("judge_backend", "heuristic")
		checks["judge_mode"] = "heuristic"
	else:
		judged = judge_checks_with_expert_llm(
			text,
			settings=ExpertJudgeSettings(
				model_name=ecfg.EXPERT_JUDGE_MODEL_NAME or str(ecfg.EXPERT_JUDGE_MODEL_PATH),
				model_path=str(ecfg.EXPERT_JUDGE_MODEL_PATH),
				local_files_only=ecfg.EXPERT_JUDGE_LOCAL_ONLY,
			),
			prompt_context=prompt_context or "",
			ground_truth_explanation=ground_truth_explanation,
		)
		checks = judged
		checks["judge_mode"] = mode
		checks.setdefault("judge_backend", "embedding_format_components")

		if (mode == "auto" and not checks.get("judge_parse_ok", True) and 
		    checks.get("judge_backend") != "embedding_format_components"):
			heur = heuristic_checks(text, prompt_context=prompt_context or "")
			heur.update({
				"judge_mode": "auto",
				"judge_fallback_from": str(checks.get("judge_backend", "expert_llm")),
				"judge_primary_parse_ok": bool(checks.get("judge_parse_ok", False)),
				"judge_primary_format_score": float(checks.get("format_score", 0.0)),
			})
			checks = heur

	checks["judge_backend_used"] = str(checks.get("judge_backend", "unknown"))

	actual_citation_count, actual_unique_count = count_citations(str(text or ""))
	checks["citation_count"] = actual_citation_count
	checks["unique_citation_count"] = actual_unique_count

	if ground_truth_explanation:
		checks["total_embedding_similarity"] = total_text_embedding_similarity(text, ground_truth_explanation)

	from LLM_module.utils.hallucination_scoring import compute_hallucination_metrics

	hall_score, hall_details = compute_hallucination_metrics(
		text=text,
		ground_truth_explanation=ground_truth_explanation,
		prompt_context=prompt_context or "",
	)
	checks["hallucination_score"] = hall_score
	checks["hallucination_details"] = hall_details
	checks["faithfulness_score"] = hall_details.get("faithfulness_score", 0.0)
	checks["gt_faithfulness"] = hall_details.get("gt_faithfulness", 0.0)
	checks["kg_faithfulness"] = hall_details.get("kg_faithfulness", 0.0)

	if actual_citation_count < 2:
		checks["grounding_ok"] = False

	if bool(ecfg.EVAL_DEBUG_JUDGE):
		print(
			f"[judge_counts] citations={actual_citation_count} unique={actual_unique_count} "
			f"grounding_ok={bool(checks.get('grounding_ok'))} format_score={checks.get('format_score')}"
		)

	raw_recall = feat_eval.get("recall") or 0.0
	gate = float(checks.get("format_score", 0.0)) if (checks.get("grounding_ok") and actual_citation_count >= 2) else 0.0

	return {
		"feature_embed_f1_raw": raw_recall,
		"feature_embed_recall_only": raw_recall,
		"feature_embed_f1_raw_full": feat_eval.get("f1"),
		"feature_embed_f1_raw_topk_p50": feat_eval.get("topk", {}).get("p50", {}).get("f1"),
		"feature_embed_f1_raw_topk_p75": feat_eval.get("topk", {}).get("p75", {}).get("f1"),
		"hallucination_score": checks.get("hallucination_score"),
		"faithfulness_score": checks.get("faithfulness_score"),
		"kg_faithfulness": checks.get("kg_faithfulness"),
		"feature_embed_f1": float(raw_recall) * gate,
		"feature_embed_gate": gate,
		"grounded_feature_score": grounded_feature_score(raw_recall, checks),
		"grounded_feature_parts": grounded_feature_parts(raw_recall, checks),
		"feature_embed_details": feat_eval,
		"feature_embed_precision_raw": feat_eval.get("precision"),
		"feature_embed_recall_raw": feat_eval.get("recall"),
		"checks": checks,
		"effective_model": effective_model,
	}
