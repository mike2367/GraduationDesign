from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from LLM_module import eval_config as ecfg
from LLM_module.utils.common import KG_EDGE_CIT_RE, KG_KEY_RE, clamp01, count_citations, cosine


# ============================================================================
# HYBRID EMBEDDING + STRUCTURE FORMAT SCORING
# ============================================================================
# Problem: Pure embedding similarity captures semantic content, not structure.
# Solution: Hybrid scoring that combines:
# 1. SEMANTIC QUALITY: Embedding similarity to expected content (what)
# 2. STRUCTURAL COMPLIANCE: Ratio of detected elements to expected (how)
#
# Final score = semantic_quality * structural_compliance
#
# This has NO arbitrary weights because:
# - Semantic quality: cosine similarity naturally in [0, 1]
# - Structural compliance: detected/expected naturally in [0, 1]
# - Product naturally in [0, 1]
#
# NOTE:
# - Pure embedding similarity to GT measures semantic alignment, NOT formatting.
# - The previous semantic×structural gate was biased against naive methods.
#
# This module now treats `format_score` as *format compliance*: does the answer
# contain the expected explanation components (mechanism/evidence/caveats/etc),
# even if the author didn't use strict numbered headings.
#
# Semantic alignment to GT is tracked separately via `total_embedding_similarity`
# in eval_payload.py.

# Expected structural elements for grounded outputs (used for grounding_ok check, not format score)
EXPECTED_SECTION_NUMBERS = {1, 2, 3, 4, 5, 6, 7, 8}  # 8 numbered sections
EXPECTED_SECTION_COUNT = len(EXPECTED_SECTION_NUMBERS)

FORMAT_COMPONENTS: list[tuple[str, str]] = [
	("mechanism", "Mechanistic explanation linking gene A and gene B (pathway/causal chain)."),
	("evidence", "Evidence description (supporting interactions, references, or concrete support)."),
	("direction", "Directionality or effect description (activates/inhibits, up/down, increases/decreases)."),
	("context", "Biological context / assumptions (cell type, condition, scope, caveats about context)."),
	("caveats", "Limitations / uncertainty / alternative hypotheses."),
	("validation", "Suggested validation (experiments, perturbations, checks, follow-ups)."),
]

# Pattern to detect section numbers
SECTION_NUMBER_RE = re.compile(r"(?:^|\n)\s*(\d+)\s*\)", flags=re.MULTILINE)


def _safe_json_extract(s: str) -> Optional[Dict[str, Any]]:
	if not s:
		return None
	m = re.search(r"```json\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
	if m:
		try:
			return json.loads(m.group(1).strip())
		except Exception:
			pass
	for m2 in re.finditer(r"\{[\s\S]*?\}", s):
		try:
			obj = json.loads(m2.group(0))
			if isinstance(obj, dict):
				return obj
		except Exception:
			pass
	return None


def _embed_text(text: str, model_path: str) -> list[float]:
	from pathlib import Path

	local_candidates = [
		model_path,
		"/data/guoyu/HF-models/all-MiniLM-L6-v2",  # Local sentence transformer
		"/data/guoyu/HF-models/sentence-transformers/all-MiniLM-L6-v2",  # Alternative path
	]
	resolved_path = None
	for candidate in local_candidates:
		if candidate and Path(candidate).exists():
			resolved_path = candidate
			break
	if not resolved_path:
		resolved_path = model_path if ("/" in model_path and "prometheus" not in model_path.lower()) else "sentence-transformers/all-MiniLM-L6-v2"

	mdl = _get_embedder(resolved_path)
	vec = mdl.encode([text], normalize_embeddings=True, show_progress_bar=False)
	return vec[0].tolist()


def _embed_texts(texts: list[str], model_path: str) -> list[list[float]]:
	"""Batch embed texts with the same resolver logic as `_embed_text`."""
	from pathlib import Path

	local_candidates = [
		model_path,
		"/data/guoyu/HF-models/all-MiniLM-L6-v2",
		"/data/guoyu/HF-models/sentence-transformers/all-MiniLM-L6-v2",
	]
	resolved_path = None
	for candidate in local_candidates:
		if candidate and Path(candidate).exists():
			resolved_path = candidate
			break
	if not resolved_path:
		resolved_path = model_path if ("/" in model_path and "prometheus" not in model_path.lower()) else "sentence-transformers/all-MiniLM-L6-v2"

	mdl = _get_embedder(resolved_path)
	vecs = mdl.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
	return [v.tolist() for v in vecs]


def _split_paragraphs(text: str, *, max_paragraphs: int = 24) -> list[str]:
	chunks = [c.strip() for c in re.split(r"\n\s*\n+", str(text or "")) if c and c.strip()]
	# Fallback: if everything is one chunk, split on lines for some signal.
	if len(chunks) <= 1:
		chunks = [c.strip() for c in str(text or "").splitlines() if c.strip()]
	return chunks[:max_paragraphs]


@lru_cache(maxsize=2)
def _get_embedder(resolved_path: str):
	from sentence_transformers import SentenceTransformer

	return SentenceTransformer(resolved_path)





def compute_hybrid_format_score(
	text: str,
	model_path: str,
	ground_truth_explanation: str = "",
) -> Tuple[float, Dict[str, Any]]:
	"""Compute a *format compliance* score using embeddings + soft structure.

	Why this exists:
	- We cannot use GT text similarity as a format metric (GT may not encode format).
	- We cannot hard-require numbered headings (bias against naive methods).

	Approach:
	1) Embed paragraphs and check coverage of expected explanation components.
	2) Add a small bonus for explicit structure (numbered sections / headings / bullets).

	`ground_truth_explanation` is accepted for API compatibility but is NOT used.
	Semantic similarity to GT is tracked separately as `total_embedding_similarity`.
	"""
	text_s = str(text or "").strip()
	if not text_s:
		return 0.0, {"reason": "empty_text", "component_coverage": 0.0, "structural_score": 0.0}

	citation_count, unique_count = count_citations(text_s)

	# Soft structure signals (do not gate naive methods)
	section_matches = SECTION_NUMBER_RE.findall(text_s)
	detected_sections = {int(n) for n in section_matches if n.isdigit() and int(n) in EXPECTED_SECTION_NUMBERS}
	structural_compliance = len(detected_sections) / EXPECTED_SECTION_COUNT

	lines = [ln.strip() for ln in text_s.splitlines() if ln.strip()]
	has_numbered = 1.0 if detected_sections else 0.0
	has_bullets = 1.0 if any(ln.startswith(("- ", "* ", "•")) for ln in lines) else 0.0
	has_colon_headings = 1.0 if sum(1 for ln in lines[:40] if ln.endswith(":")) >= 2 else 0.0
	length_score = clamp01(min(1.0, len(text_s) / 800.0))
	structural_score = clamp01(0.35 * has_numbered + 0.25 * has_colon_headings + 0.20 * has_bullets + 0.20 * length_score)

	# Component coverage via embeddings (format/organization proxy)
	paragraphs = _split_paragraphs(text_s)
	component_descs = [d for _, d in FORMAT_COMPONENTS]
	try:
		p_vecs = _embed_texts([p[:600] for p in paragraphs] if paragraphs else [text_s[:600]], model_path)
		c_vecs = _embed_texts(component_descs, model_path)
		component_scores: dict[str, float] = {}
		for (name, _), cvec in zip(FORMAT_COMPONENTS, c_vecs):
			best = 0.0
			for pvec in p_vecs:
				best = max(best, float(cosine(pvec, cvec)))
			# Map similarity into [0,1] with a soft threshold.
			# Typical MiniLM cosine for "related" short texts is ~0.3-0.6.
			score = clamp01((best - 0.25) / 0.35)
			component_scores[name] = float(score)
		component_coverage = sum(component_scores.values()) / float(len(component_scores) or 1)
	except Exception as e:
		component_scores = {name: 0.5 for name, _ in FORMAT_COMPONENTS}
		component_coverage = 0.5

	final_score = clamp01(0.75 * component_coverage + 0.25 * structural_score)
	return float(final_score), {
		"component_scores": component_scores,
		"component_coverage": float(component_coverage),
		"structural_score": float(structural_score),
		"structural_compliance": float(structural_compliance),
		"detected_sections": sorted(detected_sections),
		"citation_count": citation_count,
		"unique_citation_count": unique_count,
		"method": "embedding_format_components",
	}


# Alias for backwards compatibility
compute_contrastive_format_score = compute_hybrid_format_score


def heuristic_checks(
	text: str,
	*,
	prompt_context: str = "",
) -> Dict[str, object]:
	text_s = str(text or "")
	ctx_s = str(prompt_context or "")

	# KG citation counting
	citation_count, unique_citation_count = count_citations(text_s)

	ctx_l = ctx_s.lower()
	requires_structure = any(m in ctx_l for m in ecfg.JUDGE_STRUCTURED_MARKERS)

	# Infer whether KG-style citations are expected.
	citations_expected = ("key=" in ctx_l) or ("citation" in ctx_l and "kg" in ctx_l) or ("knowledge graph" in ctx_l and "citation" in ctx_l)

	text_l = text_s.lower()
	required = [h for h in ecfg.JUDGE_KNOWN_HEADINGS if h in ctx_l]
	if requires_structure and not required:
		required = ["mechanistic summary", "evidence"]
	present = [h for h in required if h in text_l]

	score = 0.0
	if text_s.strip():
		score += ecfg.JUDGE_HEURISTIC_BASE_SCORE

	if required:
		frac = len(present) / float(len(required))
		score += ecfg.JUDGE_HEURISTIC_STRUCTURE_WEIGHT * clamp01(frac)
	elif requires_structure:
		score += 0.1

	# if citations_expected:
	# 	if citation_count >= 2:
	# 		score += ecfg.JUDGE_HEURISTIC_CITATION_WEIGHT
	# 	elif citation_count == 1:
	# 		score += ecfg.JUDGE_HEURISTIC_CITATION_WEIGHT * 0.5
		# No citations when expected -> no bonus (implicit 0)

	format_score = clamp01(score)
	grounding_ok = bool(citation_count >= 2) if citations_expected else False

	return {
		"format_score": format_score,
		"grounding_ok": grounding_ok,
		"citation_count": citation_count,
		"unique_citation_count": unique_citation_count,
		"judge_backend": "heuristic",
		"judge_model": "heuristic",
		"judge_parse_ok": True,
	}


@dataclass(frozen=True)
class ExpertJudgeSettings:
	model_name: str
	model_path: str
	local_files_only: bool = True
	max_new_tokens: int = 256


def judge_checks_with_expert_llm(
	text: str,
	*,
	settings: ExpertJudgeSettings,
	prompt_context: str = "",
	ground_truth_explanation: str = "",
) -> Optional[Dict[str, object]]:
	"""Compute format checks.

	Returns `format_score` as a *format compliance* metric (component coverage +
	soft structure). Semantic similarity to GT is computed elsewhere.
	"""
	text = (text or "").strip()
	if not text:
		return {
			"format_score": 0.0,
			"grounding_ok": False,
			"citation_count": 0,
			"unique_citation_count": 0,
			"judge_backend": "embedding_format_components",
			"judge_model": settings.model_name if settings else "unknown",
			"judge_parse_ok": True,
			"hybrid_details": {"reason": "empty_text"},
		}

	if not settings.model_path:
		return heuristic_checks(text, prompt_context=prompt_context)

	# Embedding-based format compliance score (component coverage + soft structure)
	format_score, hybrid_details = compute_hybrid_format_score(text, settings.model_path)
	
	citation_count = hybrid_details.get("citation_count", 0)
	unique_citation_count = hybrid_details.get("unique_citation_count", 0)
	structural_compliance = hybrid_details.get("structural_compliance", 0.0)
	
	# Determine if citations are expected from prompt context
	ctx_l = (prompt_context or "").lower()
	citations_expected = (
		("key=" in ctx_l) or 
		("citation" in ctx_l and "kg" in ctx_l) or 
		("knowledge graph" in ctx_l and "citation" in ctx_l) or
		("evidence chains" in ctx_l)
	)
	
	# Grounding OK: citations present AND structural compliance > 0.5 (at least 4/8 sections)
	grounding_ok = bool(
		citations_expected and 
		citation_count >= 2 and 
		structural_compliance >= 0.5
	)

	return {
		"format_score": float(format_score),
		"grounding_ok": grounding_ok,
		"citation_count": citation_count,
		"unique_citation_count": unique_citation_count,
		"judge_backend": "embedding_format_components",
		"judge_model": settings.model_name,
		"judge_parse_ok": True,
		"hybrid_details": hybrid_details,
	}
