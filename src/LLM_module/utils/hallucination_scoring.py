"""
Faithfulness & Evidence Quality Scoring Module (v2 — Research-backed redesign)

Addresses key issues from v1:
- NLI premise truncation at 512 chars lost most context
- KG faithfulness used noisy NLI on edge tables (not natural language)
- gt_faithfulness was silently conflated with kg_faithfulness
- Evidence quality was purely citation count (biased against naive)

This module implements FOUR orthogonal dimensions:

1. **GT-Faithfulness** (AlignScore-style chunked NLI, Zha et al. ACL 2023):
   - Splits long premises into overlapping chunks
   - Per-sentence max across chunks (best-aligned chunk wins)
   - Bidirectional: forward (GT→text) and reverse (text→GT)
   - Final score: arithmetic mean of forward and reverse

2. **KG-Faithfulness** (SummaC-style, Laban et al. TACL 2022):
   - Embeds each generated sentence and each KG edge independently
   - Per-sentence: max cosine similarity to any KG edge
   - Score: mean of per-sentence max similarities (soft coverage)
   - Properly measures KG grounding without NLI truncation

3. **Evidence Quality** (Semantic Information Density):
   - Fraction of generated sentences semantically grounded in KG edges
   - Uses same embeddings as KG-faithfulness (no extra compute)
   - FAIR to both naive and grounded methods (no citation-count bias)

4. **Hallucination Score**: 1 - gt_faithfulness

References:
- AlignScore (Zha et al., ACL 2023): Chunked NLI for long-document faithfulness
- SummaC (Laban et al., TACL 2022): NLI-based factual consistency
- BERTScore (Zhang et al., ICLR 2020): Embedding-based soft matching
- RAGAS (Es et al., 2023): Multi-dimensional RAG evaluation
- UniEval (Zhong et al., EMNLP 2022): Multi-aspect text evaluation
- TRUE (Honovich et al., NAACL 2022): Standardized factual consistency
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from LLM_module import eval_config as ecfg
from LLM_module.utils.common import clamp01, count_citations, strip_citations, resolve_device


# Sentence splitting regex
SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# Pattern to locate Section 2 (Mechanistic Summary) in structured output
_SECTION2_START = re.compile(
    r'(?:^|\n)\s*2\s*\)\s*[Mm]echanistic\s+[Ss]ummary',
    flags=re.MULTILINE,
)
_NEXT_SECTION = re.compile(
    r'(?:^|\n)\s*3\s*\)',
    flags=re.MULTILINE,
)


def _extract_mechanistic_section(text: str) -> str:
    """Return the text of Section 2 (Mechanistic Summary) if present,
    otherwise return the full text (for naive outputs without sections)."""
    m_start = _SECTION2_START.search(text)
    if not m_start:
        return text          # unstructured (e.g. naive) — score everything
    begin = m_start.end()
    m_end = _NEXT_SECTION.search(text, begin)
    section = text[begin : m_end.start()] if m_end else text[begin:]
    section = section.strip()
    return section if len(section) >= 60 else text


def _split_sentences(text: str, min_chars: int = 25) -> List[str]:
    """Split text into sentences, filtering short ones."""
    text_s = str(text or "").strip()
    if not text_s:
        return []
    text_clean = strip_citations(text_s)
    sentences = SENT_RE.split(text_clean)
    return [s.strip() for s in sentences if len(s.strip()) >= min_chars]


@lru_cache(maxsize=1)
def _get_nli_pipeline():
    from transformers import pipeline
    from LLM_module.utils.common import get_pipeline_device

    device_str = resolve_device(ecfg.HALLUCINATION_NLI_DEVICE)
    local_only = bool(getattr(ecfg, "HALLUCINATION_NLI_LOCAL_ONLY", False))
    return pipeline(
        "text-classification",
        model=ecfg.HALLUCINATION_NLI_MODEL_PATH,
        tokenizer=ecfg.HALLUCINATION_NLI_MODEL_PATH,
        device=get_pipeline_device(device_str),
        top_k=None,
        local_files_only=local_only,
    )


# ──────────────────────────────────────────────────────────────────────────────
# AlignScore-style Chunked NLI  (Zha et al., ACL 2023)
# ──────────────────────────────────────────────────────────────────────────────

_CHUNK_SIZE = 1200     # characters per premise chunk (~300 tokens for PubMedBERT)
_CHUNK_OVERLAP = 300   # overlap between consecutive chunks


def _sigmoid01(x: float, *, center: float, scale: float) -> float:
    """Map a scalar to (0, 1) with a calibrated logistic.

    This avoids hard cutoffs from linear clamp normalization while still
    producing a bounded, monotonic score.

    Args:
        x: Input scalar (e.g., cosine similarity).
        center: Value that maps to 0.5.
        scale: Controls slope; smaller = steeper.
    """
    if scale <= 0:
        return 0.0

    z = (x - center) / scale
    # prevent overflow in exp for extreme z
    z = max(min(z, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-z))


def _cosine_to_unit_interval(cos_sim: float) -> float:
    """Map cosine similarity in [-1, 1] to [0, 1] without extra calibration.

    Uses angular similarity:
        score = 1 - arccos(cos_sim) / pi

    Properties:
    - Monotonic in cos_sim
    - score(-1)=0, score(0)=0.5, score(1)=1
    - No arbitrary (center/scale) parameters
    """
    x = float(cos_sim)
    # numerical stability
    if x > 1.0:
        x = 1.0
    elif x < -1.0:
        x = -1.0
    return 1.0 - (math.acos(x) / math.pi)


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE,
                overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks for AlignScore-style scoring."""
    text = str(text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _compute_nli_scores_single_premise(
    premise: str,
    hypothesis_sentences: List[str],
) -> List[Dict[str, float]]:
    """Run NLI for one premise against multiple hypotheses."""
    if not hypothesis_sentences or not premise:
        return []

    pipe = _get_nli_pipeline()
    results = []

    for hypo in hypothesis_sentences[:ecfg.HALLUCINATION_MAX_SENTENCES]:
        hypo_trunc = hypo[:400]
        nli_input = f"{premise} </s></s> {hypo_trunc}"
        try:
            result = pipe(nli_input, truncation=True, max_length=512)
            scores_list = result[0] if isinstance(result, list) and result else result
            scores = {r["label"].lower(): r["score"] for r in scores_list}
            ent = scores.get("entailment", scores.get("entail", 0.0))
            neu = scores.get("neutral", 0.0)
            con = scores.get("contradiction", scores.get("contradict", 0.0))
            results.append({"entailment": ent, "neutral": neu, "contradiction": con})
        except Exception:
            results.append({"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0})

    return results


def _compute_chunked_nli_scores(
    premise: str,
    hypothesis_sentences: List[str],
) -> Dict[str, Any]:
    """
    AlignScore-style chunked NLI (Zha et al., ACL 2023).

    Split the premise into overlapping chunks, run NLI per chunk,
    and for each hypothesis sentence take the MAX entailment score
    across all chunks.
    """
    if not hypothesis_sentences:
        return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0, "n_sentences": 0}

    chunks = _chunk_text(premise)
    if not chunks:
        return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0, "n_sentences": 0}

    n_hypos = min(len(hypothesis_sentences), ecfg.HALLUCINATION_MAX_SENTENCES)

    best_ent = [0.0] * n_hypos
    best_con = [1.0] * n_hypos
    best_neu = [1.0] * n_hypos

    for chunk in chunks:
        chunk_results = _compute_nli_scores_single_premise(
            chunk, hypothesis_sentences[:n_hypos]
        )
        for i, res in enumerate(chunk_results):
            ent_i = res["entailment"]
            con_i = res["contradiction"]
            neu_i = res["neutral"]
            if ent_i > best_ent[i]:
                best_ent[i] = ent_i
                best_con[i] = con_i
                best_neu[i] = neu_i

    mean_ent = sum(best_ent) / n_hypos
    mean_con = sum(best_con) / n_hypos
    mean_neu = sum(best_neu) / n_hypos

    sentence_scores = []
    for i, hypo in enumerate(hypothesis_sentences[:n_hypos]):
        sentence_scores.append({
            "sentence": hypo[:100] + "..." if len(hypo) > 100 else hypo,
            "entailment": best_ent[i],
            "neutral": best_neu[i],
            "contradiction": best_con[i],
        })

    return {
        "entailment": mean_ent,
        "neutral": mean_neu,
        "contradiction": mean_con,
        "n_sentences": n_hypos,
        "n_chunks": len(chunks),
        "worst_sentences": sorted(sentence_scores, key=lambda x: x["entailment"])[:ecfg.HALLUCINATION_WORST_K],
        "means": {"entailment": mean_ent, "neutral": mean_neu, "contradiction": mean_con},
    }


def compute_faithfulness_score(
    *,
    text: str,
    ground_truth_explanation: str,
    prompt_context: str = "",
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute GT-faithfulness with AlignScore-style chunked NLI.
    (Zha et al., ACL 2023)

    Two directions:
    1. Forward (GT->text): premise=GT chunks, hypothesis=generated sentences.
    2. Reverse (text->GT): premise=generated text chunks, hypothesis=GT sentences.

    Final: arithmetic mean (UniEval-style, less punitive than geometric mean).
    """
    text_s = str(text or "").strip()
    gt_s = str(ground_truth_explanation or "").strip()
    citation_count, unique_count = count_citations(text_s)

    base_details: Dict[str, Any] = {
        "backend": "chunked_nli",
        "citation_count": citation_count,
        "unique_citation_count": unique_count,
        "has_citations": citation_count >= 2,
        "gt_faithfulness": 0.0,
        "faithfulness_score": 0.0,
        "enabled": False,
    }

    if not gt_s:
        return 0.0, {**base_details, "reason": "no_ground_truth"}

    # Forward: GT -> text (section-aware)
    mech_section = _extract_mechanistic_section(text_s)
    fwd_sentences = _split_sentences(mech_section)
    if not fwd_sentences:
        fwd_sentences = _split_sentences(text_s)
    if not fwd_sentences:
        return 0.0, {**base_details, "reason": "no_sentences"}

    fwd_nli = _compute_chunked_nli_scores(premise=gt_s, hypothesis_sentences=fwd_sentences)
    fwd_ent = fwd_nli.get("entailment", 0.0)
    fwd_con = fwd_nli.get("contradiction", 0.0)
    fwd_net = float(fwd_ent) - float(fwd_con)
    fwd_faith = clamp01((fwd_net + 1.0) / 2.0)

    # Reverse: text -> GT
    gt_sentences = _split_sentences(gt_s, min_chars=15)
    if gt_sentences:
        text_premise = strip_citations(text_s)
        rev_nli = _compute_chunked_nli_scores(premise=text_premise, hypothesis_sentences=gt_sentences)
        rev_ent = rev_nli.get("entailment", 0.0)
        rev_con = rev_nli.get("contradiction", 0.0)
        rev_net = float(rev_ent) - float(rev_con)
        rev_faith = clamp01((rev_net + 1.0) / 2.0)
    else:
        rev_faith = fwd_faith
        rev_nli = {}

    # Arithmetic mean (UniEval, Zhong et al., EMNLP 2022)
    gt_faithfulness = clamp01(0.5 * fwd_faith + 0.5 * rev_faith)

    return gt_faithfulness, {
        "backend": "chunked_nli",
        "model_path": ecfg.HALLUCINATION_NLI_MODEL_PATH,
        "device": ecfg.HALLUCINATION_NLI_DEVICE,
        "n_sentences": fwd_nli.get("n_sentences", 0),
        "n_gt_sentences": len(gt_sentences) if gt_sentences else 0,
        "n_chunks_fwd": fwd_nli.get("n_chunks", 1),
        "n_chunks_rev": rev_nli.get("n_chunks", 1) if rev_nli else 0,
        "premise_len": len(gt_s),
        "means": fwd_nli.get("means", {}),
        "worst_sentences": fwd_nli.get("worst_sentences", []),
        "citation_count": citation_count,
        "unique_citation_count": unique_count,
        "has_citations": citation_count >= 2,
        "components": {
            "mean_entailment": fwd_ent,
            "per_gt_sentence_entailment": [
                s.get("entailment", 0.0) for s in (rev_nli.get("worst_sentences") or [])
            ] if rev_nli else [],
            "forward_faithfulness": fwd_faith,
            "reverse_faithfulness": rev_faith,
        },
        "gt_faithfulness": gt_faithfulness,
        "faithfulness_score": gt_faithfulness,
        "enabled": True,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SummaC-style Embedding-based KG Faithfulness  (Laban et al., TACL 2022)
# ──────────────────────────────────────────────────────────────────────────────

_KG_EDGE_RE = re.compile(
    r"([A-Za-z0-9_\-/\s]+?)\s*->\s*([A-Za-z0-9_\-/\s]+?)\s*\|\s*([^\|]+?)\s*\|",
)


def _extract_kg_edges_as_sentences(prompt_context: str) -> List[str]:
    """Extract KG edges and convert to natural-language fragments for embedding."""
    if not prompt_context:
        return []

    edges = []
    seen = set()

    for m in _KG_EDGE_RE.finditer(prompt_context):
        src = m.group(1).strip()
        tgt = m.group(2).strip()
        rel = m.group(3).strip().replace("_", " ").strip()
        edge_text = f"{src} {rel} {tgt}"
        key = edge_text.lower()
        if key not in seen and len(edge_text) >= 8:
            seen.add(key)
            edges.append(edge_text)

    # Also extract descriptive lines (node/pathway descriptions)
    for line in prompt_context.split("\n"):
        line = line.strip()
        if not line or len(line) < 15:
            continue
        if line.startswith("Edges") or line.startswith("Nodes") or "->" in line:
            continue
        if line.startswith("##") or line.startswith("==="):
            continue
        if ":" in line and 20 < len(line) < 500:
            desc = line.split(":", 1)[1].strip()
            if desc and len(desc) >= 15:
                key = desc.lower()[:80]
                if key not in seen:
                    seen.add(key)
                    edges.append(desc)

    return edges[:200]


@lru_cache(maxsize=1)
def _get_st_embedder():
    """Load SentenceTransformer for embedding-based similarity."""
    from sentence_transformers import SentenceTransformer
    from pathlib import Path

    candidates = [
        ecfg.EXPERT_JUDGE_MODEL_PATH,
        "/data/guoyu/HF-models/all-MiniLM-L6-v2",
        "/data/guoyu/HF-models/sentence-transformers/all-MiniLM-L6-v2",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return SentenceTransformer(p)
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def compute_kg_faithfulness(
    *,
    text: str,
    prompt_context: str,
) -> Tuple[float, Dict[str, Any]]:
    """Compute KG-faithfulness using SummaC-style embedding similarity.
    (Laban et al., TACL 2022, adapted for KG grounding)

    Embeds generated sentences and KG edges, computes per-sentence max
    cosine similarity. Self-refine/CoVe should score higher because they
    tighten claims to match KG edges more precisely.
    """
    text_s = str(text or "").strip()
    if not prompt_context or not text_s:
        return 0.0, {"reason": "no_kg_context", "kg_faithfulness": 0.0}

    mech_section = _extract_mechanistic_section(text_s)
    sentences = _split_sentences(mech_section)
    if not sentences:
        sentences = _split_sentences(text_s)
    if not sentences:
        return 0.0, {"reason": "no_sentences", "kg_faithfulness": 0.0}

    kg_edges = _extract_kg_edges_as_sentences(prompt_context)
    if not kg_edges:
        return 0.0, {"reason": "no_kg_edges", "kg_faithfulness": 0.0}

    try:
        import numpy as np
        embedder = _get_st_embedder()
        sent_vecs = embedder.encode(
            sentences[:ecfg.HALLUCINATION_MAX_SENTENCES],
            normalize_embeddings=True, show_progress_bar=False,
        )
        edge_vecs = embedder.encode(
            kg_edges, normalize_embeddings=True, show_progress_bar=False,
        )
    except Exception as e:
        return 0.0, {"reason": f"embedding_error: {e}", "kg_faithfulness": 0.0}

    sim_matrix = np.dot(sent_vecs, edge_vecs.T)
    per_sent_max = sim_matrix.max(axis=1)

    kg_faith_raw = float(np.mean(per_sent_max))

    # Parameter-free mapping for cosine similarity -> [0, 1]
    kg_faith_scaled = float(_cosine_to_unit_interval(kg_faith_raw))

    grounding_threshold = float(getattr(ecfg, "HALLUCINATION_GROUNDING_SIM_THRESHOLD", 0.35))
    n_grounded = int(np.sum(per_sent_max >= grounding_threshold))
    n_total = len(per_sent_max)

    return kg_faith_scaled, {
        "kg_faithfulness": kg_faith_scaled,
        "kg_faithfulness_raw": kg_faith_raw,
        "normalization": {
            "type": "angular_cosine",
            "formula": "1 - arccos(cos_sim)/pi",
        },
        "n_sentences": n_total,
        "n_kg_edges": len(kg_edges),
        "n_grounded_sentences": n_grounded,
        "grounding_ratio": n_grounded / max(n_total, 1),
        "per_sent_max_sim": {
            "mean": float(np.mean(per_sent_max)),
            "median": float(np.median(per_sent_max)),
            "min": float(np.min(per_sent_max)),
            "max": float(np.max(per_sent_max)),
        },
        "method": "summac_embedding",
    }


def compute_hallucination_metrics(
    *,
    text: str,
    ground_truth_explanation: str,
    prompt_context: str = "",
) -> Tuple:
    """Compute all faithfulness/hallucination metrics.

    Returns THREE clearly separated dimensions:
    - gt_faithfulness: NLI-based faithfulness against ground truth
    - kg_faithfulness: Embedding-based faithfulness against KG context
    - hallucination_score: 1 - gt_faithfulness
    """
    # GT faithfulness (AlignScore-style chunked NLI)
    gt_faith, details = compute_faithfulness_score(
        text=text,
        ground_truth_explanation=ground_truth_explanation,
        prompt_context=prompt_context,
    )

    # KG faithfulness (SummaC-style embedding similarity)
    kg_faith, kg_details = compute_kg_faithfulness(
        text=text,
        prompt_context=prompt_context,
    )

    # Set ALL fields explicitly
    details["hallucination_score"] = 1.0 - gt_faith
    details["faithfulness_score"] = gt_faith
    details["gt_faithfulness"] = gt_faith
    details["kg_faithfulness"] = kg_faith
    details["kg_details"] = kg_details

    return 1.0 - gt_faith, details
