from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
import threading
from typing import Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from LLM_module import eval_config as ecfg
from LLM_module.utils.common import clamp01, strip_citations, normalize_text, resolve_device, cosine


_WORD_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)


def _is_digit_token(tok: str) -> bool:
    return bool(tok) and tok.isdigit()


def normalize_feature(feature: str) -> str:
    return normalize_text(feature)


def split_feature_field(feature_field: str) -> List[str]:
    raw = str(feature_field or "")
    parts: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        for chunk in re.split(r"[;,]", line):
            c = normalize_feature(chunk)
            if c:
                parts.append(c)
    return parts


def _tokens(s: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(s or "")]


def feature_in_text(text: str, feature: str, *, max_span: int = 80) -> bool:
    t = normalize_text(text)
    f = normalize_feature(feature)
    if not t or not f:
        return False
    ftoks = _tokens(f)
    if not ftoks:
        return False
    if len(ftoks) == 1:
        tok = re.escape(ftoks[0])
        return re.search(rf"\b{tok}\b", t) is not None
    if len(ftoks) == 2:
        a, b = (re.escape(ftoks[0]), re.escape(ftoks[1]))
        pat1 = rf"\b{a}\b.{{0,{max_span}}}\b{b}\b"
        pat2 = rf"\b{b}\b.{{0,{max_span}}}\b{a}\b"
        return re.search(pat1, t) is not None or re.search(pat2, t) is not None
    for tok in ftoks:
        et = re.escape(tok)
        if re.search(rf"\b{et}\b", t) is None:
            return False
    return True


def extract_lexicon_features(text: str, feature_pool: Collection[str], *, max_span: int = 80) -> List[str]:
    hits: List[str] = []
    for f in feature_pool:
        nf = normalize_feature(f)
        if not nf:
            continue
        if feature_in_text(text, nf, max_span=max_span):
            hits.append(nf)
    return sorted(set(hits))


def feature_precision_recall_f1(predicted: Collection[str], ground_truth: Collection[str]) -> Mapping[str, float | None]:
    pred = {normalize_feature(f) for f in predicted if f}
    gt = {normalize_feature(f) for f in ground_truth if f}

    if not gt:
        return {"precision": None, "recall": None, "f1": None}

    tp = len(pred & gt)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _mean_pool_last_hidden(last_hidden, attention_mask):
    import torch  # type: ignore

    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1.0)
    return summed / counts
def _l2_normalize(vec):
    import math
    s = math.sqrt(sum(float(x) * float(x) for x in vec))
    if s <= 1e-9:
        return [0.0 for _ in vec]
    inv = 1.0 / s
    return [float(x) * inv for x in vec]


def total_text_embedding_similarity(text: str, ground_truth_text: str) -> Optional[float]:
    import torch  # type: ignore
    try:
        t_a = str(text or "").strip()
        t_b = str(ground_truth_text or "").strip()
        if not t_a or not t_b:
            return None
        tok, model, device_name = _load_hf_encoder(ecfg.FEATURE_EMBED_MODEL_PATH, ecfg.FEATURE_EMBED_DEVICE)
        dev = resolve_device(device_name)
        max_len = int(getattr(ecfg, "TOTAL_SIM_MAX_LENGTH", ecfg.FEATURE_EMBED_MAX_LENGTH))
        enc_a = tok(t_a, return_tensors="pt", truncation=True, max_length=max_len)
        enc_b = tok(t_b, return_tensors="pt", truncation=True, max_length=max_len)
        try:
            enc_a = {k: v.to(dev) for k, v in enc_a.items()}
            enc_b = {k: v.to(dev) for k, v in enc_b.items()}
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                # Fall back to CPU for this batch
                torch.cuda.empty_cache()
                dev = "cpu"
                enc_a = {k: v.to("cpu") for k, v in enc_a.items()}
                enc_b = {k: v.to("cpu") for k, v in enc_b.items()}
            else:
                raise
        with torch.no_grad():
            out_a = model(**enc_a)
            out_b = model(**enc_b)
        vec_a = _mean_pool_last_hidden(out_a.last_hidden_state, enc_a["attention_mask"]).reshape(-1).tolist()
        vec_b = _mean_pool_last_hidden(out_b.last_hidden_state, enc_b["attention_mask"]).reshape(-1).tolist()
        na = _l2_normalize(vec_a)
        nb = _l2_normalize(vec_b)
        sim = cosine(na, nb)
        return clamp01(sim)
    except Exception:
        return None



@lru_cache(maxsize=2)
def _load_hf_encoder(model_path: str, device: str) -> Tuple[object, object, str]:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    from transformers.utils import logging as hf_logging  # type: ignore
    import torch  # type: ignore
    import os
    import io
    import contextlib
    import warnings

    dev = resolve_device(device)

    prev_verbosity = hf_logging.get_verbosity()
    prev_env = os.environ.get("TRANSFORMERS_VERBOSITY", None)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    hf_logging.set_verbosity_error()
    
    local_only = bool(os.path.exists(model_path))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
                load_kwargs: dict[str, object] = {
                    "low_cpu_mem_usage": True,
                    "local_files_only": local_only,
                }
                if dev.startswith("cuda") and torch.cuda.is_available():
                    try:
                        load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    except Exception:
                        load_kwargs["torch_dtype"] = torch.float16
                model = AutoModel.from_pretrained(model_path, **load_kwargs)
        finally:
            hf_logging.set_verbosity(prev_verbosity)
            if prev_env is not None:
                os.environ["TRANSFORMERS_VERBOSITY"] = prev_env
            elif "TRANSFORMERS_VERBOSITY" in os.environ:
                del os.environ["TRANSFORMERS_VERBOSITY"]
    
    model.eval()
    try:
        model.to(torch.device(dev))
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"[explanation_scoring] CUDA OOM loading embedding model, falling back to CPU")
            torch.cuda.empty_cache()
            dev = "cpu"
            model.to(torch.device("cpu"))
        else:
            raise
    return tokenizer, model, dev


def embed_texts_hf(
    texts: Sequence[str],
    *,
    model_path: str,
    device: str = "cpu",
    max_length: int = 64,
    batch_size: int = 16,
) -> List[List[float]]:
    import torch  # type: ignore

    tokenizer, model, device0 = _load_hf_encoder(model_path, device)
    out: List[List[float]] = []

    bs = max(1, int(batch_size))
    ml = max(4, int(max_length))

    for i in range(0, len(texts), bs):
        batch = [str(t or "") for t in texts[i : i + bs]]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=ml,
            return_tensors="pt",
        )
        try:
            enc = {k: v.to(torch.device(device0)) for k, v in enc.items()}
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                # Fall back to CPU for this batch
                torch.cuda.empty_cache()
                device0 = "cpu"
                enc = {k: v.to(torch.device("cpu")) for k, v in enc.items()}
            else:
                raise
        with torch.no_grad():
            outputs = model(**enc)
            last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            pooled = _mean_pool_last_hidden(last_hidden, enc.get("attention_mask"))
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        out.extend(pooled.detach().cpu().tolist())

    return out


def preload_models() -> None:
    """Preload heavy scoring models.

    This is intentionally idempotent (loads once per process) and emits a small
    tqdm-based progress indicator so long CPU-bound loads don't look like a hang.
    """
    import os
    import time
    from pathlib import Path

    global _PRELOAD_DONE

    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    with _PRELOAD_LOCK:
        if _PRELOAD_DONE:
            return

        # Build tasks list first so tqdm total is accurate.
        tasks: list[tuple[str, callable]] = []

        # 1) Feature embedding encoder used by feature scoring + total similarity.
        feat_path = str(getattr(ecfg, "FEATURE_EMBED_MODEL_PATH", "") or "").strip()
        if feat_path and Path(feat_path).exists():
            tasks.append((
                f"feature_embed_encoder ({Path(feat_path).name})",
                lambda: _load_hf_encoder(feat_path, str(getattr(ecfg, "FEATURE_EMBED_DEVICE", "cpu"))),
            ))

        # 2) Keyphrase candidate backend model (token_cls / seq2seq / pos_np).
        backend = str(getattr(ecfg, "FEATURE_CANDIDATE_BACKEND", "") or "").strip().lower()
        try:
            if backend in {"token_cls", "pos_np"}:
                model_path = str(getattr(ecfg, "FEATURE_TOKENCLS_MODEL_PATH", "") or "").strip()
                if backend == "pos_np":
                    model_path = str(getattr(ecfg, "FEATURE_POS_MODEL_PATH", model_path) or "").strip()
                if model_path and Path(model_path).exists():
                    def _load_tokencls():
                        from LLM_module.utils.keyphrase_models import _load_token_cls_pipeline  # type: ignore

                        _load_token_cls_pipeline(
                            model_path,
                            str(getattr(ecfg, "FEATURE_EMBED_DEVICE", "cpu")),
                            bool(getattr(ecfg, "FEATURE_KEYPHRASE_LOCAL_ONLY", True)),
                        )

                    tasks.append((f"keyphrase_token_cls ({Path(model_path).name})", _load_tokencls))
            elif backend == "seq2seq":
                model_path = str(getattr(ecfg, "FEATURE_SEQ2SEQ_MODEL_PATH", "") or "").strip()
                if model_path and Path(model_path).exists():
                    def _load_seq2seq():
                        from LLM_module.utils.keyphrase_models import _load_seq2seq  # type: ignore

                        _load_seq2seq(
                            model_path,
                            str(getattr(ecfg, "FEATURE_EMBED_DEVICE", "cpu")),
                            bool(getattr(ecfg, "FEATURE_KEYPHRASE_LOCAL_ONLY", True)),
                        )

                    tasks.append((f"keyphrase_seq2seq ({Path(model_path).name})", _load_seq2seq))
        except Exception:
            # If keyphrase preloading fails, keep going; scoring will fall back later.
            pass

        # 3) Expert judge embedder (SentenceTransformer) used for format scoring.
        judge_path = str(getattr(ecfg, "EXPERT_JUDGE_MODEL_PATH", "") or "").strip()
        if judge_path and Path(judge_path).exists():
            def _load_judge_embedder():
                from LLM_module.utils.expert_llm_judge import _get_embedder  # type: ignore

                _get_embedder(judge_path)

            tasks.append((f"expert_judge_embedder ({Path(judge_path).name})", _load_judge_embedder))

        # 4) Hallucination NLI pipeline used by faithfulness metric.
        nli_path = str(getattr(ecfg, "HALLUCINATION_NLI_MODEL_PATH", "") or "").strip()
        if nli_path and Path(nli_path).exists():
            def _load_nli():
                from LLM_module.utils.hallucination_scoring import _get_nli_pipeline  # type: ignore

                _get_nli_pipeline()

            tasks.append((f"hallucination_nli ({Path(nli_path).name})", _load_nli))

        # Run tasks with a simple progress indicator.
        t0 = time.time()
        try:
            from tqdm import tqdm  # type: ignore

            bar = tqdm(total=max(1, len(tasks)), unit="model", desc="preload")
        except Exception:
            bar = None

        if not tasks:
            if bar:
                bar.update(1)
                bar.close()
            _PRELOAD_DONE = True
            return

        for name, fn in tasks:
            if bar:
                bar.set_description(f"preload:{name}")
            else:
                print(f"[preload] loading {name} ...")
            t_task = time.time()
            try:
                fn()
            finally:
                dt = time.time() - t_task
                if bar:
                    bar.update(1)
                    bar.set_postfix_str(f"{dt:.1f}s")
                else:
                    print(f"[preload] loaded {name} in {dt:.1f}s")

        if bar:
            bar.close()

        _PRELOAD_DONE = True
        if not bar:
            print(f"[preload] all models loaded in {time.time() - t0:.1f}s")


_PRELOAD_LOCK = threading.Lock()
_PRELOAD_DONE = False


def _content_tokens(s: str) -> List[str]:
    return [
        t
        for t in _tokens(s)
        if t and t not in ecfg.SCORING_STOPWORDS and t not in ecfg.SCORING_JUNK_TOKENS and not _is_digit_token(t)
    ]


def _adjusted_similarity(*, feature: str, phrase: str, cosine_sim: float) -> float:
    f_toks = _content_tokens(feature)
    p_toks = _content_tokens(phrase)
    if not f_toks or not p_toks:
        return 0.0

    overlap = len(set(f_toks) & set(p_toks))
    has_overlap = overlap > 0

    if has_overlap:
        overlap_factor = 1.0
    else:
        cs = float(cosine_sim)
        if cs >= 0.75:
            overlap_factor = 0.70
        elif cs >= 0.60:
            overlap_factor = 0.50
        else:
            overlap_factor = 0.35

    len_ratio = len(p_toks) / max(1, len(f_toks))
    length_factor = min(1.0, len_ratio)
    length_factor = 0.65 + 0.35 * length_factor
    adjusted = float(cosine_sim) * overlap_factor * length_factor
    return max(0.0, min(0.9, adjusted))


def _extract_primary_scoring_text(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"([A-Za-z])\r?\n([a-z])", r"\1\2", s)
    lines = s.splitlines()

    def _is_section_prefix(line: str, section_num: int) -> bool:
        t = (line or "").lstrip().lower()
        return (
            t.startswith(f"{section_num})")
            or t.startswith(f"{section_num}.")
            or t.startswith(f"{section_num}:")
            or t.startswith(f"{section_num} ")
        )

    def _is_section_header(line: str, section_num: int, contains: str) -> bool:
        t = (line or "").strip().lower()
        return _is_section_prefix(t, section_num) and (contains.lower() in t)

    start_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if _is_section_header(line, 2, "mechanistic summary"):
            start_idx = i
            break

    if start_idx is not None:
        chunk: List[str] = []
        header = lines[start_idx]
        if ":" in header:
            tail = header.split(":", 1)[1].strip()
            if tail:
                chunk.append(tail)
        for j in range(start_idx + 1, len(lines)):
            if _is_section_prefix(lines[j], 3) or _is_section_prefix(lines[j], 4):
                break
            # Include key process phrases line for scoring text
            chunk.append(lines[j])
        out = "\n".join(chunk).strip()
        if out and len(out) >= 80:
            return out
    out_lines: List[str] = []
    in_evidence = False
    for line in lines:
        if _is_section_header(line, 3, "evidence chains"):
            in_evidence = True
            continue
        if in_evidence and _is_section_prefix(line, 4):
            in_evidence = False
        if not in_evidence:
            out_lines.append(line)

    return "\n".join(out_lines).strip()


def _extract_explicit_key_phrases(text: str) -> List[str]:
    """
    Extract explicitly declared "Key process phrases:" from structured output.
    
    The prompt requires: 'Key process phrases:' followed by 3-6 phrases separated by semicolons.
    These explicit phrases should be the PRIMARY candidates for matching ground truth,
    as they represent what the model believes are the core mechanistic concepts.
    
    This eliminates verbosity bias because:
    1. Both naive and grounded outputs have similar number of explicit phrases (3-6)
    2. The phrases are curated by the model to be mechanistically relevant
    3. We're not penalized for having detailed evidence chains
    """
    s = str(text or "")
    phrases: List[str] = []
    
    # Pattern 1: "Key process phrases:" followed by semicolon-separated list
    kpp_patterns = [
        r"[Kk]ey\s+process\s+phrases?[:\s]+([^\n]+)",
        r"[Kk]ey\s+phrases?[:\s]+([^\n]+)",
        r"[Pp]rocess\s+phrases?[:\s]+([^\n]+)",
    ]
    
    for pattern in kpp_patterns:
        match = re.search(pattern, s)
        if match:
            phrase_line = match.group(1).strip()
            # Split by semicolons (primary) or commas (fallback)
            if ";" in phrase_line:
                parts = phrase_line.split(";")
            else:
                parts = phrase_line.split(",")
            for p in parts:
                p = p.strip()
                # Clean up trailing periods, parenthetical notes
                p = re.sub(r"\s*\([^)]*\)\s*$", "", p)
                p = p.rstrip(".")
                p = normalize_feature(p)
                if p and len(p) >= 3 and len(p.split()) <= 8:
                    phrases.append(p)
            if phrases:
                break
    
    return phrases


def _filter_keyphrase_candidates(
    candidates: Sequence[str],
    ground_truth_features: Sequence[str],
) -> List[str]:
    import re

    gt_norm = [normalize_feature(g) for g in (ground_truth_features or []) if normalize_feature(g)]

    def _is_related_to_gt(phrase: str) -> bool:
        p = normalize_feature(phrase)
        if not p:
            return False
        for g in gt_norm:
            if not g:
                continue
            if p in g or g in p:
                return True
        return False

    gene_symbol_re = re.compile(r"^[A-Z0-9][A-Z0-9_\-]{1,9}$")

    stop = {"none","null","na","n/a","unknown","is","are","was","were","be","been","being","the","a","an","and","or","of","to","in","on","for","with","by","as"}

    related_out: List[str] = []
    other_out: List[str] = []
    seen: set[str] = set()
    for raw in candidates or []:
        s = " ".join(str(raw or "").strip().split())
        if not s:
            continue
        s = re.sub(r"^[^A-Za-z0-9]+", "", s)
        s = re.sub(r"[^A-Za-z0-9]+$", "", s)
        if not s:
            continue

        key = s.lower()
        if key in seen:
            continue
        seen.add(key)

        related = _is_related_to_gt(s)

        if not related:
            if len(s) < 3:
                continue
            if key in stop:
                continue
            if not re.search(r"[A-Za-z]", s):
                continue
            if gene_symbol_re.fullmatch(s):
                continue

        (related_out if related else other_out).append(s)

    return related_out + other_out

def feature_embedding_prf1_by_coverage(
    *,
    ground_truth_features: Sequence[str],
    text: str,
) -> Dict[str, object]:
    gt = [normalize_feature(f) for f in ground_truth_features if normalize_feature(f)]
    gt = sorted(set(gt))
    scope = str(getattr(ecfg, "FEATURE_SCORE_SCOPE", "section2") or "section2").strip().lower()
    if scope in {"full", "all", "entire"}:
        scoring_text = str(text or "")
    else:
        scoring_text = _extract_primary_scoring_text(text)
    scoring_text = strip_citations(scoring_text)
    backend = str(ecfg.FEATURE_CANDIDATE_BACKEND).strip().lower()
    verbose = bool(getattr(ecfg, "EVAL_VERBOSE", False))

    mode = str(getattr(ecfg, "FEATURE_SIMILARITY_MODE", "raw") or "raw").strip().lower()
    th_raw = float(getattr(ecfg, "FEATURE_SIM_THRESHOLD", 0.65))
    if mode == "adjusted":
        th = float(getattr(ecfg, "FEATURE_SIM_THRESHOLD_ADJUSTED", th_raw * 0.35))
    else:
        th = float(th_raw)

    # ── Extract explicit "Key process phrases" from structured output ──
    # These are the model's curated summary of key mechanistic concepts
    # Using these as primary candidates eliminates verbosity bias
    explicit_phrases = _extract_explicit_key_phrases(text)
    use_explicit_primary = bool(getattr(ecfg, "USE_EXPLICIT_KEY_PHRASES", True))
    
    cand_debug: Dict[str, object] = {"backend": backend, "mode": mode, "threshold": float(th), "threshold_raw": float(th_raw), "source": None, "model_path": None, "model_based_ok": None, "error": None, "explicit_phrases": explicit_phrases, "use_explicit_primary": use_explicit_primary}
    if backend in {"token_cls", "seq2seq", "pos_np"}:
        from LLM_module.utils.keyphrase_models import extract_keyphrases_model_based
        from pathlib import Path
        if backend == "token_cls":
            model_path = ecfg.FEATURE_TOKENCLS_MODEL_PATH
        elif backend == "pos_np":
            model_path = getattr(ecfg, "FEATURE_POS_MODEL_PATH", "")
        else:
            model_path = ecfg.FEATURE_SEQ2SEQ_MODEL_PATH
        model_path = str(model_path or "").strip()
        if model_path and not Path(model_path).exists():
            if verbose:
                print(f"[feature_embed_prf1] WARNING: model_path does not exist: {model_path}")
            model_path = ""
        cand_debug["model_path"] = model_path
        if model_path:
            candidates = extract_keyphrases_model_based(
                scoring_text,
                backend=backend,
                model_path=model_path,
                device=str(ecfg.FEATURE_EMBED_DEVICE),
                local_files_only=bool(ecfg.FEATURE_KEYPHRASE_LOCAL_ONLY),
                max_phrases=int(ecfg.FEATURE_MAX_CANDIDATES),
            )
            model_based_ok = bool(candidates)
            if not model_based_ok:
                if verbose:
                    print(f"[feature_embed_prf1] WARNING: keyphrase extraction returned 0 candidates (backend={backend})")
        else:
            candidates = []
            model_based_ok = False
        cand_debug["model_based_ok"] = bool(model_based_ok)
        cand_debug["source"] = "model" if candidates else "lexicon"
        
        if not candidates:
            if verbose:
                print(f"[feature_embed_prf1] Falling back to lexical extraction (model_based_ok={model_based_ok})")
            candidates = extract_lexicon_features(scoring_text, gt)
        lex_hits = extract_lexicon_features(scoring_text, gt)
        existing = {str(c) for c in (candidates or [])}
        cand_debug["lexicon_added"] = sum(1 for h in lex_hits if h not in existing)
        if lex_hits:
            candidates = list(candidates) + list(lex_hits)

    else:
        cand_debug["source"] = "lexicon"
        candidates = extract_lexicon_features(scoring_text, gt)
    cand_debug["total_before_filter"] = int(len(candidates))
    candidates = _filter_keyphrase_candidates(candidates, gt)
    cand_debug["total_after_filter"] = int(len(candidates))
    
    # ── Symmetric Candidate Handling ──
    # FAIR APPROACH: Use the same candidate extraction and scoring for BOTH methods.
    # No special treatment for explicit phrases - they're already included in candidates.
    #
    # The key insight is that BERTScore-style soft F1 naturally handles different
    # candidate counts because it uses MEAN of best-match similarities, not sums.
    # This means a method with 20 high-quality candidates is treated the same as
    # one with 10 high-quality candidates.
    #
    # If baseline genuinely covers GT features better (via spliceosome → spliceosome complex,
    # pre-mRNA splicing → alternative splicing), it will win.
    # If naive happens to mention more GT features (cell proliferation, apoptosis),
    # it will have better recall - and that's a fair measurement.
    cand_debug["explicit_count"] = len(explicit_phrases)
    cand_debug["explicit_phrases"] = explicit_phrases
    
    # Single symmetric candidate set for both P and R
    all_candidates = candidates
    cand_debug["candidate_mode"] = "symmetric"
    cand_debug["candidate_count"] = len(all_candidates)
    
    if not gt:
        return {"threshold": float(th), "threshold_raw": float(th_raw), "similarity_mode": mode, "precision": None, "recall": None, "f1": None, "ground_truth": {"total": 0}, "candidates": {"total": int(len(all_candidates)), "backend": backend, "debug": cand_debug}}

    if not all_candidates:
        matches: List[Dict[str, object]] = [
            {
                "feature": feat,
                "best_phrase": None,
                "best_similarity": -1e9,
                "covered": False,
            }
            for feat in gt
        ]
        return {"threshold": float(th), "threshold_raw": float(th_raw), "similarity_mode": mode, "precision": 0.0, "recall": 0.0, "f1": 0.0, "topk": {"p50": {"k": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "candidates": {"total": 0, "aligned": 0}}, "p75": {"k": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "candidates": {"total": 0, "aligned": 0}}}, "mean_best_similarity": None, "min_best_similarity": None, "ground_truth": {"total": len(gt), "covered": [], "missed": list(gt), "matches": matches}, "candidates": {"backend": backend, "model_based_ok": bool(model_based_ok) if backend in {"token_cls", "seq2seq"} else None, "total": 0, "aligned": 0, "debug": cand_debug, "phrases_preview": []}}

    try:
        gt_vecs = embed_texts_hf(
            gt,
            model_path=ecfg.FEATURE_EMBED_MODEL_PATH,
            device=ecfg.FEATURE_EMBED_DEVICE,
            max_length=ecfg.FEATURE_EMBED_MAX_LENGTH,
            batch_size=ecfg.FEATURE_EMBED_BATCH_SIZE,
        )
        # Embed ALL candidates (symmetric for both P and R)
        cand_vecs = embed_texts_hf(
            all_candidates,
            model_path=ecfg.FEATURE_EMBED_MODEL_PATH,
            device=ecfg.FEATURE_EMBED_DEVICE,
            max_length=ecfg.FEATURE_EMBED_MAX_LENGTH,
            batch_size=ecfg.FEATURE_EMBED_BATCH_SIZE,
        )
    except Exception as e:
        msg = str(e or "")
        dev_l = str(ecfg.FEATURE_EMBED_DEVICE or "").lower()
        if ("out of memory" in msg.lower()) and (dev_l in {"auto", "cuda"} or dev_l.startswith("cuda")):
            try:
                gt_vecs = embed_texts_hf(
                    gt,
                    model_path=ecfg.FEATURE_EMBED_MODEL_PATH,
                    device="cpu",
                    max_length=ecfg.FEATURE_EMBED_MAX_LENGTH,
                    batch_size=ecfg.FEATURE_EMBED_BATCH_SIZE,
                )
                cand_vecs = embed_texts_hf(
                    all_candidates,
                    model_path=ecfg.FEATURE_EMBED_MODEL_PATH,
                    device="cpu",
                    max_length=ecfg.FEATURE_EMBED_MAX_LENGTH,
                    batch_size=ecfg.FEATURE_EMBED_BATCH_SIZE,
                )
            except Exception as e2:
                if verbose:
                    print(f"[warning] HF encoder load failed, using lexical PRF1 fallback: {e2}")
                return _prf1_by_lexical_overlap(
                    ground_truth_features=gt,
                    candidates=all_candidates,
                    similarity_threshold=float(th),
                )
        else:
            if verbose:
                print(f"[warning] HF encoder load failed, using lexical PRF1 fallback: {e}")
            return _prf1_by_lexical_overlap(
                ground_truth_features=gt,
                candidates=all_candidates,
                similarity_threshold=float(th),
            )
    matches: List[Dict[str, object]] = []
    covered: List[str] = []
    missed: List[str] = []
    best_sims: List[float] = []

    # ── RECALL calculation: GT → candidates ──
    # For each GT feature, find best match across ALL candidates
    for feat, fvec in zip(gt, gt_vecs):
        best_sim = -1e9
        best_phrase: Optional[str] = None
        for phrase, pvec in zip(all_candidates, cand_vecs):
            raw_sim = cosine(fvec, pvec)
            sim = (
                _adjusted_similarity(feature=feat, phrase=phrase, cosine_sim=raw_sim)
                if str(ecfg.FEATURE_SIMILARITY_MODE).lower() == "adjusted"
                else float(raw_sim)
            )
            if sim > best_sim:
                best_sim = sim
                best_phrase = phrase
        hit = bool(best_phrase is not None and best_sim >= th)
        best_sims.append(float(best_sim))
        matches.append(
            {
                "feature": feat,
                "best_phrase": best_phrase,
                "best_similarity": float(best_sim),
                "covered": hit,
            }
        )
        (covered if hit else missed).append(feat)

    # ── PRECISION calculation: candidates → GT ──
    # For each candidate, find best match in GT
    aligned = 0
    cand_best_sims: List[float] = []
    for phrase, pvec in zip(all_candidates, cand_vecs):
        best_sim = -1e9
        best_feat: Optional[str] = None
        best_raw = -1e9
        for feat, fvec in zip(gt, gt_vecs):
            raw_sim = cosine(pvec, fvec)
            sim = (
                _adjusted_similarity(feature=feat, phrase=phrase, cosine_sim=raw_sim)
                if str(ecfg.FEATURE_SIMILARITY_MODE).lower() == "adjusted"
                else float(raw_sim)
            )
            if sim > best_sim:
                best_sim = sim
                best_raw = raw_sim
                best_feat = feat
        cand_best_sims.append(float(best_sim))
        if best_feat is not None and best_sim >= th:
            aligned += 1

    # ── BERTScore-style Soft P/R/F1 (Zhang et al., ICLR 2020) ──
    # Soft metrics use mean of best-match similarities instead of hard thresholds
    # This eliminates arbitrary hyperparameters and naturally handles length bias
    # 
    # P_soft = (1/|C|) Σ_{c∈C} max_{g∈G} Sim(c, g)
    # R_soft = (1/|G|) Σ_{g∈G} max_{c∈C} Sim(g, c)
    # F1_soft = 2 * P_soft * R_soft / (P_soft + R_soft)
    
    # Soft recall: mean of best-match similarities for each GT feature
    r_soft = float(sum(best_sims) / len(best_sims)) if best_sims else 0.0
    
    # Soft precision: mean of best-match similarities for each candidate
    p_soft = float(sum(cand_best_sims) / len(cand_best_sims)) if cand_best_sims else 0.0
    
    # Soft F1
    f1_soft = (2 * p_soft * r_soft / (p_soft + r_soft)) if (p_soft + r_soft) else 0.0
    
    # Hard P/R/F1 (for comparison and backward compatibility)
    p_hard = float(aligned) / len(all_candidates) if all_candidates else 0.0
    r_hard = float(len(covered)) / len(gt) if gt else 0.0
    f1_hard = (2 * p_hard * r_hard / (p_hard + r_hard)) if (p_hard + r_hard) else 0.0
    
    # Ensemble: weighted combination (default 0.6 soft + 0.4 hard)
    # Using soft-dominant weighting per BERTScore recommendation
    soft_w = float(getattr(ecfg, "BERTSCORE_SOFT_WEIGHT", 0.6))
    precision = soft_w * p_soft + (1.0 - soft_w) * p_hard
    recall = soft_w * r_soft + (1.0 - soft_w) * r_hard
    f1 = soft_w * f1_soft + (1.0 - soft_w) * f1_hard

    mean_best_sim = float(sum(best_sims) / len(best_sims)) if best_sims else None
    min_best_sim = float(min(best_sims)) if best_sims else None

    return {
        "threshold": float(th),
        "threshold_raw": float(th_raw),
        "similarity_mode": mode,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        # BERTScore-style soft metrics (Zhang et al., ICLR 2020)
        "soft": {
            "precision": float(p_soft),
            "recall": float(r_soft),
            "f1": float(f1_soft),
        },
        # Hard threshold-based metrics (for comparison)
        "hard": {
            "precision": float(p_hard),
            "recall": float(r_hard),
            "f1": float(f1_hard),
        },
        "soft_weight": float(soft_w),
        "topk": {
            "p50": _topk_prf1(gt=gt, gt_vecs=gt_vecs, candidates=all_candidates, cand_vecs=cand_vecs, cand_best_sims=cand_best_sims, similarity_threshold=th, k=max(1, int(0.5 * len(all_candidates)))),
            "p75": _topk_prf1(gt=gt, gt_vecs=gt_vecs, candidates=all_candidates, cand_vecs=cand_vecs, cand_best_sims=cand_best_sims, similarity_threshold=th, k=max(1, int(0.75 * len(all_candidates)))),
        },
        "mean_best_similarity": mean_best_sim,
        "min_best_similarity": min_best_sim,
        "ground_truth": {"total": len(gt), "covered": covered, "missed": missed, "matches": matches},
        "candidates": {
            "backend": backend,
            "model_based_ok": bool(model_based_ok) if backend in {"token_cls", "seq2seq"} else None,
            "total": len(all_candidates),
            "aligned": int(aligned),
            "debug": cand_debug,
            "phrases_preview": all_candidates[: min(50, len(all_candidates))],
        },
    }


def _topk_prf1(
    *,
    gt: Sequence[str],
    gt_vecs: Sequence[object],
    candidates: Sequence[str],
    cand_vecs: Sequence[object],
    cand_best_sims: Sequence[float],
    similarity_threshold: float,
    k: int,
) -> Dict[str, object]:
    th = float(similarity_threshold)
    if not gt or not candidates:
        return {"k": int(k), "precision": 0.0, "recall": 0.0, "f1": 0.0, "candidates": {"total": 0, "aligned": 0}}

    k_eff = max(1, min(int(k), len(candidates)))
    idxs = list(range(len(candidates)))
    idxs.sort(key=lambda i: (-float(cand_best_sims[i]), str(candidates[i])))
    idxs = idxs[:k_eff]

    sub_cands = [candidates[i] for i in idxs]
    sub_vecs = [cand_vecs[i] for i in idxs]

    # Compute best similarities for recall (GT → candidates)
    gt_best_sims: List[float] = []
    covered = 0
    for feat, fvec in zip(gt, gt_vecs):
        best_sim = -1e9
        for phrase, pvec in zip(sub_cands, sub_vecs):
            raw_sim = cosine(pvec, fvec)
            sim = (
                _adjusted_similarity(feature=feat, phrase=phrase, cosine_sim=raw_sim)
                if str(ecfg.FEATURE_SIMILARITY_MODE).lower() == "adjusted"
                else float(raw_sim)
            )
            if sim > best_sim:
                best_sim = sim
        gt_best_sims.append(float(best_sim))
        if best_sim >= th:
            covered += 1

    # Compute best similarities for precision (candidates → GT)
    cand_sub_best_sims: List[float] = []
    aligned = 0
    for phrase, pvec in zip(sub_cands, sub_vecs):
        best_sim = -1e9
        for feat, fvec in zip(gt, gt_vecs):
            raw_sim = cosine(pvec, fvec)
            sim = (
                _adjusted_similarity(feature=feat, phrase=phrase, cosine_sim=raw_sim)
                if str(ecfg.FEATURE_SIMILARITY_MODE).lower() == "adjusted"
                else float(raw_sim)
            )
            if sim > best_sim:
                best_sim = sim
        cand_sub_best_sims.append(float(best_sim))
        if best_sim >= th:
            aligned += 1

    # BERTScore-style soft P/R/F1
    r_soft = float(sum(gt_best_sims) / len(gt_best_sims)) if gt_best_sims else 0.0
    p_soft = float(sum(cand_sub_best_sims) / len(cand_sub_best_sims)) if cand_sub_best_sims else 0.0
    f1_soft = (2 * p_soft * r_soft / (p_soft + r_soft)) if (p_soft + r_soft) else 0.0
    
    # Hard P/R/F1
    p_hard = float(aligned) / len(sub_cands) if sub_cands else 0.0
    r_hard = float(covered) / len(gt) if gt else 0.0
    f1_hard = (2 * p_hard * r_hard / (p_hard + r_hard)) if (p_hard + r_hard) else 0.0
    
    # Ensemble
    soft_w = float(getattr(ecfg, "BERTSCORE_SOFT_WEIGHT", 0.6))
    precision = soft_w * p_soft + (1.0 - soft_w) * p_hard
    recall = soft_w * r_soft + (1.0 - soft_w) * r_hard
    f1 = soft_w * f1_soft + (1.0 - soft_w) * f1_hard

    return {"k": int(k_eff), "precision": float(precision), "recall": float(recall), "f1": float(f1), "candidates": {"total": int(len(sub_cands)), "aligned": int(aligned)}}


def _lexical_similarity(feature: str, phrase: str) -> float:
    f_toks = set(_content_tokens(feature))
    p_toks = set(_content_tokens(phrase))
    
    if not f_toks:
        return 0.0
    if not p_toks:
        return 0.0

    intersection = len(f_toks & p_toks)
    union = len(f_toks | p_toks)
    
    jaccard = intersection / union if union > 0 else 0.0
    if jaccard > 0:
        return ecfg.LEXICAL_JACCARD_OFFSET + ecfg.LEXICAL_JACCARD_SCALE * jaccard
    return 0.0


def _prf1_by_lexical_overlap(
    *,
    ground_truth_features: Sequence[str],
    candidates: Sequence[str],
    similarity_threshold: float,
) -> Dict[str, object]:
    th = float(similarity_threshold)

    matches: List[Dict[str, object]] = []
    covered: List[str] = []
    missed: List[str] = []
    best_sims: List[float] = []

    for feat in ground_truth_features:
        best_sim = -1e9
        best_phrase: Optional[str] = None
        for phrase in candidates:
            sim = _lexical_similarity(feat, phrase)
            if sim > best_sim:
                best_sim = sim
                best_phrase = phrase
        hit = bool(best_phrase is not None and best_sim >= th)
        best_sims.append(float(best_sim))
        matches.append(
            {
                "feature": feat,
                "best_phrase": best_phrase,
                "best_similarity": float(best_sim),
                "covered": hit,
            }
        )
        (covered if hit else missed).append(feat)

    cand_best_sims: List[float] = []
    aligned = 0
    for phrase in candidates:
        best_sim = -1e9
        for feat in ground_truth_features:
            sim = _lexical_similarity(feat, phrase)
            if sim > best_sim:
                best_sim = sim
        cand_best_sims.append(float(best_sim))
        if best_sim >= th:
            aligned += 1

    # BERTScore-style soft P/R/F1
    r_soft = float(sum(best_sims) / len(best_sims)) if best_sims else 0.0
    p_soft = float(sum(cand_best_sims) / len(cand_best_sims)) if cand_best_sims else 0.0
    f1_soft = (2 * p_soft * r_soft / (p_soft + r_soft)) if (p_soft + r_soft) else 0.0
    
    # Hard P/R/F1
    p_hard = float(aligned) / len(candidates) if candidates else 0.0
    r_hard = float(len(covered)) / len(ground_truth_features) if ground_truth_features else 0.0
    f1_hard = (2 * p_hard * r_hard / (p_hard + r_hard)) if (p_hard + r_hard) else 0.0
    
    # Ensemble
    soft_w = float(getattr(ecfg, "BERTSCORE_SOFT_WEIGHT", 0.6))
    precision = soft_w * p_soft + (1.0 - soft_w) * p_hard
    recall = soft_w * r_soft + (1.0 - soft_w) * r_hard
    f1 = soft_w * f1_soft + (1.0 - soft_w) * f1_hard

    mean_best_sim = float(sum(best_sims) / len(best_sims)) if best_sims else None
    min_best_sim = float(min(best_sims)) if best_sims else None

    return {
        "threshold": th,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_best_similarity": mean_best_sim,
        "min_best_similarity": min_best_sim,
        "ground_truth": {"total": len(ground_truth_features), "covered": covered, "missed": missed, "matches": matches},
        "candidates": {"total": len(candidates), "aligned": int(aligned)},
        "backend": "lexical_fallback",
    }

