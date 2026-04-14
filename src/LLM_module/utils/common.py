from __future__ import annotations

import re
from typing import Tuple


KG_EDGE_CIT_RE = re.compile(
    r"\([^)]*(?:->|<->)[^)]*\|[^)]*\|[^)]*(?:\bkey\s*=\s*\d+)?[^)]*\)",
    flags=re.IGNORECASE | re.DOTALL
)

KG_KEY_RE = re.compile(r"\bkey\s*=\s*(\d+)\b", flags=re.IGNORECASE)

# KG node IDs and Reactome IDs
KG_NODE_ID_RE = re.compile(
    r"\b(?:gene|pathway|protein|drug|disease|cohort)\s*:\s*[a-z0-9_.\-]+\b",
    flags=re.IGNORECASE
)

KG_REACTOME_ID_RE = re.compile(r"\bR-HSA-\d+\b", flags=re.IGNORECASE)

# Citation boilerplate patterns
URL_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)
BRACKET_CIT_RE = re.compile(r"\[(?:\s*\d+\s*(?:[,;\-]\s*\d+\s*)*)\]", flags=re.IGNORECASE)
PMID_RE = re.compile(r"\bpmid\s*[:#]?\s*\d+\b", flags=re.IGNORECASE)
DOI_RE = re.compile(r"\bdoi\s*[:#]?\s*\S+", flags=re.IGNORECASE)
REF_SECTION_RE = re.compile(r"\b(references|bibliography)\b", flags=re.IGNORECASE)


def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def normalize_text(s: str) -> str:
    return " ".join(str(s or "").strip().split())


def count_citations(text: str) -> Tuple[int, int]:
    text_s = str(text or "")
    all_cits = KG_EDGE_CIT_RE.findall(text_s)
    keys = set()
    for c in all_cits:
        m = KG_KEY_RE.search(c)
        if m:
            keys.add(m.group(1))
    return len(all_cits), len(keys)


def strip_citations(text: str) -> str:
    s = str(text or "")
    
    # Drop reference section entirely
    m = REF_SECTION_RE.search(s)
    if m:
        s = s[:m.start()]
    
    # Remove inline citations
    s = URL_RE.sub(" ", s)
    s = BRACKET_CIT_RE.sub(" ", s)
    s = PMID_RE.sub(" ", s)
    s = DOI_RE.sub(" ", s)
    s = KG_EDGE_CIT_RE.sub(" ", s)
    s = KG_NODE_ID_RE.sub(" ", s)
    s = KG_REACTOME_ID_RE.sub(" ", s)
    
    return s


def resolve_device(device: str) -> str:
    import torch
    
    dev = str(device or "cpu").strip().lower() or "cpu"
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return dev


def get_pipeline_device(device: str) -> int:
    device_arg = -1
    if str(device).lower().startswith("cuda"):
        try:
            device_arg = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
        except Exception:
            device_arg = 0
    return device_arg


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        xn = " ".join(str(x or "").strip().split())
        if not xn:
            continue
        k = xn.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(xn)
    return out


def cosine(a, b) -> float:
    """Compute cosine similarity between two vectors (assumed normalized)."""
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))
