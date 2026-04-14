from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

# ═════════════════════════════════════════════════════════════════════════════
# Weight / rank dictionaries are loaded from the GNN's learned_weights.json
# (produced by GNN training).
#
# IMPORTANT: This module is intentionally strict. If learned weights are not
# present, we raise early instead of silently falling back to arbitrary
# heuristics.
# ═════════════════════════════════════════════════════════════════════════════

_WEIGHTS_FILE = Path("/data/guoyu/KG-LLM-XSL/output/GNN_checkpoints/learned_weights.json")


_learned = json.loads(_WEIGHTS_FILE.read_text(encoding="utf-8"))


LEARNED_WEIGHTS_PATH: str = str(_WEIGHTS_FILE)
LEARNED_WEIGHTS_META: dict = _learned.get("_meta", {})
LEARNED_WEIGHTS_MODE: str = str(LEARNED_WEIGHTS_META.get("weight_mode", "unknown"))


def _require_dict(name: str) -> Dict:
    v = _learned.get(name)
    if not isinstance(v, dict) or not v:
        raise RuntimeError(
            f"learned_weights.json is missing required non-empty dict section {name!r}: {_WEIGHTS_FILE}"
        )
    return v


# ── Loaded (learned) ONLY ───────────────────────────────────────────────
EDGE_TYPE_WEIGHT: Dict[str, float] = _require_dict("EDGE_TYPE_WEIGHT")
EDGE_RELATION_WEIGHT: Dict[str, float] = _require_dict("EDGE_RELATION_WEIGHT")
EDGE_SOURCE_WEIGHT: Dict[str, float] = _require_dict("EDGE_SOURCE_WEIGHT")
EDGE_TYPE_PRIORITY: Dict[str, int] = _require_dict("EDGE_TYPE_PRIORITY")
NODE_TYPE_RANK: Dict[str, int] = _require_dict("NODE_TYPE_RANK")

# Require explicit fixed types (no downstream fallbacks).
_required_edge_types = {"SL_pair"}
missing_req = {
    "EDGE_TYPE_WEIGHT": sorted(_required_edge_types - set(EDGE_TYPE_WEIGHT.keys())),
    "EDGE_RELATION_WEIGHT": sorted(_required_edge_types - set(EDGE_RELATION_WEIGHT.keys())),
    "EDGE_TYPE_PRIORITY": sorted(_required_edge_types - set(EDGE_TYPE_PRIORITY.keys())),
}
missing_req = {k: v for k, v in missing_req.items() if v}
if missing_req:
    raise RuntimeError(
        "learned_weights.json is missing required fixed edge types (no fallbacks allowed). "
        f"Missing: {missing_req}. Re-run GNN training to regenerate {_WEIGHTS_FILE}."
    )

# ── Scoring formula meta-weights (algorithm hyper-params, NOT learned) ───
PATH_RANK_WEIGHTS: Dict[str, float] = {"len": 0.35, "switch": 0.35, "prob": 0.20, "ppr": 0.10}
NEIGHBOR_RANK_WEIGHTS: Dict[str, float] = {"hop": 0.25, "evidence": 0.40, "prob": 0.10, "ppr": 0.05, "semantic": 0.20}
SUBGRAPH_NODE_RANK_WEIGHTS: Dict[str, float] = {"dist": 0.35, "evidence": 0.45, "ppr": 0.08, "prob": 0.07, "semantic": 0.05}

# ── Dynamic Subgraph Expansion (GNN-driven) ──────────────────────────────
# Instead of arbitrary per-type limits, we rely on the GNN's learned importance 
# to rank neighbors globally.
# Hard caps are only for computational safety / context window fit.
SUBGRAPH_MAX_NODES: int = 50
SUBGRAPH_IMPORTANCE_THRESHOLD: float = 0.0  # Optional: min score to include

# ── Inference ─────────────────────────────────────────────────────────────
INFERENCE_DEFAULT_MAX_HOPS: int = 3
INFERENCE_DEFAULT_TOP_K: int = 4
INFERENCE_MAX_PATHS_CONSIDERED: int = 40
EXPLANATION_HARD_MAX_HOPS: int = 3
PATH_PROB_EPS: float = 1e-12

# ── PPR ───────────────────────────────────────────────────────────────────
PPR_ALPHA: float = 0.85
PPR_MAX_ITER: int = 200
PPR_TOL: float = 1e-9

# ── Neighborhood ──────────────────────────────────────────────────────────
NEIGHBOR_DEFAULT_MAX_HOPS: int = 2
GO_ANNOTATION_MAX_TERMS: int = 5
USE_PROB_IN_NEIGHBOUR: bool = True
NEIGHBOR_CANDIDATE_CAP_MULTIPLIER: int = 6
NEIGHBOR_CANDIDATE_MIN: int = 80
NEIGHBOR_MAX_EXPLORED_MULTIPLIER: int = 10

# ── Semantic ──────────────────────────────────────────────────────────────
SEMANTIC_NEIGHBOR_MODEL_PATH: str = "/data/guoyu/HF-models/MedCPT-Query-Encoder"
SEMANTIC_EMBED_BATCH_SIZE: int = 16
SEMANTIC_EMBED_MAX_LENGTH: int = 64
SEMANTIC_NEIGHBOR_MMR_LAMBDA: float = 0.7

# ── Adaptive stop ─────────────────────────────────────────────────────────
ADAPTIVE_STOP_MIN_NODES: int = 24
ADAPTIVE_STOP_MIN_TYPES: int = 6
NEIGHBOUR_RESTRICTION: int = 4

# ── Subgraph extraction ──────────────────────────────────────────────────
INFINITE = 10**9
# SUBGRAPH_MAX_NODES: int = 40
# SUBGRAPH_KEEP_FRACTION_HOP1: float = 1.0
# SUBGRAPH_KEEP_FRACTION_HOP2: float = 0.40
# SUBGRAPH_MAX_DRUGS_PER_CORE_HOP1: int = -1
# SUBGRAPH_EXCLUDE_GENE_BEYOND_1_HOP: bool = True
# SUBGRAPH_MAX_EDGES: int = 20
# SUBGRAPH_DRUGS_PER_GENE_CAP_STEP: int = 4
# SUBGRAPH_MIN_GENE_NEIGHBORS_PER_CORE: int = 10
# SUBGRAPH_MAX_TF_NEIGHBORS_PER_CORE: int = 12
# SUBGRAPH_GENE_CAP_TRIM_PASSES: int = 6
SUBGRAPH_MAX_NODES: int = INFINITE
SUBGRAPH_KEEP_FRACTION_HOP1: float = 1.0
SUBGRAPH_KEEP_FRACTION_HOP2: float = 0.40
SUBGRAPH_MAX_DRUGS_PER_CORE_HOP1: int = -1
SUBGRAPH_EXCLUDE_GENE_BEYOND_1_HOP: bool = True
SUBGRAPH_MAX_EDGES: int = INFINITE
SUBGRAPH_DRUGS_PER_GENE_CAP_STEP: int = 4
SUBGRAPH_MIN_GENE_NEIGHBORS_PER_CORE: int = INFINITE
SUBGRAPH_MAX_TF_NEIGHBORS_PER_CORE: int = INFINITE
SUBGRAPH_GENE_CAP_TRIM_PASSES: int = 6