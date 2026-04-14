"""Data loading, feature construction, and negative sampling for SL prediction."""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data


# ── Graph loading ─────────────────────────────────────────────────────────

def load_knowledge_graph(
    graphml_path: str,
) -> Tuple[nx.MultiDiGraph, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a full knowledge graph and extract node / edge tables + curated SL pairs.

    IMPORTANT (leakage prevention)
    ------------------------------
    The returned graph is **sanitized for message passing**:
    all SL-labeled edges between gene nodes are removed before we build:
      - node features (degree / relation counts / PageRank)
      - relation vocabulary (edge_type ids)
      - edge_index/edge_type tensors

    Curated SL edges are still returned separately via ``sl_pairs_df`` and are
    used only as supervision labels (train/val positives).

    Returns
    -------
    G : nx.MultiDiGraph
    nodes_df : pd.DataFrame  (columns: node_id, type)
    edges_df : pd.DataFrame  (columns: src, dst, key, type, source, is_sl_curated)
    sl_pairs_df : pd.DataFrame  (columns: gene_a, gene_b, attrs)
    """
    G_full = nx.read_graphml(graphml_path)
    G_full = G_full if isinstance(G_full, nx.MultiDiGraph) else nx.MultiDiGraph(G_full)

    def _is_gene_node(n: str) -> bool:
        attrs = G_full.nodes.get(n, {}) or {}
        t = str(attrs.get("type", "")).strip().lower()
        if not t and isinstance(n, str) and ":" in n:
            t = n.split(":", 1)[0].strip().lower()
        return "gene" in t

    # ── SL edges ──
    # Collect curated SL pairs for supervision AND remove all SL-labeled gene-gene edges
    # from the message-passing graph to prevent leakage.
    curated_sl_edges: list = []
    sl_edges_to_remove: list[tuple[str, str, object]] = []
    for u, v, k, attrs in G_full.edges(keys=True, data=True):
        attrs = attrs or {}
        etype = str(
            attrs.get("type", attrs.get("edge_type", attrs.get("relation", "")))
        ).lower()
        src = str(attrs.get("source", "")).lower()

        # Remove ALL SL-labeled edges from the message-passing graph.
        # (Even if node types are missing/mis-typed, this remains safe.)
        is_sl_like = ("sl" in etype)
        if is_sl_like:
            sl_edges_to_remove.append((str(u), str(v), k))

            # Keep only curated gene-gene SL pairs as supervision labels.
            is_gene_gene = _is_gene_node(str(u)) and _is_gene_node(str(v))
            if is_gene_gene and ("curated" in src):
                curated_sl_edges.append((str(u), str(v), attrs))

    # Build the graph used for message passing / features.
    G = G_full.copy()
    for u, v, k in sl_edges_to_remove:
        # MultiDiGraph needs key for deterministic removal.
        if G.has_edge(u, v, k):
            G.remove_edge(u, v, k)

    # ── nodes_df ──
    node_rows = []
    for n, attrs in G.nodes(data=True):
        attrs = attrs or {}
        ntype = str(attrs.get("type", "")).strip().lower()
        if not ntype and isinstance(n, str) and ":" in n:
            ntype = n.split(":", 1)[0]
        node_rows.append({"node_id": str(n), "type": ntype or "unknown"})
    nodes_df = pd.DataFrame(node_rows)

    # ── edges_df ──
    edge_rows = []
    for u, v, k, attrs in G.edges(keys=True, data=True):
        attrs = attrs or {}
        etype = str(attrs.get("type", attrs.get("edge_type", attrs.get("relation", "related_to"))))
        source = str(attrs.get("source", "unknown"))
        edge_rows.append({
            "src": str(u),
            "dst": str(v),
            "key": str(k),
            "type": etype,
            "source": source,
            # After sanitization, this should always be False, but we keep the flag
            # for backward compatibility with downstream code.
            "is_sl_curated": (etype.lower() == "sl_pair" and source.lower() == "curated"),
        })
    edges_df = pd.DataFrame(edge_rows)

    sl_pairs_df = pd.DataFrame(
        [{"gene_a": u, "gene_b": v, "attrs": a} for u, v, a in curated_sl_edges]
    )
    return G, nodes_df, edges_df, sl_pairs_df


# ── Feature construction ──────────────────────────────────────────────────

def build_node_features(
    nodes_df: pd.DataFrame,
    G: nx.MultiDiGraph,
) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str]]:
    """
    Build node feature matrix:
      - node-type one-hot
      - normalised log-degree
      - per-relation-type edge-count (log-normalised)
      - PageRank score

    Returns (x, node_to_idx, idx_to_node).
    """
    node_ids = nodes_df["node_id"].tolist()
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    idx_to_node = {i: nid for nid, i in node_to_idx.items()}
    num_nodes = len(node_to_idx)
    UG = G.to_undirected(as_view=True)

    # 1. type one-hot
    types = sorted(nodes_df["type"].unique())
    type_to_idx = {t: i for i, t in enumerate(types)}
    x_type = np.zeros((num_nodes, len(types)), dtype=np.float32)
    for i, t in enumerate(nodes_df["type"]):
        x_type[i, type_to_idx[t]] = 1.0

    # 2. normalised log-degree
    deg = np.array([UG.degree(n) for n in node_ids], dtype=np.float32)
    log_deg = np.log1p(deg)
    log_deg = (log_deg - log_deg.min()) / (log_deg.max() - log_deg.min() + 1e-9)

    # 3. per-relation-type edge counts (undirected, log-normalised)
    etypes = sorted({
        str((attrs or {}).get("type", "unknown"))
        for _, _, attrs in G.edges(data=True)
    })
    etype_to_col = {et: j for j, et in enumerate(etypes)}
    x_rel = np.zeros((num_nodes, len(etypes)), dtype=np.float32)
    for u, v, attrs in G.edges(data=True):
        et = str((attrs or {}).get("type", "unknown"))
        col = etype_to_col[et]
        if u in node_to_idx:
            x_rel[node_to_idx[u], col] += 1
        if v in node_to_idx:
            x_rel[node_to_idx[v], col] += 1
    for j in range(len(etypes)):
        mx = x_rel[:, j].max()
        if mx > 0:
            x_rel[:, j] = np.log1p(x_rel[:, j]) / np.log1p(mx)

    # 4. PageRank
    pr = nx.pagerank(G, alpha=0.85)
    x_pr = np.array([pr.get(nid, 0.0) for nid in node_ids], dtype=np.float32)
    x_pr = (x_pr - x_pr.min()) / (x_pr.max() - x_pr.min() + 1e-9)

    x = torch.FloatTensor(np.c_[x_type, log_deg[:, None], x_rel, x_pr[:, None]])
    return x, node_to_idx, idx_to_node


def build_relation_vocab(edges_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build relation vocabulary.  Expects a ``relation`` column."""
    unique = sorted(edges_df["relation"].unique())
    rel_to_idx = {r: i for i, r in enumerate(unique)}
    idx_to_rel = {i: r for r, i in rel_to_idx.items()}
    return rel_to_idx, idx_to_rel


def build_edge_tensors(
    edges_df: pd.DataFrame,
    node_to_idx: Dict[str, int],
    rel_to_idx: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build edge_index and edge_type tensors, excluding SL-labeled edges.

    Returns (edge_index [2, E], edge_type [E], sl_excluded_count).
    """
    edge_list: list = []
    edge_type_list: list = []
    sl_excluded = 0

    for _, row in edges_df.iterrows():
        if row["src"] in node_to_idx and row["dst"] in node_to_idx:
            etype = str(row.get("type", "")).lower()
            if bool(row.get("is_sl_curated", False)) or ("sl" in etype):
                sl_excluded += 1
                continue
            edge_list.append([node_to_idx[row["src"]], node_to_idx[row["dst"]]])
            edge_type_list.append(rel_to_idx[row["relation"]])

    edge_index = torch.LongTensor(edge_list).t()
    edge_type = torch.LongTensor(edge_type_list)
    return edge_index, edge_type, sl_excluded


def build_sl_pairs(
    sl_pairs_df: pd.DataFrame,
    node_to_idx: Dict[str, int],
) -> torch.Tensor:
    """Deduplicated unique SL pair tensor [2, P]."""
    pairs: list = []
    seen: Set[Tuple[int, int]] = set()
    for _, row in sl_pairs_df.iterrows():
        if row["gene_a"] in node_to_idx and row["gene_b"] in node_to_idx:
            a, b = node_to_idx[row["gene_a"]], node_to_idx[row["gene_b"]]
            key = tuple(sorted([a, b]))
            if key not in seen:
                seen.add(key)
                pairs.append([a, b])
    return torch.LongTensor(pairs).t()


# ── Negative sampling ─────────────────────────────────────────────────────

class NegativeSampler:
    """
    Hard + distant negative sampling for SL prediction.

    - Hard negatives : gene pairs connected by non-SL edges
    - Distant negatives : curated genes not directly connected
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        pos_sl_pairs: torch.Tensor,
        nodes_df: pd.DataFrame,
        node_to_idx: Dict[str, int],
    ):
        self.pos_set: Set[Tuple[int, int]] = set()
        for i in range(pos_sl_pairs.shape[1]):
            a, b = pos_sl_pairs[0, i].item(), pos_sl_pairs[1, i].item()
            self.pos_set.add(tuple(sorted([a, b])))

        # gene indices
        gene_idx: Set[int] = set()
        for nid, idx in node_to_idx.items():
            row = nodes_df[nodes_df["node_id"] == nid]
            if not row.empty and "gene" in row["type"].iloc[0]:
                gene_idx.add(idx)

        # connected non-SL gene pairs (hard)
        connected: Set[Tuple[int, int]] = set()
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            if s in gene_idx and d in gene_idx and s != d:
                key = tuple(sorted([s, d]))
                if key not in self.pos_set:
                    connected.add(key)
        self.connected_pairs: List[Tuple[int, int]] = list(connected)

        # curated genes
        curated: Set[int] = set()
        for a, b in self.pos_set:
            curated.add(a)
            curated.add(b)
        curated_list = list(curated)

        # distant pairs
        all_direct = connected | self.pos_set
        distant: list = []
        for i, g1 in enumerate(curated_list):
            for g2 in curated_list[i + 1 :]:
                key = tuple(sorted([g1, g2]))
                if key not in all_direct:
                    distant.append(key)
        self.distant_pairs: List[Tuple[int, int]] = distant

    def sample(self, num_neg: int, hard_ratio: float = 0.7) -> torch.Tensor:
        """Return [2, N] tensor of negative gene pairs."""
        neg: list = []
        num_hard = int(num_neg * hard_ratio)
        num_distant = num_neg - num_hard

        if num_hard > 0 and self.connected_pairs:
            idxs = np.random.choice(
                len(self.connected_pairs),
                size=min(num_hard, len(self.connected_pairs)),
                replace=False,
            )
            neg.extend([list(self.connected_pairs[i]) for i in idxs])

        if num_distant > 0 and self.distant_pairs:
            idxs = np.random.choice(
                len(self.distant_pairs),
                size=min(num_distant, len(self.distant_pairs)),
                replace=False,
            )
            neg.extend([list(self.distant_pairs[i]) for i in idxs])

        if neg:
            return torch.LongTensor(neg).t()
        return torch.empty((2, 0), dtype=torch.long)


# ── Train / val split ─────────────────────────────────────────────────────

def prepare_train_val_split(
    pos_sl: torch.Tensor,
    sampler: NegativeSampler,
    val_ratio: float = 0.15,
    neg_multiplier: int = 3,
    hard_ratio: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split positive SL pairs into train / val and sample initial negatives.

    Returns (train_pos, train_neg, val_pos, val_neg).
    """
    num_pos = pos_sl.shape[1]
    perm = torch.randperm(num_pos)
    num_val = max(10, int(val_ratio * num_pos))

    val_pos = pos_sl[:, perm[:num_val]]
    train_pos = pos_sl[:, perm[num_val:]]

    train_neg = sampler.sample(train_pos.shape[1] * neg_multiplier, hard_ratio)
    val_neg = sampler.sample(val_pos.shape[1] * 2, hard_ratio)
    return train_pos, train_neg, val_pos, val_neg
