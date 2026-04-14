"""Training entry-point for the R-GCN SL predictor."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.data import Data
from tqdm.auto import tqdm

# ── path setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
for _p in (str(SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from GNN_algo_module import gnn_config as gcfg
from GNN_algo_module.data import (
    NegativeSampler,
    build_edge_tensors,
    build_node_features,
    build_relation_vocab,
    build_sl_pairs,
    load_knowledge_graph,
    prepare_train_val_split,
)
from GNN_algo_module.model import RGCN_SL_Predictor


def set_global_seed(seed: int, *, deterministic_cuda: bool = True, strict: bool = False) -> None:
    """Best-effort determinism across Python/numpy/torch.

    Note: Some CUDA ops can still be nondeterministic depending on build.
    """
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_cuda:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        if strict:
            # Strict determinism can fail on some CUDA ops unless additional env vars
            # are set (e.g., CUBLAS_WORKSPACE_CONFIG). Use only if you know you need it.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


def _calibrate_weights(
    raw: dict[str, float],
    *,
    lo: float,
    hi: float,
    method: str,
    alpha: float,
) -> dict[str, float]:
    """Map raw positive scores into a bounded, more separable weight dict."""
    if not raw:
        return {}
    m = (method or "none").strip().lower()
    if m == "none":
        # Expect raw already in roughly [0, 1] or [0, 2]
        return {k: float(v) for k, v in raw.items()}

    keys = list(raw.keys())
    vals = np.array([float(raw[k]) for k in keys], dtype=np.float64)
    eps = 1e-12

    # Mean-center to allow symmetric spreading.
    centered = vals - vals.mean()

    if m == "sphere":
        denom = float(np.linalg.norm(centered)) + eps
        # unit-sphere direction; scale by sqrt(d) to keep magnitudes usable
        spread = centered / denom * float(np.sqrt(len(vals)))
        # Map spread -> (0, 1) -> [lo, hi]
        p = 1.0 / (1.0 + np.exp(-float(alpha) * spread))
        w = float(lo) + (float(hi) - float(lo)) * p

    elif m == "zscore":
        std = float(centered.std()) + eps
        spread = centered / std
        # Map z-score -> sigmoid -> [lo, hi]
        p = 1.0 / (1.0 + np.exp(-float(alpha) * spread))
        w = float(lo) + (float(hi) - float(lo)) * p

    elif m == "minmax":
        # Simple min-max normalization to [lo, hi]
        min_val, max_val = vals.min(), vals.max()
        if max_val - min_val < eps:
            return {k: float(hi) for k in raw}
        w = float(lo) + (vals - min_val) / (max_val - min_val) * (float(hi) - float(lo))
        
    elif m == "sigmoid":
        # Direct sigmoid on raw values (if raw values are logits)
        # usually raw attention is already softmaxed (0-1) or logits
        # Here we assume we want to spread them.
        # Let's use centered sigmoid
        p = 1.0 / (1.0 + np.exp(-float(alpha) * centered))
        w = float(lo) + (float(hi) - float(lo)) * p

    else:
        raise ValueError(f"Unknown WEIGHT_CALIBRATION={method!r}")
    return {k: round(float(v), 4) for k, v in zip(keys, w)}


def _avg_norms_by_type_and_source(
    model: RGCN_SL_Predictor,
    idx_to_rel: dict[int, str],
) -> tuple[dict[str, float], dict[str, float]]:
    norms = model._relation_norms()  # relation_idx -> norm
    by_type: dict[str, list[float]] = {}
    by_src: dict[str, list[float]] = {}
    for r_idx, norm_val in norms.items():
        compound = idx_to_rel.get(int(r_idx), f"unknown_{r_idx}")
        parts = compound.split("|||")
        rel_type = parts[0]
        source = parts[1] if len(parts) > 1 else "unknown"
        by_type.setdefault(rel_type, []).append(float(norm_val))
        by_src.setdefault(source, []).append(float(norm_val))

    avg_type = {k: float(np.mean(v)) for k, v in by_type.items() if v}
    avg_src = {k: float(np.mean(v)) for k, v in by_src.items() if v}
    return avg_type, avg_src


def _relation_indices_for_type(idx_to_rel: dict[int, str], rel_type: str) -> np.ndarray:
    rel_type = str(rel_type)
    out: list[int] = []
    for r_idx, compound in idx_to_rel.items():
        parts = str(compound).split("|||", 1)
        t = parts[0]
        if t == rel_type:
            out.append(int(r_idx))
    return np.asarray(out, dtype=np.int64)


def _relation_indices_for_source(idx_to_rel: dict[int, str], source: str) -> np.ndarray:
    source = str(source)
    out: list[int] = []
    for r_idx, compound in idx_to_rel.items():
        parts = str(compound).split("|||", 1)
        s = parts[1] if len(parts) > 1 else "unknown"
        if s == source:
            out.append(int(r_idx))
    return np.asarray(out, dtype=np.int64)


@torch.no_grad()
def _cf_drop_for_relation_indices(
    model: RGCN_SL_Predictor,
    data: Data,
    pairs_pos: torch.Tensor,
    base_probs: torch.Tensor,
    drop_rel_indices: np.ndarray,
) -> float:
    """Mean probability drop on positives when masking all edges of some relation indices."""
    if drop_rel_indices.size == 0:
        return 0.0

    # Build a boolean keep-mask over edges, based on edge_type (relation index)
    et = data.edge_type.detach().cpu().numpy()
    keep_np = ~np.isin(et, drop_rel_indices)
    if bool(keep_np.all()):
        return 0.0

    keep = torch.from_numpy(keep_np).to(device=data.edge_type.device)
    masked = Data(
        x=data.x,
        edge_index=data.edge_index[:, keep],
        edge_type=data.edge_type[keep],
        num_nodes=data.num_nodes,
    )
    scores, _ = model(masked, pairs_pos)
    probs = torch.sigmoid(scores)
    drop = (base_probs - probs).mean().item()
    return float(drop)


@torch.no_grad()
def _compute_cf_importance(
    model: RGCN_SL_Predictor,
    data: Data,
    val_pos: torch.Tensor,
    idx_to_rel: dict[int, str],
    *,
    edge_types: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (cf_drop_by_type, cf_drop_by_source)."""
    model.eval()
    val_pos = val_pos.to(data.x.device)
    base_scores, _ = model(data, val_pos)
    base_probs = torch.sigmoid(base_scores)

    # per-type CF
    cf_by_type: dict[str, float] = {}
    for t in edge_types:
        if str(t) == "SL_pair":
            continue
        drop_rel = _relation_indices_for_type(idx_to_rel, t)
        drop = _cf_drop_for_relation_indices(model, data, val_pos, base_probs, drop_rel)
        cf_by_type[str(t)] = float(drop)

    # per-source CF
    sources = sorted({(str(v).split("|||", 1)[1] if "|||" in str(v) else "unknown") for v in idx_to_rel.values()})
    cf_by_source: dict[str, float] = {}
    for s in sources:
        drop_rel = _relation_indices_for_source(idx_to_rel, s)
        drop = _cf_drop_for_relation_indices(model, data, val_pos, base_probs, drop_rel)
        cf_by_source[str(s)] = float(drop)

    return cf_by_type, cf_by_source


# ── training loop ─────────────────────────────────────────────────────────

def train_model(
    model: RGCN_SL_Predictor,
    data: Data,
    train_pos: torch.Tensor,
    val_pos: torch.Tensor,
    val_neg: torch.Tensor,
    sampler: NegativeSampler,
    *,
    num_epochs: int = gcfg.NUM_EPOCHS,
    lr: float = gcfg.LR,
    weight_decay: float = gcfg.WEIGHT_DECAY,
    hard_ratio: float = gcfg.HARD_RATIO,
    neg_multiplier: int = gcfg.NEG_MULTIPLIER,
    device: torch.device | str = "cpu",
) -> dict:
    """Train + validate, returning a history dict."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # Cosine annealing with linear warmup
    warmup = LinearLR(
        optimizer, start_factor=0.01, total_iters=gcfg.WARMUP_EPOCHS,
    )
    cosine = CosineAnnealingLR(
        optimizer, T_max=num_epochs - gcfg.WARMUP_EPOCHS, eta_min=gcfg.LR_MIN,
    )
    scheduler = SequentialLR(
        optimizer, [warmup, cosine], milestones=[gcfg.WARMUP_EPOCHS],
    )

    history: dict = {"train_loss": [], "val_auc": [], "val_ap": [], "lr": []}
    train_pos = train_pos.to(device)
    val_pos = val_pos.to(device)
    val_neg = val_neg.to(device)

    # pre-build val tensors once
    vp = torch.cat([val_pos, val_neg], dim=1)
    vl = torch.cat([
        torch.ones(val_pos.shape[1]),
        torch.zeros(val_neg.shape[1]),
    ]).cpu().numpy()

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        train_neg_fresh = sampler.sample(
            train_pos.shape[1] * neg_multiplier, hard_ratio,
        ).to(device)

        pairs = torch.cat([train_pos, train_neg_fresh], dim=1)
        labels = torch.cat([
            torch.ones(train_pos.shape[1]),
            torch.zeros(train_neg_fresh.shape[1]),
        ]).to(device)

        scores, _ = model(data, pairs)
        loss = criterion(scores, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gcfg.GRADIENT_CLIP)
        optimizer.step()
        scheduler.step()
        history["train_loss"].append(loss.item())

        # validate every epoch
        model.eval()
        with torch.no_grad():
            vs, _ = model(data, vp)
            vprobs = torch.sigmoid(vs).cpu().numpy()
            auc = roc_auc_score(vl, vprobs)
            ap = average_precision_score(vl, vprobs)
        history["val_auc"].append(auc)
        history["val_ap"].append(ap)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        pbar.set_postfix(
            Loss=f"{loss.item():.4f}",
            AUC=f"{auc:.4f}",
            LR=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return history


# ── full pipeline ─────────────────────────────────────────────────────────

def run_training_pipeline(graphml_path: str = gcfg.FULL_GRAPHML) -> tuple:
    """
    End-to-end: load → build features → train → extract weights → save.

    Returns (model, data, history, context_dict).
    """
    set_global_seed(
        int(getattr(gcfg, "RANDOM_SEED", 42)),
        deterministic_cuda=bool(getattr(gcfg, "DETERMINISTIC_CUDA", True)),
        strict=bool(getattr(gcfg, "STRICT_DETERMINISM", False)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # load
    print("Loading knowledge graph …")
    G, nodes_df, edges_df, sl_pairs_df = load_knowledge_graph(graphml_path)
    edges_df["relation"] = edges_df["type"] + "|||" + edges_df["source"]

    # Safety check: SL edges must not be present in the message-passing graph.
    if not edges_df.empty:
        sl_mask = edges_df["type"].astype(str).str.lower().str.contains("sl")
        if bool(sl_mask.any()):
            bad = edges_df.loc[sl_mask, ["type", "source"]].head(5).to_dict("records")
            raise RuntimeError(
                "Data leakage risk: SL-labeled edges are present in the message-passing graph. "
                f"Examples: {bad}"
            )
    print(
        f"  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}"
        f"  SL pairs: {len(sl_pairs_df)}"
    )

    # features
    x, node_to_idx, idx_to_node = build_node_features(nodes_df, G)
    rel_to_idx, idx_to_rel = build_relation_vocab(edges_df)
    edge_index, edge_type, sl_excluded = build_edge_tensors(
        edges_df, node_to_idx, rel_to_idx,
    )
    pos_sl = build_sl_pairs(sl_pairs_df, node_to_idx)

    num_relations = len(rel_to_idx)
    num_bases = min(num_relations, gcfg.NUM_BASES)

    data = Data(
        x=x, edge_index=edge_index, edge_type=edge_type,
        num_nodes=len(node_to_idx),
    ).to(device)

    print(
        f"  Features: {x.shape}  Relations: {num_relations}"
        f"  Training edges: {edge_index.shape[1]:,}  (SL excluded: {sl_excluded})"
    )
    print(f"  Unique SL pairs: {pos_sl.shape[1]}")

    # negative sampler & split
    sampler = NegativeSampler(edge_index, pos_sl, nodes_df, node_to_idx)
    train_pos, train_neg, val_pos, val_neg = prepare_train_val_split(
        pos_sl, sampler, gcfg.VAL_RATIO, gcfg.NEG_MULTIPLIER, gcfg.HARD_RATIO,
    )
    print(f"  Train: {train_pos.shape[1]} pos, {train_neg.shape[1]} neg")
    print(f"  Val:   {val_pos.shape[1]} pos, {val_neg.shape[1]} neg\n")

    # model
    model = RGCN_SL_Predictor(
        in_channels=x.shape[1],
        hidden_channels=gcfg.HIDDEN_CHANNELS,
        out_channels=gcfg.OUT_CHANNELS,
        num_relations=num_relations,
        num_layers=gcfg.NUM_LAYERS,
        num_bases=num_bases,
        dropout=gcfg.DROPOUT,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}\n")

    # train
    t0 = time.time()
    history = train_model(
        model, data, train_pos, val_pos, val_neg, sampler, device=device,
    )
    elapsed = time.time() - t0

    best_auc = max(history["val_auc"])
    best_ap = max(history["val_ap"])
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  Best AUC: {best_auc:.4f}")
    print(f"  Best AP:  {best_ap:.4f}")

    # ── extract learned weights ──────────────────────────────────────────
    print("\nExtracting learned weights …")
    learned = model.extract_all_weights(data, node_to_idx, idx_to_rel, nodes_df)

    # Optional calibration to avoid near-flat weights; also used to map CF drops
    cal_method = (getattr(gcfg, "WEIGHT_CALIBRATION", "none") or "none").strip().lower()
    cal_meta = {
        "method": cal_method,
        "alpha": float(getattr(gcfg, "CALIBRATION_ALPHA", 2.0)),
        "rel_range": [float(getattr(gcfg, "REL_WEIGHT_LO", 0.01)), float(getattr(gcfg, "REL_WEIGHT_HI", 1.0))],
        "src_range": [float(getattr(gcfg, "SRC_WEIGHT_LO", 0.01)), float(getattr(gcfg, "SRC_WEIGHT_HI", 1.0))],
    }

    try:
        # Compute both signals
        print("  Export signal: attention/norm + counterfactual (linear blend)")
        raw_type_attn, raw_src_attn = _avg_norms_by_type_and_source(model, idx_to_rel)
        
        # Use all relation types present in the graph for CF analysis to ensure no gaps.
        # (Exclude SL_pair as it is the target).
        all_rel_types = sorted([t for t in raw_type_attn.keys() if t != "SL_pair"])
        cf_types, cf_sources = _compute_cf_importance(
            model,
            data,
            val_pos,
            idx_to_rel,
            edge_types=all_rel_types,
        )
        raw_type_cf = {k: max(0.0, float(v)) for k, v in (cf_types or {}).items()}
        raw_src_cf = {k: max(0.0, float(v)) for k, v in (cf_sources or {}).items()}

        # Ensure CF dict covers all attention keys (missing => 0 drop)
        for k in raw_type_attn.keys():
            raw_type_cf.setdefault(k, 0.0)
        for k in raw_src_attn.keys():
            raw_src_cf.setdefault(k, 0.0)

        alpha_blend = float(getattr(gcfg, "WEIGHT_BLEND_ALPHA", 0.5))
        alpha_blend = max(0.0, min(1.0, alpha_blend))

        if cal_method != "none":
            rel_attn = _calibrate_weights(
                raw_type_attn,
                lo=float(getattr(gcfg, "REL_WEIGHT_LO", 0.01)),
                hi=float(getattr(gcfg, "REL_WEIGHT_HI", 1.0)),
                method=cal_method,
                alpha=float(getattr(gcfg, "CALIBRATION_ALPHA", 2.0)),
            )
            rel_cf = _calibrate_weights(
                raw_type_cf,
                lo=float(getattr(gcfg, "REL_WEIGHT_LO", 0.01)),
                hi=float(getattr(gcfg, "REL_WEIGHT_HI", 1.0)),
                method=cal_method,
                alpha=float(getattr(gcfg, "CALIBRATION_ALPHA", 2.0)),
            )
            src_attn = _calibrate_weights(
                raw_src_attn,
                lo=float(getattr(gcfg, "SRC_WEIGHT_LO", 0.01)),
                hi=float(getattr(gcfg, "SRC_WEIGHT_HI", 1.0)),
                method=cal_method,
                alpha=float(getattr(gcfg, "CALIBRATION_ALPHA", 2.0)),
            )
            src_cf = _calibrate_weights(
                raw_src_cf,
                lo=float(getattr(gcfg, "SRC_WEIGHT_LO", 0.01)),
                hi=float(getattr(gcfg, "SRC_WEIGHT_HI", 1.0)),
                method=cal_method,
                alpha=float(getattr(gcfg, "CALIBRATION_ALPHA", 2.0)),
            )
        else:
            rel_attn = {k: round(float(v), 4) for k, v in raw_type_attn.items()}
            rel_cf = {k: round(float(v), 4) for k, v in raw_type_cf.items()}
            src_attn = {k: round(float(v), 4) for k, v in raw_src_attn.items()}
            src_cf = {k: round(float(v), 4) for k, v in raw_src_cf.items()}

        # Blend in calibrated space
        rel_keys = sorted(set(rel_attn.keys()) | set(rel_cf.keys()))
        src_keys = sorted(set(src_attn.keys()) | set(src_cf.keys()))
        rel_w = {
            k: round(alpha_blend * float(rel_attn.get(k, 0.0)) + (1.0 - alpha_blend) * float(rel_cf.get(k, 0.0)), 4)
            for k in rel_keys
        }
        src_w = {
            k: round(alpha_blend * float(src_attn.get(k, 0.0)) + (1.0 - alpha_blend) * float(src_cf.get(k, 0.0)), 4)
            for k in src_keys
        }

        learned["EDGE_RELATION_WEIGHT"] = rel_w
        learned["EDGE_TYPE_WEIGHT"] = rel_w
        learned["EDGE_SOURCE_WEIGHT"] = src_w
        learned["EDGE_TYPE_PRIORITY"] = {
            t: rank
            for rank, (t, _) in enumerate(sorted(rel_w.items(), key=lambda x: -x[1]))
        }
        learned.setdefault("_meta", {})["weight_calibration"] = cal_meta
        learned.setdefault("_meta", {})["weight_blend"] = {
            "alpha_attention": round(alpha_blend, 4),
            "alpha_cf": round(1.0 - alpha_blend, 4),
        }
    except Exception as e:
        print(f"  [warn] weight export skipped: {e}")

    meta = dict(learned.get("_meta", {}) or {})
    meta.update({
        "best_auc": round(best_auc, 4),
        "best_ap": round(best_ap, 4),
        "num_epochs": gcfg.NUM_EPOCHS,
        "num_layers": gcfg.NUM_LAYERS,
        "hidden": gcfg.HIDDEN_CHANNELS,
        "out": gcfg.OUT_CHANNELS,
        "weight_mode": "blend",
    })
    learned["_meta"] = meta

    # Ensure SL_pair has a defined weight in the export.
    # Since it is the target relation not used in message passing, we assign it a fixed high confidence.
    # This prevents the algorithm module from needing heuristic fallbacks.
    for sec in ("EDGE_RELATION_WEIGHT", "EDGE_TYPE_WEIGHT"):
        if isinstance(learned.get(sec), dict):
            learned[sec]["SL_pair"] = 1.0
            
    # Also add it to priority list (put it at the top)
    if isinstance(learned.get("EDGE_TYPE_PRIORITY"), dict):
        current_max = max(learned["EDGE_TYPE_PRIORITY"].values()) if learned["EDGE_TYPE_PRIORITY"] else 0
        learned["EDGE_TYPE_PRIORITY"]["SL_pair"] = current_max + 1

    ckpt_dir = Path("/data/guoyu/KG-LLM-XSL/output/GNN_checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # save learned weights JSON
    lw_path = ckpt_dir / gcfg.LEARNED_WEIGHTS_FILENAME
    lw_path.write_text(json.dumps(learned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Learned weights → {lw_path}")

    for section, vals in learned.items():
        if section.startswith("_"):
            continue
        if isinstance(vals, dict):
            top3 = sorted(vals.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0)[:3]
            print(f"  {section}: {dict(top3)} …")

    # save model checkpoint
    ckpt_path = ckpt_dir / gcfg.CHECKPOINT_FILENAME
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node,
        "rel_to_idx": rel_to_idx,
        "idx_to_rel": idx_to_rel,
        "num_relations": num_relations,
        "in_channels": x.shape[1],
        "learned_weights": learned,
    }, ckpt_path)
    print(f"  Checkpoint → {ckpt_path}\n")

    ctx = {
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node,
        "rel_to_idx": rel_to_idx,
        "idx_to_rel": idx_to_rel,
        "edges_df": edges_df,
        "nodes_df": nodes_df,
        "sl_pairs_df": sl_pairs_df,
        "sampler": sampler,
        "train_pos": train_pos,
        "val_pos": val_pos,
        "val_neg": val_neg,
        "num_relations": num_relations,
        "learned_weights": learned,
    }
    return model, data, history, ctx


if __name__ == "__main__":
    run_training_pipeline()
