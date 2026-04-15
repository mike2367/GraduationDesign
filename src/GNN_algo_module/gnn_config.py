from __future__ import annotations

# ═════════════════════════════════════════════════════════════════════════════
# GNN Architecture & Training Configuration
#
# Weight / rank definitions are LEARNED by the trained model and saved to
# checkpoints/learned_weights.json.  algo_config.py loads them at runtime,
# falling back to built-in defaults before the first training run.
# ═════════════════════════════════════════════════════════════════════════════

# ── Graph Data ────────────────────────────────────────────────────────────
FULL_GRAPHML: str = "/data/guoyu/KG-LLM-XSL/output/ablation_graphs/full.graphml"
CKPTS_DIR: str = "/data/guoyu/KG-LLM-XSL/output/GNN_checkpoints"

# ── R-GCN Architecture ───────────────────────────────────────────────────
NUM_LAYERS: int = 4
HIDDEN_CHANNELS: int = 192
OUT_CHANNELS: int = 96
NUM_BASES: int = 16
DROPOUT: float = 0.3

# ── Training ─────────────────────────────────────────────────────────────
NUM_EPOCHS: int = 1000
LR: float = 0.003
WEIGHT_DECAY: float = 0.005
WARMUP_EPOCHS: int = 15
GRADIENT_CLIP: float = 1.0

# ── Reproducibility ─────────────────────────────────────────────────────
RANDOM_SEED: int = 42
DETERMINISTIC_CUDA: bool = True
STRICT_DETERMINISM: bool = False

# ── Negative Sampling ────────────────────────────────────────────────────
NEG_MULTIPLIER: int = 3
HARD_RATIO: float = 0.7
VAL_RATIO: float = 0.15

# ── LR Scheduler (Cosine annealing with linear warmup) ───────────────────
LR_MIN: float = 5e-5

# ── Counterfactual Analysis ──────────────────────────────────────────────
# Primary relation types for diagnostic visualization and deep-dive.
# Note: The export logic in train.py now automatically computes CF drops for 
# ALL relation types present in the graph to ensure complete coverage.
COUNTERFACTUAL_EDGE_TYPES: list = [
    "STRING_association",
    "in_pathway",
    "TF_regulates",
    "targets", 
    "members",
    "DepMap_codependency",
    "OmniPath_interaction",
    "encodes",
    "driver_in",
    "mutated_in",
]

# ── Checkpoint / Learned Weights ─────────────────────────────────────────
CHECKPOINT_FILENAME: str = "rgcn_sl_predictor.pt"
LEARNED_WEIGHTS_FILENAME: str = "learned_weights.json"

# ── Learned-weight export mode ───────────────────────────────────────────
# Which signal should be used to EXPORT the heuristic weights for the
# downstream algorithm module.
# - 'attention': use model-extracted norms/attention-like signals (fast)
# - 'cf':        use counterfactual drop on validation positives (slower, but
#                aligned with masking sensitivity)
# ── Learned-weight export blending ───────────────────────────────────────
# Exported heuristic weights are a linear blend of:
# - attention/norm-derived signal (internal gating)
# - counterfactual-drop signal (masking sensitivity)
#
# alpha=1.0 -> pure attention/norm
# alpha=0.0 -> pure counterfactual drop
WEIGHT_BLEND_ALPHA: float = 0.7

# ── Learned-weight calibration (export-time) ─────────────────────────────
# Calibrates extracted per-type/per-source scores into a bounded range.
# This is for the heuristic algorithm module; it does NOT affect training.
WEIGHT_CALIBRATION: str = "zscore"  # 'sphere', 'zscore', 'minmax', 'sigmoid' or 'none'
CALIBRATION_ALPHA: float = 2.0      # Used by 'sigmoid'/'sphere'/'zscore'
REL_WEIGHT_LO: float = 0.01
REL_WEIGHT_HI: float = 1.0
SRC_WEIGHT_LO: float = 0.01
SRC_WEIGHT_HI: float = 1.0
