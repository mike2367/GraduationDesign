from __future__ import annotations

from pathlib import Path


# Read evaluation results from the canonical data directory.
DATA_ROOT = Path("/data/guoyu/KG-LLM-XSL/output")
EVAL_RESULTS_DIR = DATA_ROOT / "eval_results"

# Write visualization artifacts (if any) to the workspace output folder,
# to avoid modifying files under /data.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_OUTPUT_ROOT = REPO_ROOT / "output"
VIS_OUTPUT_DIR = LOCAL_OUTPUT_ROOT / "visualizations"

METRICS_TO_PLOT = [
	"f1_raw",
	"f1_raw_full",
	"f1_raw_topk_p50",
	"f1_raw_topk_p75",
	"precision_raw",
	"recall_raw",
	"hallucination_score",
	"faithfulness_score",
	"total_similarity",
	"format_score",
]

DASHBOARD_METRICS = [
	"f1_raw",
	"hallucination_score",
	"total_similarity",
	"format_score",
]

BASELINE_STRATEGY_NAME = "baseline"
NAIVE_STRATEGY_NAME = "naive"
GENERATE_LEGACY_PLOTS_WHEN_PAIRED = False

PLOT_TEMPLATE = "plotly_white"
PLOT_HEIGHT = 600
PLOT_WIDTH = 1000

# Publication-style defaults (used by visualization.plot_utils)
FONT_FAMILY = "Arial"
FONT_SIZE = 13
FONT_COLOR = "#1f1f1f"

# Slightly looser margins to avoid cramped subplots and clipped labels
MARGIN_L = 70
MARGIN_R = 40
MARGIN_T = 90
MARGIN_B = 70

# Optional legend title; keep empty to reduce clutter.
LEGEND_TITLE = ""

STRATEGY_COLORS = {
	"baseline": "#44B5F6",
	"self_refine": "#2EEFBB",
	"cove": "#F34DA8",
	"naive": "#F09753",
}

DELTA_COLOR_POS = "#31CEEA"
DELTA_COLOR_NEG = "#FB9C53"

DASHBOARD_TOP_PAIRS_LIMIT = 12
DASHBOARD_MIN_HEIGHT = 720
DASHBOARD_ROW_HEIGHT = 260
DASHBOARD_MIN_WIDTH = 1400
