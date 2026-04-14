"""
Scientific Visualization Module - Individual Plot Functions
=============================================================

Modular, publication-quality individual plots following:
- Nature Methods visual communication guidelines
- Seaborn color palette best practices
- Cleveland & McGill perceptual studies

Each function produces a single, focused visualization.
Figures are sized at 1000x800 pixels (10x8 inches at 100 DPI).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import seaborn as sns

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# ============================================================================
# COLOR PALETTES - Publication-quality, colorblind-safe
# ============================================================================

# Paul Tol's colorblind-friendly palette (recommended for scientific figures)
TOLS_BRIGHT = {
    "blue": "#0468CD",
    "cyan": "#059CCF",
    "yellow": "#F08221",
    "red": "#BC081D",
    "purple": "#930A8C",
    "green": "#0B9C23",
    "grey": "#BBBBBB",
}

# Two-group comparison colors
COLOR_BASELINE = "#0468CD"  
COLOR_NAIVE = "#F08221"     

# Effect direction colors
COLOR_POSITIVE = "#4017E4" 
COLOR_NEGATIVE = "#C4172B"  
COLOR_NEUTRAL = "#1299A5"   

# Figure sizing (1000x800 pixels at 100 DPI = 10x8 inches)
FIG_WIDTH = 10
FIG_HEIGHT = 8
FIG_DPI = 100

# Typography
FONT_FAMILY = "sans-serif"
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 10
FONT_SIZE_ANNOT = 9

# Font candidates aligned with the thesis template (artratex):
# - main Chinese font: SimSun
# - sans Chinese font: SimHei
_CHINESE_FONT_CANDIDATES: list[str] = [
    "SimHei",
    "SimSun",
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
]

_WINDOWS_FONT_FILES: list[Path] = [
    Path(r"C:\\Windows\\Fonts\\simsun.ttc"),
    Path(r"C:\\Windows\\Fonts\\simhei.ttf"),
    Path(r"C:\\Windows\\Fonts\\msyh.ttc"),
    Path(r"C:\\Windows\\Fonts\\msyhbd.ttc"),
]


def _pick_available_font(preferred: list[str] = _CHINESE_FONT_CANDIDATES) -> str:
    """Pick the first available font name from preferred list.

    We use matplotlib's font resolver to avoid silent fallback (which causes
    Chinese glyphs to appear as squares when a CJK-capable font is not used).
    """
    for name in preferred:
        try:
            fm.findfont(FontProperties(family=name), fallback_to_default=False)
            return name
        except Exception:
            continue
    return preferred[0] if preferred else "SimHei"


def _ensure_chinese_font_registered() -> str:
    """Best-effort ensure CJK fonts are discoverable, then return chosen font."""
    for font_path in _WINDOWS_FONT_FILES:
        if font_path.exists():
            try:
                fm.fontManager.addfont(str(font_path))
            except Exception:
                pass
    return _pick_available_font()

# Grid styling
GRID_ALPHA = 0.35
SPINE_WIDTH = 1.0


# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

def configure_style() -> None:
    """Configure matplotlib/seaborn for publication-quality output."""
    sns.set_style("whitegrid", {
        "axes.edgecolor": "#333333",
        "axes.linewidth": SPINE_WIDTH,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.6,
        "grid.alpha": GRID_ALPHA,
    })

    chosen_font = _ensure_chinese_font_registered()
    
    plt.rcParams.update({
        # Force a specific CJK-capable font (aligned with thesis template).
        # Using a concrete font name is more reliable than relying on generic
        # families that may silently fall back to non-CJK fonts.
        "font.family": [chosen_font],
        "font.sans-serif": [chosen_font],
        "font.serif": [chosen_font],
        "axes.unicode_minus": False,
        # Bold typography across figures (requested) while keeping colors intact.
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "font.size": FONT_SIZE_TICK,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        # Embed TrueType fonts in vector outputs to avoid missing glyphs.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "figure.dpi": FIG_DPI,
    })


# ============================================================================
# CHINESE LABELS (PLOT ANNOTATIONS)
# ============================================================================

# Strategy/model identifiers are kept in CSVs; plots should display Chinese only.
STRATEGY_NAME_ZH: dict[str, str] = {
    # Unified terminology (paper + figures)
    "naive": "基线",
    "baseline": "无自精炼",
    "self_refine": "无链式验证",
    "cove": "完整模型",
}

METRIC_NAME_ZH: dict[str, str] = {
    # Note: this repo historically uses recall-like coverage under different keys.
    "f1_raw": "特征召回率",
    "recall_raw": "特征召回率",
    "precision_raw": "精确率",
    "f1_raw_full": "F1分数（完整）",
    "f1_raw_topk_p50": "F1分数（前50%）",
    "f1_raw_topk_p75": "F1分数（前75%）",
    "hallucination_score": "幻觉率",
    "faithfulness_score": "忠实度",
    "total_similarity": "语义相似度",
    "format_score": "内容覆盖度",
}


def _zh_strategy(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return STRATEGY_NAME_ZH.get(key, "策略")


def _zh_metric(metric: str) -> str:
    key = str(metric).strip().lower().replace("-", "_").replace(" ", "_")
    return METRIC_NAME_ZH.get(key, "指标")


def _zh_label(name: str) -> str:
    """Best-effort conversion of any plot-visible label to Chinese.

    If the input already contains Chinese characters, it is returned unchanged.
    Otherwise, we attempt to map common strategy names; if unknown, we fall back
    to a generic Chinese label to avoid English appearing in figures.
    """
    text = "" if name is None else str(name).strip()
    if not text:
        return "标签"

    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return text

    normalized = text.lower().replace("-", "_").replace(" ", "_")
    if normalized in STRATEGY_NAME_ZH:
        return STRATEGY_NAME_ZH[normalized]
    if "baseline" in normalized:
        return STRATEGY_NAME_ZH.get("baseline", "无自精炼")
    if "naive" in normalized:
        return STRATEGY_NAME_ZH.get("naive", "基线")
    if "self" in normalized or "refine" in normalized:
        return STRATEGY_NAME_ZH.get("self_refine", "无链式验证")
    if "cove" in normalized:
        return STRATEGY_NAME_ZH.get("cove", "完整模型")

    return "标签"


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_csv(csv_path: Path) -> list[dict]:
    """Load CSV as list of dicts for backward compatibility."""
    return pd.read_csv(csv_path).to_dict(orient="records")


def load_comparison_data(
    normal_csv: Path,
    naive_csv: Path,
    baseline_strategy: str = "baseline",
    naive_strategy: str = "naive",
) -> pd.DataFrame:
    """
    Load and merge baseline/naive CSVs on gene pairs.
    
    Returns:
        Merged DataFrame with _base and _naive suffixes for metrics.
    """
    def _normalize_pair_key(row: pd.Series) -> str:
        a = str(row.get("gene_a", "")).strip().upper()
        b = str(row.get("gene_b", "")).strip().upper()
        return "/".join(sorted([a, b]))
    
    bdf = pd.read_csv(normal_csv)
    ndf = pd.read_csv(naive_csv)
    
    # Filter by strategy if column exists
    if "strategy" in bdf.columns:
        bdf = bdf[bdf["strategy"] == baseline_strategy].copy()
    if "strategy" in ndf.columns:
        ndf = ndf[ndf["strategy"] == naive_strategy].copy()
    
    bdf["pair_key"] = bdf.apply(_normalize_pair_key, axis=1)
    ndf["pair_key"] = ndf.apply(_normalize_pair_key, axis=1)
    
    merged = bdf.merge(ndf, on="pair_key", how="inner", suffixes=("_base", "_naive"))
    
    if merged.empty:
        raise ValueError("No overlapping gene pairs found between datasets")
    
    return merged


def build_summary_dataframe(
    merged: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """
    Build a summary DataFrame with statistics for each metric.
    
    Returns:
        DataFrame with columns: Metric, Baseline_Mean, Naive_Mean, Delta, Pct_Change
    """
    rows = []
    for metric in metrics:
        base_col = f"{metric}_base"
        naive_col = f"{metric}_naive"
        
        if base_col not in merged.columns or naive_col not in merged.columns:
            continue
        
        base_vals = pd.to_numeric(merged[base_col], errors="coerce").dropna()
        naive_vals = pd.to_numeric(merged[naive_col], errors="coerce").dropna()
        
        if len(base_vals) == 0 or len(naive_vals) == 0:
            continue
        
        base_mean = base_vals.mean()
        naive_mean = naive_vals.mean()
        delta = naive_mean - base_mean
        pct_change = (delta / base_mean * 100) if base_mean != 0 else 0
        
        rows.append({
            "Metric": _clean_metric_name(metric),
            "Baseline_Mean": round(base_mean, 4),
            "Naive_Mean": round(naive_mean, 4),
            "Delta": round(delta, 4),
            "Pct_Change": f"{pct_change:+.1f}%",
        })
    
    return pd.DataFrame(rows)


def build_per_pair_dataframe(
    merged: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """
    Build a per-pair performance DataFrame.
    
    Returns:
        DataFrame with pair-level metric values.
    """
    rows = []
    for _, row in merged.iterrows():
        pair_row = {"Pair": row["pair_key"]}
        
        base_scores = []
        naive_scores = []
        
        for m in metrics:
            base_val = pd.to_numeric(row.get(f"{m}_base", np.nan), errors="coerce")
            naive_val = pd.to_numeric(row.get(f"{m}_naive", np.nan), errors="coerce")
            
            pair_row[f"{_clean_metric_name(m)[:8]}_B"] = round(base_val, 3) if not np.isnan(base_val) else np.nan
            pair_row[f"{_clean_metric_name(m)[:8]}_N"] = round(naive_val, 3) if not np.isnan(naive_val) else np.nan
            
            base_scores.append(base_val)
            naive_scores.append(naive_val)
        
        avg_base = np.nanmean(base_scores)
        avg_naive = np.nanmean(naive_scores)
        pair_row["Avg_Delta"] = round(avg_naive - avg_base, 4)
        pair_row["Winner"] = "基线" if avg_base > avg_naive else "无图证据"
        
        rows.append(pair_row)
    
    return pd.DataFrame(rows)


def _clean_metric_name(metric: str) -> str:
    """Convert metric names to human-readable labels."""
    return _zh_metric(metric)


def _metric_higher_better(metric: str) -> bool:
    """Determine if higher values are better for a metric."""
    return "hallucination" not in metric.lower()


# ============================================================================
# INDIVIDUAL PLOT FUNCTIONS
# ============================================================================

def plot_violin_comparison(
    merged: pd.DataFrame,
    metric: str,
    baseline_label: str = "基线",
    naive_label: str = "无图证据",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot violin comparison for a single metric.
    
    Args:
        merged: Merged DataFrame with _base and _naive columns
        metric: Metric name (without suffix)
        baseline_label: Label for baseline strategy
        naive_label: Label for naive strategy
        title: Optional custom title
    
    Returns:
        matplotlib Figure
    """
    configure_style()

    baseline_label = _zh_label(baseline_label)
    naive_label = _zh_label(naive_label)
    
    base_vals = pd.to_numeric(merged[f"{metric}_base"], errors="coerce").dropna()
    naive_vals = pd.to_numeric(merged[f"{metric}_naive"], errors="coerce").dropna()
    
    df_long = pd.DataFrame({
        "Strategy": [baseline_label] * len(base_vals) + [naive_label] * len(naive_vals),
        "Value": pd.concat([base_vals, naive_vals], ignore_index=True),
    })
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    sns.violinplot(
        data=df_long,
        x="Strategy",
        y="Value",
        hue="Strategy",
        palette=[COLOR_BASELINE, COLOR_NAIVE],
        ax=ax,
        inner="box",
        cut=0,
        linewidth=1.5,
        legend=False,
        alpha=0.5,
    )
    
    # Add individual data points
    sns.stripplot(
        data=df_long,
        x="Strategy",
        y="Value",
        color="black",
        alpha=0.4,
        size=6,
        ax=ax,
        jitter=0.15,
    )
    
    ax.set_xlabel("")
    ax.set_ylabel(_clean_metric_name(metric), fontsize=FONT_SIZE_LABEL, weight="bold")
    ax.set_title(title or f"分布：{_clean_metric_name(metric)}", fontsize=FONT_SIZE_TITLE, weight="bold", pad=15)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    
    # Add mean annotations
    for i, (label, vals) in enumerate([(baseline_label, base_vals), (naive_label, naive_vals)]):
        mean_val = vals.mean()
        ax.annotate(
            f"μ={mean_val:.3f}",
            xy=(i, mean_val),
            xytext=(15, 0),
            textcoords="offset points",
            fontsize=FONT_SIZE_ANNOT,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
        )
    
    plt.tight_layout()
    return fig


def plot_box_comparison(
    merged: pd.DataFrame,
    metric: str,
    baseline_label: str = "基线",
    naive_label: str = "无图证据",
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot box + jittered points comparison for a single metric."""
    configure_style()

    baseline_label = _zh_label(baseline_label)
    naive_label = _zh_label(naive_label)

    base_vals = pd.to_numeric(merged[f"{metric}_base"], errors="coerce").dropna()
    naive_vals = pd.to_numeric(merged[f"{metric}_naive"], errors="coerce").dropna()

    df_long = pd.DataFrame(
        {
            "Strategy": [baseline_label] * len(base_vals) + [naive_label] * len(naive_vals),
            "Value": pd.concat([base_vals, naive_vals], ignore_index=True),
        }
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    sns.boxplot(
        data=df_long,
        x="Strategy",
        y="Value",
        hue="Strategy",
        palette=[COLOR_BASELINE, COLOR_NAIVE],
        ax=ax,
        width=0.55,
        linewidth=1.6,
        fliersize=0,
        legend=False,
    )
    
    # Set alpha=0.7 for box patches
    for patch in ax.patches:
        patch.set_alpha(0.7)

    # Jittered points on top (helps with small-N like 4 pairs)
    sns.stripplot(
        data=df_long,
        x="Strategy",
        y="Value",
        color="#1a1a1a",
        alpha=0.55,
        size=7,
        jitter=0.14,
        ax=ax,
        zorder=3,
    )

    # Mean markers
    means = [base_vals.mean(), naive_vals.mean()]
    ax.scatter([0, 1], means, s=130, c=[COLOR_NAIVE], edgecolors="white", linewidths=1.4, zorder=4)

    ax.set_xlabel("")
    ax.set_ylabel(_clean_metric_name(metric), fontsize=FONT_SIZE_LABEL, weight="bold")
    ax.set_title(title or f"箱线图：{_clean_metric_name(metric)}", fontsize=FONT_SIZE_TITLE, weight="bold", pad=15)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    for i, mean_val in enumerate(means):
        ax.annotate(
            f"μ={mean_val:.3f}",
            xy=(i, mean_val),
            xytext=(12, 0),
            textcoords="offset points",
            fontsize=FONT_SIZE_ANNOT,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    plt.tight_layout()
    return fig


def plot_scatter_comparison(
    merged: pd.DataFrame,
    metric: str,
    baseline_label: str = "基线",
    naive_label: str = "无图证据",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot scatter comparison (baseline vs naive) for a single metric.
    
    Args:
        merged: Merged DataFrame with _base and _naive columns
        metric: Metric name (without suffix)
        baseline_label: Label for baseline strategy
        naive_label: Label for naive strategy
        title: Optional custom title
    
    Returns:
        matplotlib Figure
    """
    configure_style()

    baseline_label = _zh_label(baseline_label)
    naive_label = _zh_label(naive_label)
    
    x = pd.to_numeric(merged[f"{metric}_base"], errors="coerce")
    y = pd.to_numeric(merged[f"{metric}_naive"], errors="coerce")
    
    # Color by improvement direction
    delta = y - x
    colors = [COLOR_POSITIVE if d > 0 else COLOR_NEGATIVE if d < 0 else COLOR_NEUTRAL for d in delta]
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    ax.scatter(x, y, c=colors, s=120, alpha=0.5, edgecolors="white", linewidths=1.5, zorder=3)
    
    # Unity line
    lims = [min(x.min(), y.min()) * 0.95, max(x.max(), y.max()) * 1.05]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1.5, zorder=1, label="一致线")
    ax.fill_between(lims, lims, [lims[1], lims[1]], alpha=0.05, color=COLOR_POSITIVE, zorder=0)
    ax.fill_between(lims, [lims[0], lims[0]], lims, alpha=0.05, color=COLOR_NEGATIVE, zorder=0)
    
    # Add pair labels
    for i, (xi, yi, pair) in enumerate(zip(x, y, merged["pair_key"])):
        if not (np.isnan(xi) or np.isnan(yi)):
            ax.annotate(
                pair,
                xy=(xi, yi),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=FONT_SIZE_ANNOT - 1,
                alpha=0.5,
            )
    
    ax.set_xlabel(f"{baseline_label}", fontsize=FONT_SIZE_LABEL, weight="bold")
    ax.set_ylabel(f"{naive_label}", fontsize=FONT_SIZE_LABEL, weight="bold")
    ax.set_title(title or f"成对对比：{_clean_metric_name(metric)}", fontsize=FONT_SIZE_TITLE, weight="bold", pad=15)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=GRID_ALPHA)
    
    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_POSITIVE, alpha=0.5, label=f"{naive_label}胜出"),
        Patch(facecolor=COLOR_NEGATIVE, alpha=0.5, label=f"{baseline_label}胜出"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, framealpha=0.95)
    
    plt.tight_layout()
    return fig


def plot_effect_bars(
    merged: pd.DataFrame,
    metric: str,
    title: Optional[str] = None,
    top_n: int = 10,
) -> plt.Figure:
    """
    Plot horizontal bar chart showing per-pair effect (delta).
    
    Args:
        merged: Merged DataFrame with _base and _naive columns
        metric: Metric name (without suffix)
        title: Optional custom title
        top_n: Number of top/bottom pairs to show
    
    Returns:
        matplotlib Figure
    """
    configure_style()
    
    x = pd.to_numeric(merged[f"{metric}_base"], errors="coerce")
    y = pd.to_numeric(merged[f"{metric}_naive"], errors="coerce")
    delta = y - x
    
    df = pd.DataFrame({
        "Pair": merged["pair_key"],
        "Delta": delta,
    }).dropna().sort_values("Delta")
    
    # Select top/bottom pairs
    if len(df) > top_n:
        n_each = top_n // 2
        df_plot = pd.concat([df.head(n_each), df.tail(n_each)]).sort_values("Delta")
    else:
        df_plot = df.sort_values("Delta")
    
    colors = [COLOR_POSITIVE if d > 0 else COLOR_NEGATIVE for d in df_plot["Delta"]]
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    bars = ax.barh(range(len(df_plot)), df_plot["Delta"], color=colors, alpha=0.5, edgecolor="white", linewidth=1.2)
    
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot["Pair"], fontsize=FONT_SIZE_TICK)
    ax.axvline(0, color="#333333", linewidth=1.2, linestyle="-", alpha=0.7)
    ax.set_xlabel("效应（无图证据 − 基线）", fontsize=FONT_SIZE_LABEL, weight="bold")
    ax.set_title(title or f"逐对效应：{_clean_metric_name(metric)}", fontsize=FONT_SIZE_TITLE, weight="bold", pad=15)
    ax.grid(axis="x", alpha=GRID_ALPHA)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_plot["Delta"])):
        x_pos = val + 0.005 if val >= 0 else val - 0.005
        ha = "left" if val >= 0 else "right"
        ax.annotate(
            f"{val:+.3f}",
            xy=(x_pos, i),
            va="center",
            ha=ha,
            fontsize=FONT_SIZE_ANNOT,
            alpha=0.8,
        )
    
    plt.tight_layout()
    return fig


def plot_radar_comparison(
    merged: pd.DataFrame,
    metrics: list[str],
    baseline_label: str = "基线",
    naive_label: str = "无图证据",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot radar/spider chart comparing multiple metrics.
    
    Args:
        merged: Merged DataFrame with _base and _naive columns
        metrics: List of metric names to include
        baseline_label: Label for baseline strategy
        naive_label: Label for naive strategy
        title: Optional custom title
    
    Returns:
        matplotlib Figure
    """
    configure_style()

    baseline_label = _zh_label(baseline_label)
    naive_label = _zh_label(naive_label)
    
    # Calculate means for each metric
    base_means = []
    naive_means = []
    labels = []
    
    for m in metrics:
        base_col = f"{m}_base"
        naive_col = f"{m}_naive"
        
        if base_col not in merged.columns or naive_col not in merged.columns:
            continue
        
        base_val = pd.to_numeric(merged[base_col], errors="coerce").mean()
        naive_val = pd.to_numeric(merged[naive_col], errors="coerce").mean()
        
        if not (np.isnan(base_val) or np.isnan(naive_val)):
            base_means.append(base_val)
            naive_means.append(naive_val)
            labels.append(_clean_metric_name(m)[:12])
    
    if len(labels) < 3:
        raise ValueError("Need at least 3 metrics for radar chart")
    
    # Close the polygon
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    base_means += base_means[:1]
    naive_means += naive_means[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, base_means, "o-", linewidth=2.5, color=COLOR_BASELINE, label=baseline_label, markersize=8)
    ax.fill(angles, base_means, alpha=0.15, color=COLOR_BASELINE)
    ax.plot(angles, naive_means, "o-", linewidth=2.5, color=COLOR_NAIVE, label=naive_label, markersize=8)
    ax.fill(angles, naive_means, alpha=0.15, color=COLOR_NAIVE)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICK)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=FONT_SIZE_TICK - 1, alpha=0.7)
    ax.grid(color="#CCCCCC", linewidth=0.8, alpha=GRID_ALPHA)
    
    ax.set_title(title or "多指标对比", fontsize=FONT_SIZE_TITLE, weight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1), frameon=True, framealpha=0.95)
    
    plt.tight_layout()
    return fig


def plot_kde_distribution(
    merged: pd.DataFrame,
    metric: str,
    baseline_label: str = "基线",
    naive_label: str = "无图证据",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot KDE distribution comparison for a single metric.
    
    Args:
        merged: Merged DataFrame with _base and _naive columns
        metric: Metric name (without suffix)
        baseline_label: Label for baseline strategy
        naive_label: Label for naive strategy
        title: Optional custom title
    
    Returns:
        matplotlib Figure
    """
    configure_style()

    baseline_label = _zh_label(baseline_label)
    naive_label = _zh_label(naive_label)
    
    base_vals = pd.to_numeric(merged[f"{metric}_base"], errors="coerce").dropna()
    naive_vals = pd.to_numeric(merged[f"{metric}_naive"], errors="coerce").dropna()
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # KDE plots
    if len(base_vals) > 1:
        sns.kdeplot(data=base_vals, ax=ax, color=COLOR_BASELINE, linewidth=2.5, label=baseline_label, fill=True, alpha=0.25)
    if len(naive_vals) > 1:
        sns.kdeplot(data=naive_vals, ax=ax, color=COLOR_NAIVE, linewidth=2.5, label=naive_label, fill=True, alpha=0.25)
    
    # Add rug plot
    sns.rugplot(data=base_vals, ax=ax, color=COLOR_BASELINE, height=0.05, alpha=0.5)
    sns.rugplot(data=naive_vals, ax=ax, color=COLOR_NAIVE, height=0.05, alpha=0.5)
    
    # Add mean lines
    ax.axvline(base_vals.mean(), color=COLOR_BASELINE, linestyle="--", linewidth=2, alpha=0.8)
    ax.axvline(naive_vals.mean(), color=COLOR_NAIVE, linestyle="--", linewidth=2, alpha=0.8)
    
    metric_label = _clean_metric_name(metric)
    ax.set_xlabel(metric_label, fontsize=FONT_SIZE_LABEL, weight="bold")
    ax.set_ylabel("密度", fontsize=FONT_SIZE_LABEL, weight="bold")
    ax.set_title(title or f"分布：{metric_label}", fontsize=FONT_SIZE_TITLE, weight="bold", pad=15)
    ax.legend(frameon=True, framealpha=0.95)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    merged: pd.DataFrame,
    metrics: list[str],
    strategy: str = "base",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot correlation heatmap for metrics.
    
    Args:
        merged: Merged DataFrame with _base and _naive columns
        metrics: List of metric names to include
        strategy: "base" or "naive"
        title: Optional custom title
    
    Returns:
        matplotlib Figure
    """
    configure_style()
    
    suffix = f"_{strategy}"
    metric_cols = [f"{m}{suffix}" for m in metrics if f"{m}{suffix}" in merged.columns]
    
    if len(metric_cols) < 2:
        raise ValueError("Need at least 2 metrics for correlation heatmap")
    
    corr_matrix = merged[metric_cols].apply(pd.to_numeric, errors="coerce").corr()
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # Custom colormap (diverging)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="plasma",
        center=0,
        square=True,
        linewidths=1.5,
        cbar_kws={"shrink": 0.8, "label": "相关系数"},
        ax=ax,
        vmin=-1,
        vmax=1,
        annot_kws={"fontsize": FONT_SIZE_ANNOT},
    )
    
    labels = [_clean_metric_name(m.replace(suffix, ""))[:10] for m in metric_cols]
    ax.set_xticklabels(labels, ha="right", fontsize=FONT_SIZE_TICK)
    ax.set_yticklabels(labels, rotation=0, fontsize=FONT_SIZE_TICK)
    
    strategy_label = "基线" if strategy == "base" else "无图证据"
    ax.set_title(title or f"指标相关性（{strategy_label}）", fontsize=FONT_SIZE_TITLE, weight="bold", pad=15)
    
    plt.tight_layout()
    return fig


def plot_win_loss_bars(
    merged: pd.DataFrame,
    metrics: list[str],
    baseline_label: str = "基线",
    naive_label: str = "无图证据",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot win/loss/tie stacked bar chart.
    
    Args:
        merged: Merged DataFrame with _base and _naive columns
        metrics: List of metric names to analyze
        baseline_label: Label for baseline strategy
        naive_label: Label for naive strategy
        title: Optional custom title
    
    Returns:
        matplotlib Figure
    """
    configure_style()

    baseline_label = _zh_label(baseline_label)
    naive_label = _zh_label(naive_label)
    
    win_loss_data = []
    for m in metrics:
        base_col = f"{m}_base"
        naive_col = f"{m}_naive"
        
        if base_col not in merged.columns or naive_col not in merged.columns:
            continue
        
        base_vals = pd.to_numeric(merged[base_col], errors="coerce")
        naive_vals = pd.to_numeric(merged[naive_col], errors="coerce")
        
        if _metric_higher_better(m):
            wins = (base_vals > naive_vals).sum()
            losses = (naive_vals > base_vals).sum()
        else:
            wins = (base_vals < naive_vals).sum()
            losses = (naive_vals < base_vals).sum()
        
        ties = (base_vals == naive_vals).sum()
        total = wins + losses + ties
        
        if total > 0:
            win_loss_data.append({
                "Metric": _clean_metric_name(m)[:12],
                "wins": wins / total * 100,
                "losses": losses / total * 100,
                "ties": ties / total * 100,
            })
    
    df_wlt = pd.DataFrame(win_loss_data)
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    x = np.arange(len(df_wlt))
    
    ax.barh(x, df_wlt["wins"], color=COLOR_BASELINE, alpha=0.5, edgecolor="white", linewidth=1.2, label=f"{baseline_label}胜出")
    ax.barh(x, df_wlt["losses"], left=df_wlt["wins"], color=COLOR_NAIVE, alpha=0.5, edgecolor="white", linewidth=1.2, label=f"{naive_label}胜出")
    ax.barh(x, df_wlt["ties"], left=df_wlt["wins"] + df_wlt["losses"], color=COLOR_NEUTRAL, alpha=0.5, edgecolor="white", linewidth=1.2, label="平局")
    
    ax.set_yticks(x)
    ax.set_yticklabels(df_wlt["Metric"], fontsize=FONT_SIZE_TICK)
    ax.set_xlabel("比例（%）", fontsize=FONT_SIZE_LABEL, weight="bold")
    ax.set_title(title or "胜负平分析", fontsize=FONT_SIZE_TITLE, weight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=GRID_ALPHA)
    
    plt.tight_layout()
    return fig


def plot_cohens_d(
    merged: pd.DataFrame,
    metrics: list[str],
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Cohen's d effect sizes for each metric.
    
    Args:
        merged: Merged DataFrame with _base and _naive columns
        metrics: List of metric names to analyze
        title: Optional custom title
    
    Returns:
        matplotlib Figure
    """
    configure_style()
    
    effect_sizes = []
    for m in metrics:
        base_col = f"{m}_base"
        naive_col = f"{m}_naive"
        
        if base_col not in merged.columns or naive_col not in merged.columns:
            continue
        
        base_vals = pd.to_numeric(merged[base_col], errors="coerce").dropna()
        naive_vals = pd.to_numeric(merged[naive_col], errors="coerce").dropna()
        
        if len(base_vals) < 2 or len(naive_vals) < 2:
            continue
        
        pooled_std = np.sqrt(
            ((len(base_vals) - 1) * base_vals.var() + (len(naive_vals) - 1) * naive_vals.var()) 
            / (len(base_vals) + len(naive_vals) - 2)
        )
        
        if pooled_std > 0:
            cohens_d = (naive_vals.mean() - base_vals.mean()) / pooled_std
            effect_sizes.append((_clean_metric_name(m)[:12], cohens_d))
    
    df_effect = pd.DataFrame(effect_sizes, columns=["指标", "科恩d"]).sort_values("科恩d")
    
    colors = [COLOR_POSITIVE if d > 0 else COLOR_NEGATIVE for d in df_effect["科恩d"]]
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    bars = ax.barh(df_effect["指标"], df_effect["科恩d"], color=colors, alpha=0.5, edgecolor="white", linewidth=1.2)
    
    ax.axvline(0, color="#333333", linewidth=1.5, linestyle="-")
    # Slightly stronger background reference lines for readability
    ax.axvline(-0.2, color="#888888", linewidth=1, linestyle=":", alpha=0.95)
    ax.axvline(0.2, color="#888888", linewidth=1, linestyle=":", alpha=0.95)
    ax.axvline(-0.8, color="#888888", linewidth=1, linestyle=":", alpha=0.95)
    ax.axvline(0.8, color="#888888", linewidth=1, linestyle=":", alpha=0.95)
    
    ax.set_xlabel("科恩d（效应量）", fontsize=FONT_SIZE_LABEL, weight="bold")
    ax.set_title(title or "效应量分析", fontsize=FONT_SIZE_TITLE, weight="bold", pad=15)
    ax.grid(axis="x", alpha=GRID_ALPHA)
    
    # Add effect size labels
    ax.text(0.1, -0.08, "小", transform=ax.transAxes, fontsize=FONT_SIZE_ANNOT, alpha=0.6, ha="center")
    ax.text(0.5, -0.08, "中", transform=ax.transAxes, fontsize=FONT_SIZE_ANNOT, alpha=0.6, ha="center")
    ax.text(0.9, -0.08, "大", transform=ax.transAxes, fontsize=FONT_SIZE_ANNOT, alpha=0.6, ha="center")
    
    # Add value labels on bars
    for bar, val in zip(bars, df_effect["科恩d"]):
        x_pos = val + 0.05 if val >= 0 else val - 0.05
        ha = "left" if val >= 0 else "right"
        ax.annotate(f"{val:.2f}", xy=(x_pos, bar.get_y() + bar.get_height()/2), va="center", ha=ha, fontsize=FONT_SIZE_ANNOT)
    # Legend showing effect direction
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_POSITIVE, label="无图证据 > 基线"),
        Patch(facecolor=COLOR_NEGATIVE, label="基线 > 无图证据"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, framealpha=0.95)
    plt.tight_layout()
    return fig


def plot_ranking_bars(
    merged: pd.DataFrame,
    metrics: list[str],
    baseline_label: str = "基线",
    naive_label: str = "无图证据",
    top_n: int = 10,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot grouped bar chart of top performing pairs.
    
    Args:
        merged: Merged DataFrame with _base and _naive columns
        metrics: List of metric names to average
        baseline_label: Label for baseline strategy
        naive_label: Label for naive strategy
        top_n: Number of top pairs to show
        title: Optional custom title
    
    Returns:
        matplotlib Figure
    """
    configure_style()
    
    ranking = []
    for _, row in merged.iterrows():
        base_avg = np.nanmean([pd.to_numeric(row.get(f"{m}_base", np.nan), errors="coerce") for m in metrics])
        naive_avg = np.nanmean([pd.to_numeric(row.get(f"{m}_naive", np.nan), errors="coerce") for m in metrics])
        ranking.append((row["pair_key"], base_avg, naive_avg))
    
    ranking.sort(key=lambda x: max(x[1], x[2]), reverse=True)
    top_ranking = ranking[:min(top_n, len(ranking))]
    
    pairs = [r[0] for r in top_ranking]
    base_scores = [r[1] for r in top_ranking]
    naive_scores = [r[2] for r in top_ranking]
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    x = np.arange(len(pairs))
    width = 0.38
    
    ax.barh(x + width/2, base_scores, width, label=baseline_label, color=COLOR_BASELINE, alpha=0.5, edgecolor="white", linewidth=1.2)
    ax.barh(x - width/2, naive_scores, width, label=naive_label, color=COLOR_NAIVE, alpha=0.5, edgecolor="white", linewidth=1.2)
    
    ax.set_yticks(x)
    ax.set_yticklabels(pairs, fontsize=FONT_SIZE_TICK)
    ax.set_xlabel("平均得分", fontsize=FONT_SIZE_LABEL, weight="bold")
    ax.set_title(title or "表现最佳的基因对", fontsize=FONT_SIZE_TITLE, weight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.grid(axis="x", alpha=GRID_ALPHA)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MULTI-STRATEGY (ABLATION) UTILITIES
# ============================================================================

DEFAULT_STRATEGY_ORDER = ["baseline", "self_refine", "cove", "naive"]
DEFAULT_STRATEGY_COLORS = {
    "baseline": "#31ABF3",
    "self_refine": "#8734FC",
    "cove": "#FC9B2D",
    "naive": "#F4371A",
}


# ============================================================================
# MULTI-STRATEGY DISTRIBUTION PLOTS (KDE, Violin all in one)
# ============================================================================


def plot_kde_all_strategies(
    df: pd.DataFrame,
    metric: str,
    *,
    strategies: Optional[list[str]] = None,
    colors: Optional[dict[str, str]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot KDE distribution for all strategies in a single figure.

    Args:
        df: Long-form DataFrame with 'strategy' column.
        metric: Metric name.
        strategies: List of strategies to include (or auto-detect).
        colors: Mapping from strategy to color.
        title: Optional custom title.

    Returns:
        matplotlib Figure.
    """
    configure_style()

    strategies = strategies or get_strategies_present(df)
    colors = colors or DEFAULT_STRATEGY_COLORS

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    for strat in strategies:
        vals = pd.to_numeric(df.loc[df["strategy"] == strat, metric], errors="coerce").dropna()
        if len(vals) < 2:
            continue
        col = colors.get(strat, TOLS_BRIGHT["grey"])
        sns.kdeplot(data=vals, ax=ax, color=col, linewidth=2.5, label=_zh_strategy(strat), fill=True, alpha=0.2)
        # Mean line
        ax.axvline(vals.mean(), color=col, linestyle="--", linewidth=1.8, alpha=0.75)

    ax.set_xlabel(_clean_metric_name(metric), fontsize=FONT_SIZE_LABEL, weight="bold")
    ax.set_ylabel("密度", fontsize=FONT_SIZE_LABEL, weight="bold")
    ax.set_title(title or f"核密度分布：{_clean_metric_name(metric)}", fontsize=FONT_SIZE_TITLE, weight="bold", pad=15)
    ax.legend(frameon=True, framealpha=0.95)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    plt.tight_layout()
    return fig


def plot_violin_all_strategies(
    df: pd.DataFrame,
    metric: str,
    *,
    strategies: Optional[list[str]] = None,
    colors: Optional[dict[str, str]] = None,
    show_legend: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot violin distribution for all strategies in a single figure.

    Args:
        df: Long-form DataFrame with 'strategy' column.
        metric: Metric name.
        strategies: List of strategies to include (or auto-detect).
        colors: Mapping from strategy to color.
        title: Optional custom title.

    Returns:
        matplotlib Figure.
    """
    configure_style()

    strategies = strategies or get_strategies_present(df)
    colors = colors or DEFAULT_STRATEGY_COLORS

    plot_df = df[df["strategy"].isin(strategies)].copy()
    plot_df["策略"] = plot_df["strategy"].map(_zh_strategy)
    plot_df["Value"] = pd.to_numeric(plot_df.get(metric), errors="coerce")
    plot_df = plot_df.dropna(subset=["Value"])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    order_zh = [_zh_strategy(s) for s in strategies]
    palette_map = {_zh_strategy(s): colors.get(s, TOLS_BRIGHT["grey"]) for s in strategies}

    sns.violinplot(
        data=plot_df,
        x="策略",
        y="Value",
        order=order_zh,
        hue="策略",
        palette=palette_map,
        dodge=False,
        ax=ax,
        inner="box",
        cut=0,
        linewidth=1.5,
        alpha=0.55,
    )

    # Legend (explicitly shows 'naive' color)
    # Note: seaborn may or may not auto-create a legend when hue == x.
    # We always create a manual legend so the strategy-to-color mapping is visible.
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    if show_legend:
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=palette_map[_zh_strategy(s)], edgecolor="white", alpha=0.7, label=_zh_strategy(s)) for s in strategies]
        ax.legend(
            handles=handles,
            title="",
            loc="upper right",
            frameon=True,
            framealpha=0.95,
        )

    # Stripplot overlay
    sns.stripplot(
        data=plot_df,
        x="策略",
        y="Value",
        order=order_zh,
        color="#1a1a1a",
        alpha=0.45,
        size=5,
        jitter=0.15,
        ax=ax,
        zorder=3,
    )

    # Add mean annotations
    for i, strat in enumerate(strategies):
        vals = pd.to_numeric(df.loc[df["strategy"] == strat, metric], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        mean_val = vals.mean()
        ax.annotate(
            f"μ={mean_val:.3f}",
            xy=(i, mean_val),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=FONT_SIZE_ANNOT,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    ax.set_xlabel("")
    ax.set_ylabel(_clean_metric_name(metric), fontsize=FONT_SIZE_LABEL, weight="bold")
    ax.set_title(title or f"小提琴图：{_clean_metric_name(metric)}", fontsize=FONT_SIZE_TITLE, weight="bold", pad=15)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    # Tick label font styling (labels are already Chinese via the "策略" column).
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    plt.tight_layout()
    return fig


def plot_scatter_grid_vs_naive(
    df: pd.DataFrame,
    metrics: list[str],
    *,
    strategies: Optional[list[str]] = None,
    naive_strategy: str = "naive",
    compare_order: tuple[str, ...] = ("baseline", "self_refine", "cove"),
    colors: Optional[dict[str, str]] = None,
    title: Optional[str] = None,
    label_alpha: float = 0.8,
) -> plt.Figure:
    """3×N grid: 3 strategies (rows) × N metrics (columns) scatter plots vs naive.

    - Points are colored by winner using strategy palette colors.
    - Background regions colored by winner (above/below diagonal).
    - "Isolated" points are defined as outliers far from the main cluster using the
      1.5×IQR rule on residuals from the diagonal. These points are annotated
      with their gene-pair names.
    """
    configure_style()

    colors = colors or DEFAULT_STRATEGY_COLORS
    strategies = strategies or get_strategies_present(df)

    compare_strats = [s for s in compare_order if s in strategies and s != naive_strategy]
    for s in strategies:
        if s != naive_strategy and s not in compare_strats:
            compare_strats.append(s)
    compare_strats = compare_strats[:3]
    if not compare_strats:
        raise ValueError("No non-naive strategies found")

    n_rows = len(compare_strats)
    n_cols = len(metrics)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5.5 * n_cols, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    tie_color = TOLS_BRIGHT["green"]

    for row_idx, strat in enumerate(compare_strats):
        # row-level colors
        naive_color = colors.get(naive_strategy, COLOR_NAIVE)
        strat_color = colors.get(strat, TOLS_BRIGHT["grey"])

        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            
            wide = df.pivot_table(index="pair_key", columns="strategy", values=metric, aggfunc="first")
            if naive_strategy not in wide.columns or strat not in wide.columns:
                ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel(_zh_strategy(strat), fontsize=FONT_SIZE_LABEL - 1, weight="bold")
                ax.set_ylabel(_zh_strategy(naive_strategy) if col_idx == 0 else "", fontsize=FONT_SIZE_LABEL - 1, weight="bold")
                continue

            x = pd.to_numeric(wide.get(strat), errors="coerce")
            y = pd.to_numeric(wide.get(naive_strategy), errors="coerce")

            both = x.notna() & y.notna()
            x_only = x.notna() & y.isna()
            y_only = x.isna() & y.notna()


            xv = x[both]
            yv = y[both]
            delta = yv - xv

            all_vals = pd.concat([xv, yv]).dropna()
            if len(all_vals) > 0:
                vmin = float(all_vals.min())
                vmax = float(all_vals.max())
                pad = 0.07 * (vmax - vmin + 1e-9)
                lims = [vmin - pad, vmax + pad]

                # Background regions: faint shading to indicate winner regions
                ax.fill_between(lims, lims, [lims[1]] * 2, color=naive_color, alpha=0.1, zorder=0)
                ax.fill_between(lims, [lims[0]] * 2, lims, color=strat_color, alpha=0.1, zorder=0)

                ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1.3, zorder=1)
                ax.set_xlim(lims)
                ax.set_ylim(lims)

            point_colors = [naive_color if d > 0 else strat_color if d < 0 else tie_color for d in delta]
            ax.scatter(xv, yv, c=point_colors, s=75, alpha=0.62, edgecolors="white", linewidths=0.9, zorder=3)

            # outliers far from the diagonal (1.5×IQR on residuals)
            outlier_idx: list[str] = []
            if len(delta) >= 4 and float(delta.max() - delta.min()) > 0:
                q1 = float(delta.quantile(0.25))
                q3 = float(delta.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    low = q1 - 1.5 * iqr
                    high = q3 + 1.5 * iqr
                    outlier_idx = delta[(delta < low) | (delta > high)].index.astype(str).tolist()

            if len(all_vals) > 0 and outlier_idx:
                for pk in outlier_idx:
                    if pk not in xv.index or pk not in yv.index:
                        continue
                    ax.annotate(
                        str(pk),
                        (float(xv.loc[pk]), float(yv.loc[pk])),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=FONT_SIZE_ANNOT - 2,
                        alpha=label_alpha,
                    )

            # missing points on margins
            if len(all_vals) > 0:
                iso_pad = 0.06 * (lims[1] - lims[0] + 1e-9)
                if x_only.any():
                    x_iso = x[x_only]
                    y_iso = np.full(len(x_iso), lims[0] - iso_pad)
                    ax.scatter(x_iso, y_iso, c=strat_color, s=60, alpha=0.8, marker="^", edgecolors="white", linewidths=0.7, zorder=4)

                if y_only.any():
                    y_iso = y[y_only]
                    x_iso = np.full(len(y_iso), lims[0] - iso_pad)
                    ax.scatter(x_iso, y_iso, c=naive_color, s=60, alpha=0.8, marker="<", edgecolors="white", linewidths=0.7, zorder=4)

            # Only show the `naive vs {strategy}` y-label on the first column; clear others.
            if col_idx == 0:
                ax.set_ylabel(f"{_zh_strategy(naive_strategy)}对比{_zh_strategy(strat)}", fontsize=FONT_SIZE_LABEL - 1, weight="bold")
            else:
                ax.set_ylabel("", fontsize=FONT_SIZE_LABEL - 1, weight="bold")

            # Metric title on the top row for each column
            if row_idx == 0:
                ax.set_title(_clean_metric_name(metric), fontsize=FONT_SIZE_TITLE - 2, weight="bold", pad=8)
            
            ax.grid(True, which="both", alpha=GRID_ALPHA, linestyle="--", linewidth=0.6)

            # end per-axis

        # add a legend for this row on the rightmost axis
        try:
            from matplotlib.patches import Patch
            right_ax = axes[row_idx][n_cols - 1]
            right_ax.legend(
                handles=[
                    Patch(facecolor=strat_color, alpha=0.7, label=f"{_zh_strategy(strat)}胜出"),
                    Patch(facecolor=naive_color, alpha=0.7, label=f"{_zh_strategy(naive_strategy)}胜出"),
                    Patch(facecolor=tie_color, alpha=0.7, label="平局"),
                ],
                loc="lower right",
                frameon=True,
                framealpha=0.95,
                fontsize=FONT_SIZE_LABEL - 2,
            )
        except Exception:
            pass

    if title:
        fig.suptitle(title, fontsize=FONT_SIZE_TITLE, weight="bold", y=0.995)
    plt.tight_layout()
    return fig


def load_eval_longform(
    normal_csv: Path,
    naive_csv: Optional[Path] = None,
    *,
    normal_default_strategy: str = "baseline",
    naive_default_strategy: str = "naive",
) -> pd.DataFrame:
    """Load evaluation CSV(s) into a single long-form DataFrame.

    - `normal_csv` may contain multiple strategies (baseline/self_refine/cove).
    - `naive_csv` (if provided) is appended as an additional strategy.
    """

    def _normalize_pair_key(row: pd.Series) -> str:
        a = str(row.get("gene_a", "")).strip().upper()
        b = str(row.get("gene_b", "")).strip().upper()
        return "/".join(sorted([a, b]))

    normal_df = pd.read_csv(normal_csv)
    if "strategy" not in normal_df.columns:
        normal_df = normal_df.copy()
        normal_df["strategy"] = normal_default_strategy

    dfs = [normal_df]

    if naive_csv is not None:
        naive_df = pd.read_csv(naive_csv)
        # IMPORTANT:
        # Naive CSVs in this repo historically used `strategy=baseline`.
        # When we concatenate, that silently merges naive rows into the normal
        # baseline bucket and the notebook “loses” the naive method.
        #
        # Always force the naive CSV to be labeled as `naive`.
        naive_df = naive_df.copy()
        naive_df["strategy"] = naive_default_strategy
        dfs.append(naive_df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.copy()
    df["pair_key"] = df.apply(_normalize_pair_key, axis=1)
    df["strategy"] = df["strategy"].astype(str)
    return df


def get_strategies_present(
    df: pd.DataFrame,
    *,
    preferred_order: Optional[list[str]] = None,
) -> list[str]:
    """Return strategies present in the DataFrame, in a stable preferred order."""
    preferred = preferred_order or DEFAULT_STRATEGY_ORDER
    present = {str(s) for s in df.get("strategy", pd.Series([], dtype=str)).dropna().unique().tolist()}
    ordered = [s for s in preferred if s in present]
    ordered += sorted(present - set(ordered))
    return ordered


def build_strategy_summary_table(
    df: pd.DataFrame,
    metrics: list[str],
    *,
    strategies: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute mean score per metric per strategy (wide table)."""
    strategies = strategies or get_strategies_present(df)
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        row: dict[str, object] = {"Metric": _clean_metric_name(metric)}
        for s in strategies:
            vals = pd.to_numeric(df.loc[df["strategy"] == s, metric], errors="coerce")
            row[s] = float(vals.mean()) if vals.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_box_by_strategy(
    df: pd.DataFrame,
    metric: str,
    *,
    strategies: Optional[list[str]] = None,
    title: Optional[str] = None,
    colors: Optional[dict[str, str]] = None,
) -> plt.Figure:
    """Box + jitter plot for a metric across all strategies present."""
    configure_style()

    strategies = strategies or get_strategies_present(df)
    colors = colors or DEFAULT_STRATEGY_COLORS

    plot_df = df[df["strategy"].isin(strategies)].copy()
    plot_df["策略"] = plot_df["strategy"].map(_zh_strategy)
    plot_df["Value"] = pd.to_numeric(plot_df.get(metric), errors="coerce")
    plot_df = plot_df.dropna(subset=["Value"])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    order_zh = [_zh_strategy(s) for s in strategies]
    palette_map = {_zh_strategy(s): colors.get(s, TOLS_BRIGHT["grey"]) for s in strategies}

    # Create boxes with alpha=0.7 transparency
    box_patches = sns.boxplot(
        data=plot_df,
        x="策略",
        y="Value",
        order=order_zh,
        hue="策略",
        palette=palette_map,
        dodge=False,
        ax=ax,
        width=0.6,
        linewidth=1.6,
        fliersize=0,
    )
    
    # Set alpha for all box patches
    for patch in ax.patches:
        patch.set_alpha(0.7)

    # hue=x triggers a redundant legend; keep plots clean.
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    sns.stripplot(
        data=plot_df,
        x="策略",
        y="Value",
        order=order_zh,
        color=COLOR_POSITIVE,
        alpha=0.3,
        size=6,
        jitter=0.16,
        ax=ax,
        zorder=3,
    )

    ax.set_xlabel("")
    ax.set_ylabel(_clean_metric_name(metric), fontsize=FONT_SIZE_LABEL, weight="bold")
    ax.set_title(title or f"{_clean_metric_name(metric)}（按策略）", fontsize=FONT_SIZE_TITLE, weight="bold", pad=15)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    # Tick label font styling (labels are already Chinese via the "策略" column).
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    plt.tight_layout()
    return fig


def plot_radar_strategies(
    df: pd.DataFrame,
    metrics: list[str],
    *,
    strategies: Optional[list[str]] = None,
    colors: Optional[dict[str, str]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Radar/spider chart: mean metric values for multiple strategies."""
    configure_style()

    strategies = strategies or get_strategies_present(df)
    colors = colors or DEFAULT_STRATEGY_COLORS

    labels: list[str] = []
    metric_means_by_strategy: dict[str, list[float]] = {s: [] for s in strategies}

    for metric in metrics:
        if metric not in df.columns:
            continue
        vals_any = pd.to_numeric(df[metric], errors="coerce")
        if not vals_any.notna().any():
            continue
        labels.append(_clean_metric_name(metric)[:12])
        for s in strategies:
            vals = pd.to_numeric(df.loc[df["strategy"] == s, metric], errors="coerce")
            metric_means_by_strategy[s].append(float(vals.mean()) if vals.notna().any() else np.nan)

    if len(labels) < 3:
        raise ValueError("Need at least 3 metrics with data for radar chart")

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), subplot_kw=dict(projection="polar"))

    for s in strategies:
        values = metric_means_by_strategy[s]
        if len(values) != len(labels):
            continue
        # Close polygon
        values = values + values[:1]
        ax.plot(angles, values, "o-", linewidth=2.3, color=colors.get(s, TOLS_BRIGHT["grey"]), label=_zh_strategy(s), markersize=7)
        ax.fill(angles, values, alpha=0.12, color=colors.get(s, TOLS_BRIGHT["grey"]))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICK)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=FONT_SIZE_TICK - 1, alpha=0.7)
    ax.grid(color="#CCCCCC", linewidth=0.8, alpha=GRID_ALPHA)
    ax.set_title(title or "多策略雷达图（均值）", fontsize=FONT_SIZE_TITLE, weight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.1), frameon=True, framealpha=0.95)

    plt.tight_layout()
    return fig


def plot_ablation_mean_improvement(
    df: pd.DataFrame,
    metrics: list[str],
    *,
    baseline: str = "baseline",
    strategies: Optional[list[str]] = None,
    title: Optional[str] = None,
    colors: Optional[dict[str, str]] = None,
) -> plt.Figure:
    """Ablation bar chart: mean improvement vs baseline for each metric.

    Improvement is signed so that positive means "better" for all metrics.
    """
    configure_style()

    strategies = strategies or [s for s in get_strategies_present(df) if s != baseline]
    colors = colors or DEFAULT_STRATEGY_COLORS
    rows: list[dict[str, object]] = []

    for metric in metrics:
        if metric not in df.columns:
            continue
        wide = df.pivot_table(index="pair_key", columns="strategy", values=metric, aggfunc="first")
        if baseline not in wide.columns:
            continue
        base_vals = pd.to_numeric(wide[baseline], errors="coerce")
        for s in strategies:
            if s not in wide.columns:
                continue
            other_vals = pd.to_numeric(wide[s], errors="coerce")
            common = pd.concat([base_vals, other_vals], axis=1).dropna()
            if common.empty:
                continue
            base = common.iloc[:, 0]
            other = common.iloc[:, 1]
            if _metric_higher_better(metric):
                imp = other - base
            else:
                imp = base - other

            rows.append({"Metric": _clean_metric_name(metric), "Improvement": float(imp.mean()), "Strategy": s})

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    if not rows:
        ax.text(0.5, 0.5, "无可绘制数据", ha="center", va="center")
        ax.set_axis_off()
        return fig

    df_rows = pd.DataFrame(rows)
    import seaborn as sns

    # Use Chinese strategy labels for the legend and palette keys.
    df_rows = df_rows.copy()
    df_rows["策略"] = df_rows["Strategy"].map(_zh_strategy)
    palette = {_zh_strategy(s): colors.get(s, TOLS_BRIGHT["grey"]) for s in strategies}

    sns.barplot(
        data=df_rows,
        x="Metric",
        y="Improvement",
        hue="策略",
        palette=palette,
        ax=ax,
        edgecolor="white",
        linewidth=1.0,
    )

    ax.axhline(0, color="#333333", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("")
    ax.set_ylabel("相对基线的平均提升", fontsize=FONT_SIZE_LABEL, weight="bold")
    ax.set_title(title or "消融：相对基线的平均提升", fontsize=FONT_SIZE_TITLE, weight="bold", pad=15)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(title="", frameon=True, framealpha=0.95)
    ax.tick_params(axis="x")

    plt.tight_layout()
    return fig


def plot_ablation_winrate_heatmap(
    df: pd.DataFrame,
    metrics: list[str],
    *,
    baseline: str = "baseline",
    strategies: Optional[list[str]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Heatmap: win-rate of each strategy vs baseline per metric."""
    configure_style()
    strategies = strategies or [s for s in get_strategies_present(df) if s != baseline]

    winrate: dict[str, dict[str, float]] = {}
    for s in strategies:
        winrate[s] = {}
        for metric in metrics:
            if metric not in df.columns:
                continue
            wide = df.pivot_table(index="pair_key", columns="strategy", values=metric, aggfunc="first")
            if baseline not in wide.columns or s not in wide.columns:
                continue
            common = wide[[baseline, s]].apply(pd.to_numeric, errors="coerce").dropna()
            if common.empty:
                continue
            base = common[baseline]
            other = common[s]
            if _metric_higher_better(metric):
                wins = (other > base).mean()
            else:
                wins = (other < base).mean()
            winrate[s][_clean_metric_name(metric)[:14]] = float(wins)

    hm = pd.DataFrame(winrate).T
    if hm.empty:
        raise ValueError("No win-rates computed (check strategies/metrics)")

    # Display Chinese strategy names on the y-axis while keeping values intact.
    hm = hm.copy()
    hm.index = [_zh_strategy(s) for s in hm.index]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    sns.heatmap(
        hm,
        annot=True,
        fmt=".2f",
        cmap="plasma",
        vmin=0,
        vmax=1,
        linewidths=1.0,
        linecolor="white",
        cbar_kws={"label": "相对基线胜率"},
        ax=ax,
        annot_kws={"fontsize": FONT_SIZE_ANNOT},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title or "消融：相对基线胜率", fontsize=FONT_SIZE_TITLE, weight="bold", pad=15)

    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    plt.tight_layout()
    return fig

