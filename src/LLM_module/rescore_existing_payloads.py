#!/usr/bin/env python3
"""
Rescore existing evaluation payloads with updated metrics.

This script:
1. Finds all JSON payloads in /data/guoyu/KG-LLM-XSL/output/eval_results/<gene_pair>/
2. Re-applies the scoring pipeline with redesigned metrics
3. Modifies the JSON files IN PLACE (same payload format)
4. Updates the corresponding CSV files (eval_naive_*_<model>.csv, eval_normal_*_<model>.csv)

Key changes in updated metrics:
- Format score: Embedding similarity to GT explanation (FAIR to both grounded and naive)
- Hallucination score: NLI-based faithfulness (unchanged)
- Grounding score: Requires structure AND citations (for grounded only)

Usage:
    python rescore_existing_payloads.py --model gpt-3.5-turbo
    python rescore_existing_payloads.py --model deepseek-chat
    python rescore_existing_payloads.py --dry-run               # rescore all detected models
    python rescore_existing_payloads.py --model gpt-3.5-turbo --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup paths
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from LLM_module.utils.eval_payload import score_text_metrics


# Configuration
EVAL_RESULTS_DIR = Path("/data/guoyu/KG-LLM-XSL/output/eval_results")


def infer_model_from_filename(json_path: Path) -> Optional[str]:
    """Infer model name from a payload filename.

    Expected examples:
      - KRAS_SNRPD3_gpt-3.5-turbo.json
      - KRAS_SNRPD3_naive_deepseek-chat.json

    Returns:
      Model string (e.g., 'gpt-3.5-turbo', 'deepseek-chat') or None if unknown.
    """
    stem = json_path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None

    # pattern: <geneA>_<geneB>_<model>
    # or:      <geneA>_<geneB>_naive_<model>
    if parts[2].lower() == "naive" and len(parts) >= 4:
        return "_".join(parts[3:])
    return "_".join(parts[2:])


def find_payload_files(base_dir: Path, model: Optional[str]) -> List[Tuple[Path, str, str, bool, str]]:
    """
    Find all JSON payload files for a specific model (or all models if model is None).
    
    Returns:
        List of (path, gene_a, gene_b, is_naive, model) tuples
    """
    payloads = []
    
    if not base_dir.exists():
        print(f"[ERROR] Directory not found: {base_dir}")
        return payloads
    
    for gene_pair_dir in sorted(base_dir.iterdir()):
        if not gene_pair_dir.is_dir():
            continue
        
        # Parse gene pair from directory name (e.g., "AKT1_CDK6")
        dir_name = gene_pair_dir.name
        if "_" not in dir_name:
            continue
        
        parts = dir_name.split("_")
        if len(parts) < 2:
            continue
        gene_a, gene_b = parts[0], parts[1]
        
        # Find JSON files
        for json_file in sorted(gene_pair_dir.glob("*.json")):
            inferred_model = infer_model_from_filename(json_file)
            if not inferred_model:
                continue
            if model is not None and inferred_model != model:
                continue
            is_naive = "_naive_" in json_file.name.lower()
            payloads.append((json_file, gene_a, gene_b, is_naive, inferred_model))
    
    return payloads


def rescore_payload(path: Path, *, forced_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load a payload JSON and rescore all text entries.
    
    Returns:
        Rescored payload dict, or None on error
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return None
    
    texts = payload.get("texts", {})
    metrics = payload.get("metrics", {})
    gt = payload.get("ground_truth", {})
    
    gt_features = gt.get("features") or []
    gt_explanation = gt.get("explanation") or ""
    
    # Rescore each text entry
    for key, text in texts.items():
        if key == "prompt":
            continue
        
        if not isinstance(text, str) or not text.strip():
            continue
        
        # Get existing metric entry or create new one
        existing_metrics = metrics.get(key) if isinstance(metrics.get(key), dict) else {}
        effective_model = forced_model or existing_metrics.get("effective_model")
        
        # Re-score with updated metrics
        new_metrics = score_text_metrics(
            ground_truth_features=gt_features,
            ground_truth_explanation=gt_explanation,
            text=text,
            effective_model=effective_model,
            prompt_context=texts.get("prompt") or "",
        )
        
        # Replace metrics (not merge - use new scoring completely)
        metrics[key] = {
            **new_metrics,
            "rescored": True,
        }
    
    payload["metrics"] = metrics
    
    return payload


def extract_csv_row(
    payload: Dict[str, Any],
    gene_a: str,
    gene_b: str,
    strategy: str,
    model: str,
) -> Dict[str, Any]:
    """
    Extract key metrics for CSV row (same format as original eval).
    """
    metrics = payload.get("metrics", {})
    m = metrics.get(strategy, {})
    checks = m.get("checks", {})
    
    return {
        "gene_a": gene_a,
        "gene_b": gene_b,
        "model": model,
        "strategy": strategy,
        "f1_raw": m.get("feature_embed_f1_raw"),
        "f1_raw_full": m.get("feature_embed_f1_raw_full"),
        "f1_raw_topk_p50": m.get("feature_embed_f1_raw_topk_p50"),
        "f1_raw_topk_p75": m.get("feature_embed_f1_raw_topk_p75"),
        "precision_raw": m.get("feature_embed_precision_raw"),
        "recall_raw": m.get("feature_embed_recall_raw"),
        "grounded_score": m.get("grounded_feature_score"),
        "faithfulness": checks.get("faithfulness_score"),
        "gt_faithfulness": checks.get("gt_faithfulness"),
        "kg_faithfulness": checks.get("kg_faithfulness"),
        "hallucination_score": checks.get("hallucination_score"),
        "total_similarity": checks.get("total_embedding_similarity"),
        "format_score": checks.get("format_score"),
        "citation_count": checks.get("citation_count"),
    }


def update_csv_file(csv_path: Path, rows: List[Dict[str, Any]], dry_run: bool = False) -> None:
    """
    Rewrite a CSV file with the provided rows.

    This rescoring script recomputes *all* rows for a given run, so merging with
    older rows can lead to duplicates (e.g., when naive strategy labeling changes).
    """
    # Fixed column order matching GPT-3.5 reference format
    fieldnames = [
        "gene_a", "gene_b", "model", "strategy",
        "f1_raw", "f1_raw_full", "f1_raw_topk_p50", "f1_raw_topk_p75",
        "precision_raw", "recall_raw",
        "grounded_score",
        "faithfulness", "gt_faithfulness", "kg_faithfulness",
        "hallucination_score", "total_similarity", "format_score",
        "citation_count",
    ]
    
    if not rows:
        print(f"[WARN] No rows for {csv_path}, skipping")
        return
    
    # Ensure all rows have all fields
    for r in rows:
        for k in fieldnames:
            r.setdefault(k, "")
    
    # Sort rows by gene_a, gene_b, strategy
    sorted_rows = sorted(rows, key=lambda r: (r.get("gene_a", ""), r.get("gene_b", ""), r.get("strategy", "")))
    
    if dry_run:
        print(f"[DRY-RUN] Would write {len(sorted_rows)} rows to {csv_path}")
        return
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)
    
    print(f"[OK] Updated {csv_path} with {len(sorted_rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore existing evaluation payloads")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files, just show what would change")
    args = parser.parse_args()

    model = 'gemini-2.5-flash'
    # model: Optional[str] = "gpt-3.5-turbo"
    # model: Optional[str] = "deepseek-chat"
    dry_run = args.dry_run

    print(f"[rescore] Scanning {EVAL_RESULTS_DIR} for model={model or 'ALL'}")
    if dry_run:
        print("[rescore] DRY-RUN mode - no files will be modified")
    
    payloads = find_payload_files(EVAL_RESULTS_DIR, model)
    print(f"[rescore] Found {len(payloads)} payload files")
    
    if not payloads:
        print("[rescore] No payloads found. Exiting.")
        return
    
    naive_rows: List[Dict[str, Any]] = []
    normal_rows: List[Dict[str, Any]] = []
    from tqdm import tqdm
    
    for path, gene_a, gene_b, is_naive, payload_model in tqdm(payloads):
        # print(f"[rescore] Processing {path.name}...")

        rescored = rescore_payload(path, forced_model=payload_model)
        if rescored is None:
            continue
        
        # Save rescored payload in place
        if not dry_run:
            path.write_text(json.dumps(rescored, ensure_ascii=False, indent=2), encoding="utf-8")
            # print(f"  -> Updated {path.name}")
        # else:
        #     print(f"  -> [DRY-RUN] Would update {path.name}")
        
        # Extract rows for CSV
        metrics = rescored.get("metrics", {})
        strategies_found = [k for k in metrics if k not in ("prompt",)]

        if is_naive:
            # Naive payloads typically only have a single metrics entry keyed by
            # "baseline". For visualization, we MUST label these rows as
            # strategy="naive" so they don't collide with normal baseline.
            baseline_key = "baseline" if "baseline" in metrics else (strategies_found[0] if strategies_found else None)
            if baseline_key:
                row = extract_csv_row(rescored, gene_a, gene_b, baseline_key, payload_model)
                row["strategy"] = "naive"
                naive_rows.append(row)
        else:
            for strat in strategies_found:
                row = extract_csv_row(rescored, gene_a, gene_b, strat, payload_model)
                normal_rows.append(row)
    
    # Count pairs
    naive_pairs = len(set((r["gene_a"], r["gene_b"]) for r in naive_rows))
    normal_pairs = len(set((r["gene_a"], r["gene_b"]) for r in normal_rows))
    
    # Update CSV files (per model when rescoring ALL)
    def _write_by_model(rows: List[Dict[str, Any]], *, prefix: str) -> None:
        if not rows:
            return
        models = sorted(set(r.get("model", "") for r in rows if r.get("model")))
        if not models:
            return
        for mname in models:
            mrows = [r for r in rows if r.get("model") == mname]
            mpairs = len(set((r["gene_a"], r["gene_b"]) for r in mrows))
            csv_path = EVAL_RESULTS_DIR / f"{prefix}_{mpairs}pairs_{mname}.csv"
            update_csv_file(csv_path, mrows, dry_run=dry_run)

    _write_by_model(naive_rows, prefix="eval_naive")
    _write_by_model(normal_rows, prefix="eval_normal")
    
    # Print summary statistics
    print("\n=== Rescore Summary ===")
    models_found = sorted(set(p[4] for p in payloads))
    print(f"Model(s): {', '.join(models_found) if models_found else 'UNKNOWN'}")
    print(f"Naive payloads: {len([p for p in payloads if p[3]])}")
    print(f"Normal payloads: {len([p for p in payloads if not p[3]])}")
    
    if naive_rows:
        print(f"\nNaive results ({len(naive_rows)} rows):")
        _print_stats(naive_rows)
    
    if normal_rows:
        print(f"\nNormal results ({len(normal_rows)} rows):")
        _print_stats(normal_rows)
        # Per-strategy breakdown
        strats = sorted(set(r.get("strategy", "") for r in normal_rows))
        for s in strats:
            s_rows = [r for r in normal_rows if r.get("strategy") == s]
            print(f"\n  --- {s} ({len(s_rows)} rows) ---")
            _print_stats(s_rows)


def _print_stats(rows: List[Dict[str, Any]]) -> None:
    """Print summary statistics for a list of rows."""
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0
    
    recall_vals = [r["recall_raw"] for r in rows]
    fmt_vals = [r["format_score"] for r in rows]
    hall_vals = [r["hallucination_score"] for r in rows]
    gt_faith_vals = [r.get("gt_faithfulness") for r in rows]
    kg_faith_vals = [r["kg_faithfulness"] for r in rows]
    gs_vals = [r["grounded_score"] for r in rows]
    
    print(f"  Recall (mean):            {mean(recall_vals):.4f}")
    print(f"  Format score (mean):      {mean(fmt_vals):.4f}")
    print(f"  GT-Faithfulness (mean):   {mean(gt_faith_vals):.4f}")
    print(f"  KG-Faithfulness (mean):   {mean(kg_faith_vals):.4f}")
    print(f"  Hallucination (mean):     {mean(hall_vals):.4f}")
    print(f"  Grounded score (mean):    {mean(gs_vals):.4f}")


if __name__ == "__main__":
    main()
