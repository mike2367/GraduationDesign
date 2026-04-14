from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
import time
import os
from pathlib import Path
from tqdm import tqdm  # type: ignore



ROOT = Path(__file__).resolve().parent.parent.parent
for p in (str(ROOT / "src"), str(ROOT)):
	if p not in sys.path:
		sys.path.insert(0, p)


from graph_module import graph_config as gcfg
from LLM_module import eval_config as ecfg
from LLM_module.utils.eval_payload import make_pair_payload, score_text_metrics
from LLM_module.utils.explanation_scoring import split_feature_field, preload_models
from LLM_module.utils.llm_client import get_default_client
from LLM_module.utils.llm_strategies import run_baseline, run_cove, run_self_refine


if getattr(ecfg, "CUDA_VISIBLE_DEVICES", None):
	os.environ["CUDA_VISIBLE_DEVICES"] = str(ecfg.CUDA_VISIBLE_DEVICES)


def _extract_pair_from_filename(path: Path) -> tuple[str, str] | None:
	"""Extract (gene_a, gene_b) from a payload filename."""
	parts = path.stem.split("_")
	if len(parts) < 3:
		return None
	return parts[0], parts[1]


def _load_json(path: Path) -> dict[str, object] | None:
	"""Best-effort JSON loader for on-disk payloads."""
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except (OSError, UnicodeDecodeError, json.JSONDecodeError):
		return None


def _row_from_metrics(pair: tuple[str, str], strategy: str, m: dict[str, object], model_name: str) -> dict[str, object]:
	checks = (m.get("checks") or {}) if isinstance(m, dict) else {}
	return {
		"gene_a": pair[0],
		"gene_b": pair[1],
		"model": model_name,
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


def _collect_csv_rows_from_json(out_root: Path, model_name: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
	"""Build CSV rows by scanning model-specific JSON payloads.

	This avoids cross-model contamination and guarantees the pair count reflects
	actual JSON payloads on disk (same idea as `rescore_existing_payloads.py`).
	"""
	normal_rows: list[dict[str, object]] = []
	naive_rows: list[dict[str, object]] = []

	# Normal payloads
	for path in out_root.rglob(f"*_{model_name}.json"):
		if "_naive_" in path.name.lower():
			continue
		pair = _extract_pair_from_filename(path)
		if not pair:
			continue
		payload = _load_json(path)
		if not payload:
			continue
		metrics = payload.get("metrics") or {}
		if not isinstance(metrics, dict):
			continue
		for strat, m in metrics.items():
			if strat == "prompt":
				continue
			if not isinstance(m, dict):
				continue
			normal_rows.append(_row_from_metrics(pair, str(strat), m, model_name))

	# Naive payloads
	for path in out_root.rglob(f"*_naive_{model_name}.json"):
		pair = _extract_pair_from_filename(path)
		if not pair:
			continue
		payload = _load_json(path)
		if not payload:
			continue
		metrics = payload.get("metrics") or {}
		if not isinstance(metrics, dict):
			continue
		m = metrics.get("baseline")
		if isinstance(m, dict):
			naive_rows.append(_row_from_metrics(pair, "naive", m, model_name))

	return normal_rows, naive_rows


def _write_csv_atomically(csv_path: Path, rows: list[dict[str, object]]) -> None:
	"""Write CSV with atomic replace to prevent partial writes."""
	# Fixed column order matching GPT-3.5 reference format
	fieldnames = [
		"gene_a", "gene_b", "model", "strategy",
		"f1_raw", "f1_raw_full", "f1_raw_topk_p50", "f1_raw_topk_p75",
		"precision_raw", "recall_raw",
		"grounded_score",
		"faithfulness", "kg_faithfulness",
		"hallucination_score", "total_similarity", "format_score",
		"citation_count",
	]
	
	for r in rows:
		for k in fieldnames:
			r.setdefault(k, "")
	
	tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
	with tmp_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)
	tmp_path.replace(csv_path)


def _cleanup_old_model_csvs(csv_dir: Path, model_name: str, keep: set[Path]) -> None:
	"""Remove outdated per-model CSV files, preserving those in keep set."""
	for pattern in [f"eval_normal_*pairs_{model_name}.csv", f"eval_naive_*pairs_{model_name}.csv"]:
		for old_csv in csv_dir.glob(pattern):
			if old_csv not in keep:
				try:
					old_csv.unlink()
				except OSError:
					pass


def _write_model_csvs(csv_out_dir: Path, out_root: Path, model_name: str) -> None:
	"""Generate per-model summary CSVs from JSON payloads; cleanup old versions."""
	rows_normal, rows_naive = _collect_csv_rows_from_json(out_root, model_name)
	
	if not rows_normal and not rows_naive:
		print(f"[csv] WARNING: no rows found for model={model_name}; keeping existing CSVs")
		return
	
	written_csvs: set[Path] = set()
	
	# Write normal strategy results
	if rows_normal:
		pair_count = len({(r["gene_a"], r["gene_b"]) for r in rows_normal})
		csv_path = csv_out_dir / f"eval_normal_{pair_count}pairs_{model_name}.csv"
		_write_csv_atomically(csv_path, rows_normal)
		written_csvs.add(csv_path)
		print(f"Normal results CSV: {csv_path} ({pair_count} pairs)")
	
	# Write naive baseline results
	if rows_naive:
		pair_count = len({(r["gene_a"], r["gene_b"]) for r in rows_naive})
		csv_path = csv_out_dir / f"eval_naive_{pair_count}pairs_{model_name}.csv"
		_write_csv_atomically(csv_path, rows_naive)
		written_csvs.add(csv_path)
		print(f"Naive results CSV: {csv_path} ({pair_count} pairs)")
	
	# Remove outdated CSVs for this model only
	_cleanup_old_model_csvs(csv_out_dir, model_name, written_csvs)


def _get_completed_pairs(out_root: Path, model_name: str, strategies: list[str], check_naive: bool) -> set[tuple[str, str]]:
	"""Scan output directory to find gene pairs with complete results."""
	completed = set()
	if not out_root.exists():
		return completed
	
	for pair_dir in out_root.iterdir():
		if not pair_dir.is_dir():
			continue
		parts = pair_dir.name.split("_")
		if len(parts) < 2:
			continue
		gene_a, gene_b = parts[0], parts[1]
		
		# Check main strategies JSON exists and has all required strategies
		main_json = pair_dir / f"{gene_a}_{gene_b}_{model_name}.json"
		if main_json.exists():
			try:
				payload = json.loads(main_json.read_text(encoding="utf-8"))
				texts = payload.get("texts", {})
				metrics = payload.get("metrics", {})
				# Check all required strategies are present with non-empty text and metrics
				all_present = all(
					texts.get(s) and metrics.get(s)
					for s in strategies
				)
				if not all_present:
					continue
			except Exception:
				continue
		
		# Check naive JSON if required
		if check_naive:
			naive_json = pair_dir / f"{gene_a}_{gene_b}_naive_{model_name}.json"
			if not naive_json.exists():
				continue
			try:
				np = json.loads(naive_json.read_text(encoding="utf-8"))
				if not (np.get("texts", {}).get("baseline") and np.get("metrics", {}).get("baseline")):
					continue
			except Exception:
				continue
		
		completed.add((gene_a, gene_b))
	
	return completed


def _print_cuda_device_once() -> None:
	import torch  # type: ignore
	cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
	if torch.cuda.is_available():
		# When CUDA_VISIBLE_DEVICES is set to a single GPU id, torch renumbers it as cuda:0.
		idx = int(torch.cuda.current_device())
		name = str(torch.cuda.get_device_name(idx))
		visible = f"cuda:{idx}"
		mapped = None
		if cuda_vis and "," not in cuda_vis:
			mapped = f"cuda:{cuda_vis}"
		print(f"[cuda] visible={visible} mapped={mapped} name={name}")
	else:
		print(f"[cuda] CUDA_VISIBLE_DEVICES={cuda_vis!r} cuda_available=False")


def main() -> None:
	start_all = time.time()
	def _now() -> float:
		return time.time()

	all_pairs = json.loads(gcfg.SL_PAIRS_COMMON_FILE.read_text(encoding="utf-8"))
	all_pairs_tuples = [(pair["gene_a"], pair["gene_b"]) for pair in all_pairs[:112]]
	root = Path(__file__).resolve().parents[2]
	out_root = (root / ecfg.DEFAULT_EVAL_OUT_DIR).resolve()
	
	continue_mode = bool(getattr(ecfg, "EVAL_CONTINUE_FROM_EXISTING", False))
	limit = int(getattr(ecfg, "EVAL_PAIR_LIMIT", 0) or 0)
	
	if continue_mode:
		# Determine which strategies to check based on config
		model_name_tmp = str(ecfg.MODEL).split("/")[-1]
		strategies_to_check = [k for k in ["baseline", "self_refine", "cove"] if ecfg.EVAL_STRATEGY in {k, "all"}]
		check_naive = ecfg.EVAL_STRATEGY in {"baseline", "all"} and bool(getattr(ecfg, "EVAL_RUN_NAIVE", True))
		
		completed = _get_completed_pairs(out_root, model_name_tmp, strategies_to_check, check_naive)
		print(f"[continue] Found {len(completed)} completed pairs (will skip)")
		
		# Filter out completed pairs
		pending = [p for p in all_pairs_tuples if p not in completed]
		print(f"[continue] {len(pending)} pairs remaining to evaluate")
		
		if limit > 0:
			pairs = pending[:limit]
			print(f"[continue] Processing next {len(pairs)} pairs (EVAL_PAIR_LIMIT={limit})")
		else:
			pairs = pending
			print(f"[continue] Processing all {len(pairs)} remaining pairs")
	else:
		# Normal mode: just apply limit to all pairs
		if limit > 0:
			pairs = all_pairs_tuples[:limit]
		else:
			pairs = all_pairs_tuples
	
	if not pairs:
		print("[run] No pairs to evaluate (all complete or empty list). Exiting.")
		return

	print(f"[run] Evaluating {len(pairs)} gene pairs")
	_print_cuda_device_once()
	
	# Preload scoring models once at startup (can be CPU-heavy, so keep it visible).
	print("[preload] Starting local model preload...")
	t_pre = time.time()
	preload_models()
	print(f"[preload] Done in {time.time() - t_pre:.1f}s")

	reuse = bool(ecfg.EVAL_REUSE_EXISTING)
	deterministic = bool(ecfg.EVAL_DETERMINISTIC)
	ts = datetime.utcnow().isoformat()

	# Load ground truth once.
	gt: dict[tuple[str, str], dict[str, object]] = {}
	gt_path = root / ecfg.EVAL_GROUND_TRUTH_PATH
	if gt_path.exists():
		with gt_path.open("r", encoding="utf-8") as f:
			for row in csv.DictReader(f):
				a, b = (row.get("geneA") or "").strip().upper(), (row.get("geneB") or "").strip().upper()
				if a and b:
					gt[(a, b)] = {
						"features": split_feature_field(row.get("important_features") or ""),
						"explanation": (row.get("explanation") or "").strip(),
					}

	client = get_default_client()
	temperature = 0.0 if deterministic else ecfg.TEMPERATURE
	top_p = 1.0 if deterministic else ecfg.TOP_P
	if deterministic:
		print("EVAL_DETERMINISTIC=1: forcing temperature=0.0, top_p=1.0")

	provider = str(getattr(ecfg, "LLM_PROVIDER", "aigcbest") or "aigcbest").strip().lower()
	if provider == "local":
		endpoint_repr = getattr(ecfg, "LOCAL_MODEL_PATH", "") or ""
	elif provider in {"aigc", "aigcbest", "api2"}:
		endpoint_repr = ecfg.AIGC_BEST_BASE_URL
	else:
		endpoint_repr = ""
	model_payload = {
		"endpoint": endpoint_repr,
		"model": ecfg.MODEL,
		"temperature": temperature,
		"top_p": top_p,
		"max_tokens": ecfg.MAX_TOKENS,
	}
	model_name = str(ecfg.MODEL).split("/")[-1]

	strategies = {
		"baseline": run_baseline,
		"self_refine": run_self_refine,
		"cove": run_cove,
	}
	selected_keys = [k for k in strategies.keys() if ecfg.EVAL_STRATEGY in {k, "all"}]
	do_naive = ecfg.EVAL_STRATEGY in {"baseline", "all"} and bool(getattr(ecfg, "EVAL_RUN_NAIVE", True))
	stages_per_pair = len(selected_keys) + (1 if do_naive else 0)
	stage_total = max(1, len(pairs) * max(1, stages_per_pair))
	bar = tqdm(total=stage_total, unit="stage") if tqdm else None

	# CSV output for visualization
	# NOTE: CSVs are generated at the end by scanning JSON payloads for the
	# current model. This avoids any cross-model contamination.

	# Clear output dir at start (prevents mixing old/new results), unless continuing
	# if not continue_mode and not reuse and bool(getattr(ecfg, "EVAL_CLEAR_OUTPUT_DIR", True)):
	# 	print("[run] Clearing output directory (fresh run)")
	# 	shutil.rmtree(out_root, ignore_errors=True)
	out_root.mkdir(parents=True, exist_ok=True)

	def _load_prompt(pair: tuple[str, str]) -> tuple[str, str | None, str | None]:
		prompt_path = root / ecfg.EVAL_PROMPTS_DIR / f"{pair[0]}_{pair[1]}" / f"{pair[0]}_{pair[1]}_prompt.txt"
		msg_path = root / ecfg.EVAL_PROMPTS_DIR / f"{pair[0]}_{pair[1]}" / f"{pair[0]}_{pair[1]}_prompt_messages.json"
		system_prompt: str | None = None
		user_prompt: str | None = None
		if msg_path.exists():
			try:
				obj = json.loads(msg_path.read_text(encoding="utf-8"))
				msgs = obj.get("messages") if isinstance(obj, dict) else None
				if isinstance(msgs, list):
					for m in msgs:
						if not isinstance(m, dict):
							continue
						role = str(m.get("role") or "").strip().lower()
						content = m.get("content")
						if not isinstance(content, str):
							continue
						if role == "system" and system_prompt is None:
							system_prompt = content
						elif role == "user" and user_prompt is None:
							user_prompt = content
			except Exception:
				system_prompt, user_prompt = None, None

		if prompt_path.exists():
			prompt_txt = prompt_path.read_text(encoding="utf-8")
			stability = (
				"\n\nStability constraints (follow strictly):\n"
				"- Be concise and avoid repetition (target <= ~650 words).\n"
				"- Prefer mechanistic processes/phenotypes over repeating entity names.\n"
				"- In the Mechanistic Summary, include at least TWO explicit downstream phenotype/outcome statements and tie them causally to the mechanism.\n"
				"  - Use explicit directionality words (e.g., 'reduced', 'increased', 'impaired') so it is scorable.\n"
				"  - Examples (not an exhaustive list): reduced cell proliferation; increased apoptosis; cell cycle arrest; reduced viability.\n"
				"- In the 'Key process phrases:' line, include at least ONE phenotype/outcome phrase (e.g., 'reduced cell proliferation' or 'increased apoptosis').\n"
			)
			if system_prompt is not None and user_prompt is not None:
				system_prompt = (system_prompt.rstrip() + stability).strip()
				return user_prompt, system_prompt, str(msg_path)
			prompt_txt = (prompt_txt + stability).strip()
			return prompt_txt, None, str(prompt_path)
		fallback = f"You are a computational biologist specializing in mechanistic explanations of synthetic lethality (SL). Explain the Synthetic Lethality(SL) Mechanism between {pair[0]} and {pair[1]}."
		print(f"[prompt] WARNING: missing KG prompt at {prompt_path}; using naive fallback")
		return fallback, None, None

	def _run_strategy(payload: dict[str, object], truth: dict[str, object], key: str, fn, prompt_text: str, system_prompt: str | None):
		text = str(payload.get("texts", {}).get(key) or "") if reuse else ""
		tr = None
		if not text:
			tr = fn(
				client,
				prompt_text,
				system_prompt=system_prompt,
				temperature=temperature,
				top_p=top_p,
				max_tokens=ecfg.MAX_TOKENS,
			)
			text = (tr.final.text if getattr(tr, "final", None) else tr.initial.text)

		payload.setdefault("texts", {})
		payload.setdefault("metrics", {})
		payload["texts"][key] = text
		payload["metrics"][key] = score_text_metrics(
			ground_truth_features=truth.get("features") or [],
			ground_truth_explanation=truth.get("explanation") or "",
			text=text,
			effective_model=getattr(getattr(tr, "final", None), "model", None),
			prompt_context=(prompt_text if system_prompt is None else (str(system_prompt).rstrip() + "\n\n" + str(prompt_text).lstrip())),
		)
		return tr

	for pair in pairs:
		pair_name = f"{pair[0]}_{pair[1]}"
		out_dir = (root / ecfg.DEFAULT_EVAL_OUT_DIR / f"{pair[0]}_{pair[1]}")
		out_dir.mkdir(parents=True, exist_ok=True)
		prompt, system_prompt, prompt_path_s = _load_prompt(pair)
		truth = gt.get((pair[0], pair[1]), {})

		out_path = out_dir / f"{pair[0]}_{pair[1]}_{model_name}.json"
		payload = make_pair_payload(
			gene_a=pair[0],
			gene_b=pair[1],
			prompt_text=(prompt if system_prompt is None else (str(system_prompt).rstrip() + "\n\n" + str(prompt).lstrip())),
			model_payload=model_payload,
			ground_truth_available=(pair in gt),
			ground_truth_features=truth.get("features") or [],
			ground_truth_explanation=truth.get("explanation") or "",
			prompt_path=prompt_path_s,
		)
		if reuse and out_path.exists():
			old = json.loads(out_path.read_text(encoding="utf-8"))
			payload["texts"].update(old.get("texts") or {})
			payload["model"] = old.get("model") or payload["model"]
			payload["rescored_at"], payload["rescoring_mode"] = ts, "EVAL_REUSE_EXISTING"

		for k in selected_keys:
			fn = strategies[k]
			if bar:
				bar.set_description(f"{pair_name}:{k}")
			_run_strategy(payload, truth, k, fn, prompt, system_prompt)
			if bar:
				bar.update(1)

		out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
		
		if do_naive:
			naive_prompt = (
				f"You are a computational biologist specializing in mechanistic explanations of synthetic lethality (SL). "
				f"Explain the Synthetic Lethality (SL) Mechanism between {pair[0]} and {pair[1]}. "
				"Use general biological knowledge only. "
				"No knowledge-graph tables are provided for this naive baseline; do NOT invent KG edges or KG citations. "
				"Be concise."
			)
			naive_path = out_dir / f"{pair[0]}_{pair[1]}_naive_{model_name}.json"
			np = make_pair_payload(
				gene_a=pair[0],
				gene_b=pair[1],
				prompt_text=naive_prompt,
				model_payload=model_payload,
				ground_truth_available=(pair in gt),
				ground_truth_features=truth.get("features") or [],
				ground_truth_explanation=truth.get("explanation") or "",
			)

			if reuse and naive_path.exists():
				old = json.loads(naive_path.read_text(encoding="utf-8"))
				np["texts"].update(old.get("texts") or {})
				np["model"] = old.get("model") or np["model"]
				np["rescored_at"], np["rescoring_mode"] = ts, "EVAL_REUSE_EXISTING"

			text = str(np.get("texts", {}).get("baseline") or "") if reuse else ""
			tr = None
			if not text:
				tr = run_baseline(
					client,
					naive_prompt,
					temperature=temperature,
					top_p=top_p,
					max_tokens=ecfg.MAX_TOKENS,
				)
				text = tr.final.text

			np.setdefault("texts", {})["baseline"] = text
			np.setdefault("metrics", {})["baseline"] = score_text_metrics(
				ground_truth_features=truth.get("features") or [],
				ground_truth_explanation=truth.get("explanation") or "",
				text=text,
				effective_model=getattr(getattr(tr, "final", None), "model", None),
				prompt_context=naive_prompt,
			)
			naive_path.write_text(json.dumps(np, ensure_ascii=False, indent=2), encoding="utf-8")
			
			if bar:
				bar.set_description(f"{pair_name}:naive")
				bar.update(1)

	if bar:
		bar.close()

	# Save CSV results for visualization (per-model, rebuilt from JSON payloads)
	csv_out_dir = root / ecfg.DEFAULT_EVAL_OUT_DIR
	csv_out_dir.mkdir(parents=True, exist_ok=True)
	_write_model_csvs(csv_out_dir, out_root, model_name)

	print(f"Total evaluation time (all pairs): {_now() - start_all:.2f} seconds")


if __name__ == "__main__":
	main()
