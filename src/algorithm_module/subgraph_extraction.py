from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import networkx as nx
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
	if p not in sys.path:
		sys.path.insert(0, p)

from algorithm_module import algo_config as acfg
from graph_module import graph_config as gcfg
from algorithm_module import output_config as ocfg
from algorithm_module.graph_search_algo import build_explanation_subgraph, gene_node
from algorithm_module.utils.graph_search_utils import load_graphml, node_type
from graph_module.utils.graph_vis import graph_vis
from algorithm_module.utils.prompt_utils import format_row_kv, nonempty_columns, fmt_scalar
from algorithm_module.utils.scoring import edge_score as _edge_score, linear_rank_score
from algorithm_module.utils.node_annotation import annotate_subgraph_nodes
from algorithm_module.graph_search_core import calculate_necessity_score, personalized_pagerank_score


def _load_gnn_scaffolding() -> Mapping[str, object]:
	path = getattr(ocfg, "GNN_SCAFFOLDING_REFERENCE_PATH", None)
	p = Path(path)
	obj = json.loads(p.read_text(encoding="utf-8"))
	if not isinstance(obj, dict):
		raise TypeError("GNN scaffolding payload must be a JSON object.")
	return obj


def _extract_edge_type_scores(obj: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
	"""Return mapping: edge_type -> {"attn": float, "cf_drop": float}."""
	cmp_rows = obj["type_comparison"]
	if not isinstance(cmp_rows, list):
		raise TypeError("`type_comparison` must be a list.")

	by_type: Dict[str, Dict[str, float]] = {}
	for r in cmp_rows:
		if not isinstance(r, dict):
			raise TypeError("Each `type_comparison` row must be an object.")
		etype = str(r["edge_type"]).strip()
		by_type[etype] = {"attn": float(r["attention"]), "cf_drop": float(r["cf_drop"])}
	return by_type


def _extract_source_scores(obj: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
	"""Return mapping: source -> {"attn": float, "cf_drop": float}."""
	attn = obj["attention_by_source"]
	if not isinstance(attn, dict):
		raise TypeError("`attention_by_source` must be an object.")
	attn_map: Dict[str, float] = {str(k).strip(): float(v) for k, v in attn.items()}

	cf = obj["counterfactual_by_source"]
	if not isinstance(cf, list):
		raise TypeError("`counterfactual_by_source` must be a list.")
	cf_map: Dict[str, float] = {str(r["masked_source"]).strip(): float(r["mean_score_drop"]) for r in cf}

	by_src: Dict[str, Dict[str, float]] = {}
	for s in sorted(set(attn_map) | set(cf_map)):
		by_src[s] = {"attn": attn_map[s], "cf_drop": cf_map[s]}
	return by_src


def _annotate_rows_with_gnn(
	*,
	node_rows: Sequence[Mapping[str, object]],
	edge_rows: Sequence[Mapping[str, object]],
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
	obj = _load_gnn_scaffolding()
	type_scores = _extract_edge_type_scores(obj)
	src_scores = _extract_source_scores(obj)

	# IMPORTANT: GNN scaffolding is computed over the *sanitized* message-passing graph
	# (see `src/GNN_algo_module/data.py`), which removes curated/seed SL edges to
	# prevent leakage. Therefore keys like source='curated' and type='SL_pair'
	# are intentionally absent from the scaffolding payload.
	# For node rows, the exported `source` field is node provenance (e.g., seed/curated),
	# which is not an edge provenance and should not be looked up in `attention_by_source`.
	# Instead, derive node-level GNN source signals from incident edge sources.
	node_incident_sources: Dict[str, Set[str]] = {}
	for er in edge_rows:
		s = str(er.get("source", "")).strip()
		u = str(er.get("src", "")).strip()
		v = str(er.get("dst", "")).strip()
		if u:
			node_incident_sources.setdefault(u, set()).add(s)
		if v:
			node_incident_sources.setdefault(v, set()).add(s)

	n_out: List[Dict[str, object]] = []
	for r in node_rows:
		r2 = dict(r)
		nid = str(r2.get("node_id", "")).strip()
		sources = node_incident_sources.get(nid, set())
		if sources:
			# Use max to reflect the strongest attached provenance signal.
			a = max(src_scores[s]["attn"] for s in sources)
			c = max(src_scores[s]["cf_drop"] for s in sources)
		else:
			a = 0.0
			c = 0.0
		r2["gnn_src_attn"] = float(a)
		r2["gnn_src_cf_drop"] = float(c)
		n_out.append(r2)

	e_out: List[Dict[str, object]] = []
	for r in edge_rows:
		r2 = dict(r)
		src = str(r2["source"]).strip()
		a, c = src_scores[src]["attn"], src_scores[src]["cf_drop"]
		r2["gnn_src_attn"] = a
		r2["gnn_src_cf_drop"] = c

		edge_type = str(r2["type"]).strip()
		ta, tc = type_scores[edge_type]["attn"], type_scores[edge_type]["cf_drop"]
		r2["gnn_type_attn"] = ta
		r2["gnn_type_cf_drop"] = tc
		e_out.append(r2)

	return n_out, e_out


def _order_cols(cols: Sequence[str], preferred_prefix: Sequence[str]) -> List[str]:
	preferred = [c for c in preferred_prefix if c in cols]
	return preferred + [c for c in cols if c not in preferred]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for r in rows:
			w.writerow(r)


def export_llm_csv(
	sub: nx.MultiDiGraph,
	cores: Set[str],
	path_nodes: Set[str],
	out_dir: Path,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
	def _is_sl_like_edge(attrs: Mapping[str, object]) -> bool:
		t = str(attrs.get("type") or attrs.get("edge_type") or attrs.get("relation") or "").strip().lower()
		return bool(t) and (t == "sl_pair" or "sl" in t)

	ug = sub.to_undirected(as_view=True)
	core_dists = {c: nx.single_source_shortest_path_length(ug, c) for c in cores}
	dist = {n: min((core_dists[c].get(n) for c in cores if n in core_dists[c]), default=len(sub.nodes)) for n in sub.nodes}
	weights = dict(getattr(acfg, "SUBGRAPH_NODE_RANK_WEIGHTS", {}))
	enabled = {"dist": True, "evidence": True, "ppr": True, "prob": True, "semantic": False}

	# Pre-calculate baseline PPRs for necessity calculation
	# This avoids recalculating it for every node if we can just reuse or compute once
	# However, calculate_necessity_score needs a graph and takes care of baseline calculation if not provided.
	# We will compute it on the fly or pass it if expensive.
	# Since 'sub' is small, we can afford it.
	
	# Identify src/dst genes for flow calculation (Counterfactual logic usually applies to flow between the SL pair)
	# 'cores' has gene_a and gene_b. 
	# We need to arbitrarily pick direction or check flow between them.
	# Simplified approach: If len(cores) == 2, check flow A->B and B->A. 
	# Or just pick the max necessity.
	core_list = list(cores)
	gene_mapping = {n: attrs.get("symbol") for n, attrs in sub.nodes(data=True) if n in cores}
	
	node_rows: List[Dict[str, object]] = []
	for n, attrs in sub.nodes(data=True):
		attrs = attrs or {}
		role = "core" if n in cores else ("path" if n in path_nodes else "other")
		ppr_a = float(attrs.get("ppr_from_core_a") or 0.0)
		ppr_b = float(attrs.get("ppr_from_core_b") or 0.0)
		ppr_m = max(ppr_a, ppr_b, float(attrs.get("ppr_from_core_max") or 0.0))
		prob_a = float(attrs.get("prob_from_core_a") or 0.0)
		prob_b = float(attrs.get("prob_from_core_b") or 0.0)
		prob_m = max(prob_a, prob_b, float(attrs.get("prob_from_core_max") or 0.0))
		evd = max(
			(
				_edge_score((a or {}), source_weight=acfg.EDGE_SOURCE_WEIGHT, relation_weight=acfg.EDGE_RELATION_WEIGHT)
				for u, v, a in sub.edges(n, data=True)
				if (u in cores or v in cores) and not _is_sl_like_edge((a or {}))
			),
			default=0.0,
		)
		df = 1.0 - (min(dist[n], max(dist.values())) / max(dist.values()))

		# --- Part A/B Integration: Necessity Score ---
		# Only calculate for path nodes or potential intermediaries (skip cores themselves or disconnected)
		necessity = 0.0
		if role != "core" and len(core_list) >= 2:
			# Calculate flow intervention A->B
			nec_ab = calculate_necessity_score(sub, core_list[0], core_list[1], intervention_node=n)
			# Calculate flow intervention B->A
			nec_ba = calculate_necessity_score(sub, core_list[1], core_list[0], intervention_node=n)
			necessity = max(nec_ab, nec_ba)
		
		# Store raw necessity for normalization later
		attrs["_raw_necessity"] = necessity
	
	# Normalize necessity scores in the subgraph (Min-Max Scaling to 0-1)
	# This fixes the issue where raw scores are ~1e-10 and ignored.
	max_nec = max((float(d.get("_raw_necessity", 0.0)) for _, d in sub.nodes(data=True)), default=0.0)
	
	# Inject necessity weight (User requested "Weight Fix")
	# We give it a high weight to make it impactful for research.
	weights["necessity"] = 2.0 
	enabled["necessity"] = True

	node_rows: List[Dict[str, object]] = []
	for n, attrs in sub.nodes(data=True):
		attrs = attrs or {}
		role = "core" if n in cores else ("path" if n in path_nodes else "other")
		ppr_a = float(attrs.get("ppr_from_core_a") or 0.0)
		ppr_b = float(attrs.get("ppr_from_core_b") or 0.0)
		ppr_m = max(ppr_a, ppr_b, float(attrs.get("ppr_from_core_max") or 0.0))
		prob_a = float(attrs.get("prob_from_core_a") or 0.0)
		prob_b = float(attrs.get("prob_from_core_b") or 0.0)
		prob_m = max(prob_a, prob_b, float(attrs.get("prob_from_core_max") or 0.0))
		evd = max(
			(
				_edge_score((a or {}), source_weight=acfg.EDGE_SOURCE_WEIGHT, relation_weight=acfg.EDGE_RELATION_WEIGHT)
				for u, v, a in sub.edges(n, data=True)
				if (u in cores or v in cores) and not _is_sl_like_edge((a or {}))
			),
			default=0.0,
		)
		df = 1.0 - (min(dist[n], max(dist.values())) / max(dist.values()))

		# Retrieve raw necessity and normalize
		raw_nec = float(attrs.get("_raw_necessity", 0.0))
		norm_nec = (raw_nec / max_nec) if max_nec > 1e-20 else 0.0

		nscore, _uf, _uw = linear_rank_score({
			"dist": df, 
			"evidence": evd, 
			"ppr": ppr_m, 
			"prob": prob_m, 
			"necessity": norm_nec
		}, weights=weights, enabled=enabled)
		
		node_rows.append(
			{
				"node_id": n,
				"type": node_type(sub, n),
				"label": attrs.get("symbol") or attrs.get("name") or n,
				"role": role,
				"ppr_core_a": ppr_a,
				"ppr_core_b": ppr_b,
				"ppr_max": ppr_m,
				"prob_core_a": prob_a,
				"prob_core_b": prob_b,
				"prob_max": prob_m,
				"evidence_to_cores": evd,
				"core_dist": dist[n],
				"node_score": float(nscore),
				"necessity_score": float(norm_nec), # Changed to export NORMALIZED score
				"symbol": attrs.get("symbol", ""),
				"name": attrs.get("name", ""),
				"ensembl_gene_id": attrs.get("ensembl_gene_id", ""),
				"ensembl_biotype": attrs.get("ensembl_biotype", ""),
				"phase": attrs.get("phase", ""),
				"source": attrs.get("source", ""),
				"summary": attrs.get("summary", ""),
				"function_summary": attrs.get("function_summary", ""),
				"go_terms": attrs.get("go_terms", ""),
				"protein_name": attrs.get("protein_name", ""),
			}
		)

	edge_rows: List[Dict[str, object]] = []
	for u, v, k, attrs in sub.edges(keys=True, data=True):
		attrs = attrs or {}
		if _is_sl_like_edge(attrs):
			# SL edges are supervision labels; exclude from export/prompt evidence.
			continue
		edge_rows.append({
			"src": u,
			"dst": v,
			"type": attrs.get("type") or attrs.get("edge_type") or attrs.get("relation") or "related_to",
			"source": attrs.get("source", ""),
			"evidence_score": _edge_score(attrs, source_weight=acfg.EDGE_SOURCE_WEIGHT, relation_weight=acfg.EDGE_RELATION_WEIGHT),
			"key": k,
			"context": attrs.get("context", ""),
			"cohort": attrs.get("cohort", ""),
			"condition": attrs.get("condition", ""),
			"mutations": attrs.get("mutations", ""),
			"mechanism": attrs.get("mechanism", ""),
			"disease": attrs.get("disease", ""),
			"score": attrs.get("score", ""),
			"corr": attrs.get("corr", ""),
			"sign": attrs.get("sign", ""),
			"note": attrs.get("note", ""),
		})

	# Annotate with GNN diagnostics before exporting to CSV.
	# (This uses the sanitized-GNN scaffolding reference and may yield 0.0 for
	# sources/types that were intentionally absent during GNN training.)
	node_rows_gnn, edge_rows_gnn = _annotate_rows_with_gnn(node_rows=node_rows, edge_rows=edge_rows)

	write_csv(
		out_dir / "nodes.csv",
		ocfg.NODE_CSV_FIELDS,
		node_rows_gnn,
	)
	write_csv(
		out_dir / "edges.csv",
		ocfg.EDGE_CSV_FIELDS,
		edge_rows_gnn,
	)
	return node_rows, edge_rows


def build_prompt(
	gene_a: str,
	gene_b: str,
	*,
	node_rows: Sequence[Mapping[str, object]],
	edge_rows: Sequence[Mapping[str, object]],
) -> str:
	node_rows2, edge_rows2 = _annotate_rows_with_gnn(node_rows=node_rows, edge_rows=edge_rows)
	node_cols = nonempty_columns(node_rows2, always=ocfg.PROMPT_NODE_ALWAYS_FIELDS)
	edge_cols = nonempty_columns(edge_rows2, always=ocfg.PROMPT_EDGE_ALWAYS_FIELDS)

	node_cols = _order_cols(node_cols, list(ocfg.PROMPT_NODE_ALWAYS_FIELDS) + ["gnn_src_attn", "gnn_src_cf_drop"])
	edge_cols = _order_cols(edge_cols, list(ocfg.PROMPT_EDGE_ALWAYS_FIELDS) + ["gnn_type_attn", "gnn_type_cf_drop", "gnn_src_attn", "gnn_src_cf_drop"])

	top_nodes = sorted(node_rows2, key=lambda r: float(r.get("node_score") or 0.0), reverse=True)[:ocfg.PROMPT_TOP_NODES]
	top_edges = sorted(edge_rows2, key=lambda r: float(r.get("evidence_score") or 0.0), reverse=True)[:ocfg.PROMPT_TOP_EDGES]
	
	# Check if any cohort nodes are present in the subgraph
	has_cohort = any(str(r.get("type", "")) == "cohort" for r in node_rows)
	cohort_context = ""
	if has_cohort:
		cohort_context = "\n" + ocfg.COHORT_CONTEXT_NOTE
	
	def row_to_tsv(row: Mapping[str, object], cols: Sequence[str]) -> str:
		return "\t".join(fmt_scalar(row.get(c, "")) for c in cols)
	return ocfg.PROMPT_TEMPLATE.format(
		gene_a=gene_a,
		gene_b=gene_b,
		cohort_context=cohort_context,
		node_cols="\t".join(node_cols),
		edge_cols="\t".join(edge_cols),
		node_lines="\n".join(row_to_tsv(r, node_cols) for r in top_nodes),
		edge_lines="\n".join(row_to_tsv(r, edge_cols) for r in top_edges),
	)


def build_chat_prompts(
	gene_a: str,
	gene_b: str,
	*,
	node_rows: Sequence[Mapping[str, object]],
	edge_rows: Sequence[Mapping[str, object]],
) -> tuple[str, str]:
	node_rows2, edge_rows2 = _annotate_rows_with_gnn(node_rows=node_rows, edge_rows=edge_rows)
	node_cols = nonempty_columns(node_rows2, always=ocfg.PROMPT_NODE_ALWAYS_FIELDS)
	edge_cols = nonempty_columns(edge_rows2, always=ocfg.PROMPT_EDGE_ALWAYS_FIELDS)

	node_cols = _order_cols(node_cols, list(ocfg.PROMPT_NODE_ALWAYS_FIELDS) + ["gnn_src_attn", "gnn_src_cf_drop"])
	edge_cols = _order_cols(edge_cols, list(ocfg.PROMPT_EDGE_ALWAYS_FIELDS) + ["gnn_type_attn", "gnn_type_cf_drop", "gnn_src_attn", "gnn_src_cf_drop"])

	top_nodes = sorted(node_rows2, key=lambda r: float(r.get("node_score") or 0.0), reverse=True)[:ocfg.PROMPT_TOP_NODES]
	top_edges = sorted(edge_rows2, key=lambda r: float(r.get("evidence_score") or 0.0), reverse=True)[:ocfg.PROMPT_TOP_EDGES]
	
	# Check if any cohort nodes are present in the subgraph
	has_cohort = any(str(r.get("type", "")) == "cohort" for r in node_rows)
	cohort_context = ""
	if has_cohort:
		cohort_context = "\n" + ocfg.COHORT_CONTEXT_NOTE
	
	def row_to_tsv(row: Mapping[str, object], cols: Sequence[str]) -> str:
		return "\t".join(fmt_scalar(row.get(c, "")) for c in cols)
	system_template = getattr(ocfg, "PROMPT_SYSTEM_TEMPLATE_COUNTERFACTUAL", ocfg.PROMPT_SYSTEM_TEMPLATE)
	return system_template, ocfg.PROMPT_USER_EVIDENCE_TEMPLATE.format(
		gene_a=gene_a,
		gene_b=gene_b,
		cohort_context=cohort_context,
		node_cols="\t".join(node_cols),
		edge_cols="\t".join(edge_cols),
		node_lines="\n".join(row_to_tsv(r, node_cols) for r in top_nodes),
		edge_lines="\n".join(row_to_tsv(r, edge_cols) for r in top_edges),
	)


def export_graph(
	sub: nx.MultiDiGraph,
	cores: Set[str],
	path_nodes: Set[str],
	gene_a: str,
	gene_b: str,
	out_dir: Path,
) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)
	annotate_subgraph_nodes(sub, cache_path=out_dir / "node_annotation_cache.json")
	ann = {}
	for n, attrs in sub.nodes(data=True):
		attrs = attrs or {}
		ann[str(n)] = {"type": attrs.get("type"), "symbol": attrs.get("symbol"), "name": attrs.get("name"), "summary": attrs.get("summary"), "go_terms": attrs.get("go_terms"), "protein_name": attrs.get("protein_name"), "function_summary": attrs.get("function_summary")}
	(out_dir / "node_annotations.json").write_text(json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8")
	nx.write_graphml(sub, out_dir / "subgraph.graphml")
	graph_vis(sub, out_dir / "subgraph.html", title=f"{sorted(cores)} subgraph")
	n_rows, e_rows = export_llm_csv(sub, cores, path_nodes, out_dir)
	(out_dir / f"{gene_a}_{gene_b}_prompt.txt").write_text(build_prompt(gene_a, gene_b, node_rows=n_rows, edge_rows=e_rows), encoding="utf-8")
	sys_prompt, user_prompt = build_chat_prompts(gene_a, gene_b, node_rows=n_rows, edge_rows=e_rows)
	(out_dir / f"{gene_a}_{gene_b}_prompt_messages.json").write_text(
		json.dumps({"gene_a": gene_a, "gene_b": gene_b, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]}, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)


def main() -> None:
	GENE_NUM = 112
	graph = load_graphml(gcfg.OUT_DIR / "ablation_graphs" / "full.graphml")

	for gene_a, gene_b in tqdm([(pair["gene_a"], pair["gene_b"]) for pair in json.loads(gcfg.SL_PAIRS_COMMON_FILE.read_text(encoding="utf-8"))[:GENE_NUM]], desc="Extract subgraphs"):
		sub, meta = build_explanation_subgraph(graph, gene_a, gene_b, neigh_max_hops=getattr(acfg, "NEIGHBOR_DEFAULT_MAX_HOPS", 2), max_nodes=None)
		export_graph(sub, {gene_node(gene_a), gene_node(gene_b)}, {str(n) for p in meta.get("paths", []) for n in p.get("nodes", [])}, gene_a, gene_b, Path(ocfg.SUBGRAPH_OUTPUT_DIR) / f"{gene_a}_{gene_b}")


if __name__ == "__main__":
	main()



