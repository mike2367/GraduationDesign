from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import math
import networkx as nx

from algorithm_module import algo_config
from algorithm_module import graph_search_core
from algorithm_module.graph_search_core import InferencePath, InferenceStep, gene_node
from algorithm_module.utils.graph_search_utils import (
	bounded_simple_paths,
	dominant_type_subset,
	edge_candidates,
	load_graphml,
	min_core_distance,
	neighbors_by_type,
	node_type,
	path_rank_score_linear,
	path_sources_and_switches,
	pick_step,
	weighted_undirected,
	best_edge_score_between,
)
from algorithm_module.utils.subgraph_pruning import (
	assemble_subgraph,
	prepare_node_metrics,
	prune_selected_nodes,
)
from algorithm_module.utils.neighborhood_search import khop_ranked_neighbors


def find_inference_paths(
	graph: nx.MultiDiGraph,
	src: str,
	dst: str,
	*,
	max_hops: Optional[int] = None,
	top_k: Optional[int] = None,
	max_paths_considered: Optional[int] = None,
	allow_undirected_search: bool = True,
	hard_hop_limit: int = 10,
) -> List[InferencePath]:
	max_hops_i = min(algo_config.INFERENCE_DEFAULT_MAX_HOPS if max_hops is None else max_hops, hard_hop_limit)
	top_k_i = algo_config.INFERENCE_DEFAULT_TOP_K if top_k is None else top_k
	max_paths_considered_i = algo_config.INFERENCE_MAX_PATHS_CONSIDERED if max_paths_considered is None else max_paths_considered

	wg = weighted_undirected(graph) if allow_undirected_search else graph
	paths = bounded_simple_paths(wg, src, dst, hops=max_hops_i, limit=max_paths_considered_i)
	deg = graph_search_core.undirected_degrees(graph)

	ppr = graph_search_core.personalized_pagerank_score(graph, src, dst)
	candidates: List[InferencePath] = []

	for node_path in paths:
		steps = [pick_step(graph, a, b) for a, b in zip(node_path, node_path[1:])]
		prob = graph_search_core.path_probability(graph, node_path, deg)
		if prob <= 0.0:
			continue
		sources, switches = path_sources_and_switches(steps)
		score = path_rank_score_linear(
			path_len=len(steps),
			max_hops=max_hops_i,
			source_switches=switches,
			path_probability=prob,
			ppr_score=ppr,
		)
		candidates.append(
			InferencePath(
				nodes=tuple(node_path),
				steps=tuple(steps),
				path_probability=prob,
				ppr_score=ppr,
				sources_in_path=sources,
				source_switches=switches,
				rank_score=score,
			)
		)

	return sorted(candidates, key=lambda p: (-p.rank_score, len(p.steps), p.source_switches))[:top_k_i]

def expand_neighbors_by_score(
	graph: nx.MultiDiGraph,
	node: str,
	*,
	max_total: Optional[int] = None,
	min_per_type: int = 0,  # Deprecated/Ignored
	max_gene_neighbors: Optional[int] = None,
) -> Set[str]:
	# Get all neighbors
	candidates = set(graph.neighbors(node))
	
	# Apply gene hub cap if needed
	if max_gene_neighbors is not None:
		genes = {n for n in candidates if node_type(graph, n) == "gene"}
		if len(genes) > max_gene_neighbors:
			# Rank genes by score to keep best ones
			scored_genes = sorted(
				[(best_edge_score_between(graph, node, g), g) for g in genes],
				key=lambda x: -x[0]
			)
			keep_genes = {g for _, g in scored_genes[:max_gene_neighbors]}
			candidates = (candidates - genes) | keep_genes

	if max_total is None or len(candidates) <= max_total:
		return candidates

	# Rank all by edge score (GNN weight)
	scored = []
	for n in candidates:
		s = best_edge_score_between(graph, node, n)
		scored.append((s, n))
	
	# Primary Sort: Score (descending)
	# Secondary Sort: Node Name (provenance stability)
	scored.sort(key=lambda x: (-x[0], x[1]))
	
	return {n for _, n in scored[:max_total]}


def adaptive_balanced_neighborhood(
	graph: nx.MultiDiGraph,
	start: str,
	*,
	max_hops: int = 3,
	stop_min_nodes: Optional[int] = None,
	stop_min_types: Optional[int] = None,
	max_nodes: int = 250,
	max_neighbors_per_node_per_hop: Optional[int] = None,
	max_gene_neighbors_per_gene: Optional[int] = None,
) -> Tuple[Set[str], int, Set[str]]:
	stop_min_nodes_i = algo_config.ADAPTIVE_STOP_MIN_NODES if stop_min_nodes is None else stop_min_nodes
	stop_min_types_i = algo_config.ADAPTIVE_STOP_MIN_TYPES if stop_min_types is None else stop_min_types

	seen: Set[str] = {start}
	frontier: Set[str] = {start}
	types_seen: Set[str] = set()
	hops = 0

	ug = graph.to_undirected(as_view=True)

	def _edge_type_between(a: str, b: str) -> str:
		if not (cands := edge_candidates(graph, a, b)):
			return "unknown"
		return str((min(cands, key=lambda it: (algo_config.EDGE_TYPE_PRIORITY.get(str((it[3] or {}).get("type") or "unknown"), 99), it[4], str((it[3] or {}).get("type") or "unknown")))[3] or {}).get("type") or "unknown")

	def _pick_neighbors(node: str) -> List[str]:
		nbs = sorted(set(ug.neighbors(node)))
		if max_neighbors_per_node_per_hop is None:
			return nbs
		cap = max_neighbors_per_node_per_hop
		if cap <= 0:
			return []
		if max_gene_neighbors_per_gene is not None and node_type(graph, node) == "gene":
			gene_nbs = [nb for nb in nbs if node_type(graph, nb) == "gene"]
			keep_genes = gene_nbs[:max_gene_neighbors_per_gene] if max_gene_neighbors_per_gene > 0 else []
			return keep_genes + [nb for nb in nbs if nb not in set(gene_nbs)][:max(0, cap - len(keep_genes))]
		return nbs[:cap]

	for depth in range(1, max_hops + 1):
		next_frontier: Set[str] = set()
		for node in frontier:
			for nb in _pick_neighbors(node):
				types_seen.add(_edge_type_between(node, nb))
				if nb not in seen:
					next_frontier.add(nb)

		seen.update(next_frontier)
		frontier = next_frontier
		hops = depth

		if (len(seen) - 1) >= max_nodes:
			break
		if (len(seen) - 1) >= stop_min_nodes_i and len(types_seen) >= stop_min_types_i:
			break
		if not frontier:
			break

	return seen, hops, types_seen


def khop_neighbors_by_source(
	graph: nx.MultiDiGraph,
	start: str,
	*,
	k: int = 2,
	edge_sources: Optional[Iterable[str]] = None,
) -> Dict[str, Set[str]]:
	sources = set(edge_sources) if edge_sources is not None else {str(a.get("source") or "unknown") for *_e, a in graph.edges(data=True)}
	result: Dict[str, Set[str]] = {}

	for src_name in sorted(sources):
		sg = nx.Graph()
		sg.add_node(start)
		for u, v, attrs in graph.edges(data=True):
			if str(attrs.get("source") or "unknown") != src_name:
				continue
			sg.add_edge(u, v)
		seen = frontier = {start}
		for _ in range(k):
			frontier = set().union(*(sg.neighbors(n) for n in frontier)) - seen
			seen |= frontier
		neighbors = seen - {start}
		homog = dominant_type_subset(neighbors, graph)
		if homog:
			result[src_name] = set(homog)

	return result


def khop_neighbors_cross_source(
	graph: nx.MultiDiGraph,
	start: str,
	*,
	k: int = 2,
	max_per_neighbor: int = 1,
	max_neighbors: int = 12,
	compute_paths: bool = True,
) -> Dict[str, List[InferencePath]]:
	ug = graph.to_undirected(as_view=True)
	seen = frontier = {start}
	for _ in range(k):
		frontier = set().union(*(ug.neighbors(n) for n in frontier)) - seen
		seen |= frontier

	neighbors = dominant_type_subset(seen - {start}, graph, limit=max_neighbors)
	out: Dict[str, List[InferencePath]] = {}
	for nb in neighbors:
		if not compute_paths:
			out[nb] = []
			continue
		if (paths := find_inference_paths(graph, start, nb, max_hops=k, top_k=max_per_neighbor, max_paths_considered=40)):
			out[nb] = paths
	return out

def build_explanation_subgraph(
	graph: nx.MultiDiGraph,
	gene_a: str,
	gene_b: str,
	*,
	max_path_hops: int = 3,
	top_k_paths: int = 5,
	neigh_max_hops: int = 2,
	max_gene_neighbors_per_gene: Optional[int] = None,
	max_nodes: Optional[int] = None,
	# Override the neighborhood expansion cap used by adaptive_balanced_neighborhood.
	# If None: uses a multiplier of `max_nodes`.
	neigh_max_nodes: Optional[int] = None,
) -> Tuple[nx.MultiDiGraph, dict]:
	hard_hop_limit = getattr(algo_config, "EXPLANATION_HARD_MAX_HOPS", algo_config.INFERENCE_DEFAULT_MAX_HOPS)
	max_path_hops = min(max_path_hops, hard_hop_limit)
	neigh_max_hops = min(neigh_max_hops, hard_hop_limit)

	max_gene_neighbors_per_gene_i = algo_config.NEIGHBOUR_RESTRICTION if max_gene_neighbors_per_gene is None else max_gene_neighbors_per_gene
	max_nodes_i = algo_config.SUBGRAPH_MAX_NODES if max_nodes is None else max_nodes
	ga = gene_a if gene_a.startswith("gene:") else gene_node(gene_a)
	gb = gene_b if gene_b.startswith("gene:") else gene_node(gene_b)
	cores = {ga, gb}

	paths = find_inference_paths(graph, ga, gb, max_hops=max_path_hops, top_k=top_k_paths)
	path_nodes = {n for p in paths for n in p.nodes}

	def _core_function_neighbors(core: str) -> Set[str]:
		return {v for _u, v, attrs in graph.out_edges(core, data=True) if str((attrs or {}).get("type") or "") == "in_pathway" and node_type(graph, v) == "pathway"}

	def _core_driver_cohort_neighbors(core: str) -> Set[str]:
		"""Return cohort neighbors for a core gene, but ONLY if the gene shows differential driver status.
		
		If the gene is a driver in ALL considered cohorts (universal), return empty set.
		If the gene is a driver in SOME cohorts (differential), return those cohort nodes.
		"""
		# Collect all cohorts where this gene is a driver
		gene_cohorts = {
			v
			for _u, v, attrs in graph.out_edges(core, data=True)
			if str((attrs or {}).get("type") or "") == "driver_in" and node_type(graph, v) == "cohort"
		}
		
		# Get all considered cohorts from graph_config
		from graph_module import graph_config as gcfg
		all_considered_cohorts = {f"cohort:{name}" for name in getattr(gcfg, "COHORT_TO_TUMOR_TYPE", {}).keys()}
		
		# Find which considered cohorts are actually in the graph
		available_cohorts = {c for c in all_considered_cohorts if c in graph}
		
		if not available_cohorts:
			return gene_cohorts  # No comparison possible, include all
		
		# If gene is driver in ALL available cohorts → universal → return empty (not informative)
		if gene_cohorts and available_cohorts and gene_cohorts >= available_cohorts:
			return set()
		
		# Otherwise differential → return the specific cohorts
		return gene_cohorts

	core_function_nodes = _core_function_neighbors(ga) | _core_function_neighbors(gb)
	core_cohort_nodes = _core_driver_cohort_neighbors(ga) | _core_driver_cohort_neighbors(gb)

	neigh_cap = neigh_max_nodes
	if neigh_cap is None:
		mult = getattr(algo_config, "NEIGHBOR_CANDIDATE_CAP_MULTIPLIER", 6)
		min_cap = getattr(algo_config, "NEIGHBOR_CANDIDATE_MIN", 80)
		neigh_cap = max(min_cap, max_nodes_i * max(1, mult))

	exclude_gene_beyond_1_hop = bool(algo_config.SUBGRAPH_EXCLUDE_GENE_BEYOND_1_HOP)

	use_prob_in_neigh = bool(algo_config.USE_PROB_IN_NEIGHBOUR)

	na, hops_a, *_ = khop_ranked_neighbors(
		graph,
		ga,
		k=neigh_max_hops,
		max_nodes=int(neigh_cap),
		gene_a=ga,
		gene_b=gb,
		use_prob_features=use_prob_in_neigh,
		traversal="bfs",
		max_explored=neigh_cap * algo_config.NEIGHBOR_MAX_EXPLORED_MULTIPLIER,
	)
	nb, hops_b, *_ = khop_ranked_neighbors(
		graph,
		gb,
		k=neigh_max_hops,
		max_nodes=int(neigh_cap),
		gene_a=ga,
		gene_b=gb,
		use_prob_features=use_prob_in_neigh,
		traversal="bfs",
		max_explored=neigh_cap * algo_config.NEIGHBOR_MAX_EXPLORED_MULTIPLIER,
	)

	selected_nodes: Set[str] = set(cores) | path_nodes | na | nb
	selected_nodes |= core_function_nodes
	selected_nodes |= core_cohort_nodes
	ug = graph.to_undirected(as_view=True)

	cutoff = max(max_path_hops, neigh_max_hops) + 2
	_node_metrics = prepare_node_metrics(
		graph,
		ug,
		cores,
		selected_nodes,
		ga,
		gb,
		max_shortest_path_len=cutoff,
	)

	selected_nodes = prune_selected_nodes(
		graph,
		ug,
		cores,
		path_nodes,
		selected_nodes,
		_node_metrics,
		neigh_max_hops=neigh_max_hops,
		max_nodes=max_nodes_i,
		max_gene_neighbors_per_gene=max_gene_neighbors_per_gene_i,
		exclude_gene_beyond_1_hop=exclude_gene_beyond_1_hop,
		use_prob_features=use_prob_in_neigh,
		gene_a=ga,
		gene_b=gb,
		protected_nodes=(core_function_nodes | core_cohort_nodes),
	)

	_node_metrics = {n: _node_metrics.get(n, {}) for n in selected_nodes}

	sub = assemble_subgraph(
		graph,
		cores,
		path_nodes,
		_node_metrics,
		selected_nodes,
		paths,
		max_gene_neighbors_per_gene=max_gene_neighbors_per_gene_i,
		protected_nodes=(core_function_nodes | core_cohort_nodes),
	)

	meta = {
		"gene_a": ga,
		"gene_b": gb,
		"selected_nodes": sub.number_of_nodes(),
		"selected_edges": sub.number_of_edges(),
		"paths": [p.to_dict() for p in paths],
		"neighbor": {ga: {"hops": hops_a, "nodes": len(na)}, gb: {"hops": hops_b, "nodes": len(nb)}},
		"core_function_nodes": len(core_function_nodes),
		"core_cohort_nodes": len(core_cohort_nodes),
		"pruning": {
			"keep_fraction_hop1": float(algo_config.SUBGRAPH_KEEP_FRACTION_HOP1),
			"keep_fraction_hop2": float(algo_config.SUBGRAPH_KEEP_FRACTION_HOP2),
			"max_drugs_per_core_hop1": int(algo_config.SUBGRAPH_MAX_DRUGS_PER_CORE_HOP1),
			"max_gene_neighbors_per_gene": max_gene_neighbors_per_gene_i,
			"max_nodes": max_nodes_i,
		},
	}

	return sub, meta


def explain_gene_pair(
	graph: nx.MultiDiGraph,
	gene_a: str,
	gene_b: str,
	*,
	max_hops: Optional[int] = None,
	top_k_paths: int = 5,
	khop: int = 2,
	max_neighbors: int = 12,
	compute_cross_paths: bool = True,
) -> dict:
	ga = gene_a if gene_a.startswith("gene:") else gene_node(gene_a)
	gb = gene_b if gene_b.startswith("gene:") else gene_node(gene_b)
	return {
		"gene_a": ga,
		"gene_b": gb,
		"ppr_score": graph_search_core.personalized_pagerank_score(graph, ga, gb),
		"paths": [p.to_dict() for p in find_inference_paths(graph, ga, gb, max_hops=max_hops, top_k=top_k_paths)],
		"khop_within_source": {k: sorted(v) for k, v in khop_neighbors_by_source(graph, ga, k=khop).items()},
		"khop_cross_source": {k: [p.to_dict() for p in v] for k, v in khop_neighbors_cross_source(graph, ga, k=khop, max_neighbors=max_neighbors, compute_paths=compute_cross_paths).items()},
	}


__all__ = [
	"InferencePath",
	"InferenceStep",
	"adaptive_balanced_neighborhood",
	"expand_neighbors_by_score",
	"build_explanation_subgraph",
	"explain_gene_pair",
	"find_inference_paths",
	"gene_node",
	"khop_neighbors_by_source",
	"khop_neighbors_cross_source",
	"load_graphml",
	"path_rank_score_linear",
	"personalized_pagerank_score",
]
