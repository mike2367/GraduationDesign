from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx

from algorithm_module import algo_config
from algorithm_module.graph_search_core import InferencePath, personalized_pagerank_vector
from algorithm_module.utils.graph_search_utils import (
	best_edge_score_between,
	edge_candidates,
	min_core_distance,
	node_type,
)
from algorithm_module.utils.semantic_neighbors import build_core_query_text, cosine, embed_texts, node_text
from algorithm_module.utils.scoring import edge_score as _edge_score, clamp01, linear_rank_score


def edge_score(attrs: Mapping[str, object]) -> float:
	return _edge_score(attrs, source_weight=algo_config.EDGE_SOURCE_WEIGHT, relation_weight=algo_config.EDGE_RELATION_WEIGHT)


def best_edge_between(
	graph: nx.MultiDiGraph, u: str, v: str
) -> Optional[Tuple[str, str, str, Mapping[str, object]]]:
	return max(
		((src, dst, key, (attrs or {})) for src, dst, key, attrs, _dir in edge_candidates(graph, u, v)),
		key=lambda t: edge_score(t[3]),
		default=None,
	)


def best_edge_score_to_cores(
	graph: nx.MultiDiGraph, cores: Set[str], node: str
) -> float:
	return max((edge_score(c[3]) for core in cores if (c := best_edge_between(graph, core, node)) is not None), default=0.0)

def prepare_node_metrics(
	graph: nx.MultiDiGraph,
	ug: nx.Graph,
	cores: Set[str],
	nodes: Set[str],
	gene_a: str,
	gene_b: str,
	*,
	max_shortest_path_len: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
	ppr_a = personalized_pagerank_vector(graph, gene_a)
	ppr_b = personalized_pagerank_vector(graph, gene_b)

	paths_a = nx.single_source_shortest_path(ug, gene_a, cutoff=max_shortest_path_len)
	paths_b = nx.single_source_shortest_path(ug, gene_b, cutoff=max_shortest_path_len)

	def _path_prob(node_path: Optional[Sequence[str]]) -> float:
		if not node_path:
			return 0.0
		prob = 1.0
		for u, v in zip(node_path, node_path[1:]):
			if not (graph.has_edge(u, v) or graph.has_edge(v, u)):
				return 0.0
			d = max(1, int(ug.degree(u)))
			prob *= 1.0 / float(d)
		return clamp01(prob)
	return {
		node: {
			"ppr_from_core_a": (pa := float(ppr_a.get(node, 0.0))),
			"ppr_from_core_b": (pb := float(ppr_b.get(node, 0.0))),
			"ppr_from_core_max": max(pa, pb),
			"prob_from_core_a": (pra := float(_path_prob(paths_a.get(node)))),
			"prob_from_core_b": (prb := float(_path_prob(paths_b.get(node)))),
			"prob_from_core_max": max(pra, prb),
		}
		for node in nodes
	}

def open_targets_drug_neighbors(graph: nx.MultiDiGraph, core: str) -> Set[str]:
	out: Set[str] = set()
	for u, _v, attrs in graph.in_edges(core, data=True):
		a = attrs or {}
		if a.get("type") == "targets" and a.get("source") == "OpenTargets" and node_type(graph, u) == "drug":
			out.add(u)
	return out


def dynamic_gene_cap_for_core(graph: nx.MultiDiGraph, core: str, base_cap: int) -> int:
	if base_cap <= 0:
		return base_cap
	reduce_by = int(len(open_targets_drug_neighbors(graph, core)) // max(1, int(algo_config.SUBGRAPH_DRUGS_PER_GENE_CAP_STEP)))
	return max(int(algo_config.SUBGRAPH_MIN_GENE_NEIGHBORS_PER_CORE), int(base_cap) - reduce_by)


def _is_tf_neighbor(graph: nx.MultiDiGraph, anchor: str, nb: str) -> bool:
	"""Check if *nb* is a TF-regulator neighbour of *anchor*."""
	attrs = graph.nodes.get(nb, {}) or {}
	if bool(attrs.get("is_tf")):
		return True
	for _s, _d, _k, eattrs, _dir in edge_candidates(graph, anchor, nb):
		if str((eattrs or {}).get("type") or "") == "TF_regulates":
			return True
	return False


def cap_gene_neighbors(
	graph: nx.MultiDiGraph,
	ug: nx.Graph,
	kept: Set[str],
	cores: Set[str],
	path_nodes: Set[str],
	cap: int,
	*,
	anchor_nodes: Optional[Set[str]] = None,
	tf_cap: Optional[int] = None,
	use_dynamic_cap: bool = False,
	passes: int = 1,
) -> Set[str]:
	"""Unified gene-neighbour capping (replaces three former functions).

	Parameters
	----------
	anchor_nodes : iterate only over these nodes (e.g. cores). ``None`` → all.
	tf_cap       : separate cap for TF-regulator gene neighbours; ``None`` →
	               TF nodes are excluded from capping.
	use_dynamic_cap : reduce per-core cap based on drug-neighbour count.
	passes       : iterative trim passes.
	"""
	kept = set(kept)

	def _score_to(anchor: str, nb: str) -> float:
		return best_edge_score_between(graph, anchor, nb)

	for _ in range(max(1, passes)):
		changed = False
		anchors = sorted(anchor_nodes) if anchor_nodes is not None else sorted(kept)
		for anchor in anchors:
			if anchor not in kept:
				continue
			gene_nbs = {
				nb for nb in ug.neighbors(anchor)
				if nb in kept and nb != anchor and nb not in cores
				and node_type(graph, nb) == "gene"
			}
			if not gene_nbs:
				continue

			actual_cap = dynamic_gene_cap_for_core(graph, anchor, cap) if use_dynamic_cap else cap

			if tf_cap is not None:
				# Separate TF and regular gene neighbours
				tf_nbs = {nb for nb in gene_nbs if _is_tf_neighbor(graph, anchor, nb)}
				reg_nbs = gene_nbs - tf_nbs
				# Regular
				mand = {n for n in reg_nbs if n in path_nodes}
				space = max(0, max(actual_cap, len(mand)) - len(mand))
				extras = sorted([n for n in reg_nbs if n not in mand], key=lambda n: (-_score_to(anchor, n), n))
				for nb in reg_nbs - (set(extras[:space]) | mand):
					kept.discard(nb); changed = True
				# TF
				tf_mand = {n for n in tf_nbs if n in path_nodes}
				tf_space = max(0, max(tf_cap, len(tf_mand)) - len(tf_mand))
				tf_extras = sorted([n for n in tf_nbs if n not in tf_mand], key=lambda n: (-_score_to(anchor, n), n))
				for nb in tf_nbs - (set(tf_extras[:tf_space]) | tf_mand):
					kept.discard(nb); changed = True
			else:
				# Skip TF-flagged nodes from removal
				cappable = [nb for nb in gene_nbs if not bool((graph.nodes.get(nb, {}) or {}).get("is_tf"))]
				ordered = sorted(cappable, key=lambda nb: (
					0 if nb in cores else 1, 0 if nb in path_nodes else 1,
					-_score_to(anchor, nb), str(nb),
				))
				for nb in ordered[actual_cap:]:
					if nb not in cores:
						kept.discard(nb); changed = True
		if not changed:
			break
	return kept

def apply_node_type_budgets(
	graph: nx.MultiDiGraph,
	ug: nx.Graph,
	cores: Set[str],
	path_nodes: Set[str],
	nodes: Set[str],
	node_metrics: Dict[str, Dict[str, float]],
	*,
	neigh_max_hops: int,
	max_nodes: int,
	use_prob_features: bool,
	gene_a: str,
	gene_b: str,
	protected_nodes: Optional[Set[str]] = None,
) -> Set[str]:
	# Enforce minimum size if path nodes are few.
	target_count = getattr(algo_config, "SUBGRAPH_MAX_NODES", 50)
	
	protected = set(cores) | set(path_nodes) | set(protected_nodes or set())
	candidates = sorted(list(nodes - protected))
	
	if len(protected) >= target_count:
		return protected
	
	remaining_slots = target_count - len(protected)
	
	# Prepare scoring context
	q_text = build_core_query_text(gene_a, gene_b)
	# (Caching embeddings would be ideal here if called repeatedly, but for now linear scan)
	q_vec = embed_texts([q_text], model_path=str(algo_config.SEMANTIC_NEIGHBOR_MODEL_PATH))[0]
	cand_vecs = embed_texts([node_text(graph, n).replace("_", " ") for n in candidates], model_path=str(algo_config.SEMANTIC_NEIGHBOR_MODEL_PATH))
	sem_scores = {
		n: clamp01(max(0.0, cosine(v, q_vec))) 
		for n, v in zip(candidates, cand_vecs)
	}

	weights_base = dict(getattr(algo_config, "SUBGRAPH_NODE_RANK_WEIGHTS", {}))
	
	# Score all candidates
	scored = []
	for node in candidates:
		d = int(min_core_distance(ug, cores, node, neigh_max_hops + 2))
		cutoff = max(1, int(neigh_max_hops) + 2)
		f_dist = 1.0 - (min(d, cutoff) / float(cutoff))
		f_evd = float(best_edge_score_to_cores(graph, cores, node))
		
		met = node_metrics.get(node, {})
		f_ppr = float(met.get("ppr_from_core_max", 0.0))
		f_prob = float(met.get("prob_from_core_max", 0.0))
		f_sem = float(sem_scores.get(node, 0.0))
		
		feats = {
			"dist": clamp01(f_dist),
			"evidence": clamp01(f_evd),
			"ppr": clamp01(f_ppr),
			"prob": clamp01(f_prob),
			"semantic": clamp01(f_sem),
		}
		
		# Rank score
		s, _, _ = linear_rank_score(feats, weights=weights_base, enabled=None)
		scored.append((s, node))
	
	# Pick top N
	scored.sort(key=lambda x: -x[0])
	best = {node for _, node in scored[:remaining_slots]}
	
	return protected | best


def exclude_far_gene_neighbors(
	graph: nx.MultiDiGraph,
	ug: nx.Graph,
	cores: Set[str],
	nodes: Set[str],
	path_nodes: Set[str],
	*,
	distance_limit: int,
	neigh_max_hops: int,
	protected_nodes: Optional[Set[str]] = None,
) -> Set[str]:
	protected = set(path_nodes) | set(protected_nodes or set())
	kept = set(nodes)
	for node in list(kept):
		if node in cores or node in protected:
			continue
		dist = min_core_distance(ug, cores, node, neigh_max_hops)
		if dist > distance_limit:
			kept.discard(node)
	return kept


def enforce_global_node_cap(
	graph: nx.MultiDiGraph,
	ug: nx.Graph,
	nodes: Set[str],
	cores: Set[str],
	path_nodes: Set[str],
	node_metrics: Dict[str, Dict[str, float]],
	*,
	max_nodes: int,
	neigh_max_hops: int,
	use_prob_features: bool,
	protected_nodes: Optional[Set[str]] = None,
) -> Set[str]:
	protected = set(path_nodes) | set(protected_nodes or set())
	def _rank(node: str) -> Tuple[int, int, int, int, float, float, str]:
		core_flag = 0 if node in cores else 1
		path_flag = 0 if node in protected else 1
		dist = min_core_distance(ug, cores, node, neigh_max_hops + 2)
		node_t = node_type(graph, node)
		type_val = algo_config.NODE_TYPE_RANK.get(node_t, algo_config.NODE_TYPE_RANK.get("unknown", 99))
		metrics = node_metrics.get(node, {})
		return (
			core_flag,
			path_flag,
			dist,
			type_val,
			-float(metrics.get("ppr_from_core_max", 0.0)),
			-float(metrics.get("prob_from_core_max", 0.0)),
			node,
		)

	ordered = sorted(nodes, key=_rank)
	return set(ordered[:max_nodes])


def prune_selected_nodes(
	graph: nx.MultiDiGraph,
	ug: nx.Graph,
	cores: Set[str],
	path_nodes: Set[str],
	nodes: Set[str],
	node_metrics: Dict[str, Dict[str, float]],
	*,
	neigh_max_hops: int,
	max_nodes: int,
	max_gene_neighbors_per_gene: Optional[int],
	exclude_gene_beyond_1_hop: bool,
	use_prob_features: bool,
	gene_a: str,
	gene_b: str,
	protected_nodes: Optional[Set[str]] = None,
) -> Set[str]:
	kept = cap_gene_neighbors(
		graph, ug, nodes, cores, path_nodes,
		int(max_gene_neighbors_per_gene),
		anchor_nodes=cores,
		tf_cap=int(getattr(algo_config, 'SUBGRAPH_MAX_TF_NEIGHBORS_PER_CORE', 12)),
		use_dynamic_cap=True,
	)

	kept = apply_node_type_budgets(
		graph,
		ug,
		cores,
		path_nodes,
		kept,
		node_metrics,
		neigh_max_hops=neigh_max_hops,
		max_nodes=max_nodes,
		use_prob_features=use_prob_features,
		gene_a=gene_a,
		gene_b=gene_b,
		protected_nodes=protected_nodes,
	)

	if exclude_gene_beyond_1_hop:
		kept = exclude_far_gene_neighbors(
			graph,
			ug,
			cores,
			kept,
			path_nodes,
			distance_limit=1,
			neigh_max_hops=neigh_max_hops,
			protected_nodes=protected_nodes,
		)

	kept = enforce_global_node_cap(
		graph,
		ug,
		kept,
		cores,
		path_nodes,
		node_metrics,
		max_nodes=max_nodes,
		neigh_max_hops=neigh_max_hops,
		use_prob_features=use_prob_features,
		protected_nodes=protected_nodes,
	)

	# Final safety-net: cap gene-gene hubs in the induced candidate set.
	kept = cap_gene_neighbors(
		graph, ug, kept, cores, path_nodes,
		int(max_gene_neighbors_per_gene),
		passes=int(getattr(algo_config, 'SUBGRAPH_GENE_CAP_TRIM_PASSES', 6)),
	)
	return kept


def prune_disconnected_components(sub: nx.MultiDiGraph, cores: Set[str]) -> None:
	"""Remove disconnected components that don't contribute to the main explanation.

	Rules:
	- If there is a component containing *both* cores, keep only that component.
	- Else, keep the union of components that contain at least one core.
	- Always drop components containing no core.
	"""
	if sub.number_of_nodes() == 0:
		return
	ug = sub.to_undirected(as_view=True)
	comps = list(nx.connected_components(ug))
	if not comps:
		return
	core_list = list(cores)
	both = None
	if len(core_list) >= 2:
		for c in comps:
			if core_list[0] in c and core_list[1] in c:
				both = c
				break
	if both is not None:
		keep = set(both)
	else:
		keep = set().union(*[c for c in comps if (c & cores)])
	for n in list(sub.nodes):
		if n not in keep:
			sub.remove_node(n)



def add_nodes_with_metrics(
	sub: nx.MultiDiGraph,
	graph: nx.MultiDiGraph,
	nodes: Set[str],
	node_metrics: Dict[str, Dict[str, float]],
) -> None:
	for node in nodes:
		attrs = dict(graph.nodes[node] or {})
		attrs.update(node_metrics.get(node, {}))
		sub.add_node(node, **attrs)


def mandatory_path_edges(
	graph: nx.MultiDiGraph,
	paths: Sequence["InferencePath"],
) -> Tuple[List[Tuple[str, str, str, Mapping[str, object]]], Set[Tuple[str, str, str]]]:
	mandatory: List[Tuple[str, str, str, Mapping[str, object]]] = []
	keys: Set[Tuple[str, str, str]] = set()
	for path in paths:
		for u, v in zip(path.nodes, path.nodes[1:]):
			candidate = best_edge_between(graph, u, v)
			if candidate is None:
				continue
			src, dst, key, attrs = candidate
			key_tuple = (src, dst, str(key))
			if key_tuple in keys:
				continue
			keys.add(key_tuple)
			mandatory.append((src, dst, str(key), dict(attrs)))
	return mandatory, keys


def collect_candidate_edges(
	graph: nx.MultiDiGraph,
	nodes: Set[str],
	mandatory_keys: Set[Tuple[str, str, str]],
) -> List[Tuple[str, str, str, Mapping[str, object]]]:
	candidates: List[Tuple[str, str, str, Mapping[str, object]]] = []
	for u, v, key, attrs in graph.edges(keys=True, data=True):
		if u not in nodes or v not in nodes:
			continue
		if (u, v, str(key)) in mandatory_keys:
			continue
		candidates.append((u, v, str(key), attrs or {}))
	return candidates


def apply_open_targets_drug_cap(
	graph: nx.MultiDiGraph,
	cores: Set[str],
	path_nodes: Set[str],
	edges: List[Tuple[str, str, str, Mapping[str, object]]],
	max_drugs_per_core_hop1: int,
) -> List[Tuple[str, str, str, Mapping[str, object]]]:
	drugs_to_keep: Set[str] = set()
	for core in cores:
		candidates: List[Tuple[float, str]] = []
		for src, dst, _key, attrs in edges:
			if dst != core:
				continue
			phase = graph.nodes.get(src, {}).get("phase")
			phase_val = float(phase) if isinstance(phase, (int, float)) else -1.0
			candidates.append((phase_val, src))
		for _, name in sorted(candidates, key=lambda item: (-item[0], item[1]))[:max_drugs_per_core_hop1]:
			drugs_to_keep.add(name)

	filtered: List[Tuple[str, str, str, Mapping[str, object]]] = []
	for src, dst, key, attrs in edges:
		if node_type(graph, src) == "drug" and src not in drugs_to_keep and src not in path_nodes:
			continue
		filtered.append((src, dst, key, attrs))
	return filtered


def add_edges_sorted(
	sub: nx.MultiDiGraph,
	cores: Set[str],
	edges: List[Tuple[str, str, str, Mapping[str, object]]],
	keep_fraction_hop1: float,
	keep_fraction_hop2: float,
) -> None:
	def _edge_key(item: Tuple[str, str, str, Mapping[str, object]]) -> Tuple[float, str, str, str]:
		src, dst, key, attrs = item
		return (-edge_score(attrs), str(src), str(dst), str(key))

	frac_h1 = float(keep_fraction_hop1)
	frac_h2 = float(keep_fraction_hop2)

	hop1: List[Tuple[str, str, str, Mapping[str, object]]] = []
	hop2p: List[Tuple[str, str, str, Mapping[str, object]]] = []
	for item in edges:
		src, dst, _key, _attrs = item
		if src in cores or dst in cores:
			hop1.append(item)
		else:
			hop2p.append(item)

	ordered_h1 = sorted(hop1, key=_edge_key)
	ordered_h2 = sorted(hop2p, key=_edge_key)

	keep_h1 = int(len(ordered_h1) * frac_h1)
	keep_h2 = int(len(ordered_h2) * frac_h2)
	kept = ordered_h1[:keep_h1] + ordered_h2[:keep_h2]

	for src, dst, key, attrs in kept:
		sub.add_edge(src, dst, key=key, **attrs)


def enforce_gene_caps_in_subgraph(
	graph: nx.MultiDiGraph,
	sub: nx.MultiDiGraph,
	cores: Set[str],
	path_nodes: Set[str],
	max_gene_neighbors_per_gene: Optional[int],
) -> None:
	cap = int(max_gene_neighbors_per_gene)

	def _gene_neighbors_undirected(node: str) -> List[str]:
		nbs = ({*sub.successors(node), *sub.predecessors(node)} - {node})
		return sorted([nb for nb in nbs if node_type(graph, nb) == "gene"])

	def _remove_link(a: str, b: str) -> None:
		# Remove all edges between a and b in both directions.
		for _src, _dst, _key in list(sub.edges(a, keys=True)):
			if _dst == b:
				sub.remove_edge(_src, _dst, key=_key)
		for _src, _dst, _key in list(sub.edges(b, keys=True)):
			if _dst == a:
				sub.remove_edge(_src, _dst, key=_key)

	def _pair_score(a: str, b: str) -> float:
		# Best available evidence score among edges between a and b.
		best = 0.0
		for _src, _dst, _key, attrs in sub.edges(a, keys=True, data=True):
			if _dst == b:
				best = max(best, edge_score(attrs or {}))
		for _src, _dst, _key, attrs in sub.in_edges(a, keys=True, data=True):
			if _src == b:
				best = max(best, edge_score(attrs or {}))
		return best

	for _ in range(int(algo_config.SUBGRAPH_GENE_CAP_TRIM_PASSES)):
		changed = False
		for node in list(sub.nodes):
			nbs = _gene_neighbors_undirected(node)

			ordered = sorted(
				nbs,
				key=lambda nb: (
					0 if nb in cores else 1,
					0 if nb in path_nodes else 1,
					-_pair_score(node, nb),
					str(nb),
				),
			)
			keep = set(ordered[:cap])
			for nb in ordered[cap:]:
				_remove_link(node, nb)
				changed = True
		if not changed:
			break



def trim_excess_edges(sub: nx.MultiDiGraph) -> None:
	# Deterministic trimming: keep the highest-evidence edges.
	items: List[Tuple[str, str, str, Mapping[str, object]]] = []
	for u, v, k, attrs in sub.edges(keys=True, data=True):
		items.append((str(u), str(v), str(k), attrs or {}))
	items.sort(key=lambda it: (-edge_score(it[3]), it[0], it[1], it[2]))
	keep_keys = {(u, v, k) for u, v, k, _a in items[: int(algo_config.SUBGRAPH_MAX_EDGES)]}
	for u, v, k in list(sub.edges(keys=True)):
		key = (str(u), str(v), str(k))
		if key not in keep_keys:
			sub.remove_edge(u, v, key=k)


def remove_isolates(sub: nx.MultiDiGraph, cores: Set[str]) -> None:
	for node in list(sub.nodes):
		if node in cores:
			continue
		if sub.degree(node) == 0:
			sub.remove_node(node)


def assemble_subgraph(
	graph: nx.MultiDiGraph,
	cores: Set[str],
	path_nodes: Set[str],
	node_metrics: Dict[str, Dict[str, float]],
	selected_nodes: Set[str],
	paths: Sequence["InferencePath"],
	*,
	max_gene_neighbors_per_gene: Optional[int],
	protected_nodes: Optional[Set[str]] = None,
) -> nx.MultiDiGraph:
	keep_fraction_hop1 = float(algo_config.SUBGRAPH_KEEP_FRACTION_HOP1)
	keep_fraction_hop2 = float(algo_config.SUBGRAPH_KEEP_FRACTION_HOP2)
	max_drugs_per_core_hop1 = int(algo_config.SUBGRAPH_MAX_DRUGS_PER_CORE_HOP1)
	sub = nx.MultiDiGraph()
	add_nodes_with_metrics(sub, graph, selected_nodes, node_metrics)

	mandatory, mandatory_keys = mandatory_path_edges(graph, paths)
	candidates = collect_candidate_edges(graph, selected_nodes, mandatory_keys)
	candidates = apply_open_targets_drug_cap(
		graph,
		cores,
		path_nodes,
		candidates,
		max_drugs_per_core_hop1,
	)

	for src, dst, key, attrs in mandatory:
		sub.add_edge(src, dst, key=key, **attrs)

	add_edges_sorted(sub, cores, candidates, keep_fraction_hop1, keep_fraction_hop2)
	enforce_gene_caps_in_subgraph(
		graph,
		sub,
		cores,
		path_nodes,
		max_gene_neighbors_per_gene,
	)
	trim_excess_edges(sub)

	remove_isolates(sub, cores)
	prune_disconnected_components(sub, cores)
	return sub
