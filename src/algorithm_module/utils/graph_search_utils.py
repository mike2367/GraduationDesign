from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import math

import networkx as nx

from algorithm_module import algo_config
from algorithm_module.graph_search_core import InferenceStep
from algorithm_module.utils.scoring import clamp01, edge_score as _raw_edge_score


def load_graphml(path: Path) -> nx.MultiDiGraph:
	g_raw = nx.read_graphml(path)

	mg = g_raw if isinstance(g_raw, nx.MultiDiGraph) else nx.MultiDiGraph(g_raw)
	for _n, attrs in mg.nodes(data=True):
		t = (attrs or {}).get("type")
		if t is not None:
			attrs["type"] = str(t).strip().lower()
	return mg


def node_type(graph: nx.MultiDiGraph, node: str) -> str:
	attrs = graph.nodes.get(node)
	if attrs:
		t = attrs.get("type")
		if t:
			return str(t).strip().lower()
	# Fallback: infer from ID
	if isinstance(node, str) and ":" in node:
		return node.split(":", 1)[0].strip().lower()
	return "unknown"


def edge_candidates(graph: nx.MultiDiGraph, u: str, v: str) -> List[Tuple[str, str, str, dict, str]]:
	out: List[Tuple[str, str, str, dict, str]] = []
	for key, attrs in (graph.get_edge_data(u, v) or {}).items():
		out.append((u, v, str(key), attrs or {}, "forward"))
	for key, attrs in (graph.get_edge_data(v, u) or {}).items():
		out.append((v, u, str(key), attrs or {}, "reverse"))
	return out


def pick_step(graph: nx.MultiDiGraph, want_src: str, want_dst: str) -> InferenceStep:
	cands = edge_candidates(graph, want_src, want_dst)
	if not cands:
		return InferenceStep(want_src, want_dst, "related_to", "unknown", "unknown", None, {})
	src, dst, key, attrs, dir_label = sorted(cands, key=lambda item: (algo_config.EDGE_TYPE_PRIORITY.get(str((item[3] or {}).get("type") or "related_to"), 99), item[4], str((item[3] or {}).get("type") or "related_to")))[0]
	return InferenceStep(
		src=want_src,
		dst=want_dst,
		edge_type=str(attrs.get("type") or "related_to"),
		edge_source=str(attrs.get("source") or "unknown"),
		direction="both" if graph.has_edge(want_dst, want_src) and graph.has_edge(want_src, want_dst) else dir_label,
		edge_key=key,
		attrs={k: v for k, v in (attrs or {}).items() if k in {"type", "source", "mutations", "mechanism", "disease", "context"}},
	)


def path_sources_and_switches(steps: Sequence[InferenceStep]) -> Tuple[Tuple[str, ...], int]:
	sources = [s.edge_source or "unknown" for s in steps]
	switches = sum(1 for i in range(1, len(sources)) if sources[i] != sources[i - 1])
	return tuple(sources), switches


def path_rank_score_linear(
	*,
	path_len: int,
	max_hops: int,
	source_switches: int,
	path_probability: float,
	ppr_score: float,
	weights: Optional[Mapping[str, float]] = None,
) -> float:
	weights_base = dict(algo_config.PATH_RANK_WEIGHTS)
	if weights:
		weights_base.update({k: v for k, v in weights.items() if k in weights_base})
	f_len = 1.0 - (max(0, path_len - 1) / max(1, max_hops))
	f_switch = 1.0 - (source_switches / max(1, path_len - 1)) if path_len > 1 else 1.0
	f_prob = 1.0 - math.log(max(path_probability, algo_config.PATH_PROB_EPS), algo_config.PATH_PROB_EPS)
	return clamp01(weights_base["len"] * clamp01(f_len) + weights_base["switch"] * clamp01(f_switch) + weights_base["prob"] * clamp01(f_prob) + weights_base["ppr"] * clamp01(ppr_score))


def weighted_undirected(graph: nx.MultiDiGraph) -> nx.Graph:
	wg = nx.Graph()
	wg.add_nodes_from(graph.nodes(data=True))
	for u, v, _k, attrs in graph.edges(keys=True, data=True):
		etype = str((attrs or {}).get("type") or "related_to")
		# Strict weight: no fallback for missing types.
		w = float(algo_config.EDGE_TYPE_WEIGHT.get(etype, 0.0))
		if wg.has_edge(u, v):
			# If multiple edges exist between u and v, keep the one with HIGHEST importance.
			wg[u][v]["weight"] = max(float(wg[u][v].get("weight", w)), w)
		else:
			wg.add_edge(u, v, weight=w)
	return wg


def bounded_simple_paths(
	g: nx.Graph,
	start: str,
	end: str,
	*,
	hops: int,
	limit: int,
) -> List[List[str]]:
	reach = nx.single_source_shortest_path_length(g, start, cutoff=hops)
	if end not in reach:
		return []
	dist_to_end = nx.single_source_shortest_path_length(g, end, cutoff=hops)
	out: List[List[str]] = []
	stack: List[Tuple[str, List[str]]] = [(start, [start])]
	while stack and len(out) < limit:
		node, path = stack.pop()
		if len(path) - 1 > hops:
			continue
		if node == end:
			out.append(path)
			continue
		for nb in sorted(list(g.neighbors(node)), key=lambda n: (dist_to_end.get(n, 10**9), str(n))):
			if nb in path:
				continue
			remaining = hops - (len(path) - 1) - 1
			if remaining < 0 or dist_to_end.get(nb, 10**9) > remaining:
				continue
			stack.append((nb, path + [nb]))
	return out


def neighbors_by_type(graph: nx.MultiDiGraph, node: str) -> Dict[str, Set[str]]:
	grouped: Dict[str, Set[str]] = {}
	for _, v, attrs in graph.out_edges(node, data=True):
		t = str((attrs or {}).get("type") or "unknown")
		grouped.setdefault(t, set()).add(v)
	for u, _, attrs in graph.in_edges(node, data=True):
		t = str((attrs or {}).get("type") or "unknown")
		grouped.setdefault(t, set()).add(u)
	return grouped


def dominant_type_subset(nodes: Iterable[str], graph: nx.MultiDiGraph, limit: Optional[int] = None) -> List[str]:
	by_type: Dict[str, List[str]] = {}
	for n in nodes:
		t = str(graph.nodes.get(n, {}).get("type") or "unknown")
		by_type.setdefault(t, []).append(n)
	if not by_type:
		return []
	chosen_type = sorted(by_type, key=lambda t: (-len(by_type[t]), t))[0]
	ordered = sorted(by_type[chosen_type])
	return ordered if limit is None else ordered[:limit]


def min_core_distance(ug: nx.Graph, cores: Set[str], node: str, cutoff: int) -> int:
	best = cutoff + 1
	for core in cores:
		d = nx.shortest_path_length(ug, core, node)
		best = min(best, d)
	return best


def best_edge_score_between(graph: nx.MultiDiGraph, u: str, v: str) -> float:
	"""Best evidence score among all edges (both directions) between u and v."""
	return max(
		(_raw_edge_score(
			attrs, source_weight=algo_config.EDGE_SOURCE_WEIGHT,
			relation_weight=algo_config.EDGE_RELATION_WEIGHT,
		) for _, _, _, attrs, _ in edge_candidates(graph, u, v)),
		default=0.0,
	)
