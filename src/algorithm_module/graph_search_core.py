from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import networkx as nx

from algorithm_module import algo_config


@dataclass(frozen=True)
class InferenceStep:
	src: str
	dst: str
	edge_type: str
	edge_source: str
	direction: str = "unknown"
	edge_key: Optional[str] = None
	attrs: Optional[dict] = None


@dataclass(frozen=True)
class InferencePath:
	nodes: Tuple[str, ...]
	steps: Tuple[InferenceStep, ...]
	path_probability: float
	ppr_score: float
	sources_in_path: Tuple[str, ...]
	source_switches: int
	rank_score: float = 0.0

	def to_dict(self) -> dict:
		return {
			"nodes": list(self.nodes),
			"steps": [
				{
					"src": s.src,
					"dst": s.dst,
					"edge_type": s.edge_type,
					"edge_source": s.edge_source,
					"direction": s.direction,
					"edge_key": s.edge_key,
					"attrs": s.attrs or {},
				}
				for s in self.steps
			],
			"path_probability": self.path_probability,
			"ppr_score": self.ppr_score,
			"rank_score": self.rank_score,
			"sources_in_path": list(self.sources_in_path),
			"source_switches": self.source_switches,
		}


def gene_node(symbol: str) -> str:
	return f"gene:{symbol.strip()}"


def undirected_degrees(graph: nx.MultiDiGraph) -> Dict[str, int]:
	return {n: graph.to_undirected(as_view=True).degree(n) for n in graph.nodes}


def path_probability(graph: nx.MultiDiGraph, node_path: Sequence[str], degree: Mapping[str, int]) -> float:
	prob = 1.0
	for u, v in zip(node_path, node_path[1:]):
		if not (graph.has_edge(u, v) or graph.has_edge(v, u)):
			return 0.0
		prob *= 1.0 / max(1, degree.get(u, 0))
	return prob


def personalized_pagerank_vector(
	graph: nx.MultiDiGraph,
	src: str,
) -> Dict[str, float]:
	if src not in graph:
		return {}
	cache = graph.graph.setdefault("_ppr_cache", {})
	key = (src, algo_config.PPR_ALPHA, algo_config.PPR_MAX_ITER, algo_config.PPR_TOL)
	if key in cache:
		return cache[key]
	vec = dict(nx.pagerank(graph, alpha=algo_config.PPR_ALPHA, max_iter=algo_config.PPR_MAX_ITER, tol=algo_config.PPR_TOL, personalization={src: 1.0}))
	cache[key] = vec
	return vec


def personalized_pagerank_score(
	graph: nx.MultiDiGraph,
	src: str,
	dst: str,
) -> float:
	if src not in graph or dst not in graph:
		return 0.0
	return personalized_pagerank_vector(graph, src).get(dst, 0.0)


def calculate_necessity_score(
	graph: nx.MultiDiGraph,
	src: str,
	dst: str,
	intervention_node: str,
	baseline_ppr: Optional[float] = None,
) -> float:
	"""
	Calculates the necessity of a node for the communication between src and dst
	using Counterfactual Intervention (Node Removal) + PPR.
	"""
	if intervention_node not in graph or intervention_node in (src, dst):
		return 0.0

	if baseline_ppr is None:
		baseline_ppr = personalized_pagerank_score(graph, src, dst)

	if baseline_ppr <= 1e-9:
		return 0.0

	# 1. Create Counterfactual Graph (Intervention)
	g_cf = graph.copy()
	if "_ppr_cache" in g_cf.graph:
		# Important: Clear cache because graph structure changed
		del g_cf.graph["_ppr_cache"]
	
	if g_cf.has_node(intervention_node):
		g_cf.remove_node(intervention_node)

	# 2. Re-calculate PPR
	cf_ppr = personalized_pagerank_score(g_cf, src, dst)

	# 3. Compute Necessity
	# If cf_ppr drops to 0, delta = baseline, score = 1.0 (Fully Necessary)
	# If cf_ppr stays same, delta = 0, score = 0.0 (Not Necessary)
	delta = max(0.0, baseline_ppr - cf_ppr)
	return delta / baseline_ppr


__all__ = [
	"InferenceStep",
	"InferencePath",
	"gene_node",
	"path_probability",
	"personalized_pagerank_score",
	"personalized_pagerank_vector",
	"undirected_degrees",
]


