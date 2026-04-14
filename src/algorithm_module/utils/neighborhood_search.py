from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx

from algorithm_module import algo_config
from algorithm_module.graph_search_core import personalized_pagerank_vector
from algorithm_module.utils.graph_search_utils import best_edge_score_between, edge_candidates
from algorithm_module.utils.scoring import clamp01, edge_score, linear_rank_score
from algorithm_module.utils.semantic_neighbors import build_core_query_text, cosine, embed_texts, node_text


@dataclass(frozen=True)
class NeighborScore:
	node: str
	hops: int
	score: float
	features: Dict[str, float]


def khop_ranked_neighbors(
	graph: nx.MultiDiGraph,
	start: str,
	*,
	k: int,
	max_nodes: int,
	gene_a: Optional[str] = None,
	gene_b: Optional[str] = None,
	query_text: Optional[str] = None,
	weights: Optional[Mapping[str, float]] = None,
	use_prob_features: bool = True,
	traversal: str = "bfs",
	max_explored: Optional[int] = None,
) -> Tuple[Set[str], int, Set[str], List[NeighborScore]]:

	if start not in graph:
		return set(), 0, set(), []

	k_i = max(0, int(k))
	max_nodes_i = max(0, int(max_nodes))
	if k_i == 0 or max_nodes_i == 0:
		return set(), 0, set(), []

	wg = graph.to_undirected(as_view=True)

	# BFS/DFS bookkeeping.
	dist: Dict[str, int] = {start: 0}
	parent: Dict[str, str] = {}
	best_edge: Dict[str, float] = {start: 0.0}
	path_prob: Dict[str, float] = {start: 1.0}
	edge_types_seen: Set[str] = set()

	# Choose container for frontier based on traversal style.
	if traversal.lower() not in {"bfs", "dfs"}:
		raise ValueError("traversal must be 'bfs' or 'dfs'")
	frontier: List[str] = [start]

	def _pop_frontier() -> str:
		return frontier.pop(0) if traversal.lower() == "bfs" else frontier.pop()

	hops_used = 0
	explored_limit = None if max_explored is None else max(1, int(max_explored))

	while frontier:
		u = _pop_frontier()
		du = dist.get(u, 0)
		if du >= k_i:
			continue
		hops_used = max(hops_used, du + 1)

		neighbors = sorted(wg.neighbors(u), key=str)

		for v in neighbors:
			if v == start:
				continue

			cand_dist = du + 1
			if cand_dist > k_i:
				continue

			# Representative edge type for bookkeeping.
			cands = edge_candidates(graph, u, v)
			if cands:
				# Choose the best type deterministically using EDGE_TYPE_PRIORITY.
				def _etype_key(item: Tuple[str, str, str, dict, str]) -> Tuple[int, str, str]:
					_src, _dst, _key, attrs, dir_label = item
					etype = str((attrs or {}).get("type") or "unknown")
					return (algo_config.EDGE_TYPE_PRIORITY.get(etype, 99), dir_label, etype)

				best = sorted(cands, key=_etype_key)[0]
				edge_types_seen.add(str((best[3] or {}).get("type") or "unknown"))

			e = best_edge_score_between(graph, u, v)
			deg_u = max(1, int(wg.degree(u)))
			new_prob = float(path_prob.get(u, 1.0)) * (1.0 / float(deg_u))
			new_prob = clamp01(new_prob)

			if v not in dist:
				dist[v] = cand_dist
				parent[v] = u
				best_edge[v] = float(e)
				path_prob[v] = float(new_prob)
				frontier.append(v)
			else:
				# Allow improving parent choice when the same hop distance is found.
				if dist[v] == cand_dist:
					cur_e = float(best_edge.get(v, 0.0))
					if e > cur_e or (e == cur_e and str(u) < str(parent.get(v, "~"))):
						parent[v] = u
						best_edge[v] = float(e)
						path_prob[v] = float(new_prob)

			# Exploration guard (compute budget, not a preference).
			if explored_limit is not None and (len(dist) - 1) >= explored_limit:
				frontier.clear()
				break

	# Prepare optional PPR and semantic similarity.
	ppr: Dict[str, float] = {}
	if use_prob_features:
		ppr = {n: float(v) for n, v in personalized_pagerank_vector(graph, start).items()}

	if query_text is None:
		if gene_a is not None and gene_b is not None:
			query_text = build_core_query_text(gene_a, gene_b)
		else:
			query_text = str(start)

	sem: Dict[str, float] = {}
	# Semantic similarity is always enabled; negative cosines are clipped to zero.
	q_vec = embed_texts([str(query_text)], model_path=str(algo_config.SEMANTIC_NEIGHBOR_MODEL_PATH))[0]
	candidates = [n for n in dist.keys() if n != start]
	texts = [node_text(graph, n) for n in candidates]
	vecs = embed_texts(texts, model_path=str(algo_config.SEMANTIC_NEIGHBOR_MODEL_PATH))
	for node, vec in zip(candidates, vecs):
		sem[node] = clamp01(max(0.0, cosine(vec, q_vec)))

	weights_base = dict(algo_config.NEIGHBOR_RANK_WEIGHTS)
	if weights:
		weights_base.update({k: float(v) for k, v in weights.items() if k in weights_base})

	ranked: List[NeighborScore] = []
	for node, d in dist.items():
		if node == start:
			continue
		# Normalized features in [0, 1].
		f_hop = 1.0 - (max(0, int(d) - 1) / float(max(1, k_i)))
		f_edge = float(best_edge.get(node, 0.0))
		f_prob = float(path_prob.get(node, 0.0))
		f_ppr = float(ppr.get(node, 0.0))
		f_sem = float(sem.get(node, 0.0))

		features = {
			"hop": clamp01(f_hop),
			"evidence": clamp01(f_edge),
			"prob": clamp01(f_prob),
			"ppr": clamp01(f_ppr),
			"semantic": clamp01(f_sem),
		}
		enabled = {
			"hop": True,
			"evidence": True,
			"prob": bool(use_prob_features),
			"ppr": bool(use_prob_features),
			"semantic": True,
		}
		s, used_f, _used_w = linear_rank_score(features, weights=weights_base, enabled=enabled)
		ranked.append(NeighborScore(node=node, hops=int(d), score=float(s), features=dict(used_f)))

	ranked.sort(key=lambda r: (-r.score, r.hops, r.node))
	top = ranked[:max_nodes_i]
	selected = {r.node for r in top}
	return selected, hops_used, edge_types_seen, top


__all__ = [
	"NeighborScore",
	"khop_ranked_neighbors",
]
