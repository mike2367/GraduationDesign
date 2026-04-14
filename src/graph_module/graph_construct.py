import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
	if p not in sys.path:
		sys.path.insert(0, p)
import json
import time
from typing import Dict, Iterable, Optional, Set

import networkx as nx
from tqdm import tqdm
from graph_module import graph_config as config
from graph_module.construction_functions import (
	add_cbioportal_context,
	add_cancer_driver_context,
	add_depmap_context,
	add_open_targets_drugs,
	add_reactome_pathways,
	add_tf_regulation,
	build_base_graph,
	build_logger,
	expand_with_omnipath,
	expand_with_string,
	graph_vis,
)


def _normalize_and_validate_node_types(graph: nx.MultiDiGraph, logger=None) -> None:
	for _n, _attrs in graph.nodes(data=True):
		if (t := _attrs.get("type")) is not None:
			_attrs["type"] = str(t).strip().lower()


def _annotate_khop_cache(graph: nx.MultiDiGraph, cfg=config):
	if cfg.CACHE_KHOP_K <= 0:
		return
	ug = graph.to_undirected(as_view=True)
	for n in list(graph.nodes()):
		seen = {n}
		frontier = {n}
		for _depth in range(cfg.CACHE_KHOP_K):
			frontier = set().union(*(ug.neighbors(cur) for cur in frontier)) - seen
			seen |= frontier
		neighbors = seen - {n}
		if neighbors:
			by_type = {}
			for nb in neighbors:
				by_type.setdefault(str(graph.nodes[nb].get("type") or "unknown").strip().lower(), []).append(nb)
			chosen_type = sorted(by_type, key=lambda t: (-len(by_type[t]), t))[0]
			cand = sorted(by_type[chosen_type])[:cfg.CACHE_KHOP_MAX_NEIGHBORS] if cfg.CACHE_KHOP_MAX_NEIGHBORS else sorted(by_type[chosen_type])
			graph.nodes[n]["khop_cache"] = {"k": cfg.CACHE_KHOP_K, "neighbors": cand, "chosen_type": chosen_type}


def _annotate_neighbor_gene_cache(graph: nx.MultiDiGraph, cfg=config):
	ug = graph.to_undirected(as_view=True)
	for n in graph.nodes():
		if str(((attrs := graph.nodes[n]) or {}).get("type") or "").strip().lower() == "gene":
			neighbor_syms = sorted(set(nb_attrs.get("symbol") or str(nb).split(":")[-1] for nb in ug.neighbors(n) if str((nb_attrs := graph.nodes.get(nb, {})).get("type") or "").strip().lower() == "gene"))
			if cfg.CACHE_KHOP_MAX_NEIGHBORS and len(neighbor_syms) > cfg.CACHE_KHOP_MAX_NEIGHBORS:
				neighbor_syms = neighbor_syms[:cfg.CACHE_KHOP_MAX_NEIGHBORS]
			graph.nodes[n]["neighbor_genes"] = ",".join(neighbor_syms)
			graph.nodes[n]["neighbor_gene_count"] = len(neighbor_syms)


def _ensure_dirs(cfg=config):
	cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
	cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)


PIPELINE_STEPS = (
	("string", expand_with_string),
	("omnipath", expand_with_omnipath),
	("tf_regulation", add_tf_regulation),
	("depmap", add_depmap_context),
	("reactome", add_reactome_pathways),
	("cbioportal", add_cbioportal_context),
	("cancer_driver", add_cancer_driver_context),
	("opentargets", add_open_targets_drugs),
)


def _build_graph(skip: Set[str], cfg=config, logger=None, clients: Optional[Dict[str, object]] = None) -> nx.MultiDiGraph:
	clients = clients or {}
	graph = build_base_graph(cfg, logger=logger)
	for name, fn in list(PIPELINE_STEPS):
		if name in skip:
			continue
		kwargs = {"cfg": cfg, "logger": logger}
		if name == "omnipath":
			kwargs["uniprot_client"] = clients.get("uniprot")
		fn(graph, **kwargs)
	# Validate/normalize node types before any cached annotations or exports.
	_normalize_and_validate_node_types(graph, logger=logger)
	# Always add cached neighbor-gene info for downstream inspection/visualization.
	_annotate_neighbor_gene_cache(graph, cfg=cfg)
	if bool(cfg.CACHE_KHOP_NEIGHBORS):
		_annotate_khop_cache(graph, cfg=cfg)
	return graph


def build_variant(name: str, skip: Iterable[str], cfg=config, logger=None, clients=None) -> nx.MultiDiGraph:
	graph = _build_graph(set(skip), cfg=cfg, logger=logger, clients=clients)
	graph.graph["ablation_name"] = name
	return graph


def _store_outputs(graph: nx.MultiDiGraph, name: str, cfg=config, logger=None):
	for _n, _attrs in graph.nodes(data=True):
		for k, v in list((_attrs or {}).items()):
			if v is None or isinstance(v, (str, int, float, bool)):
				continue
			_attrs[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple, set)) else str(v)
	for _u, _v, _attrs in graph.edges(data=True):
		for k, v in list((_attrs or {}).items()):
			if v is None or isinstance(v, (str, int, float, bool)):
				continue
			_attrs[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple, set)) else str(v)
	nx.write_graphml(graph, (cfg.OUT_DIR / "ablation_graphs") / f"{name}.graphml")
	graph_vis(graph, (cfg.OUT_DIR / "ablation_graphs") / f"{name}.html", ablation=name)


def main():
	logger = build_logger()
	_ensure_dirs(config)
	clients: Dict[str, object] = {}
	jobs = {
		"full": set(),
	}
	for name, skip in list(jobs.items()):
		print("=" * 30, name, "=" * 30)
		graph = build_variant(name, skip, cfg=config, logger=logger, clients=clients)
		_store_outputs(graph, name, cfg=config, logger=logger)


if __name__ == "__main__":
	t0 = time.time()
	main()
	print(f"Total time: {time.time() - t0:.2f}s")



