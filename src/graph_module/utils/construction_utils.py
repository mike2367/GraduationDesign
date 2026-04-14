from __future__ import annotations

import logging
from typing import Iterable, Optional

import networkx as nx

import graph_module.graph_config as config


def _norm_type(val: object) -> str:
	return str(val or "unknown").strip().lower()


def build_logger(name: str = config.LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
	logger = logging.getLogger(name)
	if not logger.handlers:
		handler = logging.StreamHandler()
		handler.setFormatter(logging.Formatter("%(message)s"))
		logger.addHandler(handler)
	logger.setLevel(level)
	logger.propagate = False
	return logger


def node_key(prefix: str, identifier: str) -> str:
	return f"{prefix}:{identifier}"


def ensure_node(
	graph: nx.MultiDiGraph,
	prefix: str,
	identifier: str,
	ensembl_id: Optional[str] = None,
	**attrs,
) -> str:
	key = node_key(prefix, identifier)
	attrs.setdefault("type", prefix)
	attrs["type"] = _norm_type(attrs.get("type"))
	if prefix == "gene":
		attrs.setdefault("symbol", identifier)
		if ensembl_id:
			attrs.setdefault("ensembl_id", ensembl_id)

	if not graph.has_node(key):
		graph.add_node(key, **attrs)
	else:
		if "type" not in graph.nodes[key]:
			graph.nodes[key].update(attrs)

	return key


def add_gene(
	graph: nx.MultiDiGraph,
	symbol: str,
	source: str = "seed",
	ensembl_id: Optional[str] = None,
) -> str:
	return ensure_node(graph, "gene", symbol, symbol=symbol, source=source, ensembl_id=ensembl_id)


def graph_gene_symbols(graph: nx.MultiDiGraph) -> Iterable[str]:
	return sorted(
		attrs.get("symbol")
		for _, attrs in graph.nodes(data=True)
		if str(attrs.get("type") or "").strip().lower() == "gene" and attrs.get("symbol")
	)


def graph_gene_symbols_prioritized(graph: nx.MultiDiGraph) -> list[str]:
	seed: list[str] = []
	curated: list[str] = []
	other: list[str] = []
	seen: set[str] = set()
	for _nid, attrs in graph.nodes(data=True):
		attrs = attrs or {}
		if str(attrs.get("type") or "").strip().lower() != "gene":
			continue
		sym = attrs.get("symbol")
		if not sym:
			continue
		sym = str(sym)
		if sym in seen:
			continue
		seen.add(sym)
		src = str(attrs.get("source") or "").strip().lower()
		if src == "seed":
			seed.append(sym)
		elif src == "curated":
			curated.append(sym)
		else:
			other.append(sym)
	return seed + curated + other


__all__ = [
	"add_gene",
	"build_logger",
	"ensure_node",
	"graph_gene_symbols",
	"graph_gene_symbols_prioritized",
	"node_key",
]
