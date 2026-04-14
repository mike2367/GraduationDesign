from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx

import graph_module.graph_config as config


def graph_vis(graph: nx.MultiDiGraph, html_path: Path, title: Optional[str] = None, ablation: Optional[str] = None):
	"""Write an interactive HTML visualization.

	This mirrors the working PyVis approach used in playground/graph_construct.ipynb
	(i.e., `net = Network(...); net.set_options(...); net.write_html(...)`).
	"""

	from pyvis.network import Network

	def _edge_color_spec(value: object, fallback: str) -> object:
		"""Return a vis-network compatible `edge.color` value.

		vis-network accepts either:
		- a string color (e.g. "#999")
		- or an object with keys like {color, highlight, hover}.

		PyVis will JSON-serialize this into the exported HTML.
		"""
		if isinstance(value, dict):
			base = value.get("color")
			if isinstance(base, str) and base.strip():
				out: dict[str, str] = {"color": base}
				hl = value.get("highlight")
				if isinstance(hl, str) and hl.strip():
					out["highlight"] = hl
				hv = value.get("hover")
				if isinstance(hv, str) and hv.strip():
					out["hover"] = hv
				return out
		if isinstance(value, str) and value.strip():
			return value
		return fallback

	# Edge styling (colors come from config; fall back to non-black defaults)
	DEFAULT_EDGE_COLOR = _edge_color_spec(config.VIS_EDGE_COLORS.get("default"), "#999999")
	CTX_EDGE_COLOR = _edge_color_spec(config.VIS_EDGE_COLORS.get("SL_pair_context"), "#00ffdd")
	MUT_EDGE_COLOR = _edge_color_spec(config.VIS_EDGE_COLORS.get("mutated_in"), "#d21bd8")
	SL_MUT_EDGE_COLOR = _edge_color_spec(config.VIS_EDGE_COLORS.get("SL_pair_mutated"), "#d21bd8")
	HIGHLIGHT_EDGE_WIDTH = float(config.VIS_HIGHLIGHT_EDGE_WIDTH)
	CTX_EDGE_WIDTH = float(config.VIS_CONTEXT_EDGE_WIDTH)
	SL_EDGE_WIDTH = float(config.VIS_SL_EDGE_WIDTH)
	DEFAULT_EDGE_WIDTH = float(config.VIS_DEFAULT_EDGE_WIDTH)

	def _node_tooltip(nid: str, attrs: dict) -> str:
		lines: list[str] = [f"<b>{nid}</b>"]
		t = attrs.get("type")
		if t:
			lines.append(f"type: {t}")

		if t == "gene":
			for k in [
				"symbol",
				"ensembl_gene_id",
				"ensembl_biotype",
				"ensembl_contig",
				"ensembl_start",
				"ensembl_end",
				"ensembl_strand",
			]:
				v = attrs.get(k)
				if v is not None:
					lines.append(f"{k}: {v}")
			# Cached neighbor gene symbols (GraphML-safe CSV string)
			nbs = attrs.get("neighbor_genes")
			if isinstance(nbs, str) and nbs.strip():
				parts = [p.strip() for p in nbs.split(",") if p.strip()]
				if parts:
					lines.append(f"neighbor_genes ({len(parts)}): {', '.join(parts)}")
		else:
			for k in ["name", "accession", "phase", "source"]:
				v = attrs.get(k)
				if v is not None:
					lines.append(f"{k}: {v}")

		return "<br>".join(lines)

	def _edge_tooltip(u: str, v: str, attrs: dict) -> str:
		lines: list[str] = []
		et = attrs.get("type") or attrs.get("edge_type") or attrs.get("relation")
		lines.append(f"<b>{u} &rarr; {v}</b>")
		if et:
			lines.append(f"type: {et}")
		for k in (
			"context",
			"cohort",
			"condition",
			"mutations",
			"mechanism",
			"disease",
			"score",
			"corr",
			"sign",
			"source",
		):
			if k in attrs and attrs.get(k) is not None:
				lines.append(f"{k}: {attrs.get(k)}")
		return "<br>".join(lines)

	def _is_mutated_edge(attrs: dict) -> bool:
		et = attrs.get("type") or attrs.get("edge_type") or attrs.get("relation")
		return et == "mutated_in" or attrs.get("mutations") is not None

	def _is_context_edge(attrs: dict) -> bool:
		return any(attrs.get(k) is not None for k in ("context", "cohort", "condition"))

	def _edge_style(attrs: dict) -> dict:
		etype = attrs.get("type") or attrs.get("edge_type") or attrs.get("relation")
		if etype == "SL_pair":
			if _is_mutated_edge(attrs) or attrs.get("note"):
				return {"color": SL_MUT_EDGE_COLOR, "width": SL_EDGE_WIDTH, "dashes": False}
			return {"color": CTX_EDGE_COLOR, "width": SL_EDGE_WIDTH, "dashes": False}
		if _is_mutated_edge(attrs):
			return {"color": MUT_EDGE_COLOR, "width": HIGHLIGHT_EDGE_WIDTH, "dashes": False}
		if etype in config.VIS_EDGE_COLORS:
			return {"color": _edge_color_spec(config.VIS_EDGE_COLORS.get(etype), "#999999"), "width": DEFAULT_EDGE_WIDTH, "dashes": False}
		if _is_context_edge(attrs):
			return {"color": CTX_EDGE_COLOR, "width": CTX_EDGE_WIDTH, "dashes": True}
		# Unknown edge types: use a neutral default color (avoid vis.js default black).
		return {"color": DEFAULT_EDGE_COLOR, "width": DEFAULT_EDGE_WIDTH, "dashes": False}

	# Use inline CDN resources to avoid repeated "cdn_resources is 'local'" warnings.
	net = Network(height="1000px", width="100%", directed=True, notebook=True, cdn_resources="in_line")

	vis_opts = dict(config.VIS_OPTIONS or {})
	groups_opt = (
		{str(g): {"color": {"background": col.get("background"), "border": col.get("border")}} for g, col in config.VIS_GROUP_COLORS.items() if isinstance(col, dict)}
		if isinstance(config.VIS_GROUP_COLORS, dict)
		else {}
	)
	if groups_opt:
		vis_opts.setdefault("groups", {}).update(groups_opt)
	net.set_options(json.dumps(vis_opts))

	for n, attrs in graph.nodes(data=True):
		attrs = attrs or {}
		label = attrs.get("symbol") or attrs.get("name") or str(n).split(":")[-1]
		title_html = _node_tooltip(str(n), attrs)
		# Visualization group can differ from semantic node `type`.
		group = "regulation" if bool(attrs.get("is_tf")) else attrs.get("type")
		node_kwargs = {"label": label, "group": group, "title": title_html}
		# Apply configured group colors when available
		if group and isinstance(config.VIS_GROUP_COLORS, dict):
			col = config.VIS_GROUP_COLORS.get(group)
			if isinstance(col, dict):
				node_kwargs["color"] = {"background": col.get("background"), "border": col.get("border")}
		net.add_node(str(n), **node_kwargs)

	for u, v, attrs in graph.edges(data=True):
		attrs = attrs or {}
		title_html = _edge_tooltip(str(u), str(v), attrs)
		style = _edge_style(attrs)
		edge_kwargs = {
			"label": attrs.get("type", "") or attrs.get("edge_type", "") or attrs.get("relation", ""),
			"title": title_html,
			"dashes": bool(style.get("dashes")),
			"width": float(style.get("width", 1.0)),
		}
		if style.get("color") is not None:
			edge_kwargs["color"] = style.get("color")
		net.add_edge(str(u), str(v), **edge_kwargs)

	html_path.parent.mkdir(parents=True, exist_ok=True)
	net.write_html(str(html_path))


__all__ = ["graph_vis"]
