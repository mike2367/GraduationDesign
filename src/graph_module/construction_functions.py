from __future__ import annotations

import logging
import re
from typing import Optional

import networkx as nx
import pandas as pd
from tqdm import tqdm

import graph_module.graph_config as config

from graph_module.utils.construction_utils import (
	add_gene,
	build_logger,
	ensure_node,
	graph_gene_symbols,
	graph_gene_symbols_prioritized,
	node_key,
)
from graph_module.utils.graph_vis import graph_vis

from graph_module.utils.resource_cache import (
	fetch_ensembl_symbol_to_ensg,
	fetch_intogen_drivers,
	fetch_omnipath_tf_regulation,
	fetch_opentargets_known_drugs_by_gene,
	fetch_reactome_pathways_by_gene,
	fetch_uniprot_gene_to_acc,
)


def build_base_graph(cfg=config, logger: Optional[logging.Logger] = None) -> nx.MultiDiGraph:
	graph = nx.MultiDiGraph(name="SL-KG")
	# IMPORTANT: keep a stable, low-bias ordering of seed genes.
	# - core genes first (from CORE_GENE_PAIRS order)
	# - then curated genes (from SL_PAIRS_COMMON file order)
	core_seen: set[str] = set()
	core_list: list[str] = []
	for a, b in cfg.CORE_GENE_PAIRS:
		for g in (a, b):
			g = str(g).strip() if g else ""
			if not g or g in cfg.NON_GENE_ENDPOINTS or g in core_seen:
				continue
			core_seen.add(g)
			core_list.append(g)

	cur_seen: set[str] = set(core_seen)
	cur_list: list[str] = []
	for row in cfg.SL_PAIRS_COMMON:
		for g in (row.get("gene_a"), row.get("gene_b")):
			g = str(g).strip() if g else ""
			if not g or g in cfg.NON_GENE_ENDPOINTS or g in cur_seen:
				continue
			cur_seen.add(g)
			cur_list.append(g)

	for gene in core_list:
		add_gene(graph, gene, source="seed", ensembl_id=cfg.ENSEMBL_LOOKUP.get(gene))
	for gene in cur_list:
		add_gene(graph, gene, source="curated", ensembl_id=cfg.ENSEMBL_LOOKUP.get(gene))

	# Seed genes are now in the graph; keep the rest of the build logic unchanged.

	for a, b in cfg.GENE_PAIRS:
		graph.add_edge(node_key("gene", a), node_key("gene", b), type="SL_pair", source="seed")
		graph.add_edge(node_key("gene", b), node_key("gene", a), type="SL_pair", source="seed")

	for row in cfg.SL_PAIRS_COMMON:
		a = row.get("gene_a")
		b = row.get("gene_b")
		if not a or not b or a in cfg.NON_GENE_ENDPOINTS or b in cfg.NON_GENE_ENDPOINTS:
			continue
		edge_attrs = {"type": "SL_pair", "source": "curated"}
		if row.get("context"):
			edge_attrs["context"] = row["context"]
		if row.get("note"):
			edge_attrs["note"] = row["note"]
		graph.add_edge(node_key("gene", a), node_key("gene", b), **edge_attrs)
		graph.add_edge(node_key("gene", b), node_key("gene", a), **edge_attrs)

	if logger:
		logger.info("Seed graph: %s nodes, %s edges", graph.number_of_nodes(), graph.number_of_edges())
	return graph


def expand_with_string(graph: nx.MultiDiGraph, cfg=config, logger: Optional[logging.Logger] = None):
	import stringdb

	df = stringdb.get_network(
		graph_gene_symbols(graph),
		species=cfg.STRING_SPECIES,
		required_score=int(cfg.STRING_REQUIRED_SCORE),
		add_nodes=int(cfg.STRING_ADD_NODES),
	)

	for _, row in tqdm(df.iterrows(), total=len(df), desc="STRING"):
		a_id = add_gene(graph, row.get("preferredName_A"), source="STRING")
		b_id = add_gene(graph, row.get("preferredName_B"), source="STRING")
		graph.add_edge(
			a_id,
			b_id,
			type="STRING_association",
			score=float(row.get("score")) if pd.notna(row.get("score")) else None,
			source="STRING",
		)
	if logger:
		logger.info("STRING ready: %s nodes %s edges", graph.number_of_nodes(), graph.number_of_edges())


def expand_with_omnipath(graph: nx.MultiDiGraph, cfg=config, logger: Optional[logging.Logger] = None, uniprot_client=None):
	# Use a simple on-disk cache for symbol->UniProt accession.
	# IMPORTANT: avoid alphabetical truncation bias when applying the cap.
	max_genes = int(cfg.MAX_GENES_FOR_UNIPROT_IN_OMNIPATH)
	genes = graph_gene_symbols_prioritized(graph)[:max_genes]
	gene_to_acc = fetch_uniprot_gene_to_acc(
		genes,
		cache_path=cfg.UNIPROT_GENE_TO_ACC_CACHE_PATH,
		logger=logger,
	)

	op = pd.read_csv(cfg.OMNIPATH_TSV, sep="\t")
	src_col = "source" if "source" in op.columns else op.columns[0]
	tgt_col = "target" if "target" in op.columns else op.columns[1]
	acc_set = set(gene_to_acc.values())
	scoped = op.loc[
		op[src_col].str.upper().isin(acc_set) | op[tgt_col].str.upper().isin(acc_set)
	]
	max_edges = int(cfg.MAX_OMNIPATH_EDGES)
	for _, row in tqdm(scoped.head(max_edges).iterrows(), total=min(len(scoped), max_edges), desc="OmniPath"):
		sa = str(row[src_col]).upper()
		sb = str(row[tgt_col]).upper()
		s_node = ensure_node(graph, "protein", sa, accession=sa, source="OmniPath")
		t_node = ensure_node(graph, "protein", sb, accession=sb, source="OmniPath")
		graph.add_edge(s_node, t_node, type="OmniPath_interaction", source="OmniPath")

	for gene, acc in gene_to_acc.items():
		g_node = node_key("gene", gene)
		p_node = ensure_node(graph, "protein", acc, accession=acc, source="OmniPath")
		graph.add_edge(g_node, p_node, type="encodes", source="UniProt/OmniPath")

	if logger:
		logger.info("OmniPath ready: %s nodes %s edges", graph.number_of_nodes(), graph.number_of_edges())


def add_tf_regulation(graph: nx.MultiDiGraph, cfg=config, logger: Optional[logging.Logger] = None):
	"""Add TF->target regulation edges (directed, mechanistic).

	This is designed to add *within-graph* regulatory edges only (both endpoints
	already in the KG gene set), to avoid exploding the graph.
	"""
	if not bool(getattr(cfg, "ADD_TF_REGULATION_TO_KG", True)):
		return

	genes = list(graph_gene_symbols(graph))
	gene_set = {g.strip().upper() for g in genes if isinstance(g, str) and g.strip()}
	if not gene_set:
		return

	# Some OmniPath TF exports are keyed by UniProt accessions (not gene symbols).
	# If OmniPath expansion ran, we can map accessions -> gene symbols via encodes edges.
	acc_to_gene: dict[str, str] = {}
	allowed_ids = set(gene_set)
	for u, v, attrs in graph.edges(data=True):
		if str((attrs or {}).get("type")) != "encodes":
			continue
		u_type = str((graph.nodes.get(u, {}) or {}).get("type"))
		v_type = str((graph.nodes.get(v, {}) or {}).get("type"))
		if u_type != "gene" or v_type != "protein":
			continue
		gene_sym = str((graph.nodes.get(u, {}) or {}).get("symbol") or "").strip().upper()
		acc = str((graph.nodes.get(v, {}) or {}).get("accession") or "").strip().upper()
		if not acc:
			# fallback: node id like "protein:P01106"
			vs = str(v)
			if vs.startswith("protein:"):
				acc = vs.split(":", 1)[1].strip().upper()
		if gene_sym and acc:
			acc_to_gene[acc] = gene_sym
			allowed_ids.add(acc)

	allow_external_tf = bool(getattr(cfg, "TF_REGULATION_ALLOW_EXTERNAL_TF", True))
	rows = fetch_omnipath_tf_regulation(
		# Allow filtering by either gene symbols or UniProt accessions (for TSV exports).
		genes=sorted(allowed_ids),
		# IMPORTANT: some OmniPath regulation TSVs use UniProt accessions in the
		# `target` column. Filtering only by gene symbols would drop all rows
		# before we can map accessions -> symbols via `acc_to_gene`.
		targets=sorted(allowed_ids),
		tsv_path=getattr(cfg, "OMNIPATH_TF_TSV"),
		use_python_client=bool(getattr(cfg, "OMNIPATH_TF_USE_PYTHON_CLIENT", True)),
		dorothea_levels=getattr(cfg, "OMNIPATH_TF_DOROTHEA_LEVELS", ("A", "B", "C")),
		max_edges=int(getattr(cfg, "OMNIPATH_TF_MAX_EDGES", 100_000)),
		allow_external_tf=allow_external_tf,
		logger=logger,
	)
	if not rows:
		if logger:
			logger.info("TF regulation: no edges added")
		return

	_uniprot_acc_6 = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$")
	_uniprot_acc_6_sp = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$")
	_uniprot_acc_10 = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{8}$")
	_uniprot_acc_a0a = re.compile(r"^A0A[0-9A-Z]{7}$")

	def _looks_like_uniprot_accession(s: str) -> bool:
		s = str(s or "").strip().upper()
		if not s:
			return False
		# Prefer precise UniProt accession formats over heuristics to avoid
		# misclassifying gene symbols like SLC2A1.
		return bool(
			_uniprot_acc_6_sp.match(s)
			or _uniprot_acc_6.match(s)
			or _uniprot_acc_a0a.match(s)
			or _uniprot_acc_10.match(s)
		)

	max_per_target = int(getattr(cfg, "TF_REGULATION_MAX_TF_PER_TARGET", 12))
	by_target: dict[str, list[dict]] = {}
	for r in rows:
		tgt = str(r.get("target") or "").strip().upper()
		if not tgt:
			continue
		by_target.setdefault(tgt, []).append(r)

	def _level_rank(level: object) -> int:
		lv = str(level or "").strip().upper()
		lv = lv[0] if lv else ""
		return {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(lv, 9)

	added = 0
	for tgt, rs in tqdm(sorted(by_target.items()), desc="TF regulation"):
		# Keep best-confidence TFs per target deterministically.
		rs_sorted = sorted(
			rs,
			key=lambda r: (
				_level_rank(r.get("level")),
				str(r.get("source") or ""),
				str(r.get("tf") or ""),
			),
		)
		if max_per_target > 0:
			rs_sorted = rs_sorted[:max_per_target]
		for r in rs_sorted:
			tf = str(r.get("tf") or "").strip().upper()
			if not tf or not tgt:
				continue
			tf_is_acc = _looks_like_uniprot_accession(tf)
			# Map UniProt accessions to gene symbols if needed.
			if tf not in gene_set and tf in acc_to_gene:
				tf = acc_to_gene[tf]
			if tgt not in gene_set and tgt in acc_to_gene:
				tgt = acc_to_gene[tgt]
			# If external TFs are allowed, create TF nodes even if not in the seed set.
			if tgt not in gene_set:
				continue
			# Handle TF IDs robustly:
			# - If it's a gene symbol (or mapped from accession): keep as gene node.
			# - If it's a UniProt accession and unmapped: represent as protein node.
			src_node = None
			if tf in gene_set:
				src_node = node_key("gene", tf)
				# Mark TF nodes for visualization grouping (keep type=gene).
				if graph.has_node(src_node):
					graph.nodes[src_node]["is_tf"] = True
			else:
				if not allow_external_tf:
					continue
				if tf_is_acc:
					# Create TF protein node (accession) when no gene-symbol mapping exists.
					src_node = ensure_node(
						graph,
						"protein",
						tf,
						accession=tf,
						source="DoRothEA/OmniPath",
						is_tf=True,
					)
				else:
					# Add TF gene node (symbol) to KG.
					add_gene(graph, tf, source="DoRothEA/OmniPath")
					gene_set.add(tf)
					src_node = node_key("gene", tf)
					if graph.has_node(src_node):
						graph.nodes[src_node]["is_tf"] = True
			if not src_node:
				continue
			attrs = {
				"type": "TF_regulates",
				"source": "DoRothEA/OmniPath",
			}
			src = r.get("source")
			if src:
				attrs["source"] = str(src)
			sign = r.get("sign")
			if sign:
				attrs["sign"] = str(sign)
			level = r.get("level")
			if level:
				attrs["level"] = str(level)
			graph.add_edge(src_node, node_key("gene", tgt), **attrs)
			added += 1
			if added >= int(getattr(cfg, "OMNIPATH_TF_MAX_EDGES", 100_000)):
				break
		if added >= int(getattr(cfg, "OMNIPATH_TF_MAX_EDGES", 100_000)):
			break

	if logger:
		logger.info("TF regulation ready: %d edges", added)


def add_depmap_context(graph: nx.MultiDiGraph, cfg=config, logger: Optional[logging.Logger] = None):
	dep = pd.read_csv(cfg.DEPMAP_GENE_EFFECT, index_col=0)
	if dep.shape[1] > 0:
		dep = dep.T if any(str(c).upper().startswith("ACH-") for c in dep.columns[:5]) else dep
	dep.columns = [str(c).split(" ")[0] for c in dep.columns]

	genes = [g for g in graph_gene_symbols_prioritized(graph) if g in dep.columns]
	max_genes = int(cfg.MAX_GENES_FOR_DEPMAP)
	if max_genes:
		genes = genes[:max_genes]
	corr = dep[genes].corr()
	edges = []
	for i, ga in enumerate(tqdm(genes, desc="DepMap")):
		for gb in genes[i + 1 :]:
			val = corr.loc[ga, gb]
			if pd.notna(val) and abs(val) >= cfg.DEPMAP_MIN_ABS_CORR:
				edges.append((ga, gb, float(val)))

	edges.sort(key=lambda x: abs(x[2]), reverse=True)
	for ga, gb, val in edges[: cfg.DEPMAP_MAX_EDGES]:
		graph.add_edge(node_key("gene", ga), node_key("gene", gb), type="DepMap_codependency", corr=val, source="DepMap")
		graph.add_edge(node_key("gene", gb), node_key("gene", ga), type="DepMap_codependency", corr=val, source="DepMap")

	if logger:
		logger.info("DepMap ready: %s edges", len(edges) * 2)



def add_reactome_pathways(graph: nx.MultiDiGraph, cfg=config, logger: Optional[logging.Logger] = None):
	genes = graph_gene_symbols_prioritized(graph)[: int(cfg.MAX_GENES_FOR_REACTOME)]
	core_genes = {str(g).strip().upper() for pair in cfg.CORE_GENE_PAIRS for g in pair if g}
	max_pw_other = int(cfg.MAX_PATHWAYS_PER_GENE)
	max_pw_core = int(getattr(cfg, "MAX_PATHWAYS_PER_CORE_GENE", max_pw_other))

	# Fetch pathways for core genes with a larger cap to preserve functional coverage.
	core_list = [g for g in genes if str(g).strip().upper() in core_genes]
	other_list = [g for g in genes if str(g).strip().upper() not in core_genes]
	cache = {}
	if other_list:
		cache.update(
			fetch_reactome_pathways_by_gene(
				other_list,
				cache_path=cfg.REACTOME_CACHE_PATH,
				max_pathways_per_gene=max_pw_other,
				logger=logger,
			)
		)
	if core_list:
		cache.update(
			fetch_reactome_pathways_by_gene(
				core_list,
				cache_path=cfg.REACTOME_CACHE_PATH,
				max_pathways_per_gene=max_pw_core,
				logger=logger,
			)
		)

	for gene in genes:
		paths = cache.get(gene) or []
		cap = max_pw_core if str(gene).strip().upper() in core_genes else max_pw_other
		for p in (paths[:cap] if cap > 0 else paths):
			pnode = ensure_node(graph, "pathway", str(p["id"]), name=str(p.get("name") or ""), source="Reactome")
			edge_attrs = {"type": "in_pathway", "source": "Reactome"}
			# Carry any evidence-like fields through to the edge for ranking.
			for k in ("entities", "entities_fdr", "fdr", "pValue", "p_value"):
				if k in p and p.get(k) is not None:
					edge_attrs[k] = p.get(k)
			graph.add_edge(node_key("gene", gene), pnode, **edge_attrs)

	if logger:
		logger.info("Reactome ready")


def add_cbioportal_context(graph: nx.MultiDiGraph, cfg=config, logger: Optional[logging.Logger] = None):
	if not bool(getattr(cfg, "ADD_CBIOPORTAL_TO_KG", True)):
		return
	# This project previously used a stub implementation (connecting every gene
	# to every cohort). That creates strong, misleading hubs and downstream bias.
	# Keep it opt-in.
	if not bool(getattr(cfg, "CBIOPORTAL_STUB_MODE", False)):
		if logger:
			logger.info("cBioPortal: skipped (stub mode disabled)")
		return
	genes = graph_gene_symbols_prioritized(graph)[: cfg.MAX_GENES_FOR_CBIOPORTAL]

	for cohort_label in tqdm(cfg.CBIO_STUDY_CANDIDATES, desc="cBioPortal cohorts"):
		cohort_node = node_key("cohort", cohort_label)
		graph.add_node(cohort_node, type="cohort", name=cohort_label, source="cBioPortal")
		for sym in genes:
			graph.add_edge(node_key("gene", sym), cohort_node, type="mutated_in", mutations=1, source="cBioPortal", context=cohort_label)

	# Drop cohort nodes that connect to every gene or are isolated
	all_genes = {node_key("gene", g) for g in graph_gene_symbols(graph)}
	for cohort_label in list(cfg.CBIO_STUDY_CANDIDATES.keys()):
		cohort_node = node_key("cohort", cohort_label)
		if not graph.has_node(cohort_node):
			continue
		neighbors = {u for u, _ in graph.in_edges(cohort_node)} | {v for _, v in graph.out_edges(cohort_node)}
		gene_nbs = {n for n in neighbors if str(graph.nodes.get(n, {}).get("type")) == "gene"}
		if not neighbors or (all_genes and gene_nbs >= all_genes):
			graph.remove_node(cohort_node)
			if logger:
				logger.info("Dropped cohort %s (isolated or connected to all genes)", cohort_label)

	if logger:
		logger.info("cBioPortal stub ready")


def add_cancer_driver_context(graph: nx.MultiDiGraph, cfg=config, logger: Optional[logging.Logger] = None):
	"""Add cancer driver annotations as gene->cohort edges.

	Unlike SL-pair edges, driver annotations provide *contextual relevance*:
	"gene is a driver in tumor type X".
	"""
	if not bool(getattr(cfg, "ADD_CANCER_DRIVER_TO_KG", True)):
		return

	genes = list(graph_gene_symbols(graph))
	tumor_map = dict(getattr(cfg, "COHORT_TO_TUMOR_TYPE", {}))
	if not genes or not tumor_map:
		return

	# Ensure cohort nodes exist (usually created by cBioPortal step).
	for cohort_label in tumor_map.keys():
		cnode = node_key("cohort", str(cohort_label))
		if not graph.has_node(cnode):
			graph.add_node(cnode, type="cohort", name=str(cohort_label), source="IntOGen")

	tumor_types = sorted({str(v).strip().upper() for v in tumor_map.values() if v})
	driver = fetch_intogen_drivers(
		genes=genes,
		tumor_types=tumor_types,
		tsv_path=getattr(cfg, "INTOGEN_DRIVERS_TSV"),
		url=getattr(cfg, "INTOGEN_DRIVERS_URL", None),
		logger=logger,
	)
	if not driver:
		if logger:
			logger.info("IntOGen: no driver annotations loaded")
		return

	added = 0
	max_per = int(getattr(cfg, "INTOGEN_DRIVER_MAX_EDGES_PER_COHORT", 200))
	for cohort_label, tumor in tumor_map.items():
		tumor_code = str(tumor).strip().upper()
		gene_to_info = driver.get(tumor_code, {}) if isinstance(driver, dict) else {}
		if not isinstance(gene_to_info, dict) or not gene_to_info:
			continue

		cnode = node_key("cohort", str(cohort_label))
		count = 0
		for gene, info in gene_to_info.items():
			if max_per and count >= max_per:
				break
			g = str(gene).strip().upper()
			if not g:
				continue
			attrs = {
				"type": "driver_in",
				"source": "IntOGen",
				"context": str(cohort_label),
				"tumor_type": tumor_code,
			}
			if isinstance(info, dict):
				if info.get("role"):
					attrs["role"] = str(info.get("role"))
				if info.get("score") is not None:
					attrs["score"] = info.get("score")
			graph.add_edge(node_key("gene", g), cnode, **attrs)
			count += 1
			added += 1

	if logger:
		logger.info("IntOGen driver context ready: %d edges", added)

	# Drop cohort nodes that connect to every gene or are isolated
	all_genes = {node_key("gene", g) for g in graph_gene_symbols(graph)}
	for cohort_label in tumor_map.keys():
		cohort_node = node_key("cohort", str(cohort_label))
		if not graph.has_node(cohort_node):
			continue
		neighbors = {u for u, _ in graph.in_edges(cohort_node)} | {v for _, v in graph.out_edges(cohort_node)}
		gene_nbs = {n for n in neighbors if str(graph.nodes.get(n, {}).get("type")) == "gene"}
		if not neighbors or (all_genes and gene_nbs >= all_genes):
			graph.remove_node(cohort_node)
			if logger:
				logger.info("Dropped cohort %s (isolated or connected to all genes)", cohort_label)


def add_open_targets_drugs(graph: nx.MultiDiGraph, cfg=config, logger: Optional[logging.Logger] = None):
	# IMPORTANT: avoid alphabetical truncation bias when applying the cap.
	genes_all = graph_gene_symbols_prioritized(graph)
	cap = int(cfg.MAX_TARGETS_FOR_OT)
	genes = genes_all if cap <= 0 else genes_all[:cap]
	# Ensure we have symbol -> ENSG for OpenTargets.
	gene_to_ensg = (cfg.ENSEMBL_LOOKUP or {})
	if bool(cfg.ENSEMBL_AUTO_FETCH):
		gene_to_ensg = fetch_ensembl_symbol_to_ensg(
			genes,
			cache_path=cfg.ENSEMBL_LOOKUP_PATH,
			existing=gene_to_ensg,
			logger=logger,
		)
		setattr(cfg, "ENSEMBL_LOOKUP", gene_to_ensg)

	cache = fetch_opentargets_known_drugs_by_gene(
		genes,
		cache_path=cfg.OT_CACHE_PATH,
		gene_to_ensg=gene_to_ensg,
		graphql_url=cfg.OT_GRAPHQL_URL,
		force_refresh=bool(cfg.OT_FORCE_REFRESH),
		logger=logger,
	)

	for gene in tqdm(genes, desc="OpenTargets add"):
		rows = cache.get(gene) or []
		for row in rows[: int(cfg.KNOWN_DRUGS_SIZE)]:
			drug = row.get("drug")
			dnode = node_key("drug", str(drug["id"]))
			graph.add_node(
				dnode,
				type="drug",
				name=str(drug["name"]),
				phase=drug.get("maximumClinicalTrialPhase"),
				source="OpenTargets",
			)
			# Store phase on the edge (not just the node) so edge_score can use it
			phase_val = drug.get("maximumClinicalTrialPhase")
			graph.add_edge(
				dnode,
				node_key("gene", gene),
				type="targets",
				phase=phase_val,
				mechanism=row.get("mechanismOfAction") or row.get("mechanism"),
				disease=(row.get("disease") or {}).get("name") if isinstance(row.get("disease"), dict) else row.get("disease"),
				source="OpenTargets",
			)

	if logger:
		logger.info("OpenTargets cache ready")
__all__ = [
	"add_cancer_driver_context",
	"add_depmap_context",
	"add_tf_regulation",
	"add_reactome_pathways",
	"add_cbioportal_context",
	"add_open_targets_drugs",
	"add_gene",
	"build_base_graph",
	"build_logger",
	"ensure_node",
	"expand_with_omnipath",
	"expand_with_string",
	"graph_gene_symbols",
	"graph_vis",
	"node_key",
]
