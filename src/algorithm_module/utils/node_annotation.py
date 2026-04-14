from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx

from algorithm_module import algo_config
from algorithm_module import output_config as ocfg




def _load_json(path: Path, default):
	try:
		if not path.exists():
			return default
		return json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return default


def _save_json(path: Path, data) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _urlopen_json(url: str, *, timeout_seconds: int = 20) -> Optional[object]:
	req = urllib.request.Request(
		url,
		method="GET",
		headers={"Accept": "application/json", "User-Agent": "sl-kg/1.0 (urllib)"},
	)
	with urllib.request.urlopen(req, timeout=int(timeout_seconds)) as resp:
		raw = resp.read().decode("utf-8")
		return json.loads(raw) if raw else None


def _extract_mygene_best_hit(data: object) -> Optional[Mapping[str, object]]:
	if not isinstance(data, dict):
		return None
	hits = data.get("hits")
	if not isinstance(hits, list) or not hits:
		return None
	# Prefer highest score (mygene uses _score)
	best = None
	best_score = -1e18
	for h in hits:
		if not isinstance(h, dict):
			continue
		s = h.get("_score")
		try:
			sf = float(s) if s is not None else 0.0
		except Exception:
			sf = 0.0
		if sf > best_score:
			best_score = sf
			best = h
	return best


def _go_terms_from_mygene(hit: Mapping[str, object]) -> List[str]:
	out: List[str] = []
	go = hit.get("go")
	if not isinstance(go, dict):
		return out
	for aspect_key in ("BP", "MF", "CC"):
		entries = go.get(aspect_key)
		if isinstance(entries, dict):
			term = entries.get("term")
			if isinstance(term, str) and term.strip():
				out.append(term.strip())
		elif isinstance(entries, list):
			for e in entries:
				if not isinstance(e, dict):
					continue
				term = e.get("term")
				if isinstance(term, str) and term.strip():
					out.append(term.strip())
	# De-dup while preserving order
	seen = set()
	uniq = []
	for t in out:
		if t in seen:
			continue
		seen.add(t)
		uniq.append(t)
	max_terms = int(getattr(algo_config, "GO_ANNOTATION_MAX_TERMS", 5))
	return uniq[:max_terms]


def _extract_mygene_summary_name(hit: Mapping[str, object]) -> Tuple[str, str]:
	summary = hit.get("summary")
	name = hit.get("name")
	return (
		summary.strip() if isinstance(summary, str) else "",
		name.strip() if isinstance(name, str) else "",
	)


def _extract_uniprot_function(data: object) -> Tuple[Optional[str], Optional[str]]:
	"""Return (protein_name, function_text) from UniProtKB JSON."""
	if not isinstance(data, dict):
		return None, None

	protein_name = None
	pd = data.get("proteinDescription")
	if isinstance(pd, dict):
		rec = pd.get("recommendedName")
		if isinstance(rec, dict):
			full = rec.get("fullName")
			if isinstance(full, dict):
				val = full.get("value")
				if isinstance(val, str) and val.strip():
					protein_name = val.strip()

	func_text = None
	comments = data.get("comments")
	if isinstance(comments, list):
		for c in comments:
			if not isinstance(c, dict):
				continue
			if str(c.get("commentType") or "").upper() != "FUNCTION":
				continue
			texts = c.get("texts")
			if isinstance(texts, list) and texts:
				for t in texts:
					if not isinstance(t, dict):
						continue
					val = t.get("value")
					if isinstance(val, str) and val.strip():
						func_text = val.strip()
						break
				break

	return protein_name, func_text


def _extract_uniprot_search_accession(data: object) -> Optional[str]:
	"""Return the best UniProt primary accession from UniProt search JSON."""
	if not isinstance(data, dict):
		return None
	results = data.get("results")
	if not isinstance(results, list) or not results:
		return None
	for r in results:
		if not isinstance(r, dict):
			continue
		acc = r.get("primaryAccession")
		if isinstance(acc, str) and acc.strip():
			return acc.strip().upper()
	return None


_UNIPROT_ACC_6 = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$")
_UNIPROT_ACC_6_SP = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$")
_UNIPROT_ACC_10 = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{8}$")
_UNIPROT_ACC_A0A = re.compile(r"^A0A[0-9A-Z]{7}$")


def _looks_like_uniprot_accession(s: object) -> bool:
	v = str(s or "").strip().upper()
	if not v:
		return False
	return bool(
		_UNIPROT_ACC_6_SP.match(v)
		or _UNIPROT_ACC_6.match(v)
		or _UNIPROT_ACC_A0A.match(v)
		or _UNIPROT_ACC_10.match(v)
	)


def annotate_subgraph_nodes(
	sub: nx.MultiDiGraph,
	*,
	cache_path: Optional[Path] = None,
	mygene_timeout_seconds: int = 20,
	uniprot_timeout_seconds: int = 25,
	sleep_seconds: float = 0.05,
	logger=None,
) -> None:
	"""Annotate gene/protein nodes with short functional descriptions.

	Adds (when available):
	- gene: summary, gene_name, go_terms
	- protein: protein_name, function_summary

	Network behavior:
	- raises on network failure (timeouts/HTTP errors), for fail-fast debugging
	- cached to disk for reproducibility and speed
	"""
	if cache_path is None:
		cache_path = Path(str(getattr(ocfg, "SUBGRAPH_OUTPUT_DIR"))) / "node_annotation_cache.json"
	else:
		cache_path = Path(cache_path)
	cache = _load_json(cache_path, {})
	cache = cache if isinstance(cache, dict) else {}
	gene_cache = cache.get("gene") if isinstance(cache.get("gene"), dict) else {}
	prot_cache = cache.get("protein") if isinstance(cache.get("protein"), dict) else {}
	gene_uniprot_cache = cache.get("gene_uniprot") if isinstance(cache.get("gene_uniprot"), dict) else {}

	# Collect symbols/accessions.
	genes: List[str] = []	# symbols
	proteins: List[str] = []	# accessions
	tf_genes: List[str] = []	# TF symbols (gene nodes; want UniProt function like protein nodes)

	for n, attrs in sub.nodes(data=True):
		attrs = attrs or {}
		t = str(attrs.get("type") or "").strip().lower()
		if t == "gene":
			sym = attrs.get("symbol")
			sym = sym if isinstance(sym, str) and sym.strip() else (str(n).split(":", 1)[1] if str(n).startswith("gene:") else str(n))
			sym_s = str(sym).strip()
			genes.append(sym_s)
			# Robustness: sometimes TF regulation rows carry UniProt accessions.
			# If those leak in as gene symbols (e.g., Q04206), treat them as proteins
			# for function annotation.
			if _looks_like_uniprot_accession(sym_s):
				proteins.append(sym_s.upper())
			if bool(attrs.get("is_tf")):
				tf_genes.append(sym_s)
		elif t == "protein":
			acc = attrs.get("accession")
			if not (isinstance(acc, str) and acc.strip()):
				acc = str(n).split(":", 1)[1] if str(n).startswith("protein:") else None
			if isinstance(acc, str) and acc.strip():
				proteins.append(str(acc).strip().upper())

	genes = sorted({g for g in genes if g})
	proteins = sorted({p for p in proteins if p})
	tf_genes = sorted({g for g in tf_genes if g})

	# Fetch gene annotations from MyGene.info (per symbol).
	for sym in genes:
		if sym in gene_cache:
			continue
		q = urllib.parse.quote(f"symbol:{sym}")
		url = f"https://mygene.info/v3/query?q={q}&species=human&fields=summary,name,go"
		data = _urlopen_json(url, timeout_seconds=int(mygene_timeout_seconds))
		hit = _extract_mygene_best_hit(data)
		if hit is None:
			gene_cache[sym] = {}
			continue
		summary = hit.get("summary")
		name = hit.get("name")
		rows = {
			"summary": summary.strip() if isinstance(summary, str) else "",
			"gene_name": name.strip() if isinstance(name, str) else "",
			"go_terms": _go_terms_from_mygene(hit),
		}
		gene_cache[sym] = rows
		if sleep_seconds:
			time.sleep(float(sleep_seconds))

	# Fetch protein annotations from UniProt.
	for acc in proteins:
		if acc in prot_cache:
			continue
		url = f"https://rest.uniprot.org/uniprotkb/{urllib.parse.quote(acc)}.json"
		data = _urlopen_json(url, timeout_seconds=int(uniprot_timeout_seconds))
		protein_name, func = _extract_uniprot_function(data)
		row = {
			"protein_name": protein_name or "",
			"function_summary": func or "",
		}
		# Fallback: if UniProt is blocked/unavailable, try MyGene using the UniProt accession.
		# This yields a gene-style summary, but it's better than empty and works well
		# for protein nodes connected via encodes.
		if not row.get("protein_name") and not row.get("function_summary"):
			q = urllib.parse.quote(f"uniprot:{acc}")
			url2 = f"https://mygene.info/v3/query?q={q}&species=human&fields=summary,name,go"
			mdata = _urlopen_json(url2, timeout_seconds=int(mygene_timeout_seconds))
			hit = _extract_mygene_best_hit(mdata)
			if isinstance(hit, dict):
				summary, name = _extract_mygene_summary_name(hit)
				go = _go_terms_from_mygene(hit)
				if name:
					row["protein_name"] = name
				if summary:
					row["function_summary"] = summary
				if go:
					row["go_terms"] = go
		prot_cache[acc] = row
		if sleep_seconds:
			time.sleep(float(sleep_seconds))

	# For TF gene nodes, also attach UniProt function (so TFs get the same
	# function fields even when their protein nodes aren't present in the subgraph).
	for sym in tf_genes:
		key = str(sym).strip().upper()
		if key in gene_uniprot_cache:
			continue
		# UniProt search by exact gene symbol.
		q = f"(gene_exact:{key}) AND (organism_id:9606)"
		url = (
			"https://rest.uniprot.org/uniprotkb/search?"
			+ "query="
			+ urllib.parse.quote(q)
			+ "&format=json&size=1"
		)
		search_data = _urlopen_json(url, timeout_seconds=int(uniprot_timeout_seconds))
		acc = _extract_uniprot_search_accession(search_data)
		if not acc:
			gene_uniprot_cache[key] = {}
			continue
		data = _urlopen_json(
			f"https://rest.uniprot.org/uniprotkb/{urllib.parse.quote(acc)}.json",
			timeout_seconds=int(uniprot_timeout_seconds),
		)
		protein_name, func = _extract_uniprot_function(data)
		gene_uniprot_cache[key] = {
			"accession": acc,
			"protein_name": protein_name or "",
			"function_summary": func or "",
		}
		if sleep_seconds:
			time.sleep(float(sleep_seconds))

	# Apply to nodes.
	for n, attrs in sub.nodes(data=True):
		attrs = attrs or {}
		t = str(attrs.get("type") or "").strip().lower()
		if t == "gene":
			sym = attrs.get("symbol")
			sym = sym if isinstance(sym, str) and sym.strip() else (str(n).split(":", 1)[1] if str(n).startswith("gene:") else str(n))
			# If this "gene" symbol is actually a UniProt accession, apply protein annotations too.
			if _looks_like_uniprot_accession(sym):
				prow = prot_cache.get(str(sym).strip().upper(), {}) if isinstance(prot_cache, dict) else {}
				if isinstance(prow, dict):
					if prow.get("protein_name") and not attrs.get("protein_name"):
						attrs["protein_name"] = prow.get("protein_name")
					if prow.get("function_summary") and not attrs.get("function_summary"):
						attrs["function_summary"] = prow.get("function_summary")
			row = gene_cache.get(str(sym).strip(), {}) if isinstance(gene_cache, dict) else {}
			if isinstance(row, dict):
				if row.get("summary"):
					attrs["summary"] = row.get("summary")
				if row.get("gene_name") and not attrs.get("name"):
					attrs["name"] = row.get("gene_name")
				go_terms = row.get("go_terms")
				if isinstance(go_terms, list) and go_terms:
					max_terms = int(getattr(algo_config, "GO_ANNOTATION_MAX_TERMS", 5))
					attrs["go_terms"] = "; ".join(str(x) for x in go_terms[:max_terms])
			# TF gene nodes: fill UniProt-derived fields too.
			if bool(attrs.get("is_tf")) and (not attrs.get("function_summary") or not attrs.get("protein_name")):
				key = str(sym).strip().upper()
				prow = gene_uniprot_cache.get(key, {}) if isinstance(gene_uniprot_cache, dict) else {}
				if isinstance(prow, dict):
					if prow.get("protein_name") and not attrs.get("protein_name"):
						attrs["protein_name"] = prow.get("protein_name")
					if prow.get("function_summary") and not attrs.get("function_summary"):
						attrs["function_summary"] = prow.get("function_summary")
		elif t == "protein":
			acc = attrs.get("accession")
			if not (isinstance(acc, str) and acc.strip()):
				acc = str(n).split(":", 1)[1] if str(n).startswith("protein:") else ""
			row = prot_cache.get(str(acc).strip().upper(), {}) if isinstance(prot_cache, dict) else {}
			if isinstance(row, dict):
				if row.get("protein_name"):
					attrs["protein_name"] = row.get("protein_name")
				if row.get("function_summary"):
					attrs["function_summary"] = row.get("function_summary")
				# Optional: carry GO terms if available from fallback.
				go_terms = row.get("go_terms")
				if isinstance(go_terms, list) and go_terms and not attrs.get("go_terms"):
					max_terms = int(getattr(algo_config, "GO_ANNOTATION_MAX_TERMS", 5))
					attrs["go_terms"] = "; ".join(str(x) for x in go_terms[:max_terms])

	# Final fallback: for proteins still missing function_summary, try to use their
	# encoding gene's summary if an encodes edge exists in this subgraph.
	for n, attrs in sub.nodes(data=True):
		attrs = attrs or {}
		if str(attrs.get("type") or "").strip().lower() != "protein":
			continue
		if attrs.get("function_summary"):
			continue
		# Find incoming encodes edge gene -> protein.
		for u, _v, eattrs in sub.in_edges(n, data=True):
			etype = str((eattrs or {}).get("type") or "")
			if etype != "encodes":
				continue
			gattrs = sub.nodes.get(u, {}) or {}
			if str(gattrs.get("type") or "").strip().lower() != "gene":
				continue
			sym = gattrs.get("symbol") or (str(u).split(":", 1)[1] if str(u).startswith("gene:") else str(u))
			g_row = gene_cache.get(str(sym).strip(), {}) if isinstance(gene_cache, dict) else {}
			if isinstance(g_row, dict) and g_row.get("summary"):
				attrs["function_summary"] = g_row.get("summary")
				if g_row.get("gene_name") and not attrs.get("protein_name"):
					attrs["protein_name"] = g_row.get("gene_name")
				break

	# Persist cache.
	cache["gene"] = gene_cache
	cache["protein"] = prot_cache
	cache["gene_uniprot"] = gene_uniprot_cache
	_save_json(cache_path, cache)
	if logger:
		logger.info("node annotation cache updated: %s", str(cache_path))


__all__ = ["annotate_subgraph_nodes"]
