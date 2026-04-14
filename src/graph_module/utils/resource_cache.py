from __future__ import annotations

import concurrent.futures
import json
import random
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import graph_module.graph_config as cfg


def load_json(path: Path, default):
	if not path.exists():
		return default
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return default


def save_json(path: Path, data) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _host_resolves(host: str, port: int = 443) -> bool:
	try:
		socket.getaddrinfo(host, port)
		return True
	except Exception:
		return False


def _head_ok(url: str) -> bool:
	try:
		low = str(url).lower()
		if "reactome.org" in low:
			timeout = int(getattr(cfg, "REACTOME_HEAD_TIMEOUT_SECONDS", int(cfg.HTTP_HEAD_TIMEOUT_SECONDS)))
		elif "opentargets" in low:
			timeout = int(getattr(cfg, "OPENTARGETS_HEAD_TIMEOUT_SECONDS", int(cfg.HTTP_HEAD_TIMEOUT_SECONDS)))
		else:
			timeout = int(cfg.HTTP_HEAD_TIMEOUT_SECONDS)
		req = urllib.request.Request(url, method="HEAD")
		with urllib.request.urlopen(req, timeout=timeout):
			return True
	except Exception:
		return False


def _sleep_backoff(attempt: int) -> None:
	# exponential backoff with small jitter
	delay = min(float(cfg.HTTP_BACKOFF_CAP_SECONDS), float(cfg.HTTP_BACKOFF_BASE_SECONDS) * (2 ** max(0, attempt)))
	delay = delay * (0.85 + 0.3 * random.random())
	time.sleep(delay)


def _urlopen_json(
	url: str,
	*,
	method: str = "GET",
	data: Optional[bytes] = None,
	headers: Optional[dict] = None,
	logger=None,
	label: str = "request",
	sleep_seconds: float = 0.0,
) -> Optional[object]:
	# Keep the public API tiny: pick service-specific defaults based on the label.
	# (Callers should configure behavior via config.graph_config.)
	lab = str(label)
	if lab.startswith("Ensembl("):
		timeout_seconds = int(getattr(cfg, "ENSEMBL_REST_TIMEOUT_SECONDS", int(cfg.HTTP_DEFAULT_TIMEOUT_SECONDS)))
		retries = int(getattr(cfg, "ENSEMBL_REST_RETRIES", int(cfg.HTTP_DEFAULT_RETRIES)))
	elif lab.startswith("Reactome("):
		timeout_seconds = int(getattr(cfg, "REACTOME_REQUEST_TIMEOUT_SECONDS", int(cfg.HTTP_DEFAULT_TIMEOUT_SECONDS)))
		retries = int(getattr(cfg, "REACTOME_REQUEST_RETRIES", int(cfg.HTTP_DEFAULT_RETRIES)))
	elif lab.startswith("OpenTargets("):
		timeout_seconds = int(getattr(cfg, "OPENTARGETS_CACHE_TIMEOUT_SECONDS", int(cfg.HTTP_DEFAULT_TIMEOUT_SECONDS)))
		retries = int(getattr(cfg, "OPENTARGETS_CACHE_RETRIES", int(cfg.HTTP_DEFAULT_RETRIES)))
	else:
		timeout_seconds = int(cfg.HTTP_DEFAULT_TIMEOUT_SECONDS)
		retries = int(cfg.HTTP_DEFAULT_RETRIES)
	headers = dict(headers or {})
	headers.setdefault("Accept", "application/json")
	headers.setdefault("User-Agent", "sl-kg/1.0 (urllib)")

	last_err: Optional[Exception] = None
	for attempt in range(retries + 1):
		try:
			if sleep_seconds:
				time.sleep(float(sleep_seconds))
			req = urllib.request.Request(url, data=data, headers=headers, method=method)
			with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
				raw = resp.read().decode("utf-8")
				return json.loads(raw) if raw else None
		except urllib.error.HTTPError as e:
			last_err = e
			# Retry common transient codes (rate limit / gateway)
			code = int(e.code) if hasattr(e, "code") and e.code is not None else 0
			if code in {408, 429, 500, 502, 503, 504} and attempt < retries:
				_sleep_backoff(attempt)
				continue
			break
		except Exception as e:
			last_err = e
			if attempt < retries:
				_sleep_backoff(attempt)
				continue
			break

	if logger and last_err is not None:
		logger.warning("%s failed after retries: %s", label, last_err)
	return None


def fetch_ensembl_symbol_to_ensg(
	symbols: Iterable[str],
	*,
	cache_path: Path,
	existing: Optional[Dict[str, str]] = None,
	logger=None,
) -> Dict[str, str]:
	"""Fetch missing gene symbol -> ENSG mappings via Ensembl REST and save JSON.

	This is network-bound; keep max_workers modest to avoid rate limiting.
	"""
	lookup: Dict[str, str] = {}
	if isinstance(existing, dict):
		lookup.update({str(k): str(v) for k, v in existing.items() if k and v})
	cached = load_json(cache_path, {})
	if isinstance(cached, dict):
		lookup.update({str(k): str(v) for k, v in cached.items() if k and v})

	missing = [s.strip() for s in symbols if isinstance(s, str) and s.strip() and s.strip() not in lookup]
	if not missing:
		return lookup

	base = "https://rest.ensembl.org/xrefs/symbol/homo_sapiens/"

	def _fetch_one(symbol: str) -> Tuple[str, Optional[str]]:
		url = f"{base}{urllib.parse.quote(symbol)}?content-type=application/json"
		data = _urlopen_json(
			url,
			method="GET",
			logger=logger,
			label=f"Ensembl({symbol})",
		)
		for item in data:
			if not isinstance(item, dict):
				continue
			if item.get("type") == "gene":
				_id = item.get("id")
				if isinstance(_id, str) and _id.startswith("ENSG"):
					return symbol, _id

	if logger:
		logger.info("Ensembl: fetching %d missing symbol→ENSG", len(missing))

	with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(cfg.ENSEMBL_AUTO_FETCH_MAX_WORKERS))) as ex:
		futs = [ex.submit(_fetch_one, s) for s in missing]
		for fut in concurrent.futures.as_completed(futs):
			sym, ensg = fut.result()
			if ensg:
				lookup[sym] = ensg

	save_json(cache_path, lookup)
	return lookup


def fetch_reactome_pathways_by_gene(
	genes: Iterable[str],
	*,
	cache_path: Path,
	max_pathways_per_gene: Optional[int] = None,
	logger=None,
) -> Dict[str, list[dict]]:
	"""Fetch Reactome pathways for genes and save JSON mapping gene -> [{id,name,...}].

	`max_pathways_per_gene` overrides `cfg.MAX_PATHWAYS_PER_GENE` for this call.
	"""
	cache = load_json(cache_path, {})
	cache = cache if isinstance(cache, dict) else {}

	gene_list = [g.strip() for g in genes if isinstance(g, str) and g.strip()]
	to_fetch = [g for g in gene_list if g not in cache]

	# Preflight: if DNS is broken, skip network entirely.
	if to_fetch and not _host_resolves("reactome.org", 443):
		if logger:
			logger.warning("Reactome offline (DNS): using cache only")
		return cache

	# Optional HEAD probe to avoid spinning up per-gene retries when the host is up
	# in DNS but not reachable.
	if to_fetch and not _head_ok("https://reactome.org/AnalysisService/"):
		if logger:
			logger.warning("Reactome unreachable (HEAD probe): using cache only")
		return cache

	page_size = int(cfg.MAX_PATHWAYS_PER_GENE) if max_pathways_per_gene is None else int(max_pathways_per_gene)
	page_size = max(1, page_size)

	def _fetch_gene(gene: str) -> Tuple[str, list[dict]]:
		params = {
			"interactors": "false",
			"pageSize": str(int(page_size)),
			"page": "1",
			"sortBy": "ENTITIES_FDR",
			"order": "ASC",
			"species": "Homo sapiens",
			"resource": "TOTAL",
			"pValue": "1",
			"includeDisease": "true",
		}
		url = "https://reactome.org/AnalysisService/identifier/" + urllib.parse.quote(gene)
		url = url + "?" + urllib.parse.urlencode(params)
		res = _urlopen_json(
			url,
			method="GET",
			logger=logger,
			label=f"Reactome({gene})",
			sleep_seconds=float(cfg.REACTOME_REQUEST_SLEEP_SECONDS),
		)
		res = res if isinstance(res, dict) else {}
		pathways = res.get("pathways")
		pathways = pathways if isinstance(pathways, list) else []
		out: list[dict] = []
		for p in pathways[: int(page_size)]:
			if not isinstance(p, dict):
				continue
			pid = p.get("stId") or p.get("dbId")
			name = p.get("name") or ""
			if pid:
				row = {"id": str(pid), "name": str(name)}
				# Keep optional evidence-like fields if present for later ranking.
				for k in ("entities", "entities_fdr", "fdr", "pValue", "p_value"):
					if k in p and p.get(k) is not None:
						row[k] = p.get(k)
				out.append(row)
		return gene, out

	if logger:
		logger.info("Reactome: %d/%d genes cached", len(gene_list) - len(to_fetch), len(gene_list))

	if to_fetch:
		with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(cfg.REACTOME_MAX_WORKERS))) as ex:
			futs = [ex.submit(_fetch_gene, g) for g in to_fetch]
			for fut in concurrent.futures.as_completed(futs):
				gene, rows = fut.result()
				cache[gene] = rows
		save_json(cache_path, cache)

	return cache


def _read_tsv_rows(path: Path, *, max_rows: Optional[int] = None) -> List[Dict[str, str]]:
	import csv

	rows: List[Dict[str, str]] = []
	if not path.exists():
		return rows
	with path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f, delimiter="\t")
		for i, row in enumerate(reader):
			if max_rows is not None and i >= int(max_rows):
				break
			clean = {str(k): ("" if v is None else str(v)) for k, v in (row or {}).items() if k}
			rows.append(clean)
	return rows


def fetch_omnipath_tf_regulation(
	*,
	genes: Sequence[str],
	targets: Optional[Sequence[str]] = None,
	tsv_path: Path,
	use_python_client: bool,
	dorothea_levels: Sequence[str] = ("A", "B", "C"),
	max_edges: int = 100_000,
	allow_external_tf: bool = False,
	logger=None,
) -> List[Dict[str, object]]:
	"""Fetch TF→target regulation interactions.

	Returns a list of dicts with at least: tf, target, sign, source, level.

	Design:
	- If `tsv_path` exists: parse it.
	- Else if `use_python_client`: try OmniPath python client.
	- Else: return empty list.

	We intentionally *filter to the provided gene set* to prevent the KG from
	exploding in size.
	"""
	gene_set = {g.strip().upper() for g in genes if isinstance(g, str) and g.strip()}
	target_set = {t.strip().upper() for t in (targets or []) if isinstance(t, str) and t.strip()}
	# Backward-compatible behavior: if no explicit targets are provided, use the gene set.
	if not target_set:
		target_set = set(gene_set)
	if not target_set:
		return []

	def _norm_level(x: object) -> str:
		v = str(x or "").strip().upper()
		# DoRothEA confidence levels are typically A..E; allow strings like "A,B".
		if not v:
			return ""
		return v[0]

	levels_allowed = tuple(_norm_level(lv) for lv in dorothea_levels if str(lv).strip())
	levels_allowed = tuple(lv for lv in levels_allowed if lv)

	if tsv_path.exists():
		rows = _read_tsv_rows(tsv_path)
		out: List[Dict[str, object]] = []
		for row in rows:
			tf = (row.get("source_genesymbol") or row.get("tf") or row.get("source") or "").strip().upper()
			tgt = (row.get("target_genesymbol") or row.get("target") or "").strip().upper()
			if not tf or not tgt:
				continue
			# Default behavior: filter to within-graph edges (both endpoints in gene_set).
			# External-TF behavior: allow TF outside gene_set, but always filter on targets.
			if tgt not in target_set:
				continue
			if not bool(allow_external_tf):
				if tf not in gene_set or tgt not in gene_set:
					continue
			level = row.get("dorothea_level") or row.get("level") or row.get("confidence") or ""
			level_n = _norm_level(level)
			if levels_allowed and level_n and level_n not in levels_allowed:
				continue

			sign = row.get("sign") or row.get("effect") or row.get("consensus_direction") or row.get("direction") or ""
			# Common OmniPath exports include boolean stimulation/inhibition flags.
			is_stim = row.get("is_stimulation")
			is_inhib = row.get("is_inhibition")
			try:
				is_stim_i = int(float(is_stim)) if str(is_stim).strip() else 0
			except Exception:
				is_stim_i = 0
			try:
				is_inhib_i = int(float(is_inhib)) if str(is_inhib).strip() else 0
			except Exception:
				is_inhib_i = 0
			# Prefer explicit stimulation/inhibition flags over numeric consensus columns.
			if is_stim_i or is_inhib_i:
				if is_stim_i and not is_inhib_i:
					sign = "activation"
				elif is_inhib_i and not is_stim_i:
					sign = "inhibition"
				else:
					sign = "unknown"

			src = row.get("sources") or row.get("source_db") or row.get("resource") or "OmniPath"
			out.append({"tf": tf, "target": tgt, "sign": sign, "level": (level_n or str(level)), "source": src})
			if len(out) >= int(max_edges):
				break
		return out

	if not use_python_client:
		return []

	try:
		from omnipath.interactions import Dorothea  # type: ignore
	except Exception as e:
		if logger:
			logger.warning("TF regulation: omnipath python client unavailable: %s", e)
		return []

	levels = tuple(str(x).strip().upper() for x in dorothea_levels if str(x).strip())
	try:
		df = Dorothea().get(organism=9606, genesymbols=True, dorothea_levels=list(levels))
	except Exception as e:
		if logger:
			logger.warning("TF regulation: OmniPath request failed: %s", e)
		return []

	# The client typically returns a pandas DataFrame.
	try:
		iter_rows = df.itertuples(index=False)
	except Exception:
		return []

	out: List[Dict[str, object]] = []
	for row in iter_rows:
		# Common columns: source_genesymbol, target_genesymbol, is_stimulation, is_inhibition, sources
		attrs = getattr(row, "_asdict", None)
		data = attrs() if callable(attrs) else row._asdict() if hasattr(row, "_asdict") else {}
		tf = str(data.get("source_genesymbol") or "").strip().upper()
		tgt = str(data.get("target_genesymbol") or "").strip().upper()
		if not tf or not tgt:
			continue
		# The python client fetch is global; we still filter deterministically.
		if tgt not in target_set:
			continue
		if not bool(allow_external_tf):
			if tf not in gene_set or tgt not in gene_set:
				continue
		is_stim = int(data.get("is_stimulation") or 0)
		is_inhib = int(data.get("is_inhibition") or 0)
		if is_stim and not is_inhib:
			sign = "activation"
		elif is_inhib and not is_stim:
			sign = "inhibition"
		else:
			sign = "unknown"
		sources = data.get("sources")
		src = str(sources) if sources is not None else "DoRothEA"
		out.append({"tf": tf, "target": tgt, "sign": sign, "level": ",".join(levels), "source": src})
		if len(out) >= int(max_edges):
			break
	return out


def fetch_intogen_drivers(
	*,
	genes: Sequence[str],
	tumor_types: Sequence[str],
	tsv_path: Path,
	url: Optional[str] = None,
	logger=None,
) -> Dict[str, Dict[str, Dict[str, object]]]:
	"""Fetch (tumor_type -> gene -> driver_info) from an IntOGen driver TSV.

	We intentionally only keep entries for the requested tumor_types and genes.
	The TSV is expected to contain at least gene symbol and tumor type columns,
	but we try to be permissive about exact column names.
	"""
	gene_set = {g.strip().upper() for g in genes if isinstance(g, str) and g.strip()}
	if not gene_set:
		return {}
	tumor_set = {t.strip().upper() for t in tumor_types if isinstance(t, str) and t.strip()}
	if not tumor_set:
		return {}

	# If file missing and url provided, try to download it once.
	if not tsv_path.exists() and url:
		try:
			if logger:
				logger.info("IntOGen: downloading drivers TSV")
			req = urllib.request.Request(str(url), method="GET", headers={"User-Agent": "sl-kg/1.0"})
			with urllib.request.urlopen(req, timeout=int(cfg.HTTP_DEFAULT_TIMEOUT_SECONDS)) as resp:
				raw = resp.read()
			tsv_path.parent.mkdir(parents=True, exist_ok=True)
			tsv_path.write_bytes(raw)
		except Exception as e:
			if logger:
				logger.warning("IntOGen: download failed: %s", e)

	if not tsv_path.exists():
		return {}

	import csv

	out: Dict[str, Dict[str, Dict[str, object]]] = {t: {} for t in sorted(tumor_set)}
	with tsv_path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f, delimiter="\t")
		# Heuristic column names
		fields = [str(c or "") for c in (reader.fieldnames or [])]
		field_l = {c.lower(): c for c in fields}

		gene_col = field_l.get("gene") or field_l.get("symbol") or field_l.get("hgnc_symbol")
		if gene_col is None:
			# common in some releases
			gene_col = field_l.get("gene_symbol") or field_l.get("gene_name")
		tumor_col = field_l.get("tumor_type") or field_l.get("cancer_type") or field_l.get("cohort")
		role_col = field_l.get("role") or field_l.get("mode_of_action") or field_l.get("moa")
		score_col = (
			field_l.get("qvalue")
			or field_l.get("q_value")
			or field_l.get("qvalue_combination")
			or field_l.get("q_value_combination")
			or field_l.get("fdr")
			or field_l.get("pvalue")
		)

		for row in reader:
			if not isinstance(row, dict):
				continue
			gene = str((row.get(gene_col) if gene_col else "") or "").strip().upper()
			tumor_raw = str((row.get(tumor_col) if tumor_col else "") or "").strip().upper()
			tumor = tumor_raw
			# Handle common cohort encodings like "TCGA-BRCA" or "BRCA-US".
			if tumor.startswith("TCGA-"):
				tumor = tumor[5:]
			if tumor not in tumor_set and "-" in tumor:
				tumor = tumor.split("-", 1)[0]
			if not gene or not tumor:
				continue
			if gene not in gene_set or tumor not in tumor_set:
				continue
			info: Dict[str, object] = {}
			if role_col:
				info["role"] = str(row.get(role_col) or "")
			if score_col:
				info["score"] = row.get(score_col)
			# Keep a small slice of provenance fields if present.
			for k in (
				"n_mutations",
				"mutations",
				"n_samples",
				"samples",
				"method",
				"methods",
				"combined_significance",
			):
				ck = field_l.get(k)
				if ck and row.get(ck) is not None:
					info[k] = row.get(ck)
			out.setdefault(tumor, {})[gene] = info

	return out




def fetch_opentargets_known_drugs_by_gene(
	genes: Iterable[str],
	*,
	cache_path: Path,
	gene_to_ensg: Dict[str, str],
	graphql_url: str,
	force_refresh: bool = False,
	logger=None,
) -> Dict[str, object]:
	"""Fetch OpenTargets known drugs per gene and save JSON cache.

	Cache format (simple):
	- cache["gene_to_ensg"][symbol] = ENSG
	- cache[symbol] = rows (OpenTargets knownDrugs rows)
	"""
	cache = load_json(cache_path, {})
	cache = cache if isinstance(cache, dict) else {}

	cache_gene_to_ensg = cache.get("gene_to_ensg")
	cache_gene_to_ensg = cache_gene_to_ensg if isinstance(cache_gene_to_ensg, dict) else {}
	cache_gene_to_ensg.update({str(k): str(v) for k, v in (gene_to_ensg or {}).items() if k and v})
	cache["gene_to_ensg"] = cache_gene_to_ensg

	query = """
query GetDrugs($ensg: String!, $size: Int!) {
  target(ensemblId: $ensg) {
    knownDrugs(size: $size) {
      rows {
        drug { id name maximumClinicalTrialPhase }
        disease { name }
        mechanismOfAction
      }
    }
  }
}
"""

	# Preflight DNS for OpenTargets host to avoid spamming retries when offline.
	host = urllib.parse.urlparse(graphql_url).hostname or ""
	if host and not _host_resolves(host, 443):
		if logger:
			logger.warning("OpenTargets offline (DNS): using cache only")
		return cache
	# Optional probe. Some servers may not like HEAD, so probe the GraphQL URL lightly.
	if host and not _head_ok(graphql_url):
		if logger:
			logger.warning("OpenTargets unreachable (HEAD probe): using cache only")
		return cache

	def _run_query(ensg: str) -> list[dict]:
		payload = {"query": query, "variables": {"ensg": ensg, "size": int(cfg.OPENTARGETS_CACHE_KNOWN_DRUGS_SIZE)}}
		res = _urlopen_json(
			graphql_url,
			method="POST",
			data=json.dumps(payload).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			logger=logger,
			label=f"OpenTargets({ensg})",
			sleep_seconds=float(cfg.OT_SLEEP_SECONDS or 0.0),
		)
		if not isinstance(res, dict):
			return []
		data_obj = res.get("data")
		if not isinstance(data_obj, dict):
			return []
		target_obj = data_obj.get("target")
		if not isinstance(target_obj, dict):
			return []
		known_obj = target_obj.get("knownDrugs")
		if not isinstance(known_obj, dict):
			return []
		rows = known_obj.get("rows")
		return rows if isinstance(rows, list) else []

	for gene in genes:
		if not isinstance(gene, str) or not gene.strip():
			continue
		gene = gene.strip()
		rows = cache.get(gene)
		rows_list = rows if isinstance(rows, list) else []
		needs_fetch = bool(force_refresh) or not rows_list
		if not needs_fetch:
			continue
		ensg = cache_gene_to_ensg.get(gene)
		if not isinstance(ensg, str) or not ensg:
			continue
		if cfg.OT_SLEEP_SECONDS:
			time.sleep(float(cfg.OT_SLEEP_SECONDS))
		cache[gene] = _run_query(ensg)

	save_json(cache_path, cache)
	return cache


def fetch_uniprot_gene_to_acc(
	genes: Iterable[str],
	*,
	cache_path: Path,
	logger=None,
) -> Dict[str, str]:
	"""Fetch UniProt primary accession for genes and save JSON mapping."""
	cache = load_json(cache_path, {})
	cache = cache if isinstance(cache, dict) else {}
	from bioservices import UniProt

	uniprot = UniProt()
	gene_list = [g.strip() for g in genes if isinstance(g, str) and g.strip()]
	missing = [g for g in gene_list if g not in cache]
	if logger:
		logger.info("UniProt: %d/%d genes cached", len(gene_list) - len(missing), len(gene_list))

	def _fetch_one(symbol: str) -> Tuple[str, Optional[str]]:
		try:
			res = uniprot.search(f"(gene_exact:{symbol}) AND (organism_id:9606)", frmt="json", limit=1)
			hits = res.get("results") if isinstance(res, dict) else []
			if hits:
				acc = hits[0].get("primaryAccession")
				if acc:
					return symbol, str(acc).upper()
		except Exception:
			return symbol, None
		return symbol, None

	if missing:
		with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(cfg.UNIPROT_MAX_WORKERS))) as ex:
			futs = [ex.submit(_fetch_one, g) for g in missing]
			for fut in concurrent.futures.as_completed(futs):
				gene, acc = fut.result()
				if acc:
					cache[gene] = acc
		save_json(cache_path, cache)

	return {str(k): str(v) for k, v in cache.items() if k and v}
