from __future__ import annotations

import json
import os
import re
from pathlib import Path

LOGGER_NAME = "sl-kg"

DATA_DIR = Path("/data/guoyu/KG-LLM-XSL/data")
OUT_DIR = Path("/data/guoyu/KG-LLM-XSL/output")

SL_PAIRS_COMMON_FILE = DATA_DIR / "sl_pairs_common.json"

MAX_CURATED_SL_PAIRS = 300
CURATED_SL_PAIRS_MODE = "head"
CURATED_SL_PAIRS_SAMPLE_SEED = 42

_GENE_SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")


def _load_curated_sl_pairs() -> list[dict]:
	if not SL_PAIRS_COMMON_FILE.exists():
		return []
	rows = json.loads(SL_PAIRS_COMMON_FILE.read_text(encoding="utf-8"))
	if not isinstance(rows, list):
		return []

	def _clean_gene_symbol(v: object) -> str | None:
		if not isinstance(v, str):
			return None
		v = v.strip()
		return v if v and _GENE_SYMBOL_RE.match(v) else None

	cleaned = [
		{**row, "gene_a": a, "gene_b": b}
		for row in rows
		if isinstance(row, dict)
		and (a := _clean_gene_symbol(row.get("gene_a")))
		and (b := _clean_gene_symbol(row.get("gene_b")))
	]

	if MAX_CURATED_SL_PAIRS and len(cleaned) > int(MAX_CURATED_SL_PAIRS):
		if CURATED_SL_PAIRS_MODE == "sample":
			import random

			rnd = random.Random(int(CURATED_SL_PAIRS_SAMPLE_SEED))
			cleaned = rnd.sample(cleaned, int(MAX_CURATED_SL_PAIRS))
		else:
			cleaned = cleaned[: int(MAX_CURATED_SL_PAIRS)]
	return cleaned


SL_PAIRS_COMMON = _load_curated_sl_pairs()
CORE_GENE_PAIRS = [(pair["gene_a"], pair["gene_b"]) for pair in SL_PAIRS_COMMON[:30]]
NON_GENE_ENDPOINTS = {"MSI"}
GENE_PAIRS = CORE_GENE_PAIRS

CBIO_STUDY_CANDIDATES = {
	"TCGA-BRCA": ["brca_tcga", "brca_tcga_pan_can_atlas_2018"],
	"TCGA-LUAD": ["luad_tcga", "luad_tcga_pan_can_atlas_2018"],
	"TCGA-OV": ["ov_tcga", "ov_tcga_pan_can_atlas_2018"],
	"TCGA-KIRC": ["kirc_tcga", "kirc_tcga_pan_can_atlas_2018"],
	"TCGA-SKCM": ["skcm_tcga", "skcm_tcga_pan_can_atlas_2018"],
	"TCGA-GBM": ["gbm_tcga", "gbm_tcga_pan_can_atlas_2018"],
	"TCGA-LGG": ["lgg_tcga", "lgg_tcga_pan_can_atlas_2018"],
}
MAX_GENES_FOR_CBIOPORTAL = int(MAX_CURATED_SL_PAIRS / 4)
SUPPRESS_CBIOPORTAL_OUTPUT = True
VERBOSE_CBIOPORTAL = False
ONLY_CREATE_CONNECTED_COHORT_NODES = True
ADD_CBIOPORTAL_TO_KG = True
CBIOPORTAL_STUB_MODE = False

ADD_OPEN_TARGETS_TO_KG = True
OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
OT_CACHE_PATH = DATA_DIR / "opentargets_cache.json"
OT_FORCE_REFRESH = False
KNOWN_DRUGS_SIZE = 50
MAX_TARGETS_FOR_OT = int(MAX_CURATED_SL_PAIRS / 4)
OT_SLEEP_SECONDS = 0.1
OT_TIMEOUT_SECONDS = 200

ENSEMBL_LOOKUP_PATH = DATA_DIR / "ensembl_lookup.json"
ENSEMBL_LOOKUP = json.loads(ENSEMBL_LOOKUP_PATH.read_text(encoding="utf-8")) if ENSEMBL_LOOKUP_PATH.exists() else {}
ENSEMBL_RELEASE = 110
MAX_GENES_FOR_ENSEMBL = int(MAX_CURATED_SL_PAIRS / 2)
ENSEMBL_AUTO_FETCH = True
ENSEMBL_AUTO_FETCH_MAX_GENES = 400
ENSEMBL_AUTO_FETCH_MAX_WORKERS = 8

STRING_SPECIES = 9606
STRING_REQUIRED_SCORE = 800
STRING_ADD_NODES = 10

# OmniPath
OMNIPATH_TSV = Path("/data/guoyu/OmniPath/omnipath_interactions.tsv")
MAX_OMNIPATH_EDGES = int(MAX_CURATED_SL_PAIRS / 2)
MAX_GENES_FOR_UNIPROT_IN_OMNIPATH = 20

ADD_TF_REGULATION_TO_KG = True
OMNIPATH_TF_TSV = DATA_DIR / "omnipath_tf_regulation.tsv"
OMNIPATH_TF_USE_PYTHON_CLIENT = True
OMNIPATH_TF_DOROTHEA_LEVELS = ("A", "B", "C")
OMNIPATH_TF_MAX_EDGES = int(MAX_CURATED_SL_PAIRS)
TF_REGULATION_ALLOW_EXTERNAL_TF = True
TF_REGULATION_MAX_TF_PER_TARGET = 12

# Cancer driver annotation (IntOGen / OncoKB style signals)
ADD_CANCER_DRIVER_TO_KG = True

INTOGEN_DRIVERS_TSV = DATA_DIR / "intogen_drivers.tsv"
INTOGEN_DRIVERS_URL = os.getenv("INTOGEN_DRIVERS_URL")
INTOGEN_DRIVER_MAX_EDGES_PER_COHORT = 200

# Map our cohort labels to a tumor-type code used in driver resources.
# (Keep this small and explicit to avoid accidental mismatches.)
COHORT_TO_TUMOR_TYPE = {
	"TCGA-BRCA": "BRCA",
	"TCGA-LUAD": "LUAD",
	"TCGA-OV": "OV",
	"TCGA-KIRC": "KIRC",
	"TCGA-SKCM": "SKCM",
	"TCGA-GBM": "GBM",
	"TCGA-LGG": "LGG",
}

DEPMAP_GENE_EFFECT = DATA_DIR / "CRISPRGeneEffect.csv"
DEPMAP_MIN_ABS_CORR = 0.15
DEPMAP_MAX_EDGES = 30

ADD_REACTOME_TO_KG = True
MAX_GENES_FOR_REACTOME = int(MAX_CURATED_SL_PAIRS)
REACTOME_TIMEOUT_SECONDS = 200
REACTOME_CACHE_PATH = DATA_DIR / "reactome_cache.json"
MAX_PATHWAYS_PER_GENE = 3
MAX_PATHWAYS_PER_CORE_GENE = 10
REACTOME_MAX_WORKERS = 8

MAX_GENES_FOR_UNIPROT = int(MAX_CURATED_SL_PAIRS / 2)

CACHE_KHOP_NEIGHBORS = False
CACHE_KHOP_K = 2
CACHE_KHOP_MAX_NEIGHBORS = 30

HTTP_MAX_WORKERS = 8
HTTP_HEAD_TIMEOUT_SECONDS = 10
HTTP_DEFAULT_TIMEOUT_SECONDS = 20
HTTP_DEFAULT_RETRIES = 3
HTTP_BACKOFF_BASE_SECONDS = 0.6
HTTP_BACKOFF_CAP_SECONDS = 12.0
REACTOME_HEAD_TIMEOUT_SECONDS = 8
OPENTARGETS_HEAD_TIMEOUT_SECONDS = 8
ENSEMBL_REST_TIMEOUT_SECONDS = 20
ENSEMBL_REST_RETRIES = 2
REACTOME_REQUEST_TIMEOUT_SECONDS = 15
REACTOME_REQUEST_SLEEP_SECONDS = 0.2
REACTOME_REQUEST_RETRIES = 2
OPENTARGETS_CACHE_TIMEOUT_SECONDS = 20
OPENTARGETS_CACHE_RETRIES = 3
OPENTARGETS_CACHE_KNOWN_DRUGS_SIZE = 10

UNIPROT_GENE_TO_ACC_CACHE_PATH = DATA_DIR / "uniprot_gene_to_acc.json"
UNIPROT_MAX_WORKERS = 6

MAX_GENES_FOR_DEPMAP = 400

VIS_HIGHLIGHT_EDGE_WIDTH = 7.0
VIS_CONTEXT_EDGE_WIDTH = 7.0
VIS_SL_EDGE_WIDTH = 3.0
VIS_DEFAULT_EDGE_WIDTH = 1.0

VIS_OPTIONS = {
	"physics": {
		"enabled": True,
		"barnesHut": {
			"gravitationalConstant": -8000,
			"centralGravity": 1,
			"springLength": 70,
			"springConstant": 0.001,
			"damping": 0.7,
		},
	},
	"interaction": {"dragNodes": True, "dragView": True, "zoomView": True},
}

VIS_GROUP_COLORS = {
	"gene": {
		"background": "#7eaef0",
		"border": "#7eaef0",
		"highlight": {"background": "#7eaef0", "border": "#7eaef0"},
		"hover": {"background": "#7eaef0", "border": "#7eaef0"},
	},
	"protein": {
		"background": "#f1b131",
		"border": "#f1b131",
		"highlight": {"background": "#f1b131", "border": "#f1b131"},
		"hover": {"background": "#f1b131", "border": "#f1b131"},
	},
	"pathway": {
		"background": "#fa886c",
		"border": "#fa886c",
		"highlight": {"background": "#fa886c", "border": "#fa886c"},
		"hover": {"background": "#fa886c", "border": "#fa886c"},
	},
	"cohort": {
		"background": "#ed84f0",
		"border": "#ed84f0",
		"highlight": {"background": "#ed84f0", "border": "#ed84f0"},
		"hover": {"background": "#ed84f0", "border": "#ed84f0"},
	},
	"drug": {
		"background": "#5eef87",
		"border": "#5eef87",
		"highlight": {"background": "#5eef87", "border": "#5eef87"},
		"hover": {"background": "#5eef87", "border": "#5eef87"},
	},
	"regulation": {
		"background": "#8A2BE2",
		"border": "#8A2BE2",
		"highlight": {"background": "#8A2BE2", "border": "#8A2BE2"},
		"hover": {"background": "#8A2BE2", "border": "#8A2BE2"},
	},
}

VIS_EDGE_COLORS = {
	# Default/fallback edge color (avoid vis-network default black).
	"default": {"color": "#7eaef0", "highlight": "#7eaef0"},
	"SL_pair_mutated": {"color": "#d21bd8", "highlight": "#d21bd8"},
	"SL_pair_context": {"color": "#00ffdd", "highlight": "#00ffdd"},
	"mutated_in": {"color": "#d21bd8", "highlight": "#d21bd8"},
	"DepMap_codependency": {"color": "#f9ce78", "highlight": "#f9ce78"},
	"in_pathway": {"color": "#fc789f", "highlight": "#fc789f"},
	# Edge types that were previously falling back to gray. Use existing node-group
	# colors to keep a consistent palette while improving edge-type readability.
	"targets": {
		"color": VIS_GROUP_COLORS["drug"]["border"],
		"highlight": VIS_GROUP_COLORS["drug"]["border"],
	},
	"encodes": {
		"color": VIS_GROUP_COLORS["protein"]["border"],
		"highlight": VIS_GROUP_COLORS["protein"]["border"],
	},
	"OmniPath_interaction": {
		"color": VIS_GROUP_COLORS["protein"]["border"],
		"highlight": VIS_GROUP_COLORS["protein"]["border"],
	},
	"TF_regulates": {
		"color": VIS_GROUP_COLORS["regulation"]["border"],
		"highlight": VIS_GROUP_COLORS["regulation"]["border"],
	},
	"driver_in": {
		"color": VIS_GROUP_COLORS["cohort"]["border"],
		"highlight": VIS_GROUP_COLORS["cohort"]["border"],
	},
	"STRING_association": {
		"color": VIS_GROUP_COLORS["gene"]["border"],
		"highlight": VIS_GROUP_COLORS["gene"]["border"],
	},
}

