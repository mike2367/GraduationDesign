from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional



# -----------------------------------------------------------------------------
# Defaults (may be overridden by BASELINE_CONFIG below)
# -----------------------------------------------------------------------------
LLM_PROVIDER: str = "aigcbest"  # aigcbest 

MAX_TOKENS: Optional[int] = 20000

"""LLM API settings (OpenAI-compatible).

SECURITY NOTE:
- Do NOT hard-code API keys in this repo.
- Provide secrets via environment variables.

Environment variables:
- AIGC_BEST_BASE_URL: defaults to a self-hosted OpenAI-compatible endpoint.
- AIGC_BEST_API_KEY: required for API calls.
"""


AIGC_BEST_BASE_URL: str = "" # replace with your own endpoint, e.g., "http://localhost:8000/v1"
AIGC_BEST_API_KEY: str = os.environ.get("AIGC_BEST_API_KEY", "") # replace with your own API key, e.g., "sk-..."

# MAX_TOKENS: Optional[int] = 8000
PROMPT_LIMIT: Optional[int] = 10000
LOCAL_MODEL_PATH: Optional[str] = None

# BASELINE_CONFIG = "deepseek-v3.2_config.json"
# BASELINE_CONFIG = "gpt-3.5-turbo_config.json"
# BASELINE_CONFIG = "gemini-2.5-flash-all_config.json"
BASELINE_CONFIG = "gpt-5.4_config.json"



_cfg_path = Path(__file__).parent.parent / "baseline_configs" / BASELINE_CONFIG

with open(_cfg_path, "r", encoding="utf-8") as fh:
	_cfg = json.load(fh)
MODEL = _cfg.get("model_id")
TEMPERATURE = _cfg.get("temperature")
TOP_P = _cfg.get("top_p")
# allow explicit null -> None
MAX_TOKENS = _cfg.get("max_tokens")
PROMPT_LIMIT = _cfg.get("prompt_limit")
# provider & local model path (for local inference)
if "provider" in _cfg:
	LLM_PROVIDER = _cfg["provider"]
if "model_path" in _cfg:
	LOCAL_MODEL_PATH = _cfg["model_path"]

EVAL_VERBOSE: bool = False

CUDA_VISIBLE_DEVICES: Optional[str] = None

SYSTEM_PROMPT = (
	"You are a computational biologist specializing in mechanistic explanations of synthetic lethality (SL). "
	"Follow instructions strictly and do not add external facts beyond the provided prompt content."
)

# Max attempts per LLM request. If all attempts fail, raise (do not silently continue).
LLM_MAX_RETRY: int = 3

LLM_REQUEST_TIMEOUT_S: float = 60.0 * 3

# Strategy defaults: keep conservative to reduce semantic drift.
SELF_REFINE_ROUNDS: int = 1
COVE_NUM_QUESTIONS: int = 5

DEFAULT_EVAL_OUT_DIR = Path("/data/guoyu/KG-LLM-XSL/output") / "eval_results"

EVAL_PROMPTS_DIR = Path("/data/guoyu/KG-LLM-XSL/output/gene_pairs_subgraphs") 
EVAL_PATTERN = "**/*_prompt.txt"
EVAL_GROUND_TRUTH_PATH = Path("/data/guoyu/KG-LLM-XSL/data") / "SL_MERK_groundtruth.csv"


# =====================================================================
EVAL_STRATEGY = "all"  # baseline | self_refine | cove | all
EVAL_PAIR_LIMIT = 112  # 0 = no limit
EVAL_PAIR_MODE = "prompt"  # prompt | ground_truth

# Continuation mode: when True, skip already-evaluated pairs and process the next EVAL_PAIR_LIMIT
# pairs that haven't been completed yet. Keeps existing results instead of clearing.
EVAL_CONTINUE_FROM_EXISTING: bool = True
# =====================================================================


EVAL_RUN_NAIVE: bool = True

EVAL_STAGE_TIMEOUT_S: float = 900.0

EVAL_REUSE_EXISTING: bool = False
EVAL_DETERMINISTIC: bool = False
EVAL_DEBUG_JUDGE: bool = False

# If True and EVAL_REUSE_EXISTING is False, clear the evaluation output directory
# before writing new JSON/CSV results.
EVAL_CLEAR_OUTPUT_DIR: bool = True

EVAL_JUDGE_BACKEND: str = "auto"  # expert | heuristic | auto
EVAL_JUDGE_FALLBACK_ON_PARSE_FAIL: bool = True
EVAL_JUDGE_FALLBACK_ON_ZERO_FORMAT: bool = True

FEATURE_EMBED_MODEL_PATH: str = "/data/guoyu/HF-models/MedCPT-Query-Encoder"

FEATURE_EMBED_DEVICE: str = "auto"
FEATURE_EMBED_MAX_LENGTH: int = 64
FEATURE_EMBED_BATCH_SIZE: int = 16

TOTAL_SIM_MAX_LENGTH: int = 256

FEATURE_SIMILARITY_MODE: str = "adjusted"
FEATURE_SIM_THRESHOLD: float = 0.6
FEATURE_SIM_THRESHOLD_ADJUSTED: float = 0.4

FEATURE_MAX_CANDIDATES: int = 800
FEATURE_TOKENCLS_MODEL_PATH: str = "/data/guoyu/HF-models/keyphrase-extraction-kbir-inspec"

FEATURE_SEQ2SEQ_MODEL_PATH: str = "/data/guoyu/HF-models/bart_finetuned_keyphrase_extraction"

FEATURE_CANDIDATE_BACKEND: str = "token_cls"
FEATURE_KEYPHRASE_LOCAL_ONLY: bool = True

FEATURE_SCORE_SCOPE: str = "section2"
EXPERT_JUDGE_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EXPERT_JUDGE_MODEL_PATH: str = "/data/guoyu/HF-models/all-MiniLM-L6-v2"
EXPERT_JUDGE_DEVICE: str = "auto"
EXPERT_JUDGE_LOCAL_ONLY: bool = True

# ------------------------------
# Hallucination / faithfulness metric
# ------------------------------
# NLI model: use local weights to avoid any network calls.
HALLUCINATION_NLI_MODEL_PATH: str = "/data/guoyu/HF-models/PubMedBERT-MNLI-MedNLI"
HALLUCINATION_NLI_LOCAL_ONLY: bool = True
HALLUCINATION_NLI_DEVICE: str = "auto"

HALLUCINATION_MAX_SENTENCES: int = 40
HALLUCINATION_MIN_SENTENCE_CHARS: int = 25

# Cosine similarity threshold above which a generated sentence is considered
# "grounded" in KG evidence (used by KG-faithfulness diagnostics and evidence quality).
# This should be tuned on a small dev set if you have human labels; default chosen
# to represent "moderate semantic match" for normalized MiniLM embeddings.
HALLUCINATION_GROUNDING_SIM_THRESHOLD: float = 0.35

# How much to penalize NEUTRAL (unsupported) vs CONTRADICTION.
# hallucination(sentence) = contradiction + neutral_weight * neutral
HALLUCINATION_NEUTRAL_WEIGHT: float = 0.5
HALLUCINATION_WORST_K: int = 6
"""Deprecated metric weights removed.

Historically we had a weighted composite metric (e.g., comparable quality).
That logic is now removed; keep evaluation metrics explicit and orthogonal.
"""

KEYPHRASE_SEQ2SEQ_MAX_LENGTH = 512
KEYPHRASE_STOPWORDS = frozenset({"the", "a", "an", "of", "in", "on", "at", "to", "for"})
POS_NOUN_TAGS = frozenset({"NOUN", "PROPN", "ADJ", "NAM", "NOM"})
POS_ALL_TAGS = frozenset({"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NAM", "NOM"})

JUDGE_STRUCTURED_MARKERS = frozenset({
	"mechanistic summary",
	"evidence",
	"references",
	"output format",
	"required format",
	"json",
})

JUDGE_KNOWN_HEADINGS = (
	"mechanistic summary",
	"evidence",
	"references",
	"caveat",
	"limitations",
)

JUDGE_HEURISTIC_BASE_SCORE = 0.2
JUDGE_HEURISTIC_STRUCTURE_WEIGHT = 0.55
JUDGE_HEURISTIC_CITATION_WEIGHT = 0.30

LLM_RETRIABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})
LLM_HTTP_HEADERS_BASE = {
	"Accept": "application/json",
	"Content-Type": "application/json",
}

# BERTScore-style soft F1 weight (Zhang et al., ICLR 2020)
# Final F1 = BERTSCORE_SOFT_WEIGHT * F1_soft + (1 - BERTSCORE_SOFT_WEIGHT) * F1_hard
# Set to 1.0 for pure soft F1, 0.0 for pure hard F1
# Literature: soft F1 eliminates threshold sensitivity and handles length bias naturally
BERTSCORE_SOFT_WEIGHT: float = 0.5

# Use explicitly declared "Key process phrases" as primary candidates for F1
# This eliminates verbosity bias by scoring only the model's curated summary phrases
# instead of all extracted keyphrases (which penalizes verbose structured output)
USE_EXPLICIT_KEY_PHRASES: bool = True

LEXICAL_JACCARD_OFFSET = 0.6
LEXICAL_JACCARD_SCALE = 0.4

SCORING_STOPWORDS = frozenset({"a","an","and","are","as","at","be","because","by","can","could","do","does","for","from","has","have","in","into","is","it","its","may","might","of","on","or","such","that","the","their","then","these","this","to","was","were","with","without","while","will","would"})

SCORING_JUNK_TOKENS = frozenset({"pmid","doi","http","https","www","com","org","arxiv","figure","fig","table","supplementary","reference","references","bibliography","citation","citations","et","al","gene","pathway","protein","drug","disease","cohort","src","dst","key","evidence","evidence_score","score","edges","nodes","reactome","string","uniprot","omnipath","curated","seed","sl_pair","in_pathway","string_association","encodes"})