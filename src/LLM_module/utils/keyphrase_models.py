from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List
import io
import contextlib
import sentencepiece as spm
from pathlib import Path

from LLM_module import eval_config as ecfg
from LLM_module.utils.common import (
    normalize_text,
    dedupe_keep_order,
    resolve_device,
    get_pipeline_device,
)

# -----------------------------------------------------------------------------
# WORKAROUND for transformers 5.0.0.dev0 CamembertTokenizer bug
# The dev version's slow tokenizer has a bug in _vocab loading (too many values to unpack).
# We patch the __init__ method to sanitize vocab entries to be strict 2-tuples.
# -----------------------------------------------------------------------------
try:
	import transformers
	from transformers.models.camembert import tokenization_camembert
	
	# Store the original __init__ method
	_original_camembert_init = tokenization_camembert.CamembertTokenizer.__init__
	
	def _fixed_camembert_init(self, *args, **kwargs):
		"""Wrapper for __init__ that sanitizes vocab to be strict 2-tuples."""
		# If vocab is provided, sanitize it
		if 'vocab' in kwargs and kwargs['vocab'] is not None:
			vocab = kwargs['vocab']
			
			# Check if vocab is a dict (from tokenizer.json)
			if isinstance(vocab, dict):
				# Convert dict to list of (token, score) tuples
				# Use vocab_file to get scores if available
				if 'vocab_file' in kwargs and kwargs['vocab_file']:
					vocab_file = Path(kwargs['vocab_file'])
					if vocab_file.exists() and vocab_file.name == 'sentencepiece.bpe.model':
						# Load scores from sentencepiece model
						sp = spm.SentencePieceProcessor()
						sp.Load(str(vocab_file))
						# Build vocab list correctly using sentencepiece
						kwargs['vocab'] = [(sp.id_to_piece(i), float(sp.get_score(i))) 
										   for i in range(sp.get_piece_size())]
					else:
						# No sentencepiece file, use default scores
						sorted_items = sorted(vocab.items(), key=lambda x: x[1])
						kwargs['vocab'] = [(token, 0.0) for token, idx in sorted_items]
				else:
					# No vocab_file, use default scores
					sorted_items = sorted(vocab.items(), key=lambda x: x[1])
					kwargs['vocab'] = [(token, 0.0) for token, idx in sorted_items]
			
			# Check if vocab is already a list/tuple
			elif isinstance(vocab, (list, tuple)):
				# Ensure all entries are exactly 2-tuples
				kwargs['vocab'] = [(entry[0], float(entry[1])) 
								   for entry in vocab 
								   if isinstance(entry, (tuple, list)) and len(entry) >= 2]
		
		# If vocab_file is provided but no vocab, load it properly
		elif 'vocab_file' in kwargs and kwargs['vocab_file'] is not None:
			vocab_file = Path(kwargs['vocab_file'])
			if vocab_file.exists() and vocab_file.name == 'sentencepiece.bpe.model':
				# Load vocab from sentencepiece model file
				sp = spm.SentencePieceProcessor()
				sp.Load(str(vocab_file))
				# Build vocab list correctly (strictly 2-tuples)
				kwargs['vocab'] = [(sp.id_to_piece(i), float(sp.get_score(i))) 
								   for i in range(sp.get_piece_size())]
		
		# Call the original __init__
		_original_camembert_init(self, *args, **kwargs)
	
	# Apply patch to the class
	tokenization_camembert.CamembertTokenizer.__init__ = _fixed_camembert_init
except Exception:
	# If we can't patch, let it fail naturally later
	pass
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyphraseModelSettings:
	model_path: str
	device: str = "cpu"
	local_files_only: bool = True

	# For seq2seq generation
	max_new_tokens: int = 64
	num_beams: int = 4


@lru_cache(maxsize=4)
def _load_token_cls_pipeline(model_path: str, device: str, local_files_only: bool):
	from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline  # type: ignore
	import torch

	dev = resolve_device(device)
	device_arg = get_pipeline_device(dev)
	
	tok = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only, use_fast=False)
	mdl = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=local_files_only)
	
	# Try to create pipeline with requested device, fall back to CPU on CUDA OOM
	try:
		return pipeline("token-classification", model=mdl, tokenizer=tok, aggregation_strategy="simple", device=device_arg)
	except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
		if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
			print(f"[keyphrase] CUDA OOM loading token_cls pipeline, falling back to CPU")
			# Clear CUDA cache and use CPU
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			return pipeline("token-classification", model=mdl, tokenizer=tok, aggregation_strategy="simple", device=-1)
		raise



@lru_cache(maxsize=2)
def _load_seq2seq(model_path: str, device: str, local_files_only: bool):
	import torch
	from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, BartForConditionalGeneration  # type: ignore
	from transformers.utils import logging as hf_logging  # type: ignore

	dev = resolve_device(device)

	# Silence model loading noise
	with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
		cfg = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
		# Force slow tokenizer for seq2seq models too (avoid transformers dev version bugs)
		tok = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only, use_fast=False)

	load_kwargs = {
		"config": cfg,
		"local_files_only": local_files_only,
		"ignore_mismatched_sizes": True,
		"low_cpu_mem_usage": True,
	}
	if dev.startswith("cuda") and torch.cuda.is_available():
		load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

	# Load model (prefer BART class if applicable)
	prev_verbosity = hf_logging.get_verbosity()
	hf_logging.set_verbosity_error()
	try:
		with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
			if str(getattr(cfg, "model_type", "")).lower() == "bart":
				mdl = BartForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
			else:
				mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, **load_kwargs)
	finally:
		hf_logging.set_verbosity(prev_verbosity)
	
	mdl.eval()
	if dev.startswith("cuda") and torch.cuda.is_available():
		try:
			mdl = mdl.to(dev)
		except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
			if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
				print(f"[keyphrase] CUDA OOM loading seq2seq model, keeping on CPU")
				torch.cuda.empty_cache()
				# Model stays on CPU
			else:
				raise
	
	return tok, mdl


def extract_keyphrases_token_cls(text: str, *, settings: KeyphraseModelSettings, max_phrases: int = 64) -> List[str]:
	"""Extract keyphrases using a token-classification model.
	
	If the model is a POS tagger, this will chunk consecutive NOUN/ADJ tokens into noun phrases.
	If the model is a proper keyphrase/NER model with entity_group labels, it will use those spans.
	"""
	pipe = _load_token_cls_pipeline(settings.model_path, settings.device, settings.local_files_only)
	preds = pipe(text)
	
	# Check if this is a POS tagger (has 'entity' field with POS tags like NOUN, ADJ, VERB)
	# vs a keyphrase/NER model (has 'entity_group' or similar with semantic labels)
	first_pred = preds[0] if isinstance(preds, list) and preds else {}
	entity_label = str(first_pred.get("entity") or first_pred.get("entity_group") or "").upper()
	
	is_pos_tagger = any(tag in entity_label for tag in ecfg.POS_ALL_TAGS)
	
	if is_pos_tagger:
		# POS tagger: build noun phrases from consecutive entities
		# Note: The tokenizer may produce subword tokens, so we use start/end positions
		# from the original text to reconstruct proper words
		phrases = []
		
		# Group entities by looking at gaps in character positions
		i = 0
		while i < len(preds):
			pred = preds[i]
			if not isinstance(pred, dict):
				i += 1
				continue
			
			entity = str(pred.get("entity") or pred.get("entity_group") or "").upper()
			start = pred.get("start")
			end = pred.get("end")
			
			if start is None or end is None:
				i += 1
				continue
			
			# Check if this is a noun-like entity
			is_noun_like = any(tag in entity for tag in ecfg.POS_NOUN_TAGS)
			
			if is_noun_like:
				# Start of a potential phrase - look ahead to find the full extent
				phrase_start = start
				phrase_end = end
				j = i + 1
				
				# Look ahead for adjacent noun entities
				while j < len(preds):
					next_pred = preds[j]
					if not isinstance(next_pred, dict):
						j += 1
						continue
					
					next_entity = str(next_pred.get("entity") or next_pred.get("entity_group") or "").upper()
					next_start = next_pred.get("start")
					next_end = next_pred.get("end")
					
					if next_start is None or next_end is None:
						j += 1
						continue
					
					next_is_noun = any(tag in next_entity for tag in ecfg.POS_NOUN_TAGS)
					
					# If next entity is noun-like and close enough, include it
					if next_is_noun and next_start <= phrase_end + 3:
						phrase_end = max(phrase_end, next_end)
						j += 1
					else:
						break
				
				# Extract the phrase from original text
				phrase_text = text[phrase_start:phrase_end].strip()
				
				# Filter out short phrases and stopwords
				if len(phrase_text) > 2 and phrase_text.lower() not in ecfg.KEYPHRASE_STOPWORDS:
					phrases.append(phrase_text)
				
				i = j  # Skip to after the phrase
			else:
				i += 1
		
		return dedupe_keep_order(phrases)[:max(1, int(max_phrases))]
	else:
		# Keyphrase/NER model: extract spans directly using start/end positions
		phrases = []
		for p in preds:
			if not isinstance(p, dict):
				continue
			# Prefer the already-decoded span when available.
			phrase_text = str(p.get("word") or "").strip()
			if not phrase_text:
				start = p.get("start")
				end = p.get("end")
				if start is not None and end is not None:
					phrase_text = text[start:end].strip()
			if phrase_text:
				phrases.append(phrase_text)
		return dedupe_keep_order(phrases)[:max(1, int(max_phrases))]


def _parse_generated_keyphrases(text: str) -> List[str]:
	s = " ".join(str(text or "").strip().split())
	s = s.replace("<sep>", ";").replace("<SEP>", ";")
	for sep in (",", "|", "\n"):
		s = s.replace(sep, ";")
	
	cleaned = []
	for p in s.split(";"):
		p = p.strip().lstrip("-")
		while p and p[0].isdigit():
			p = p[1:]
		p = " ".join(p.lstrip(").:" ).strip().split())
		if p:
			cleaned.append(p)
			
	return dedupe_keep_order(cleaned)


def extract_keyphrases_seq2seq(text: str, *, settings: KeyphraseModelSettings, max_phrases: int = 64) -> List[str]:
	"""Extract keyphrases using a seq2seq keyphrase generator."""
	import torch
	from transformers import GenerationConfig  # type: ignore

	tok, mdl = _load_seq2seq(settings.model_path, settings.device, settings.local_files_only)
	inputs = tok(text, return_tensors="pt", truncation=True, max_length=ecfg.KEYPHRASE_SEQ2SEQ_MAX_LENGTH)
	
	# Move inputs to model device
	model_device = next(mdl.parameters()).device
	inputs = {k: v.to(model_device) for k, v in inputs.items()}

	gen_cfg = GenerationConfig.from_model_config(mdl.config)
	gen_cfg.max_new_tokens = settings.max_new_tokens
	gen_cfg.num_beams = settings.num_beams
	gen_cfg.early_stopping = True

	with torch.no_grad():
		out = mdl.generate(**inputs, generation_config=gen_cfg)
		
	gen = tok.decode(out[0], skip_special_tokens=True)
	return _parse_generated_keyphrases(gen)[:max(1, int(max_phrases))]


def extract_keyphrases_model_based(
	text: str,
	*,
	backend: str,
	model_path: str,
	device: str = "cpu",
	local_files_only: bool = True,
	max_phrases: int = 64,
) -> List[str]:
	backend = str(backend or "").strip().lower()
	settings = KeyphraseModelSettings(model_path=model_path, device=device, local_files_only=local_files_only)
	if backend in {"token_cls"}:
		return extract_keyphrases_token_cls(text, settings=settings, max_phrases=max_phrases)
	if backend in {"seq2seq"}:
		return extract_keyphrases_seq2seq(text, settings=settings, max_phrases=max_phrases)
	return []
