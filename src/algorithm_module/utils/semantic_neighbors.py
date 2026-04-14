from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Mapping, Optional, Sequence

from algorithm_module import algo_config


@dataclass(frozen=True)
class SemanticModel:
	tokenizer: object
	model: object
	device: str


@lru_cache(maxsize=2)
def load_semantic_model(model_path: str, device: str = "cpu") -> SemanticModel:
	from transformers import AutoModel, AutoTokenizer  # type: ignore
	import torch  # type: ignore

	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model = AutoModel.from_pretrained(model_path)
	model.eval()
	model.to(torch.device(device))
	return SemanticModel(tokenizer=tokenizer, model=model, device=device)


def _mean_pool_last_hidden(last_hidden, attention_mask):
	import torch  # type: ignore

	mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
	summed = torch.sum(last_hidden * mask, dim=1)
	counts = torch.clamp(mask.sum(dim=1), min=1.0)
	return summed / counts


def embed_texts(
	texts: Sequence[str],
	*,
	model_path: str,
	device: str = "cpu",
) -> List[List[float]]:
	import torch
	m = load_semantic_model(model_path, device=device)
	out: List[List[float]] = []
	for i in range(0, len(texts), max(1, algo_config.SEMANTIC_EMBED_BATCH_SIZE)):
		enc = {k: v.to(torch.device(device)) for k, v in m.tokenizer(list(texts[i:i + algo_config.SEMANTIC_EMBED_BATCH_SIZE]), padding=True, truncation=True, max_length=algo_config.SEMANTIC_EMBED_MAX_LENGTH, return_tensors="pt").items()}
		with torch.no_grad():
			pooled = torch.nn.functional.normalize(_mean_pool_last_hidden(m.model(**enc).last_hidden_state, enc["attention_mask"]), p=2, dim=1)
			out.extend([[float(x) for x in row] for row in pooled.detach().cpu().tolist()])
	return out


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
	return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def mmr_select(
	items: Sequence[str],
	*,
	item_embeddings: Mapping[str, Sequence[float]],
	query_embedding: Sequence[float],
	k: int,
) -> List[str]:
	if k <= 0 or not items:
		return []
	lam = min(1.0, max(0.0, algo_config.SEMANTIC_NEIGHBOR_MMR_LAMBDA))
	pool = [it for it in items if it in item_embeddings]
	selected: List[str] = []
	remaining = set(pool)
	while remaining and len(selected) < k:
		best_item = max(remaining, key=lambda it: lam * cosine(item_embeddings[it], query_embedding) - (1.0 - lam) * (max(cosine(item_embeddings[it], item_embeddings[s]) for s in selected) if selected else 0.0), default=None)
		if best_item is None:
			break
		selected.append(best_item)
		remaining.remove(best_item)
	return selected


def node_text(graph, node: str) -> str:
	attrs = graph.nodes.get(node, {}) if hasattr(graph, "nodes") else {}
	node_type = str((attrs or {}).get("type") or "unknown")

	# Prefer semantically meaningful fields if present.
	for key in (
		"name",
		"label",
		"symbol",
		"title",
		"protein_name",
		"description",
		"summary",
		"function",
		"function_summary",
		"go_summary",
	):
		val = (attrs or {}).get(key)
		if isinstance(val, str) and val.strip():
			return f"{val.strip()} [{node_type}]"

	# Fall back to node id heuristics.
	if node.startswith("gene:"):
		return f"{node.split(':', 1)[1]} [gene]"
	return f"{node} [{node_type}]"


def build_core_query_text(gene_a: str, gene_b: str) -> str:
	a = gene_a.split(":", 1)[1] if gene_a.startswith("gene:") else gene_a
	b = gene_b.split(":", 1)[1] if gene_b.startswith("gene:") else gene_b
	return f"{a} {b} synthetic lethality mechanism"
