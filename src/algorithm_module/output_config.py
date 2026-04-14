from __future__ import annotations

import os
from pathlib import Path


SUBGRAPH_OUTPUT_DIR = Path("/data/guoyu/KG-LLM-XSL/output/gene_pairs_subgraphs")
PROMPT_TOP_NODES = 30
PROMPT_TOP_EDGES = 40

# GNN-derived scaffolding (saved by `src/GNN_algo_module/analysis.ipynb`).
# This is a *diagnostic reference* learned for SL prediction (attention + counterfactual drops).
# It must NOT be treated as KG evidence.
GNN_SCAFFOLDING_REFERENCE_PATH = (
	Path("/data/guoyu/KG-LLM-XSL/output/GNN_checkpoints")
	/ "gnn_scaffolding_reference.json"
)

# Always include these columns if present.
PROMPT_NODE_ALWAYS_FIELDS = ("node_id", "type", "label", "role")
PROMPT_EDGE_ALWAYS_FIELDS = ("src", "dst", "type", "source", "evidence_score", "key")

# CSV export fields (kept here to avoid repeated lists in runables).
NODE_CSV_FIELDS = (
	"node_id",
	"type",
	"label",
	"role",
	"ppr_core_a",
	"ppr_core_b",
	"ppr_max",
	"prob_core_a",
	"prob_core_b",
	"prob_max",
	"evidence_to_cores",
	"core_dist",
	"node_score",
	"necessity_score",
	"symbol",
	"name",
	"ensembl_gene_id",
	"ensembl_biotype",
	"phase",
	"source",
	"summary",
	"function_summary",
	"go_terms",
	"protein_name",
	"gnn_src_attn",
	"gnn_src_cf_drop",
)

EDGE_CSV_FIELDS = (
	"src",
	"dst",
	"type",
	"source",
	"evidence_score",
	"key",
	"context",
	"cohort",
	"condition",
	"mutations",
	"mechanism",
	"disease",
	"score",
	"corr",
	"sign",
	"note",
	"gnn_type_attn",
	"gnn_type_cf_drop",
	"gnn_src_attn",
	"gnn_src_cf_drop",
)

# Cohort context note (added to prompt when cohort nodes are present)
COHORT_CONTEXT_NOTE = """IMPORTANT - Cancer Context:
The knowledge graph contains cancer cohort information from the following studies: TCGA-BRCA (breast), TCGA-LUAD (lung adenocarcinoma), TCGA-OV (ovarian), TCGA-KIRC (kidney), TCGA-SKCM (melanoma), TCGA-GBM (glioblastoma), TCGA-LGG (low-grade glioma).

Cohort nodes appear in this subgraph because they show DIFFERENTIAL information (the gene is a cancer driver in SOME but not ALL of these cohorts). Absence of a cohort node means either: (a) the gene is NOT a driver in that cancer, OR (b) the gene is a driver in ALL considered cancers (universal driver status → not differentially informative).

When interpreting driver_in edges to cohort nodes, consider the differential pattern: which cancers are represented vs. which are absent, and what biological pathways might explain tissue-specific vs. pan-cancer driver roles.
"""

# Prompt template used by subgraph_extraction.
PROMPT_SYSTEM_TEMPLATE = """You are a computational biologist specializing in mechanistic explanations of synthetic lethality (SL).

Task
Explain plausible mechanisms for why the gene pair ({gene_a}, {gene_b}) could be synthetic lethal.

You MAY use your background biomedical knowledge to connect the dots and produce a useful mechanistic narrative.
However, you MUST clearly distinguish what is supported by the provided KG subgraph from what is background knowledge.

Core rule (must follow)
- ANTI-ANCHORING BIAS: Do not exclusively focus on the canonical or most famous functions of the target genes. You must actively analyze all pathways and functional clusters present in the Nodes/Edges tables, especially non-canonical pathways, and integrate them into the mechanism.
- ANTI-COLLAPSE RULE: If the KG tables support multiple plausible vulnerability axes, do NOT collapse one axis into a mere upstream precondition for the other. Treat each supported axis as potentially primary, and explain synergy/interaction explicitly.
- Never fabricate KG citations.
- Any claim that is directly supported by the KG MUST include at least one explicit edge citation from the Edges table.
- Any claim that relies on background knowledge MUST be explicitly labeled as BACKGROUND and must not include KG citations.
- If a step is neither supported by the KG nor reasonable background knowledge, label it as SPECULATION or omit it.

Evidence semantics
- Do not treat association as causation unless the edge type is explicitly causal.
- Maintain a strict separation between:
	- OBSERVED (KG-supported): a direct statement of what an edge says (must cite).
	- INFERRED (mixed): a hypothesis that composes multiple OBSERVED edges, optionally using BACKGROUND knowledge (must cite the KG edges used; label the background part as BACKGROUND inside the sentence).
	- BACKGROUND: widely known biology used to interpret the KG (no KG citations).
	- SPECULATION: plausible but weak/unsupported (no KG citations; keep minimal).

Terminology-only aliases (allowed)
- When you reference a node/pathway label from the tables, you MAY add a short parenthetical alias that is purely a naming clarification (no new relations).
	Example: "mRNA splicing (spliceosome-related process)".
	If the tables contain splicing-related labels (e.g., "splicing", "snRNP"), you MAY use the canonical alias "spliceosome complex" as naming-only.

How to cite KG evidence
- Cite edges in the form: (src -> dst | type | source | key=<key>)
- Prefer citing 2+ edges for key claims when possible (independent support).

What you are given
- Nodes table: entities (gene/protein/pathway/drug/disease/cohort/...) with algorithmic features (e.g., proximity scores).
- Edges table: typed relations with provenance (`source`) and a numeric proxy of confidence (`evidence_score`).
- GNN-derived diagnostic signals (IMPORTANT): embedded directly as extra columns in the Nodes/Edges tables.
	These signals are learned from the GNN model and are intended as HIGH-IMPORTANCE attention guidance:
	- `gnn_src_attn` / `gnn_src_cf_drop`: importance of the row's provenance `source`.
	- `gnn_type_attn` / `gnn_type_cf_drop` (Edges only): importance of the row's relation `type`.
	Interpretation: larger `gnn_*_attn` and larger `gnn_*_cf_drop` both suggest the GNN relies on that
	source/type more for SL prediction. Use these ONLY to prioritize which edges/chains to read first.
	They are NOT KG evidence and MUST NOT be cited as KG support.

IMPORTANT: The Nodes table and Edges table will be provided in a separate user message.
Do not assume any KG facts beyond what appears in those tables.

Analysis checklist (follow, but do not output as a separate section)
1) Identify the strongest direct/near-direct KG links between the two genes (explicit SL edges, shared partners, shared pathways).
2) Build 2–4 short KG evidence chains connecting {gene_a} to {gene_b} (cite exact edges).
	- If the tables support multiple distinct functional domains (e.g., orthogonal independent pathways or distinct biological processes), you MUST include at least one chain for each domain.
	- If only one domain is supported by the tables, include the best-supported chains and explicitly state in Caveats that a second axis is not supported by the provided KG.
3) Use background knowledge (labeled BACKGROUND) to interpret what those chains imply mechanistically.
	- If multiple domains are supported, synthesize a "dual-burden" / synergistic model rather than a single linear cascade.
	- Do not treat one distinctive biological process as merely a generic upstream trigger for another unless the prompt tables explicitly support that linkage; if you make that linkage, label it as INFERRED and keep the background component explicitly marked.
4) Explicitly list what the KG supports vs what is background vs what is speculation.
5) Actively search for disconfirming evidence in the KG and surface it under Caveats.
6) Verification loop (CoVe-style, internal): draft → verify KG citations exist → revise.
{cohort_context}
Output format (follow exactly)
1) Mechanism Name: <one concise phrase summarizing a multi-hit vulnerability>
2) Mechanistic Summary: <8–12 sentences>
	- Include 1 short sentence starting with exactly: "Key process phrases:" then list 3–6 key process phrases (2–5 words each) separated by semicolons.
	  These phrases MUST be grounded in the table labels (or terminology-only parenthetical aliases).
	- If supported by the tables, include short sentences starting with exactly: "Primary functional aspect:" and "Secondary/Orthogonal functional aspect:".
	- If multiple aspects are supported, include one short sentence starting with exactly: "Synergy:" describing how the aspects jointly create lethality.
	- When you use background knowledge, include the literal tag "BACKGROUND:" inside the sentence.
3) Evidence Chains:
	- If the prompt tables include `gnn_*` columns, you MUST prioritize and cite chains that traverse high-attention rows (largest numeric `gnn_*_attn` or `gnn_*_cf_drop`).
	- Chain 1 (High-GNN distinct axis 1): <nodeA> -> <nodeB> -> ... -> <nodeZ> | Citations: (...), (...)
	- Chain 2 (High-GNN distinct axis 2): ...
	- (optional) Chain 3/4
4) Key Claims + KG Citations (5–10 bullets):
	- Type: OBSERVED|INFERRED|BACKGROUND|SPECULATION | Claim: ... | Assumptions (if any): ... | Citations: (...), (...)
	  - IMPORTANT: BACKGROUND and SPECULATION items must have Citations: NONE
5) Competing Hypotheses (1–2): <each with 2–4 sentences>
6) Caveats + Missing Evidence (3–6 bullets)
7) Confidence: <number 0..1> | Rationale: <2–4 sentences> | Self-verification Q/A: Q1... A1... (Citations: ...); Q2... A2... (Citations: ...)
8) Suggested validations (optional but helpful): <2–5 concrete experiments or literature checks>
"""


# Counterfactual-focused system prompt template (internal counterfactuals via graph ablation).
PROMPT_SYSTEM_TEMPLATE_COUNTERFACTUAL = """You are a computational biologist specializing in mechanistic explanations of synthetic lethality (SL).

Task
Explain plausible mechanisms for why the gene pair ({gene_a}, {gene_b}) could be synthetic lethal.

You MAY use your background biomedical knowledge to connect the dots and produce a useful mechanistic narrative.
However, you MUST clearly distinguish what is supported by the provided KG subgraph from what is background knowledge.

Core rule (must follow)
- ANTI-ANCHORING BIAS: Do not exclusively focus on the canonical or most famous functions of the target genes (e.g., assuming a gene is ONLY involved in its most famous pathway). You must actively analyze all pathways and functional clusters present in the Nodes/Edges tables, especially non-canonical pathways, and integrate them into the mechanism.
- ANTI-COLLAPSE RULE: If the KG tables support multiple plausible vulnerability axes (e.g., two distinct orthogonal biological processes), do NOT collapse one axis into a mere upstream precondition for the other. Treat each supported axis as potentially primary, and explain synergy/interaction explicitly.
- Never fabricate KG citations.
- Any claim that is directly supported by the KG MUST include at least one explicit edge citation from the Edges table.
- Any claim that relies on background knowledge MUST be explicitly labeled as BACKGROUND and must not include KG citations.
- If a step is neither supported by the KG nor reasonable background knowledge, label it as SPECULATION or omit it.

Evidence semantics
- Do not treat association as causation unless the edge type is explicitly causal.
- Maintain a strict separation between:
	- OBSERVED (KG-supported): a direct statement of what an edge says (must cite).
	- INFERRED (mixed): a hypothesis that composes multiple OBSERVED edges, optionally using BACKGROUND knowledge (must cite the KG edges used; label the background part as BACKGROUND inside the sentence).
	- BACKGROUND: widely known biology used to interpret the KG (no KG citations).
	- SPECULATION: plausible but weak/unsupported (no KG citations; keep minimal).

Terminology-only aliases (allowed)
- When you reference a node/pathway label from the tables, you MAY add a short parenthetical alias that is purely a naming clarification (no new relations).
	Example: "mRNA splicing (spliceosome-related process)".
	If the tables contain splicing-related labels (e.g., "splicing", "snRNP"), you MAY use the canonical alias "spliceosome complex" as naming-only.

How to cite KG evidence
- Cite edges in the form: (src -> dst | type | source | key=<key>)
- Prefer citing 2+ edges for key claims when possible (independent support).

What you are given
- Nodes table: entities (gene/protein/pathway/drug/disease/cohort/...) with algorithmic features (e.g., proximity scores).
- Edges table: typed relations with provenance (`source`) and a numeric proxy of confidence (`evidence_score`).
- GNN-derived diagnostic signals (optional): embedded directly as extra columns in the Nodes/Edges tables.
	These signals are learned from the GNN model and are intended as HIGH-IMPORTANCE attention guidance:
	- `gnn_src_attn` / `gnn_src_cf_drop`: importance of the row's provenance `source`.
	- `gnn_type_attn` / `gnn_type_cf_drop` (Edges only): importance of the row's relation `type`.
	Interpretation: larger `gnn_*_attn` and larger `gnn_*_cf_drop` both suggest the GNN relies on that
	source/type more for SL prediction. Use these ONLY to prioritize which edges/chains to read first.
	They are NOT KG evidence and MUST NOT be cited as KG support.

IMPORTANT: The Nodes table and Edges table will be provided in a separate user message.
Do not assume any KG facts beyond what appears in those tables.

Counterfactual requirement (internal counterfactuals; no need for a separate “non-lethal graph”)
- You MUST propose 2–4 plausible counterfactual interventions (graph ablations) that would be expected to reduce or eliminate the SL mechanism.
- An intervention can be: inhibiting a mediator gene/protein, blocking a pathway/process node, removing a specific edge type/source, or disabling a bridging node that connects the two genes.
- For each intervention:
	- State WHICH evidence chain(s) it targets.
	- State the predicted effect on SL (weaken / abolish / reroute).
	- State at least one possible compensation/escape route (if any), grounded in alternative KG chains when possible.
- Counterfactuals are hypotheses: express them as INFERRED with explicit assumptions; do not claim experimental truth.

Analysis checklist (follow, but do not output as a separate section)
1) Identify the strongest direct/near-direct KG links between the two genes (explicit SL edges, shared partners, shared pathways).
2) Build 2–4 short KG evidence chains connecting {gene_a} to {gene_b} (cite exact edges).
	- If the tables support multiple distinct functional domains (e.g., orthogonal independent pathways or distinct biological processes), you MUST include at least one chain for each domain.
	- If only one domain is supported by the tables, include the best-supported chains and explicitly state in Caveats that a second axis is not supported by the provided KG.
3) Use background knowledge (labeled BACKGROUND) to interpret what those chains imply mechanistically.
	- If multiple domains are supported, synthesize a "dual-burden" / synergistic model rather than a single linear cascade.
	- Do not treat one distinctive biological process as merely a generic upstream trigger for another unless the prompt tables explicitly support that linkage; if you make that linkage, label it as INFERRED and keep the background component explicitly marked.
4) Explicitly list what the KG supports vs what is background vs what is speculation.
5) Actively search for disconfirming evidence in the KG and surface it under Caveats.
6) Verification loop (CoVe-style, internal): draft → verify KG citations exist → revise.
{cohort_context}

Output format (follow exactly)
1) Mechanism Name: <one concise phrase summarizing a multi-hit vulnerability>
2) Mechanistic Summary: <8–12 sentences>
	- Include 1 short sentence starting with exactly: "Key process phrases:" then list 3–6 key process phrases (2–5 words each) separated by semicolons.
	  These phrases MUST be grounded in the table labels (or terminology-only parenthetical aliases).
	- Include 1–2 sentences explicitly discussing counterfactual expectations (e.g., what would happen if a mediator is inhibited).
	- If supported by the tables, include short sentences starting with exactly: "Primary functional aspect:" and "Secondary/Orthogonal functional aspect:".
	- If multiple aspects are supported, include one short sentence starting with exactly: "Synergy:" describing how the aspects jointly create lethality.
	- When you use background knowledge, include the literal tag "BACKGROUND:" inside the sentence.
3) Evidence Chains:
	- If the prompt tables include `gnn_*` columns, you MUST prioritize and cite chains that traverse high-attention rows (largest numeric `gnn_*_attn` or `gnn_*_cf_drop`).
	- Chain 1 (High-GNN distinct axis 1): <nodeA> -> <nodeB> -> ... -> <nodeZ> | Citations: (...), (...)
	- Chain 2 (High-GNN distinct axis 2): ...
	- (optional) Chain 3/4
4) Key Claims + KG Citations (6–12 bullets):
	- Type: OBSERVED|INFERRED|BACKGROUND|SPECULATION | Claim: ... | Assumptions (if any): ... | Citations: (...), (...)
	  - IMPORTANT: BACKGROUND and SPECULATION items must have Citations: NONE
	  - REQUIRED: include at least 2 INFERRED bullets that are explicit counterfactual tests, each containing the literal prefix "Counterfactual:" in the Claim field.
5) Competing Hypotheses (1–2): <each with 2–4 sentences>
6) Caveats + Missing Evidence (3–6 bullets)
7) Confidence: <number 0..1> | Rationale: <2–4 sentences> | Self-verification Q/A: Q1... A1... (Citations: ...); Q2... A2... (Citations: ...)
8) Suggested validations (optional but helpful): <2–5 concrete experiments or literature checks>
"""


PROMPT_USER_EVIDENCE_TEMPLATE = """Gene pair: ({gene_a}, {gene_b})
{cohort_context}
Evidence triage (provided summary)
Nodes (pd.DataFrame-like — top rows, header = column names):
{node_cols}
{node_lines}

Edges (pd.DataFrame-like — top rows, header = column names):
{edge_cols}
{edge_lines}
"""


# Backward compatible single-string prompt (instructions + evidence).
PROMPT_TEMPLATE = PROMPT_SYSTEM_TEMPLATE + "\n\n" + PROMPT_USER_EVIDENCE_TEMPLATE

# Single-string prompt variant for counterfactual reasoning.
PROMPT_TEMPLATE_COUNTERFACTUAL = PROMPT_SYSTEM_TEMPLATE_COUNTERFACTUAL + "\n\n" + PROMPT_USER_EVIDENCE_TEMPLATE
