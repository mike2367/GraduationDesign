
# SL-Lex (SL-LLM-explainer)

SL-Lex is an LLM-based explanation framework for interpreting the biological mechanisms behind **Synthetic Lethality (SL)**. It combines (1) knowledge graph (KG) evidence aggregation, (2) subgraph-centric evidence packaging, and (3) LLM prompting + evaluation to produce mechanism-oriented study output.

## Background Reading
We highly recommend the following readings which can provide better understandings of the project background and motives

- [Synthetic lethality in cancer drug discovery: challenges and opportunities](https://www.nature.com/articles/s41573-025-01273-7)
- [SL-MERK: Synthetic Lethality Mechanism Explainer based on GraphRAG and Knowledge Graph](https://pubmed.ncbi.nlm.nih.gov/41337148/)
- [SynLethDB 2.0: a web-based knowledge graph database on synthetic lethality for novel anticancer drug discovery](https://pmc.ncbi.nlm.nih.gov/articles/PMC9216587/)

## What’s inside

- **KG construction**: build a multi-source biological KG and export it to `GraphML` + interactive `HTML`.
- **Subgraph extraction**: retrieve an explanation subgraph around each SL gene pair and export:
	- `nodes.csv` / `edges.csv`
	- `subgraph.html`
	- LLM prompts (`*_prompt.txt` and `*_prompt_messages.json`)
- **LLM explanation & evaluation**: run multiple prompting strategies and score outputs (faithfulness / hallucination / feature-matching metrics).
- **GNN module**: train an Attention-RGCN and use its learned weights as guidance for graph search and prompt scaffolding.



## Repository layout

- `src/graph_module/`: KG construction utilities and visualization
- `src/algorithm_module/`: graph search + subgraph extraction + prompt builders
- `src/LLM_module/`: LLM clients/strategies + evaluation + scoring modules
- `src/GNN_algo_module/`: GNN data/model/train code
- `src/visualization/`: plotting utilities for evaluation results
- `src/baseline_configs/`: LLM baseline config presets (model id / temperature / etc.)

## Installation

### 1) Create an environment

```bash
python -m venv .venv
```

### 2) Install Python dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

### 3) Download local checkpoints

Several modules expect local checkpoints or cached Hugging Face models. 
We recommend you download them and substitute to your own repo directory.

| Purpose | Default local path | Source |
| --- | --- | --- |
| Semantic embedding model (**required**) | `/data/guoyu/HF-models/MedCPT-Query-Encoder` | [ncbi/MedCPT-Query-Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder) |
| Keyphrase token-classification model (**optional**) | `/data/guoyu/HF-models/keyphrase-extraction-kbir-inspec` | [ml6team/keyphrase-extraction-kbir-inspec](https://huggingface.co/ml6team/keyphrase-extraction-kbir-inspec) |
| Keyphrase seq2seq model (**optional**) | `/data/guoyu/HF-models/bart_finetuned_keyphrase_extraction` | [aglazkova/bart_finetuned_keyphrase_extraction](https://huggingface.co/aglazkova/bart_finetuned_keyphrase_extraction) |
| Expert judge encoder (**optional**) | `/data/guoyu/HF-models/all-MiniLM-L6-v2` | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Hallucination / NLI model (**optional**) | `/data/guoyu/HF-models/PubMedBERT-MNLI-MedNLI` | [pritamdeka/PubMedBERT-MNLI-MedNLI](https://huggingface.co/pritamdeka/PubMedBERT-MNLI-MedNLI) |

**optional** models are only used in evaluation stage, if you have no intention to run LLM module, we recommend you downloading **MedCPT only**.

## Getting started

This repo currently uses **absolute paths** in configuration files (e.g., `/data/guoyu/KG-LLM-XSL/...`). On a new machine (especially Windows), you should update these paths before running.

### Step 0 — Configure paths (required)

Update the following config files to point to your local data/output folders:

- `src/graph_module/graph_config.py`: `DATA_DIR`, `OUT_DIR`, and source resource paths
- `src/algorithm_module/output_config.py`: `SUBGRAPH_OUTPUT_DIR`, `GNN_SCAFFOLDING_REFERENCE_PATH`
- `src/LLM_module/eval_config.py`: `DEFAULT_EVAL_OUT_DIR`, `EVAL_PROMPTS_DIR`, `EVAL_GROUND_TRUTH_PATH`, and local HF model paths
- `src/visualization/vis_config.py`: `DATA_ROOT` / `EVAL_RESULTS_DIR` (if you want to run plotting)

Also note:

- `src/algorithm_module/algo_config.py` **requires** a GNN-produced `learned_weights.json` (loaded at import time). If you don’t have it yet, run GNN training first or update `_WEIGHTS_FILE` to an existing checkpoint.

### Step 1 — Build the knowledge graph

Build the full KG and export `GraphML` + interactive `HTML`:

```bash
python src/graph_module/graph_construct.py
```

Outputs (by default) go to `OUT_DIR / ablation_graphs/` as:

- `full.graphml`
- `full.html`

> [!NOTE]
> The list of **300** gene pairs used to build the original graph is provided in `data/sl_pairs_common.json` if you intend to add your own pair to the full graph, please update the json file or take reference from the case study pipeline we provided in `src\case_study`.

### Step 2 — Extract subgraphs and generate prompts

```bash
python src/algorithm_module/subgraph_extraction.py
```

For each gene pair, this writes a folder under `SUBGRAPH_OUTPUT_DIR/GENE_A_GENE_B/`, including:

- `nodes.csv`, `edges.csv`
- `subgraph.graphml`, `subgraph.html`
- `GENE_A_GENE_B_prompt.txt`
- `GENE_A_GENE_B_prompt_messages.json` (system+user messages)

### Step 3 — Run LLM evaluation

Set your API credentials (for the default OpenAI-compatible client):

- Windows PowerShell

```powershell
$env:AIGC_BEST_BASE_URL = "http://<your-endpoint>/v1"
$env:AIGC_BEST_API_KEY = "<your-api-key>"
```

- macOS/Linux

```bash
export AIGC_BEST_BASE_URL="http://<your-endpoint>/v1"
export AIGC_BEST_API_KEY="<your-api-key>"
```

Then run:

```bash
python src/LLM_module/evaluate_llm_strategies.py
```

You can switch models/decoding settings by changing `BASELINE_CONFIG` in `src/LLM_module/eval_config.py` (it loads a preset from `src/baseline_configs/`).

> [!NOTE]
>The list of **112** gene pairs which have definitive literature support is provided in `data\SL_MERK_groundtruth_pairs.json`, but the **exact explanation will not be provided**, if you intend to run those evaluation yourself, please contact the authors of [**SL-MERK**](https://pubmed.ncbi.nlm.nih.gov/41337148/) for those explanation groundtruth.


