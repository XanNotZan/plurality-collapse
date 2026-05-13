# Plurality Collapse

Research project investigating whether large language models flatten the diversity of human moral reasoning. The pipeline embeds human and LLM rationales from the Sachdeva et al. r/AmITheAsshole corpus into the Kaleido-XL moral encoder and measures the geometric dimensionality and per-dilemma diversity of each source.

Milestone 2 reports the corpus-level dimensionality gap of 35 to 41 percent and shows it survives matched-sample subsampling, length stratification, kernel PCA, and a Sentence-BERT encoder swap. Milestone 3 expands the human side with 68,391 ArcticShift Reddit comments across 1,991 dilemmas and runs five falsification probes on the original claim. Three of those probes show the corpus gap is largely register-confounded, but a per-dilemma analysis finds a content-grounded gap of roughly two-to-one that survives every surface, length, and author control. The final framing decomposition shows the within-cluster residual is a length-robust signature of personal, concrete, and partisan framing in humans versus abstract and balanced framing in LLMs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create a `.env` file with `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`, `LLM_BASE_URL`, `LLM_API_KEY`, and `HF_TOKEN`. The LLM verdict-filter step uses Ollama with `qwen2.5:14b` over its OpenAI-compatible API at `http://localhost:11434/v1`. Kaleido-XL embedding extraction and the Qwen 2.5 paraphrase and qualitative probe runs need CUDA.

## Layout

- `data_collection/` Reddit PRAW pulls and ArcticShift archive gathering
- `embedding/` Kaleido-XL embedding extraction for all sources
- `milestone2/` corpus-level dimensionality analysis and robustness probes
- `milestone3/` per-dilemma diversity, falsification probes, framing decomposition
- `figures/` figure assembly scripts feeding the LaTeX reports
- `report/` LaTeX sources and compiled PDFs
- `data/` all run outputs, gitignored

## Reports

- `report/milestone2.pdf` corpus-level dimensionality gap
- `report/milestone3.pdf` falsification probes and framing decomposition
- `report/report.pdf` longer combined draft

## Pipeline

All scripts run from the project root. Embeddings must exist before any analysis script will run, and a few analyses are chained.

**Milestone 2.** Run `embedding/extract_embeddings.py` to produce one numpy array and metadata file per source under `data/embeddings/`. Then `milestone2/analyze_embeddings.py` writes the PCA summary table that `milestone2/robustness_checks.py` consumes. `milestone2/cross_space_projection.py`, `milestone2/consensus_stratification.py`, `milestone2/alternative_encoder.py`, and `milestone2/frequency_geometry.py` only need the embeddings. `milestone2/inspect_unexplained_variance.py` must run before `milestone2/inspect_llm_unexplained.py`, and `milestone2/value_diversity_gradient.py` must run before `milestone2/compare_value_frequencies.py` and `milestone2/cluster_value_labels.py`.

**Milestone 3.** Run `data_collection/arctic_shift_gather.py` to pull the ArcticShift archive, then `milestone3/arctic_within_dilemma.py` to embed the human comments and compute the per-dilemma diversity baseline. Every other `arctic_*` script in `milestone3/` depends on those embeddings. The corpus probes (`milestone3/milestone3_diagnostics.py`, `milestone3/milestone3_extra.py`, `milestone3/milestone3_more.py`) only need the original Sachdeva embeddings. The modernity and register-paraphrase pipeline runs as `milestone3/milestone3_modern_gen.py` then `milestone3/milestone3_clean_paraphrase.py` then `milestone3/milestone3_analyze_modern.py` and `milestone3/milestone3_paraphrase_clusters.py`. The within-cluster framing decomposition runs as `milestone3/milestone3_lexical_features.py`, `milestone3/milestone3_diff_pca.py`, and `milestone3/milestone3_qualitative_probe.py` and is assembled by `figures/milestone3_framing_synthesis.py`. The two milestone-3 summary figures come from `figures/milestone3_summary_figure.py` and `figures/milestone3_framing_synthesis.py`.
