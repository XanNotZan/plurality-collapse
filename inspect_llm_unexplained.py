"""Investigate what the 13% of LLM variance that human PCs miss represents."""

import csv
import json
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_NAME = "ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas"
DATASET_SPLIT = "test"
EMBEDDINGS_DIR = "data/embeddings"
OUTPUT_DIR = "data/analysis"
HIDDEN_DIM = 2048
MODEL_NAME = "allenai/kaleido-xl"
FALLBACK_TEMPLATE = "[Generate]:\tAction: ACTION"
GEN_BATCH_SIZE = 16
GEN_MAX_NEW_TOKENS = 128
N_EXEMPLARS = 100

ALL_SOURCES = ["human", "gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
LLM_SOURCE_ORDER = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]

SOURCE_COLORS = {
    "human":   "#2ca02c",
    "gpt3.5":  "#1f77b4",
    "gpt4":    "#aec7e8",
    "claude":  "#ff7f0e",
    "bison":   "#ffbb78",
    "gemma":   "#d62728",
    "mistral": "#9467bd",
    "llama":   "#8c564b",
    "all_llm": "#e377c2",
}


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("inspect_llm_unexplained")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_embeddings(logger: logging.Logger) -> dict[str, np.ndarray]:
    """Load all available embedding matrices from disk."""
    embeddings = {}
    for source in ALL_SOURCES:
        path = os.path.join(EMBEDDINGS_DIR, f"{source}.npy")
        if not os.path.exists(path):
            logger.warning("Missing embedding file: %s", path)
            continue
        matrix = np.load(path)
        if matrix.ndim != 2 or matrix.shape[1] != HIDDEN_DIM:
            logger.error("Unexpected shape for %s: %s, skipping", source, matrix.shape)
            continue
        embeddings[source] = matrix
        logger.info("Loaded %s: shape %s", source, matrix.shape)
    if not embeddings:
        raise SystemExit(f"No embedding files found in {EMBEDDINGS_DIR}/")
    return embeddings


def load_metadata(logger: logging.Logger) -> dict[str, list[dict]]:
    """Load metadata JSON for each source."""
    metadata = {}
    for source in ALL_SOURCES:
        path = os.path.join(EMBEDDINGS_DIR, f"{source}_meta.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            metadata[source] = json.load(f)
        logger.info("Loaded metadata for %s: %d entries", source, len(metadata[source]))
    return metadata


def build_text_lookup(logger: logging.Logger) -> dict[tuple[str, str], str]:
    """Build (submission_id, column) -> text lookup from HF dataset."""
    logger.info("Loading dataset: %s", DATASET_NAME)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    logger.info("Dataset loaded: %d rows", len(ds))

    lookup: dict[tuple[str, str], str] = {}
    for row in ds:
        sid = row["submission_id"]
        for col in row:
            val = row[col]
            if isinstance(val, str) and val.strip():
                lookup[(sid, col)] = val.strip()

    logger.info("Text lookup built: %d entries", len(lookup))
    return lookup


# ── Kaleido ──────────────────────────────────────────────────────────────────
def load_kaleido(logger: logging.Logger):
    """Load Kaleido-XL model and tokenizer."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This script requires a GPU.")

    device = torch.device("cuda")
    logger.info("Loading model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.to(device).eval()
    logger.info("Model loaded (fp16) on %s", torch.cuda.get_device_name(0))

    try:
        template = model.config.task_specific_params["generate"]["template"]
        logger.info("Template: %r", template)
    except (AttributeError, KeyError, TypeError):
        template = FALLBACK_TEMPLATE
        logger.warning("Using fallback template: %r", template)

    return model, tokenizer, template, device


def generate_values_batch(
    model, tokenizer, texts: list[str], template: str, device: torch.device,
) -> list[str]:
    """Generate Kaleido value expressions for a batch of rationale texts."""
    formatted = [template.replace("ACTION", t) for t in texts]
    inputs = tokenizer(
        formatted, return_tensors="pt", padding=True, truncation=True, max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=GEN_MAX_NEW_TOKENS,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # Strip "Value: " prefix
    cleaned = []
    for d in decoded:
        d = d.strip()
        if d.lower().startswith("value:"):
            d = d[len("value:"):].strip()
        cleaned.append(d)
    return cleaned


# ── Step 1: Reconstruction Error by Model ────────────────────────────────────
def compute_per_model_errors(
    embeddings: dict[str, np.ndarray], logger: logging.Logger,
) -> tuple[dict[str, np.ndarray], int]:
    """Fit PCA on human, compute reconstruction error for each LLM source.

    Returns ({source: errors_array}, k_used).
    """
    human = embeddings["human"]
    pca = PCA()
    pca.fit(human)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumulative, 0.90) + 1)
    k = min(k, len(cumulative))
    logger.info("Human PCA: k=%d for 90%% variance", k)

    components_k = pca.components_[:k]
    mean = pca.mean_

    per_model_errors: dict[str, np.ndarray] = {}
    for source in LLM_SOURCE_ORDER:
        if source not in embeddings:
            continue
        data = embeddings[source]
        centered = data - mean
        projected = centered @ components_k.T
        reconstructed = projected @ components_k
        residuals = centered - reconstructed
        errors = (residuals ** 2).sum(axis=1)
        per_model_errors[source] = errors
        logger.info("%s: n=%d, mean_error=%.4f, median=%.4f, std=%.4f",
                    source, len(errors), errors.mean(), np.median(errors), errors.std())

    return per_model_errors, k


def save_reconstruction_csv(
    per_model_errors: dict[str, np.ndarray], k: int,
    output_dir: str, logger: logging.Logger,
) -> None:
    """Save per-model reconstruction error stats."""
    path = os.path.join(output_dir, "llm_reconstruction_by_model.csv")
    fieldnames = ["source", "n_rationales", "k", "mean_error", "median_error", "std_error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for source in LLM_SOURCE_ORDER:
            if source not in per_model_errors:
                continue
            errors = per_model_errors[source]
            writer.writerow({
                "source": source,
                "n_rationales": len(errors),
                "k": k,
                "mean_error": f"{errors.mean():.4f}",
                "median_error": f"{np.median(errors):.4f}",
                "std_error": f"{errors.std():.4f}",
            })
    logger.info("Saved %s", path)


def plot_reconstruction_by_model(
    per_model_errors: dict[str, np.ndarray],
    output_dir: str, logger: logging.Logger,
) -> None:
    """Bar chart of mean reconstruction error per model."""
    sources = [s for s in LLM_SOURCE_ORDER if s in per_model_errors]
    means = [per_model_errors[s].mean() for s in sources]
    stds = [per_model_errors[s].std() for s in sources]
    colors = [SOURCE_COLORS.get(s, "#999999") for s in sources]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(sources, means, yerr=stds, color=colors,
                  edgecolor="black", linewidth=0.5, capsize=4)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{m:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Mean Squared Reconstruction Error")
    ax.set_xlabel("LLM Source")
    ax.set_title("Reconstruction Error: Human PCs Applied to Each LLM")
    fig.tight_layout()
    path = os.path.join(output_dir, "llm_reconstruction_by_model.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Step 2: Per-Model Exemplar Analysis ──────────────────────────────────────
def select_and_generate_per_model(
    per_model_errors: dict[str, np.ndarray],
    metadata: dict[str, list[dict]],
    text_lookup: dict[tuple[str, str], str],
    model, tokenizer, template: str, device: torch.device,
    logger: logging.Logger,
) -> dict[str, dict]:
    """Select exemplars per model, generate Kaleido values.

    Returns {source: {"poorly": [entries], "well": [entries]}}.
    """
    results: dict[str, dict] = {}

    for source in LLM_SOURCE_ORDER:
        if source not in per_model_errors or source not in metadata:
            continue

        errors = per_model_errors[source]
        meta = metadata[source]
        sorted_idx = np.argsort(errors)
        worst_idx = sorted_idx[-N_EXEMPLARS:][::-1]
        best_idx = sorted_idx[:N_EXEMPLARS]

        def build_entries(indices):
            entries = []
            for i in indices:
                m = meta[i]
                text = text_lookup.get((m["submission_id"], m["column"]), "")
                entries.append({
                    "index": int(i),
                    "submission_id": m["submission_id"],
                    "column": m["column"],
                    "source_model": source,
                    "reconstruction_error": float(errors[i]),
                    "rationale_text": text,
                })
            return entries

        poorly = build_entries(worst_idx)
        well = build_entries(best_idx)

        # Generate values for both groups
        for group_name, entries in [("poorly", poorly), ("well", well)]:
            texts = [e["rationale_text"] for e in entries]
            all_values = []
            for i in range(0, len(texts), GEN_BATCH_SIZE):
                batch = texts[i:i + GEN_BATCH_SIZE]
                values = generate_values_batch(model, tokenizer, batch, template, device)
                all_values.extend(values)
            for entry, val in zip(entries, all_values):
                entry["generated_values"] = val

        logger.info("%s: poorly %d unique values, well %d unique values",
                    source,
                    len(set(e["generated_values"] for e in poorly)),
                    len(set(e["generated_values"] for e in well)))

        results[source] = {"poorly": poorly, "well": well}

    return results


def save_permodel_value_diversity(
    per_model_results: dict[str, dict],
    output_dir: str, logger: logging.Logger,
) -> None:
    """Save per-model value diversity CSV."""
    path = os.path.join(output_dir, "llm_permodel_value_diversity.csv")
    fieldnames = ["source", "n_unique_poorly", "n_unique_well",
                  "top_5_poorly", "top_5_well"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for source in LLM_SOURCE_ORDER:
            if source not in per_model_results:
                continue
            poorly = per_model_results[source]["poorly"]
            well = per_model_results[source]["well"]

            def top_5(entries):
                freq: dict[str, int] = {}
                for e in entries:
                    v = e["generated_values"]
                    freq[v] = freq.get(v, 0) + 1
                top = sorted(freq.items(), key=lambda x: -x[1])[:5]
                return "; ".join(f"{v} ({c})" for v, c in top)

            writer.writerow({
                "source": source,
                "n_unique_poorly": len(set(e["generated_values"] for e in poorly)),
                "n_unique_well": len(set(e["generated_values"] for e in well)),
                "top_5_poorly": top_5(poorly),
                "top_5_well": top_5(well),
            })
    logger.info("Saved %s", path)


# ── Step 3: TF-IDF Analysis ─────────────────────────────────────────────────
def compute_tfidf_analysis(
    per_model_errors: dict[str, np.ndarray],
    metadata: dict[str, list[dict]],
    text_lookup: dict[tuple[str, str], str],
    output_dir: str, logger: logging.Logger,
) -> None:
    """TF-IDF analysis: split all LLM rationales by median reconstruction error."""
    # Concatenate errors and metadata across all LLM sources
    all_errors = []
    all_meta = []
    for source in LLM_SOURCE_ORDER:
        if source not in per_model_errors or source not in metadata:
            continue
        all_errors.append(per_model_errors[source])
        all_meta.extend(metadata[source])
    all_errors = np.concatenate(all_errors)

    logger.info("TF-IDF: %d total LLM rationales", len(all_errors))

    # Look up texts
    texts = []
    valid_mask = []
    for m in all_meta:
        t = text_lookup.get((m["submission_id"], m["column"]), "")
        texts.append(t)
        valid_mask.append(bool(t))

    texts = np.array(texts)
    valid_mask = np.array(valid_mask)
    all_errors = all_errors[valid_mask]
    texts = texts[valid_mask]
    logger.info("TF-IDF: %d rationales with text", len(texts))

    # Split by median
    median_error = np.median(all_errors)
    poorly_mask = all_errors >= median_error
    well_mask = ~poorly_mask
    logger.info("TF-IDF: %d poorly-captured, %d well-captured (median=%.4f)",
                poorly_mask.sum(), well_mask.sum(), median_error)

    # Fit TF-IDF on all texts
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", min_df=5)
    tfidf_matrix = vectorizer.fit_transform(texts.tolist())
    feature_names = vectorizer.get_feature_names_out()

    # Mean TF-IDF per group
    mean_poorly = np.asarray(tfidf_matrix[poorly_mask].mean(axis=0)).flatten()
    mean_well = np.asarray(tfidf_matrix[well_mask].mean(axis=0)).flatten()
    diff = mean_poorly - mean_well

    # Sort by difference
    sorted_idx = np.argsort(diff)
    top_poorly_idx = sorted_idx[-30:][::-1]  # highest diff = most associated with poorly
    top_well_idx = sorted_idx[:30]  # lowest diff = most associated with well

    # Save CSV
    csv_path = os.path.join(output_dir, "llm_tfidf_distinguishing_terms.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "term", "mean_tfidf_poorly", "mean_tfidf_well", "difference", "direction"])
        writer.writeheader()
        for idx in top_poorly_idx:
            writer.writerow({
                "term": feature_names[idx],
                "mean_tfidf_poorly": f"{mean_poorly[idx]:.6f}",
                "mean_tfidf_well": f"{mean_well[idx]:.6f}",
                "difference": f"{diff[idx]:.6f}",
                "direction": "poorly_captured",
            })
        for idx in top_well_idx:
            writer.writerow({
                "term": feature_names[idx],
                "mean_tfidf_poorly": f"{mean_poorly[idx]:.6f}",
                "mean_tfidf_well": f"{mean_well[idx]:.6f}",
                "difference": f"{diff[idx]:.6f}",
                "direction": "well_captured",
            })
    logger.info("Saved %s", csv_path)

    # Diverging bar chart
    plot_poorly_idx = sorted_idx[-20:][::-1]
    plot_well_idx = sorted_idx[:20][::-1]  # reverse so largest magnitude at top

    fig, ax = plt.subplots(figsize=(10, 10))

    # Poorly-captured terms (positive side)
    poorly_terms = [feature_names[i] for i in plot_poorly_idx]
    poorly_diffs = [diff[i] for i in plot_poorly_idx]

    # Well-captured terms (negative side)
    well_terms = [feature_names[i] for i in plot_well_idx]
    well_diffs = [diff[i] for i in plot_well_idx]

    all_terms = poorly_terms + well_terms
    all_diffs = poorly_diffs + well_diffs
    y_pos = np.arange(len(all_terms))
    colors = ["#d62728" if d > 0 else "#1f77b4" for d in all_diffs]

    ax.barh(y_pos, all_diffs, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_terms, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("TF-IDF Difference (poorly - well captured)")
    ax.set_title("Distinguishing Terms: Poorly vs Well Captured LLM Rationales")

    # Add legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#d62728", label="More in poorly-captured"),
        Patch(color="#1f77b4", label="More in well-captured"),
    ], loc="lower right", fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, "llm_tfidf_terms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Step 4: Value Overlap Analysis ───────────────────────────────────────────
def compute_value_overlap(
    per_model_results: dict[str, dict],
    output_dir: str, logger: logging.Logger,
) -> None:
    """Compute Jaccard overlap between LLM-unexplained and human-unexplained values."""
    # Load human-unexplained values
    human_path = os.path.join(output_dir, "unexplained_human_exemplars.json")
    if not os.path.exists(human_path):
        logger.warning("Cannot compute overlap: %s not found", human_path)
        return

    with open(human_path, encoding="utf-8") as f:
        human_exemplars = json.load(f)
    human_values = set(e["generated_values"] for e in human_exemplars if e.get("generated_values"))

    # Collect LLM poorly-captured values across all models
    llm_values = set()
    for source in LLM_SOURCE_ORDER:
        if source not in per_model_results:
            continue
        for e in per_model_results[source]["poorly"]:
            if e.get("generated_values"):
                llm_values.add(e["generated_values"])

    intersection = human_values & llm_values
    union = human_values | llm_values
    jaccard = len(intersection) / len(union) if union else 0.0

    human_only = human_values - llm_values
    llm_only = llm_values - human_values

    logger.info("Value overlap: Jaccard=%.3f", jaccard)
    logger.info("  Human-unexplained unique: %d", len(human_values))
    logger.info("  LLM-unexplained unique: %d", len(llm_values))
    logger.info("  Shared: %d, Human-only: %d, LLM-only: %d",
                len(intersection), len(human_only), len(llm_only))

    # Save CSV
    path = os.path.join(output_dir, "value_overlap_analysis.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "jaccard", "n_human_unique", "n_llm_unique", "n_shared",
            "n_human_only", "n_llm_only",
            "shared_values", "human_only_values", "llm_only_values",
        ])
        writer.writeheader()
        writer.writerow({
            "jaccard": f"{jaccard:.3f}",
            "n_human_unique": len(human_values),
            "n_llm_unique": len(llm_values),
            "n_shared": len(intersection),
            "n_human_only": len(human_only),
            "n_llm_only": len(llm_only),
            "shared_values": "; ".join(sorted(intersection)),
            "human_only_values": "; ".join(sorted(human_only)),
            "llm_only_values": "; ".join(sorted(llm_only)),
        })
    logger.info("Saved %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    logger = setup_logging()
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data (once)
    embeddings = load_embeddings(logger)
    metadata = load_metadata(logger)
    text_lookup = build_text_lookup(logger)

    if "human" not in embeddings:
        raise SystemExit("Human embeddings not found.")

    # ── Step 1: Reconstruction error by model ────────────────────────────────
    logger.info("=== Step 1: Reconstruction error by model ===")
    per_model_errors, k = compute_per_model_errors(embeddings, logger)
    save_reconstruction_csv(per_model_errors, k, OUTPUT_DIR, logger)
    plot_reconstruction_by_model(per_model_errors, OUTPUT_DIR, logger)

    # ── Step 2: Per-model exemplar analysis ──────────────────────────────────
    logger.info("=== Step 2: Per-model exemplar analysis ===")
    model, tokenizer, template, device = load_kaleido(logger)
    per_model_results = select_and_generate_per_model(
        per_model_errors, metadata, text_lookup,
        model, tokenizer, template, device, logger)
    del model
    torch.cuda.empty_cache()

    save_permodel_value_diversity(per_model_results, OUTPUT_DIR, logger)

    # ── Step 3: TF-IDF analysis ──────────────────────────────────────────────
    logger.info("=== Step 3: TF-IDF distinguishing terms ===")
    compute_tfidf_analysis(per_model_errors, metadata, text_lookup, OUTPUT_DIR, logger)

    # ── Step 4: Value overlap ────────────────────────────────────────────────
    logger.info("=== Step 4: Value overlap analysis ===")
    compute_value_overlap(per_model_results, OUTPUT_DIR, logger)

    elapsed = time.time() - start_time
    logger.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
