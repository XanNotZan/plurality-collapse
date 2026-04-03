"""TF-IDF analysis of human rationales poorly captured by LLM PCs."""

import csv
import json
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_NAME = "ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas"
DATASET_SPLIT = "test"
EMBEDDINGS_DIR = "data/embeddings"
OUTPUT_DIR = "data/analysis"
HIDDEN_DIM = 2048

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

# Stylistic markers: Reddit slang, internet abbreviations, informal/structural terms
STYLISTIC_MARKERS = {
    "edit", "update", "throwaway", "deleted", "op", "imo", "idk", "btw", "tbh",
    "lol", "lmao", "omg", "wtf", "smh", "ngl", "iirc", "tldr", "tl",
    "nta", "yta", "esh", "nah", "aita",
    "gonna", "wanna", "gotta", "kinda", "sorta", "dunno",
    "yeah", "yep", "nope", "ok", "okay",
    "shit", "damn", "hell", "ass", "crap", "fuck", "fucking", "bullshit",
    "dude", "bro", "bruh", "yikes", "wow",
    "literally", "honestly", "basically", "actually", "definitely", "probably",
    "like", "just", "really", "pretty", "super", "totally", "absolutely",
    "reddit", "post", "comment", "thread", "sub", "subreddit", "upvote", "downvote",
    "lmk", "fwiw", "afaik", "iirc",
}


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("inspect_human_unexplained_tfidf")
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


# ── Step 1: Reconstruction Error ─────────────────────────────────────────────
def compute_human_reconstruction_errors(
    embeddings: dict[str, np.ndarray], logger: logging.Logger,
) -> tuple[np.ndarray, int]:
    """Fit PCA on all_llm, project human data, return per-rationale errors and k."""
    # Build all_llm matrix
    matrices = [embeddings[s] for s in LLM_SOURCE_ORDER if s in embeddings]
    if not matrices:
        raise SystemExit("No LLM embeddings found.")
    all_llm = np.vstack(matrices)
    logger.info("all_llm matrix: shape %s", all_llm.shape)

    # Fit PCA
    pca = PCA()
    pca.fit(all_llm)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumulative, 0.90) + 1)
    k = min(k, len(cumulative))
    logger.info("all_llm PCA: k=%d for 90%% variance", k)

    # Reconstruction error for human
    human = embeddings["human"]
    components_k = pca.components_[:k]
    centered = human - pca.mean_
    projected = centered @ components_k.T
    reconstructed = projected @ components_k
    residuals = centered - reconstructed
    errors = (residuals ** 2).sum(axis=1)

    logger.info("Human reconstruction: mean=%.4f, median=%.4f, max=%.4f",
                errors.mean(), np.median(errors), errors.max())
    return errors, k


# ── Step 2: TF-IDF Analysis ─────────────────────────────────────────────────
def compute_tfidf_analysis(
    errors: np.ndarray,
    text_lookup: dict[tuple[str, str], str],
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """TF-IDF on human rationales split by median reconstruction error.

    Returns (diff, mean_poorly, mean_well, feature_names, sorted_idx).
    """
    # Load human metadata
    meta_path = os.path.join(EMBEDDINGS_DIR, "human_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # Look up texts
    texts = []
    valid_indices = []
    for i, m in enumerate(meta):
        t = text_lookup.get((m["submission_id"], m["column"]), "")
        if t:
            texts.append(t)
            valid_indices.append(i)

    valid_indices = np.array(valid_indices)
    valid_errors = errors[valid_indices]
    logger.info("TF-IDF: %d human rationales with text", len(texts))

    # Split by median
    median_error = np.median(valid_errors)
    poorly_mask = valid_errors >= median_error
    well_mask = ~poorly_mask
    logger.info("TF-IDF: %d poorly-captured, %d well-captured (median=%.4f)",
                poorly_mask.sum(), well_mask.sum(), median_error)

    # Fit TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", min_df=5)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    mean_poorly = np.asarray(tfidf_matrix[poorly_mask].mean(axis=0)).flatten()
    mean_well = np.asarray(tfidf_matrix[well_mask].mean(axis=0)).flatten()
    diff = mean_poorly - mean_well

    return diff, mean_poorly, mean_well, feature_names


def save_tfidf_csv(
    diff: np.ndarray, mean_poorly: np.ndarray, mean_well: np.ndarray,
    feature_names, output_dir: str, logger: logging.Logger,
) -> tuple[list[int], list[int]]:
    """Save TF-IDF distinguishing terms CSV. Returns (top_poorly_idx, top_well_idx)."""
    sorted_idx = np.argsort(diff)
    top_poorly_idx = sorted_idx[-30:][::-1]
    top_well_idx = sorted_idx[:30]

    path = os.path.join(output_dir, "human_tfidf_distinguishing_terms.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
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
    logger.info("Saved %s", path)
    return top_poorly_idx.tolist(), top_well_idx.tolist()


def plot_tfidf_terms(
    diff: np.ndarray, feature_names,
    output_dir: str, logger: logging.Logger,
) -> None:
    """Diverging bar chart of top TF-IDF terms."""
    sorted_idx = np.argsort(diff)
    plot_poorly_idx = sorted_idx[-20:][::-1]
    plot_well_idx = sorted_idx[:20][::-1]

    poorly_terms = [feature_names[i] for i in plot_poorly_idx]
    poorly_diffs = [diff[i] for i in plot_poorly_idx]
    well_terms = [feature_names[i] for i in plot_well_idx]
    well_diffs = [diff[i] for i in plot_well_idx]

    all_terms = poorly_terms + well_terms
    all_diffs = poorly_diffs + well_diffs
    y_pos = np.arange(len(all_terms))

    # Human green for poorly-captured (human-specific), blue for well-captured
    colors = [SOURCE_COLORS["human"] if d > 0 else SOURCE_COLORS["all_llm"]
              for d in all_diffs]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(y_pos, all_diffs, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_terms, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("TF-IDF Difference (poorly - well captured)")
    ax.set_title("Distinguishing Terms: Poorly vs Well Captured Human Rationales")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=SOURCE_COLORS["human"], label="More in poorly-captured (human-specific)"),
        Patch(color=SOURCE_COLORS["all_llm"], label="More in well-captured (LLM-like)"),
    ], loc="lower right", fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, "human_tfidf_terms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Step 3: Categorize Terms ────────────────────────────────────────────────
def categorize_term(term: str) -> str:
    """Classify a term as stylistic, content, or ambiguous."""
    t = term.lower().strip()
    if t in STYLISTIC_MARKERS:
        return "stylistic"
    # Additional heuristics: very short words (2 chars) are often stylistic
    if len(t) <= 2:
        return "stylistic"
    return "content"


def save_categorization(
    top_poorly_idx: list[int], diff: np.ndarray, feature_names,
    output_dir: str, logger: logging.Logger,
) -> dict[str, int]:
    """Categorize top 30 poorly-captured terms and save CSV."""
    path = os.path.join(output_dir, "human_tfidf_categorization.csv")
    counts = {"stylistic": 0, "content": 0, "ambiguous": 0}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["term", "category", "difference"])
        writer.writeheader()
        for idx in top_poorly_idx:
            term = feature_names[idx]
            category = categorize_term(term)
            counts[category] += 1
            writer.writerow({
                "term": term,
                "category": category,
                "difference": f"{diff[idx]:.6f}",
            })

    logger.info("Saved %s", path)
    logger.info("Categorization: %d stylistic, %d content, %d ambiguous",
                counts["stylistic"], counts["content"], counts["ambiguous"])
    return counts


# ── Step 4: Compare with LLM-side ───────────────────────────────────────────
def compare_with_llm_tfidf(
    human_counts: dict[str, int],
    output_dir: str, logger: logging.Logger,
) -> None:
    """Load LLM TF-IDF results, categorize, and save comparison."""
    llm_path = os.path.join(output_dir, "llm_tfidf_distinguishing_terms.csv")
    if not os.path.exists(llm_path):
        logger.warning("Cannot compare: %s not found", llm_path)
        return

    # Categorize LLM poorly-captured terms
    llm_counts = {"stylistic": 0, "content": 0, "ambiguous": 0}
    with open(llm_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["direction"] == "poorly_captured":
                category = categorize_term(row["term"])
                llm_counts[category] += 1

    logger.info("LLM poorly-captured categorization: %d stylistic, %d content, %d ambiguous",
                llm_counts["stylistic"], llm_counts["content"], llm_counts["ambiguous"])

    # Save comparison
    path = os.path.join(output_dir, "tfidf_comparison_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "side", "n_stylistic", "n_content", "n_ambiguous",
            "pct_stylistic", "pct_content"])
        writer.writeheader()
        for side, counts in [("human", human_counts), ("llm", llm_counts)]:
            total = sum(counts.values()) or 1
            writer.writerow({
                "side": side,
                "n_stylistic": counts["stylistic"],
                "n_content": counts["content"],
                "n_ambiguous": counts["ambiguous"],
                "pct_stylistic": f"{100 * counts['stylistic'] / total:.1f}",
                "pct_content": f"{100 * counts['content'] / total:.1f}",
            })
    logger.info("Saved %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    logger = setup_logging()
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    embeddings = load_embeddings(logger)
    text_lookup = build_text_lookup(logger)

    if "human" not in embeddings:
        raise SystemExit("Human embeddings not found.")

    # Step 1: Reconstruction errors
    logger.info("=== Step 1: Reconstruction error for human rationales ===")
    errors, k = compute_human_reconstruction_errors(embeddings, logger)

    # Step 2: TF-IDF
    logger.info("=== Step 2: TF-IDF analysis ===")
    diff, mean_poorly, mean_well, feature_names = compute_tfidf_analysis(
        errors, text_lookup, logger)
    top_poorly_idx, top_well_idx = save_tfidf_csv(
        diff, mean_poorly, mean_well, feature_names, OUTPUT_DIR, logger)
    plot_tfidf_terms(diff, feature_names, OUTPUT_DIR, logger)

    # Step 3: Categorize
    logger.info("=== Step 3: Term categorization ===")
    human_counts = save_categorization(top_poorly_idx, diff, feature_names, OUTPUT_DIR, logger)

    # Step 4: Compare with LLM side
    logger.info("=== Step 4: Compare with LLM-side TF-IDF ===")
    compare_with_llm_tfidf(human_counts, OUTPUT_DIR, logger)

    elapsed = time.time() - start_time
    logger.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
