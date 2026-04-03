"""Robustness checks for moral reasoning embedding analysis."""

import csv
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.decomposition import PCA

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_NAME = "ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas"
DATASET_SPLIT = "test"
EMBEDDINGS_DIR = "data/embeddings"
OUTPUT_DIR = "data/analysis"
SUMMARY_CSV = "data/analysis/summary_table.csv"
HIDDEN_DIM = 2048
SUBSAMPLE_SEEDS = [42, 43, 44, 45, 46]

LLM_SOURCES = {
    "gpt3.5":  ["gpt3.5_reason_1", "gpt3.5_reason_2", "gpt3.5_reason_3"],
    "gpt4":    ["gpt4_reason_1", "gpt4_reason_2"],
    "claude":  ["claude_reason_1", "claude_reason_2", "claude_reason_3"],
    "bison":   ["bison_reason_1", "bison_reason_2", "bison_reason_3"],
    "gemma":   ["gemma_reason_1", "gemma_reason_2", "gemma_reason_3"],
    "mistral": ["mistral_reason_1", "mistral_reason_2", "mistral_reason_3"],
    "llama":   ["llama_reason_1", "llama_reason_2", "llama_reason_3"],
}

ALL_SOURCES = ["human", "gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]

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

PLOT_ORDER = ["human"] + sorted(LLM_SOURCES) + ["all_llm"]


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("robustness_checks")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger


# ── Shared Helpers ─────────────────────────────────────────────────────────────
def load_pr_from_summary_csv(
    path: str, logger: logging.Logger,
) -> dict[str, float]:
    """Load participation ratios from summary_table.csv."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run analyze_embeddings.py first.")
    pr_map: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pr_map[row["source"]] = float(row["participation_ratio"])
    logger.info("Loaded PR values for %d sources from %s", len(pr_map), path)
    return pr_map


def compute_pca_metrics(matrix: np.ndarray) -> dict:
    """Run PCA and return PR, components_90, components_95."""
    pca = PCA()
    pca.fit(matrix)

    eigenvalues = pca.explained_variance_
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    pr = float((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum())

    comp90 = int(np.searchsorted(cumulative, 0.90) + 1)
    comp95 = int(np.searchsorted(cumulative, 0.95) + 1)
    comp90 = min(comp90, len(cumulative))
    comp95 = min(comp95, len(cumulative))

    return {"pr": pr, "comp90": comp90, "comp95": comp95}


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
        if matrix.shape[0] < 2:
            logger.warning("Too few rationales for %s (%d), skipping", source, matrix.shape[0])
            continue
        embeddings[source] = matrix
        logger.info("Loaded %s: shape %s", source, matrix.shape)
    if not embeddings:
        raise SystemExit(f"No embedding files found in {EMBEDDINGS_DIR}/")
    return embeddings


# ── Check 1: Rationale Length vs. PR ─────────────────────────────────────────
def compute_rationale_lengths(
    ds, logger: logging.Logger,
) -> dict[str, dict]:
    """Compute average rationale length (chars and words) per source."""
    results: dict[str, dict] = {}
    all_llm_chars: list[int] = []
    all_llm_words: list[int] = []

    # Human
    char_lens = []
    word_lens = []
    for row in ds:
        text = row.get("top_comment")
        if text and isinstance(text, str) and text.strip():
            t = text.strip()
            char_lens.append(len(t))
            word_lens.append(len(t.split()))
    results["human"] = {
        "avg_chars": np.mean(char_lens) if char_lens else 0.0,
        "avg_words": np.mean(word_lens) if word_lens else 0.0,
        "n_rationales": len(char_lens),
    }
    logger.info("human: %d rationales, avg %.0f chars, %.1f words",
                len(char_lens), results["human"]["avg_chars"], results["human"]["avg_words"])

    # LLM sources
    for source_name, columns in LLM_SOURCES.items():
        char_lens = []
        word_lens = []
        for row in ds:
            for col in columns:
                text = row.get(col)
                if text and isinstance(text, str) and text.strip():
                    t = text.strip()
                    char_lens.append(len(t))
                    word_lens.append(len(t.split()))
        results[source_name] = {
            "avg_chars": np.mean(char_lens) if char_lens else 0.0,
            "avg_words": np.mean(word_lens) if word_lens else 0.0,
            "n_rationales": len(char_lens),
        }
        all_llm_chars.extend(char_lens)
        all_llm_words.extend(word_lens)
        logger.info("%s: %d rationales, avg %.0f chars, %.1f words",
                    source_name, len(char_lens),
                    results[source_name]["avg_chars"], results[source_name]["avg_words"])

    # all_llm aggregate
    results["all_llm"] = {
        "avg_chars": np.mean(all_llm_chars) if all_llm_chars else 0.0,
        "avg_words": np.mean(all_llm_words) if all_llm_words else 0.0,
        "n_rationales": len(all_llm_chars),
    }
    logger.info("all_llm: %d rationales, avg %.0f chars, %.1f words",
                len(all_llm_chars),
                results["all_llm"]["avg_chars"], results["all_llm"]["avg_words"])

    return results


def check_length_vs_pr(logger: logging.Logger) -> None:
    """Check 1: Correlate rationale length with participation ratio."""
    # Load dataset
    logger.info("Loading dataset: %s", DATASET_NAME)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    logger.info("Dataset loaded: %d rows", len(ds))

    # Compute lengths
    length_stats = compute_rationale_lengths(ds, logger)

    # Load PR values
    pr_map = load_pr_from_summary_csv(SUMMARY_CSV, logger)

    # Build results table
    table_rows = []
    for source in ALL_SOURCES + ["all_llm"]:
        if source not in length_stats or source not in pr_map:
            continue
        table_rows.append({
            "source": source,
            "avg_chars": length_stats[source]["avg_chars"],
            "avg_words": length_stats[source]["avg_words"],
            "participation_ratio": pr_map[source],
        })

    # Log table
    logger.info("%-10s %10s %10s %8s", "source", "avg_chars", "avg_words", "PR")
    for r in table_rows:
        logger.info("%-10s %10.0f %10.1f %8.2f",
                    r["source"], r["avg_chars"], r["avg_words"], r["participation_ratio"])

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "length_vs_pr_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "avg_chars", "avg_words", "participation_ratio"])
        writer.writeheader()
        for r in table_rows:
            writer.writerow({
                "source": r["source"],
                "avg_chars": f"{r['avg_chars']:.2f}",
                "avg_words": f"{r['avg_words']:.2f}",
                "participation_ratio": f"{r['participation_ratio']:.2f}",
            })
    logger.info("Saved %s", csv_path)

    # Correlation (individual sources only, exclude all_llm)
    corr_rows = [r for r in table_rows if r["source"] != "all_llm"]
    if len(corr_rows) < 3:
        logger.warning("Too few sources (%d) for meaningful correlation", len(corr_rows))
        return

    words = np.array([r["avg_words"] for r in corr_rows])
    prs = np.array([r["participation_ratio"] for r in corr_rows])

    if HAS_SCIPY:
        pearson_r, pearson_p = pearsonr(words, prs)
        spearman_r, spearman_p = spearmanr(words, prs)
    else:
        pearson_r = float(np.corrcoef(words, prs)[0, 1])
        pearson_p = float("nan")
        spearman_r = float("nan")
        spearman_p = float("nan")
        logger.warning("scipy not available; Spearman not computed")

    logger.info("Pearson  r=%.3f  p=%.4f  (n=%d sources)", pearson_r, pearson_p, len(corr_rows))
    logger.info("Spearman r=%.3f  p=%.4f  (n=%d sources)", spearman_r, spearman_p, len(corr_rows))

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in corr_rows:
        s = r["source"]
        ax.scatter(r["avg_words"], r["participation_ratio"],
                   color=SOURCE_COLORS.get(s, "#999999"), s=80, edgecolors="black", linewidth=0.5, zorder=3)
        ax.annotate(s, (r["avg_words"], r["participation_ratio"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xlabel("Average Rationale Length (words)")
    ax.set_ylabel("Participation Ratio")
    ax.set_title(f"Rationale Length vs. PR (Pearson r={pearson_r:.3f}, Spearman \u03c1={spearman_r:.3f})")
    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "length_vs_pr.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", plot_path)


# ── Check 2: Subsampling Robustness ─────────────────────────────────────────
def check_subsampling_robustness(logger: logging.Logger) -> None:
    """Check 2: Subsample LLMs to human sample size and recompute PCA."""
    embeddings = load_embeddings(logger)

    if "human" not in embeddings:
        raise SystemExit("Human embeddings not found; cannot determine subsample size.")
    human_n = embeddings["human"].shape[0]
    logger.info("Subsample target: n=%d (human sample size)", human_n)

    results = []
    for source in ALL_SOURCES:
        if source not in embeddings:
            continue
        matrix = embeddings[source]
        n_rows = matrix.shape[0]

        if n_rows <= human_n:
            # No subsampling needed (human, or any source smaller than human)
            metrics = compute_pca_metrics(matrix)
            results.append({
                "source": source,
                "n_samples": n_rows,
                "pr_mean": metrics["pr"],
                "pr_std": 0.0,
                "comp90_mean": float(metrics["comp90"]),
                "comp90_std": 0.0,
                "comp95_mean": float(metrics["comp95"]),
                "comp95_std": 0.0,
            })
            logger.info("%s: n=%d (no subsampling), PR=%.2f, 90%%=%d, 95%%=%d",
                        source, n_rows, metrics["pr"], metrics["comp90"], metrics["comp95"])
        else:
            pr_vals = []
            comp90_vals = []
            comp95_vals = []
            for seed in SUBSAMPLE_SEEDS:
                rng = np.random.RandomState(seed)
                indices = rng.choice(n_rows, human_n, replace=False)
                sub_matrix = matrix[indices]
                metrics = compute_pca_metrics(sub_matrix)
                pr_vals.append(metrics["pr"])
                comp90_vals.append(metrics["comp90"])
                comp95_vals.append(metrics["comp95"])

            results.append({
                "source": source,
                "n_samples": human_n,
                "pr_mean": float(np.mean(pr_vals)),
                "pr_std": float(np.std(pr_vals)),
                "comp90_mean": float(np.mean(comp90_vals)),
                "comp90_std": float(np.std(comp90_vals)),
                "comp95_mean": float(np.mean(comp95_vals)),
                "comp95_std": float(np.std(comp95_vals)),
            })
            logger.info("%s: subsampled to %d, PR=%.2f\u00b1%.2f, 90%%=%.0f\u00b1%.1f, 95%%=%.0f\u00b1%.1f",
                        source, human_n,
                        np.mean(pr_vals), np.std(pr_vals),
                        np.mean(comp90_vals), np.std(comp90_vals),
                        np.mean(comp95_vals), np.std(comp95_vals))

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "subsample_table.csv")
    fieldnames = ["source", "n_samples", "pr_mean", "pr_std",
                  "comp90_mean", "comp90_std", "comp95_mean", "comp95_std"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "source": r["source"],
                "n_samples": r["n_samples"],
                "pr_mean": f"{r['pr_mean']:.2f}",
                "pr_std": f"{r['pr_std']:.2f}",
                "comp90_mean": f"{r['comp90_mean']:.1f}",
                "comp90_std": f"{r['comp90_std']:.1f}",
                "comp95_mean": f"{r['comp95_mean']:.1f}",
                "comp95_std": f"{r['comp95_std']:.1f}",
            })
    logger.info("Saved %s", csv_path)

    # Bar chart
    plot_sources = [r["source"] for r in results]
    pr_means = [r["pr_mean"] for r in results]
    pr_stds = [r["pr_std"] for r in results]
    colors = [SOURCE_COLORS.get(s, "#999999") for s in plot_sources]

    human_pr = next((r["pr_mean"] for r in results if r["source"] == "human"), None)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(plot_sources, pr_means, yerr=pr_stds, color=colors,
                  edgecolor="black", linewidth=0.5, capsize=4)

    for bar, pr in zip(bars, pr_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pr:.1f}", ha="center", va="bottom", fontsize=9)

    if human_pr is not None:
        ax.axhline(human_pr, color=SOURCE_COLORS["human"], linestyle="--",
                   linewidth=1, alpha=0.7, label="human PR")
        ax.legend(loc="upper right")

    ax.set_ylabel("Participation Ratio (effective dimensions)")
    ax.set_xlabel("Source")
    ax.set_title(f"Subsampled PR (n={human_n:,} per source, {len(SUBSAMPLE_SEEDS)} seeds)")
    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "subsample_pr_comparison.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", plot_path)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    logger = setup_logging()
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=== Check 1: Rationale length vs. participation ratio ===")
    check_length_vs_pr(logger)

    logger.info("=== Check 2: Subsampling robustness ===")
    check_subsampling_robustness(logger)

    elapsed = time.time() - start_time
    logger.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
