"""Robustness check for the consensus-stratified dimensionality gap via fixed-n bootstrap."""

import csv
import json
import logging
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_NAME = "ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas"
DATASET_SPLIT = "test"
EMBEDDINGS_DIR = "data/embeddings"
OUTPUT_DIR = "data/analysis"
HIDDEN_DIM = 2048
HUMAN_VALUES_PATH = "data/analysis/human_values_all.json"

ALL_SOURCES = ["human", "gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
LLM_SOURCE_ORDER = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]

AGREEMENT_COLS = [
    "comments_nta_agreement_weighted",
    "comments_yta_agreement_weighted",
    "comments_esh_agreement_weighted",
    "comments_nah_agreement_weighted",
]

BUCKET_ORDER = ["low", "medium", "high"]
SUBSAMPLE_SEEDS = list(range(42, 62))  # 20 seeds: 42-61
SUBSAMPLE_N = 220  # binding constraint from low-consensus human count

RARE_VALUES = ["Freedom of speech", "Humor", "Freedom"]

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

BUCKET_COLORS = {
    "low":    "#d62728",
    "medium": "#ff7f0e",
    "high":   "#2ca02c",
}


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("consensus_robustness")
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


def load_consensus_data(logger: logging.Logger) -> dict[str, dict]:
    """Load dataset and compute per-dilemma consensus metrics.

    Returns {submission_id: {"max_agreement": float, "bucket": str}}.
    """
    logger.info("Loading dataset: %s", DATASET_NAME)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    logger.info("Dataset loaded: %d rows", len(ds))

    consensus = {}
    for row in ds:
        sid = row["submission_id"]
        agreements = []
        for col in AGREEMENT_COLS:
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                agreements.append(float(val))

        if not agreements:
            continue

        max_agree = min(max(agreements), 1.0)

        if max_agree > 0.8:
            bucket = "high"
        elif max_agree >= 0.5:
            bucket = "medium"
        else:
            bucket = "low"

        consensus[sid] = {"max_agreement": max_agree, "bucket": bucket}

    bucket_counts = {"high": 0, "medium": 0, "low": 0}
    for v in consensus.values():
        bucket_counts[v["bucket"]] += 1
    for b in BUCKET_ORDER:
        logger.info("Bucket %s: %d dilemmas", b, bucket_counts[b])

    return consensus


def select_rows_by_bucket(
    meta: list[dict], consensus: dict[str, dict], bucket: str,
) -> list[int]:
    """Return indices into the embedding matrix for rows in the given bucket."""
    indices = []
    for i, m in enumerate(meta):
        sid = m["submission_id"]
        if sid in consensus and consensus[sid]["bucket"] == bucket:
            indices.append(i)
    return indices


def build_all_llm(
    embeddings: dict[str, np.ndarray],
    metadata: dict[str, list[dict]],
    logger: logging.Logger,
) -> tuple[np.ndarray, list[dict]]:
    """Concatenate LLM embeddings and metadata in extraction order."""
    matrices = []
    all_meta: list[dict] = []
    for source in LLM_SOURCE_ORDER:
        if source not in embeddings or source not in metadata:
            continue
        matrices.append(embeddings[source])
        for entry in metadata[source]:
            all_meta.append({**entry, "source_model": source})
    combined = np.vstack(matrices)
    logger.info("all_llm: shape %s, %d metadata entries", combined.shape, len(all_meta))
    return combined, all_meta


# ── PCA ────────────────────────────────────────────────────────────────────────
def compute_pca_stats(matrix: np.ndarray) -> dict:
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


# ── Combined Bootstrap ────────────────────────────────────────────────────────
def run_combined_bootstrap(
    human_emb: np.ndarray,
    human_meta: list[dict],
    allllm_emb: np.ndarray,
    allllm_meta: list[dict],
    consensus: dict[str, dict],
    logger: logging.Logger,
) -> list[dict]:
    """Run fixed-n bootstrap: PCA gap + centroid distance + within-group spread."""
    results = []

    for bucket in BUCKET_ORDER:
        human_indices = select_rows_by_bucket(human_meta, consensus, bucket)
        allllm_indices = select_rows_by_bucket(allllm_meta, consensus, bucket)

        n = min(SUBSAMPLE_N, len(human_indices))
        if n < 10:
            logger.warning("Bucket %s: only %d human embeddings, skipping", bucket, len(human_indices))
            continue
        if len(allllm_indices) < n:
            logger.warning("Bucket %s: only %d all_llm embeddings (< %d), skipping",
                           bucket, len(allllm_indices), n)
            continue

        logger.info("Bucket %s: %d human, %d all_llm indices, subsampling to n=%d",
                    bucket, len(human_indices), len(allllm_indices), n)

        gaps = []
        h_comp90s = []
        l_comp90s = []
        centroid_dists = []
        h_spreads = []
        l_spreads = []

        for seed in SUBSAMPLE_SEEDS:
            rng = np.random.RandomState(seed)

            # Subsample human
            h_sub = rng.choice(len(human_indices), n, replace=False)
            h_matrix = human_emb[[human_indices[i] for i in h_sub]]

            # Subsample all_llm
            l_sub = rng.choice(len(allllm_indices), n, replace=False)
            l_matrix = allllm_emb[[allllm_indices[i] for i in l_sub]]

            # Part A: PCA gap
            h_stats = compute_pca_stats(h_matrix)
            l_stats = compute_pca_stats(l_matrix)
            gaps.append(h_stats["comp90"] - l_stats["comp90"])
            h_comp90s.append(h_stats["comp90"])
            l_comp90s.append(l_stats["comp90"])

            # Part B: Centroid distance
            centroid = l_matrix.mean(axis=0)
            dists = cosine_distances(h_matrix, centroid.reshape(1, -1))
            centroid_dists.append(float(dists.ravel().mean()))

            # Part C: Within-group spread
            triu = np.triu_indices(n, k=1)
            h_dists = cosine_distances(h_matrix)
            h_spreads.append(float(h_dists[triu].mean()))
            l_dists = cosine_distances(l_matrix)
            l_spreads.append(float(l_dists[triu].mean()))

        results.append({
            "bucket": bucket,
            "n_samples": n,
            "gap_mean": float(np.mean(gaps)),
            "gap_std": float(np.std(gaps)),
            "h_comp90_mean": float(np.mean(h_comp90s)),
            "h_comp90_std": float(np.std(h_comp90s)),
            "l_comp90_mean": float(np.mean(l_comp90s)),
            "l_comp90_std": float(np.std(l_comp90s)),
            "centroid_dist_mean": float(np.mean(centroid_dists)),
            "centroid_dist_std": float(np.std(centroid_dists)),
            "spread_human_mean": float(np.mean(h_spreads)),
            "spread_human_std": float(np.std(h_spreads)),
            "spread_llm_mean": float(np.mean(l_spreads)),
            "spread_llm_std": float(np.std(l_spreads)),
        })

        logger.info(
            "%s: gap=%.1f\u00b1%.1f, centroid_dist=%.4f\u00b1%.4f, "
            "spread_h=%.4f\u00b1%.4f, spread_l=%.4f\u00b1%.4f",
            bucket,
            np.mean(gaps), np.std(gaps),
            np.mean(centroid_dists), np.std(centroid_dists),
            np.mean(h_spreads), np.std(h_spreads),
            np.mean(l_spreads), np.std(l_spreads),
        )

    return results


# ── Value Frequency by Bucket ────────────────────────────────────────────────
def compute_shannon_entropy(values: list[str]) -> float:
    """Compute Shannon entropy H = -sum(p * log2(p)) for value frequency distribution."""
    if not values:
        return 0.0
    freq: dict[str, int] = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    total = len(values)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def analyze_value_frequency_by_bucket(
    consensus: dict[str, dict],
    logger: logging.Logger,
) -> list[dict]:
    """Analyze value frequencies per consensus bucket using pre-generated human values."""
    if not os.path.exists(HUMAN_VALUES_PATH):
        raise SystemExit(f"{HUMAN_VALUES_PATH} not found. Run value_diversity_gradient.py first.")

    with open(HUMAN_VALUES_PATH, encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded %d human value records", len(records))

    # Group by bucket
    bucket_values: dict[str, list[str]] = {b: [] for b in BUCKET_ORDER}
    unmapped = 0
    for r in records:
        sid = r["submission_id"]
        val = r.get("generated_values", "")
        if not val:
            continue
        if sid not in consensus:
            unmapped += 1
            continue
        bucket = consensus[sid]["bucket"]
        bucket_values[bucket].append(val)

    if unmapped:
        logger.warning("%d records could not be mapped to a consensus bucket", unmapped)

    results = []
    for bucket in BUCKET_ORDER:
        values = bucket_values[bucket]
        if not values:
            logger.warning("No values for bucket %s", bucket)
            continue

        freq: dict[str, int] = {}
        for v in values:
            freq[v] = freq.get(v, 0) + 1
        total = len(values)

        # Top 20
        sorted_values = sorted(freq.items(), key=lambda x: -x[1])
        top_20 = sorted_values[:20]

        # Rare value frequencies
        rare_freqs = {}
        for rv in RARE_VALUES:
            count = freq.get(rv, 0)
            rare_freqs[rv] = count / total if total > 0 else 0.0

        entry = {
            "bucket": bucket,
            "n_rationales": total,
            "n_unique_values": len(freq),
            "shannon_entropy": compute_shannon_entropy(values),
            "top_20_values": "; ".join(f"{v} ({c})" for v, c in top_20),
        }
        for rv in RARE_VALUES:
            key = rv.lower().replace(" ", "_") + "_freq"
            entry[key] = rare_freqs[rv]

        results.append(entry)
        logger.info(
            "%s: %d rationales, %d unique values, H=%.3f",
            bucket, total, len(freq), entry["shannon_entropy"],
        )
        for rv in RARE_VALUES:
            logger.info("  %s: %.6f (%d occurrences)",
                        rv, rare_freqs[rv], freq.get(rv, 0))

    return results


# ── Outputs ────────────────────────────────────────────────────────────────────
def save_robustness_table(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Write consensus_robustness_table.csv."""
    path = os.path.join(output_dir, "consensus_robustness_table.csv")
    fieldnames = [
        "bucket", "n_samples",
        "gap_mean", "gap_std",
        "h_comp90_mean", "h_comp90_std",
        "l_comp90_mean", "l_comp90_std",
        "centroid_dist_mean", "centroid_dist_std",
        "spread_human_mean", "spread_human_std",
        "spread_llm_mean", "spread_llm_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "bucket": r["bucket"],
                "n_samples": r["n_samples"],
                "gap_mean": f"{r['gap_mean']:.2f}",
                "gap_std": f"{r['gap_std']:.2f}",
                "h_comp90_mean": f"{r['h_comp90_mean']:.1f}",
                "h_comp90_std": f"{r['h_comp90_std']:.1f}",
                "l_comp90_mean": f"{r['l_comp90_mean']:.1f}",
                "l_comp90_std": f"{r['l_comp90_std']:.1f}",
                "centroid_dist_mean": f"{r['centroid_dist_mean']:.4f}",
                "centroid_dist_std": f"{r['centroid_dist_std']:.4f}",
                "spread_human_mean": f"{r['spread_human_mean']:.4f}",
                "spread_human_std": f"{r['spread_human_std']:.4f}",
                "spread_llm_mean": f"{r['spread_llm_mean']:.4f}",
                "spread_llm_std": f"{r['spread_llm_std']:.4f}",
            })
    logger.info("Saved %s", path)


def save_value_frequency_by_bucket(
    bucket_stats: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Write consensus_value_frequency_by_bucket.csv."""
    path = os.path.join(output_dir, "consensus_value_frequency_by_bucket.csv")
    fieldnames = ["bucket", "n_rationales", "n_unique_values", "shannon_entropy", "top_20_values"]
    for rv in RARE_VALUES:
        fieldnames.append(rv.lower().replace(" ", "_") + "_freq")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in bucket_stats:
            row = {
                "bucket": r["bucket"],
                "n_rationales": r["n_rationales"],
                "n_unique_values": r["n_unique_values"],
                "shannon_entropy": f"{r['shannon_entropy']:.4f}",
                "top_20_values": r["top_20_values"],
            }
            for rv in RARE_VALUES:
                key = rv.lower().replace(" ", "_") + "_freq"
                row[key] = f"{r[key]:.6f}"
            writer.writerow(row)
    logger.info("Saved %s", path)


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_pca_gap(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Bar chart: components_90 gap by bucket with error bars."""
    buckets = [r["bucket"] for r in results]
    gaps = [r["gap_mean"] for r in results]
    errs = [r["gap_std"] for r in results]
    colors = [BUCKET_COLORS.get(b, "#999999") for b in buckets]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(buckets, gaps, yerr=errs, color=colors,
                  edgecolor="black", linewidth=0.5, capsize=4)

    for bar, g in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{g:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Consensus Level")
    ax.set_ylabel("Components_90 Gap (human \u2212 all_llm)")
    ax.set_title(f"PCA Dimensionality Gap by Consensus Level\n"
                 f"(fixed n={results[0]['n_samples']}, {len(SUBSAMPLE_SEEDS)} seeds)")
    fig.tight_layout()
    path = os.path.join(output_dir, "consensus_robustness_pca.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_centroid_distance(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Bar chart: avg cosine distance from human to LLM centroid by bucket."""
    buckets = [r["bucket"] for r in results]
    dists = [r["centroid_dist_mean"] for r in results]
    errs = [r["centroid_dist_std"] for r in results]
    colors = [BUCKET_COLORS.get(b, "#999999") for b in buckets]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(buckets, dists, yerr=errs, color=colors,
                  edgecolor="black", linewidth=0.5, capsize=4)

    for bar, d in zip(bars, dists):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{d:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Consensus Level")
    ax.set_ylabel("Avg Cosine Distance (human \u2192 LLM centroid)")
    ax.set_title(f"Human-to-LLM Centroid Distance by Consensus Level\n"
                 f"(fixed n={results[0]['n_samples']}, {len(SUBSAMPLE_SEEDS)} seeds)")
    fig.tight_layout()
    path = os.path.join(output_dir, "consensus_robustness_centroid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_spread_comparison(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Grouped bar chart: human vs all_llm within-group spread by bucket."""
    buckets = [r["bucket"] for r in results]
    h_spread = [r["spread_human_mean"] for r in results]
    h_err = [r["spread_human_std"] for r in results]
    l_spread = [r["spread_llm_mean"] for r in results]
    l_err = [r["spread_llm_std"] for r in results]

    x = np.arange(len(buckets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, h_spread, width, yerr=h_err,
           label="human", color=SOURCE_COLORS["human"],
           edgecolor="black", linewidth=0.5, capsize=4)
    ax.bar(x + width / 2, l_spread, width, yerr=l_err,
           label="all_llm", color=SOURCE_COLORS["all_llm"],
           edgecolor="black", linewidth=0.5, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel("Consensus Level")
    ax.set_ylabel("Mean Pairwise Cosine Distance")
    ax.set_title(f"Within-Group Spread by Consensus Level\n"
                 f"(fixed n={results[0]['n_samples']}, {len(SUBSAMPLE_SEEDS)} seeds)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(output_dir, "consensus_robustness_spread.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_combined(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """1x3 subplot combining PCA gap, centroid distance, and spread."""
    buckets = [r["bucket"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 0: PCA gap
    ax = axes[0]
    gaps = [r["gap_mean"] for r in results]
    gap_errs = [r["gap_std"] for r in results]
    colors = [BUCKET_COLORS.get(b, "#999999") for b in buckets]
    bars = ax.bar(buckets, gaps, yerr=gap_errs, color=colors,
                  edgecolor="black", linewidth=0.5, capsize=4)
    for bar, g in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{g:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Consensus Level")
    ax.set_ylabel("Components_90 Gap")
    ax.set_title("PCA Dimensionality Gap")

    # Panel 1: Centroid distance
    ax = axes[1]
    dists = [r["centroid_dist_mean"] for r in results]
    dist_errs = [r["centroid_dist_std"] for r in results]
    bars = ax.bar(buckets, dists, yerr=dist_errs, color=colors,
                  edgecolor="black", linewidth=0.5, capsize=4)
    for bar, d in zip(bars, dists):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{d:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Consensus Level")
    ax.set_ylabel("Avg Cosine Distance")
    ax.set_title("Human \u2192 LLM Centroid Distance")

    # Panel 2: Within-group spread
    ax = axes[2]
    x = np.arange(len(buckets))
    width = 0.35
    h_spread = [r["spread_human_mean"] for r in results]
    h_err = [r["spread_human_std"] for r in results]
    l_spread = [r["spread_llm_mean"] for r in results]
    l_err = [r["spread_llm_std"] for r in results]
    ax.bar(x - width / 2, h_spread, width, yerr=h_err,
           label="human", color=SOURCE_COLORS["human"],
           edgecolor="black", linewidth=0.5, capsize=4)
    ax.bar(x + width / 2, l_spread, width, yerr=l_err,
           label="all_llm", color=SOURCE_COLORS["all_llm"],
           edgecolor="black", linewidth=0.5, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel("Consensus Level")
    ax.set_ylabel("Mean Pairwise Cosine Distance")
    ax.set_title("Within-Group Spread")
    ax.legend(loc="upper right")

    fig.suptitle(f"Consensus Robustness (fixed n={results[0]['n_samples']}, "
                 f"{len(SUBSAMPLE_SEEDS)} seeds)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "consensus_robustness_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    logger = setup_logging()
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    embeddings = load_embeddings(logger)
    metadata = load_metadata(logger)
    consensus = load_consensus_data(logger)

    if "human" not in embeddings:
        raise SystemExit("Human embeddings not found.")

    # Build all_llm
    allllm_emb, allllm_meta = build_all_llm(embeddings, metadata, logger)

    # Parts A+B+C: Bootstrap analysis
    logger.info("=== Bootstrap PCA + centroid + spread analysis ===")
    results = run_combined_bootstrap(
        embeddings["human"], metadata["human"],
        allllm_emb, allllm_meta, consensus, logger)

    # Save and plot
    save_robustness_table(results, OUTPUT_DIR, logger)
    plot_pca_gap(results, OUTPUT_DIR, logger)
    plot_centroid_distance(results, OUTPUT_DIR, logger)
    plot_spread_comparison(results, OUTPUT_DIR, logger)
    plot_combined(results, OUTPUT_DIR, logger)

    # Part D: Value frequency by bucket
    logger.info("=== Value frequency by consensus bucket ===")
    bucket_stats = analyze_value_frequency_by_bucket(consensus, logger)
    save_value_frequency_by_bucket(bucket_stats, OUTPUT_DIR, logger)

    elapsed = time.time() - start_time
    logger.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
