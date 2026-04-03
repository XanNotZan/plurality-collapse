"""Test whether the human-LLM dimensionality gap varies with moral consensus level."""

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
from sklearn.metrics.pairwise import cosine_distances

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
HIDDEN_DIM = 2048
MIN_EMBEDDINGS = 100
MIN_LLM_PER_DILEMMA = 10

ALL_SOURCES = ["human", "gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
LLM_SOURCE_ORDER = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]

AGREEMENT_COLS = [
    "comments_nta_agreement_weighted",
    "comments_yta_agreement_weighted",
    "comments_esh_agreement_weighted",
    "comments_nah_agreement_weighted",
]

BUCKET_THRESHOLDS = {
    "high":   (0.8, 1.01),
    "medium": (0.5, 0.8),
    "low":    (0.0, 0.5),
}
BUCKET_ORDER = ["low", "medium", "high"]
SUBSAMPLE_SEEDS = [42, 43, 44, 45, 46]

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
    logger = logging.getLogger("consensus_stratification")
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

        max_agree = min(max(agreements), 1.0)  # clip to [0, 1]

        if max_agree > 0.8:
            bucket = "high"
        elif max_agree >= 0.5:
            bucket = "medium"
        else:
            bucket = "low"

        consensus[sid] = {"max_agreement": max_agree, "bucket": bucket}

    # Log bucket counts
    bucket_counts = {"high": 0, "medium": 0, "low": 0}
    for v in consensus.values():
        bucket_counts[v["bucket"]] += 1
    for b in BUCKET_ORDER:
        logger.info("Bucket %s: %d dilemmas", b, bucket_counts[b])

    return consensus


# ── Per-Bucket PCA ───────────────────────────────────────────────────────────
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


def run_per_bucket_analysis(
    embeddings: dict[str, np.ndarray],
    metadata: dict[str, list[dict]],
    consensus: dict[str, dict],
    logger: logging.Logger,
) -> list[dict]:
    """Run PCA on each source-bucket combination."""
    # Build all_llm embeddings + metadata
    all_llm_matrices = []
    all_llm_meta = []
    for source in LLM_SOURCE_ORDER:
        if source in embeddings and source in metadata:
            all_llm_matrices.append(embeddings[source])
            for m in metadata[source]:
                all_llm_meta.append(m)
    all_llm_emb = np.vstack(all_llm_matrices) if all_llm_matrices else None

    # Sources to analyze
    sources_to_analyze = {}
    for source in ALL_SOURCES:
        if source in embeddings and source in metadata:
            sources_to_analyze[source] = (embeddings[source], metadata[source])
    if all_llm_emb is not None:
        sources_to_analyze["all_llm"] = (all_llm_emb, all_llm_meta)

    results = []
    for source_name, (emb, meta) in sources_to_analyze.items():
        for bucket in BUCKET_ORDER:
            indices = select_rows_by_bucket(meta, consensus, bucket)
            n_dilemmas = len(set(meta[i]["submission_id"] for i in indices))

            if len(indices) < MIN_EMBEDDINGS:
                logger.warning("%s / %s: only %d embeddings, skipping (need %d)",
                               source_name, bucket, len(indices), MIN_EMBEDDINGS)
                continue

            subset = emb[indices]
            stats = compute_pca_stats(subset)

            results.append({
                "source": source_name,
                "bucket": bucket,
                "n_dilemmas": n_dilemmas,
                "n_embeddings": len(indices),
                "participation_ratio": stats["pr"],
                "components_90": stats["comp90"],
                "components_95": stats["comp95"],
            })
            logger.info("%s / %s: %d dilemmas, %d embeddings, PR=%.1f, 90%%=%d, 95%%=%d",
                        source_name, bucket, n_dilemmas, len(indices),
                        stats["pr"], stats["comp90"], stats["comp95"])

    return results


# ── Subsampled Per-Bucket Analysis ───────────────────────────────────────────
def run_subsampled_bucket_analysis(
    embeddings: dict[str, np.ndarray],
    metadata: dict[str, list[dict]],
    consensus: dict[str, dict],
    logger: logging.Logger,
) -> list[dict]:
    """Run PCA on subsampled source-bucket combinations to control for sample size."""
    # Build all_llm
    all_llm_matrices = []
    all_llm_meta: list[dict] = []
    for source in LLM_SOURCE_ORDER:
        if source in embeddings and source in metadata:
            all_llm_matrices.append(embeddings[source])
            all_llm_meta.extend(metadata[source])
    all_llm_emb = np.vstack(all_llm_matrices) if all_llm_matrices else None

    sources_to_analyze: dict[str, tuple[np.ndarray, list[dict]]] = {}
    for source in ALL_SOURCES:
        if source in embeddings and source in metadata:
            sources_to_analyze[source] = (embeddings[source], metadata[source])
    if all_llm_emb is not None:
        sources_to_analyze["all_llm"] = (all_llm_emb, all_llm_meta)

    # First pass: determine n_human per bucket
    human_emb, human_meta = sources_to_analyze["human"]
    human_n_per_bucket: dict[str, int] = {}
    human_indices_per_bucket: dict[str, list[int]] = {}
    for bucket in BUCKET_ORDER:
        indices = select_rows_by_bucket(human_meta, consensus, bucket)
        human_n_per_bucket[bucket] = len(indices)
        human_indices_per_bucket[bucket] = indices
        logger.info("Human in %s bucket: %d embeddings", bucket, len(indices))

    results = []
    for bucket in BUCKET_ORDER:
        n_human = human_n_per_bucket[bucket]
        if n_human < MIN_EMBEDDINGS:
            logger.warning("Bucket %s: only %d human embeddings, skipping (need %d)",
                           bucket, n_human, MIN_EMBEDDINGS)
            continue

        for source_name, (emb, meta) in sources_to_analyze.items():
            indices = select_rows_by_bucket(meta, consensus, bucket)

            if source_name == "human":
                # No subsampling — run PCA once
                subset = emb[indices]
                stats = compute_pca_stats(subset)
                results.append({
                    "source": source_name,
                    "bucket": bucket,
                    "n_samples": len(indices),
                    "pr_mean": stats["pr"],
                    "pr_std": 0.0,
                    "comp90_mean": float(stats["comp90"]),
                    "comp90_std": 0.0,
                    "comp95_mean": float(stats["comp95"]),
                    "comp95_std": 0.0,
                })
                logger.info("%s / %s: n=%d (no subsampling), PR=%.2f, 90%%=%d, 95%%=%d",
                            source_name, bucket, len(indices),
                            stats["pr"], stats["comp90"], stats["comp95"])
                continue

            if len(indices) < n_human:
                logger.warning("%s / %s: only %d embeddings (< %d human), skipping",
                               source_name, bucket, len(indices), n_human)
                continue

            # Subsample to n_human with multiple seeds
            pr_vals, comp90_vals, comp95_vals = [], [], []
            for seed in SUBSAMPLE_SEEDS:
                rng = np.random.RandomState(seed)
                sub_idx = rng.choice(len(indices), n_human, replace=False)
                subset = emb[[indices[i] for i in sub_idx]]
                stats = compute_pca_stats(subset)
                pr_vals.append(stats["pr"])
                comp90_vals.append(stats["comp90"])
                comp95_vals.append(stats["comp95"])

            results.append({
                "source": source_name,
                "bucket": bucket,
                "n_samples": n_human,
                "pr_mean": float(np.mean(pr_vals)),
                "pr_std": float(np.std(pr_vals)),
                "comp90_mean": float(np.mean(comp90_vals)),
                "comp90_std": float(np.std(comp90_vals)),
                "comp95_mean": float(np.mean(comp95_vals)),
                "comp95_std": float(np.std(comp95_vals)),
            })
            logger.info("%s / %s: subsampled to %d, PR=%.2f\u00b1%.2f, "
                        "90%%=%.0f\u00b1%.1f, 95%%=%.0f\u00b1%.1f",
                        source_name, bucket, n_human,
                        np.mean(pr_vals), np.std(pr_vals),
                        np.mean(comp90_vals), np.std(comp90_vals),
                        np.mean(comp95_vals), np.std(comp95_vals))

    return results


def save_subsampled_csv(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Save consensus_stratification_subsampled.csv."""
    path = os.path.join(output_dir, "consensus_stratification_subsampled.csv")
    fieldnames = ["source", "bucket", "n_samples", "pr_mean", "pr_std",
                  "comp90_mean", "comp90_std", "comp95_mean", "comp95_std"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "source": r["source"],
                "bucket": r["bucket"],
                "n_samples": r["n_samples"],
                "pr_mean": f"{r['pr_mean']:.2f}",
                "pr_std": f"{r['pr_std']:.2f}",
                "comp90_mean": f"{r['comp90_mean']:.1f}",
                "comp90_std": f"{r['comp90_std']:.1f}",
                "comp95_mean": f"{r['comp95_mean']:.1f}",
                "comp95_std": f"{r['comp95_std']:.1f}",
            })
    logger.info("Saved %s", path)


def plot_subsampled_by_bucket(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Grouped bar chart of subsampled components_90 by bucket."""
    by_key = {}
    for r in results:
        by_key[(r["source"], r["bucket"])] = r

    focus_sources = ["human", "all_llm"]
    available_buckets = sorted(
        set(r["bucket"] for r in results if r["source"] in focus_sources),
        key=lambda b: BUCKET_ORDER.index(b))

    if not available_buckets:
        logger.warning("No buckets with both human and all_llm data, skipping plot")
        return

    x = np.arange(len(available_buckets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, source in enumerate(focus_sources):
        vals = []
        errs = []
        for b in available_buckets:
            r = by_key.get((source, b))
            vals.append(r["comp90_mean"] if r else 0)
            errs.append(r["comp90_std"] if r else 0)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs,
                      label=source, color=SOURCE_COLORS.get(source, "#999999"),
                      edgecolor="black", linewidth=0.5, capsize=4)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\nconsensus" for b in available_buckets])
    ax.set_ylabel("Components for 90% Variance")
    ax.set_title("Subsampled Dimensionality by Consensus Level: Human vs All-LLM\n"
                 f"(matched sample sizes, {len(SUBSAMPLE_SEEDS)} seeds)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = os.path.join(output_dir, "consensus_subsampled_by_bucket.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_subsampled_gap(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Line plot of subsampled human vs all_llm components_90 with error bands."""
    by_key = {}
    for r in results:
        by_key[(r["source"], r["bucket"])] = r

    available_buckets = sorted(
        set(r["bucket"] for r in results),
        key=lambda b: BUCKET_ORDER.index(b))

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(available_buckets))

    for source in ["human", "all_llm"]:
        vals = []
        stds = []
        for b in available_buckets:
            r = by_key.get((source, b))
            vals.append(r["comp90_mean"] if r else 0)
            stds.append(r["comp90_std"] if r else 0)

        color = SOURCE_COLORS.get(source, "#999999")
        ax.plot(x_pos, vals, marker="o", linewidth=2.5, markersize=8,
                color=color, label=source)
        ax.fill_between(x_pos,
                        [v - s for v, s in zip(vals, stds)],
                        [v + s for v, s in zip(vals, stds)],
                        color=color, alpha=0.15)

    # Annotate gaps
    for i, b in enumerate(available_buckets):
        h = by_key.get(("human", b))
        l = by_key.get(("all_llm", b))
        if h and l:
            gap = h["comp90_mean"] - l["comp90_mean"]
            mid_y = (h["comp90_mean"] + l["comp90_mean"]) / 2
            ax.annotate(f"\u0394={gap:.0f}", xy=(i, mid_y),
                        fontsize=10, fontweight="bold", ha="center",
                        xytext=(i + 0.25, mid_y),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{b}\nconsensus" for b in available_buckets])
    ax.set_ylabel("Components for 90% Variance")
    ax.set_title("Subsampled Human-LLM Gap by Consensus Level\n"
                 f"(matched sample sizes, {len(SUBSAMPLE_SEEDS)} seeds)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = os.path.join(output_dir, "consensus_subsampled_gap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Per-Dilemma LLM Convergence ──────────────────────────────────────────────
def compute_llm_convergence(
    embeddings: dict[str, np.ndarray],
    metadata: dict[str, list[dict]],
    consensus: dict[str, dict],
    logger: logging.Logger,
) -> list[dict]:
    """Compute average pairwise cosine distance among LLM embeddings per dilemma."""
    # Group LLM embeddings by submission_id
    dilemma_embeddings: dict[str, list[np.ndarray]] = {}
    for source in LLM_SOURCE_ORDER:
        if source not in embeddings or source not in metadata:
            continue
        emb = embeddings[source]
        meta = metadata[source]
        for i, m in enumerate(meta):
            sid = m["submission_id"]
            if sid not in dilemma_embeddings:
                dilemma_embeddings[sid] = []
            dilemma_embeddings[sid].append(emb[i])

    results = []
    for sid, vecs in dilemma_embeddings.items():
        if len(vecs) < MIN_LLM_PER_DILEMMA:
            continue
        if sid not in consensus:
            continue

        mat = np.array(vecs)
        dists = cosine_distances(mat)
        # Average of upper triangle
        n = len(vecs)
        triu_idx = np.triu_indices(n, k=1)
        avg_dist = float(dists[triu_idx].mean())

        results.append({
            "submission_id": sid,
            "max_agreement": consensus[sid]["max_agreement"],
            "consensus_bucket": consensus[sid]["bucket"],
            "n_llm_embeddings": n,
            "avg_cosine_distance": avg_dist,
        })

    logger.info("LLM convergence: %d dilemmas with >= %d embeddings",
                len(results), MIN_LLM_PER_DILEMMA)
    return results


# ── Outputs ──────────────────────────────────────────────────────────────────
def save_stratification_csv(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Save consensus_stratification_table.csv."""
    path = os.path.join(output_dir, "consensus_stratification_table.csv")
    fieldnames = ["source", "bucket", "n_dilemmas", "n_embeddings",
                  "participation_ratio", "components_90", "components_95"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "source": r["source"],
                "bucket": r["bucket"],
                "n_dilemmas": r["n_dilemmas"],
                "n_embeddings": r["n_embeddings"],
                "participation_ratio": f"{r['participation_ratio']:.2f}",
                "components_90": r["components_90"],
                "components_95": r["components_95"],
            })
    logger.info("Saved %s", path)


def plot_pr_by_bucket(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Grouped bar chart: components_90 by bucket for human vs all_llm."""
    by_key = {}
    for r in results:
        by_key[(r["source"], r["bucket"])] = r

    focus_sources = ["human", "all_llm"]
    available = [s for s in focus_sources if any(
        (s, b) in by_key for b in BUCKET_ORDER)]

    if len(available) < 2:
        logger.warning("Need both human and all_llm for grouped bar chart, skipping")
        return

    x = np.arange(len(BUCKET_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, source in enumerate(available):
        vals = []
        for b in BUCKET_ORDER:
            r = by_key.get((source, b))
            vals.append(r["components_90"] if r else 0)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=source, color=SOURCE_COLORS.get(source, "#999999"),
                      edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        str(v), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\nconsensus" for b in BUCKET_ORDER])
    ax.set_ylabel("Components for 90% Variance")
    ax.set_title("Dimensionality by Consensus Level: Human vs All-LLM")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = os.path.join(output_dir, "consensus_pr_by_bucket.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_consensus_gap(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Line plot of human vs all_llm components_90 across consensus levels."""
    by_key = {}
    for r in results:
        by_key[(r["source"], r["bucket"])] = r

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(BUCKET_ORDER))

    for source in ["human", "all_llm"]:
        vals = []
        for b in BUCKET_ORDER:
            r = by_key.get((source, b))
            vals.append(r["components_90"] if r else None)
        if any(v is None for v in vals):
            logger.warning("Missing data for %s in some buckets", source)
            vals = [v if v is not None else 0 for v in vals]
        ax.plot(x_pos, vals, marker="o", linewidth=2.5, markersize=8,
                color=SOURCE_COLORS.get(source, "#999999"), label=source)

    # Annotate gaps
    for i, b in enumerate(BUCKET_ORDER):
        h = by_key.get(("human", b))
        l = by_key.get(("all_llm", b))
        if h and l:
            gap = h["components_90"] - l["components_90"]
            mid_y = (h["components_90"] + l["components_90"]) / 2
            ax.annotate(f"\u0394={gap}", xy=(i, mid_y),
                        fontsize=10, fontweight="bold", ha="center",
                        xytext=(i + 0.25, mid_y),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{b}\nconsensus" for b in BUCKET_ORDER])
    ax.set_ylabel("Components for 90% Variance")
    ax.set_title("Human-LLM Dimensionality Gap by Consensus Level")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = os.path.join(output_dir, "consensus_gap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def save_convergence_csv(
    convergence: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Save per-dilemma LLM convergence data."""
    path = os.path.join(output_dir, "llm_convergence_vs_consensus.csv")
    fieldnames = ["submission_id", "max_agreement", "consensus_bucket",
                  "n_llm_embeddings", "avg_cosine_distance"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in convergence:
            writer.writerow({
                "submission_id": r["submission_id"],
                "max_agreement": f"{r['max_agreement']:.4f}",
                "consensus_bucket": r["consensus_bucket"],
                "n_llm_embeddings": r["n_llm_embeddings"],
                "avg_cosine_distance": f"{r['avg_cosine_distance']:.6f}",
            })
    logger.info("Saved %s", path)


def plot_convergence(
    convergence: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Scatter plot of LLM cosine distance vs consensus."""
    if not convergence:
        logger.warning("No convergence data to plot")
        return

    agreements = np.array([r["max_agreement"] for r in convergence])
    distances = np.array([r["avg_cosine_distance"] for r in convergence])

    # Correlation
    if HAS_SCIPY:
        pearson_r, pearson_p = pearsonr(agreements, distances)
        spearman_r, spearman_p = spearmanr(agreements, distances)
    else:
        pearson_r = float(np.corrcoef(agreements, distances)[0, 1])
        pearson_p = float("nan")
        spearman_r = float("nan")
        spearman_p = float("nan")

    logger.info("Convergence correlation: Pearson r=%.3f p=%.4f, Spearman r=%.3f p=%.4f",
                pearson_r, pearson_p, spearman_r, spearman_p)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by bucket
    bucket_colors = {"low": "#d62728", "medium": "#ff7f0e", "high": "#2ca02c"}
    for r in convergence:
        ax.scatter(r["max_agreement"], r["avg_cosine_distance"],
                   color=bucket_colors.get(r["consensus_bucket"], "#999999"),
                   s=15, alpha=0.5, edgecolors="none")

    # Trend line
    z = np.polyfit(agreements, distances, 1)
    p = np.poly1d(z)
    x_line = np.linspace(agreements.min(), agreements.max(), 100)
    ax.plot(x_line, p(x_line), color="black", linewidth=1.5, linestyle="--",
            label=f"trend (slope={z[0]:.4f})")

    # Legend for buckets
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=bucket_colors["low"], label="Low consensus"),
        Patch(color=bucket_colors["medium"], label="Medium consensus"),
        Patch(color=bucket_colors["high"], label="High consensus"),
        plt.Line2D([0], [0], color="black", linestyle="--", label=f"trend (slope={z[0]:.4f})"),
    ], loc="upper right", fontsize=8)

    ax.set_xlabel("Max Agreement (consensus level)")
    ax.set_ylabel("Avg Pairwise Cosine Distance (LLM divergence)")
    ax.set_title(f"LLM Convergence vs Consensus "
                 f"(Pearson r={pearson_r:.3f}, Spearman \u03c1={spearman_r:.3f})")
    fig.tight_layout()
    path = os.path.join(output_dir, "llm_convergence_vs_consensus.png")
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

    # Per-bucket PCA analysis
    logger.info("=== Per-bucket PCA analysis ===")
    results = run_per_bucket_analysis(embeddings, metadata, consensus, logger)
    save_stratification_csv(results, OUTPUT_DIR, logger)
    plot_pr_by_bucket(results, OUTPUT_DIR, logger)
    plot_consensus_gap(results, OUTPUT_DIR, logger)

    # Subsampled per-bucket analysis
    logger.info("=== Running subsampled analysis to control for sample size differences across buckets ===")
    sub_results = run_subsampled_bucket_analysis(embeddings, metadata, consensus, logger)
    save_subsampled_csv(sub_results, OUTPUT_DIR, logger)
    plot_subsampled_by_bucket(sub_results, OUTPUT_DIR, logger)
    plot_subsampled_gap(sub_results, OUTPUT_DIR, logger)

    # Per-dilemma LLM convergence
    logger.info("=== Per-dilemma LLM convergence ===")
    convergence = compute_llm_convergence(embeddings, metadata, consensus, logger)
    save_convergence_csv(convergence, OUTPUT_DIR, logger)
    plot_convergence(convergence, OUTPUT_DIR, logger)

    elapsed = time.time() - start_time
    logger.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
