"""Analyze PCA dimensionality of moral reasoning embeddings across human and LLM sources."""

import csv
import json
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, KernelPCA

# ── Constants ──────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = "data/embeddings"
OUTPUT_DIR = "data/analysis"
HIDDEN_DIM = 2048

ALL_SOURCES = ["human", "gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]

KERNEL_PCA_MAX_SAMPLES = 5000
RANDOM_SEED = 42

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

# Display order for plots: human first, LLMs alphabetically, all_llm last
PLOT_ORDER = ["human"] + sorted(LLM_SOURCES) + ["all_llm"]


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("analyze_embeddings")
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
        if matrix.shape[0] < 2:
            logger.warning("Too few rationales for %s (%d), skipping", source, matrix.shape[0])
            continue
        embeddings[source] = matrix
        logger.info("Loaded %s: shape %s", source, matrix.shape)

    if not embeddings:
        raise SystemExit(f"No embedding files found in {EMBEDDINGS_DIR}/")

    return embeddings


# ── PCA Analysis ──────────────────────────────────────────────────────────────
def compute_pca_stats(
    matrix: np.ndarray, source_name: str, logger: logging.Logger,
) -> dict:
    """Run full PCA on an embedding matrix and return dimensionality statistics."""
    pca = PCA()
    pca.fit(matrix)

    eigenvalues = pca.explained_variance_
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    # Participation ratio: PR = (Σλ)² / Σλ²
    pr = float((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum())

    # Components needed for variance thresholds
    components_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    components_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    components_90 = min(components_90, len(cumulative))
    components_95 = min(components_95, len(cumulative))

    logger.info(
        "%s: n=%d, PR=%.1f, 90%%=%d components, 95%%=%d components",
        source_name, matrix.shape[0], pr, components_90, components_95,
    )

    return {
        "source": source_name,
        "n_rationales": matrix.shape[0],
        "participation_ratio": pr,
        "components_90pct": components_90,
        "components_95pct": components_95,
        "eigenvalues": eigenvalues.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumulative.tolist(),
    }


def build_all_llm_matrix(
    embeddings: dict[str, np.ndarray], logger: logging.Logger,
) -> np.ndarray | None:
    """Concatenate all LLM embedding matrices into one."""
    matrices = [embeddings[s] for s in LLM_SOURCES if s in embeddings]
    if not matrices:
        logger.warning("No LLM sources available; skipping all_llm.")
        return None
    combined = np.vstack(matrices)
    logger.info("all_llm combined matrix: shape %s", combined.shape)
    return combined


# ── Kernel PCA Robustness Check ───────────────────────────────────────────────
def compute_kernel_pca_pr(
    matrix: np.ndarray, source_name: str, logger: logging.Logger,
) -> float | None:
    """Compute participation ratio using Kernel PCA with RBF kernel."""
    try:
        rng = np.random.RandomState(RANDOM_SEED)
        if matrix.shape[0] > KERNEL_PCA_MAX_SAMPLES:
            indices = rng.choice(matrix.shape[0], KERNEL_PCA_MAX_SAMPLES, replace=False)
            matrix = matrix[indices]
            logger.info("%s: subsampled to %d for Kernel PCA", source_name, KERNEL_PCA_MAX_SAMPLES)

        n_components = min(matrix.shape[0], matrix.shape[1])
        kpca = KernelPCA(n_components=n_components, kernel="rbf")
        kpca.fit(matrix)

        lambdas = getattr(kpca, "eigenvalues_", None)
        if lambdas is None:
            lambdas = kpca.lambdas_

        lambdas = lambdas[lambdas > 1e-10]
        if len(lambdas) == 0:
            logger.warning("%s: no positive eigenvalues from Kernel PCA", source_name)
            return None

        pr = float((lambdas.sum() ** 2) / (lambdas ** 2).sum())
        logger.info("%s: Kernel PCA PR=%.1f", source_name, pr)
        return pr
    except Exception:
        logger.exception("%s: Kernel PCA failed", source_name)
        return None


# ── Output: CSV & JSON ────────────────────────────────────────────────────────
def save_summary_csv(
    results: list[dict],
    kernel_pca_results: dict[str, float | None],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Write summary_table.csv with one row per source."""
    path = os.path.join(output_dir, "summary_table.csv")
    fieldnames = [
        "source", "n_rationales", "participation_ratio", "kernel_pca_pr",
        "components_90pct", "components_95pct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            kpr = kernel_pca_results.get(r["source"])
            writer.writerow({
                "source": r["source"],
                "n_rationales": r["n_rationales"],
                "participation_ratio": f"{r['participation_ratio']:.2f}",
                "kernel_pca_pr": f"{kpr:.2f}" if kpr is not None else "",
                "components_90pct": r["components_90pct"],
                "components_95pct": r["components_95pct"],
            })
    logger.info("Saved %s", path)


def save_eigenvalue_spectra(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Write eigenvalue_spectra.json with full spectra per source."""
    spectra = {}
    for r in results:
        spectra[r["source"]] = {
            "eigenvalues": r["eigenvalues"],
            "explained_variance_ratio": r["explained_variance_ratio"],
            "cumulative_variance": r["cumulative_variance"],
        }
    path = os.path.join(output_dir, "eigenvalue_spectra.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spectra, f, indent=2)
    logger.info("Saved %s", path)


# ── Plotting ──────────────────────────────────────────────────────────────────
def _results_by_source(results: list[dict]) -> dict[str, dict]:
    return {r["source"]: r for r in results}


def plot_participation_ratio(
    results: list[dict],
    kernel_pca_results: dict[str, float | None],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Bar chart comparing participation ratio across sources."""
    by_source = _results_by_source(results)
    sources = [s for s in PLOT_ORDER if s in by_source]
    prs = [by_source[s]["participation_ratio"] for s in sources]
    colors = [SOURCE_COLORS.get(s, "#999999") for s in sources]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(sources, prs, color=colors, edgecolor="black", linewidth=0.5)

    # Value labels on bars
    for bar, pr in zip(bars, prs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{pr:.1f}", ha="center", va="bottom", fontsize=9,
        )

    # Reference line at human PR
    if "human" in by_source:
        ax.axhline(
            by_source["human"]["participation_ratio"],
            color=SOURCE_COLORS["human"], linestyle="--", linewidth=1, alpha=0.7,
            label="human PR",
        )
        ax.legend(loc="upper right")

    ax.set_ylabel("Participation Ratio (effective dimensions)")
    ax.set_title("Effective Dimensionality of Moral Reasoning")
    ax.set_xlabel("Source")
    fig.tight_layout()
    path = os.path.join(output_dir, "participation_ratio_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_cumulative_variance(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Line plot of cumulative explained variance curves."""
    by_source = _results_by_source(results)

    # Determine x-axis limit: 99% variance for human, capped at 500
    x_max = 500
    if "human" in by_source:
        cum = np.array(by_source["human"]["cumulative_variance"])
        idx_99 = int(np.searchsorted(cum, 0.99) + 1)
        x_max = min(max(idx_99, 50), 500)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Individual LLMs: thin dashed
    for source in sorted(LLM_SOURCES):
        if source not in by_source:
            continue
        cum = by_source[source]["cumulative_variance"]
        n = min(len(cum), x_max)
        ax.plot(
            range(1, n + 1), cum[:n],
            color=SOURCE_COLORS.get(source, "#999999"),
            linestyle="--", linewidth=1, alpha=0.5, label=source,
        )

    # Human and all_llm: thick solid
    for source, lw in [("human", 2.5), ("all_llm", 2.5)]:
        if source not in by_source:
            continue
        cum = by_source[source]["cumulative_variance"]
        n = min(len(cum), x_max)
        ax.plot(
            range(1, n + 1), cum[:n],
            color=SOURCE_COLORS.get(source, "#999999"),
            linestyle="-", linewidth=lw, label=source,
        )

    # Threshold lines
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(x_max * 0.98, 0.90, "90%", ha="right", va="bottom", fontsize=8, color="gray")
    ax.text(x_max * 0.98, 0.95, "95%", ha="right", va="bottom", fontsize=8, color="gray")

    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title("Cumulative Explained Variance")
    ax.set_xlim(1, x_max)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, "cumulative_variance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_eigenvalue_decay(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Log-scale plot of eigenvalue decay for human vs all-LLM."""
    by_source = _results_by_source(results)

    x_max = 500
    if "human" in by_source:
        cum = np.array(by_source["human"]["cumulative_variance"])
        idx_99 = int(np.searchsorted(cum, 0.99) + 1)
        x_max = min(max(idx_99, 50), 500)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Individual LLMs: thin dashed
    for source in sorted(LLM_SOURCES):
        if source not in by_source:
            continue
        eigs = by_source[source]["eigenvalues"]
        n = min(len(eigs), x_max)
        ax.plot(
            range(1, n + 1), eigs[:n],
            color=SOURCE_COLORS.get(source, "#999999"),
            linestyle="--", linewidth=1, alpha=0.5, label=source,
        )

    # Human and all_llm: thick solid
    for source, lw in [("human", 2.5), ("all_llm", 2.5)]:
        if source not in by_source:
            continue
        eigs = by_source[source]["eigenvalues"]
        n = min(len(eigs), x_max)
        ax.plot(
            range(1, n + 1), eigs[:n],
            color=SOURCE_COLORS.get(source, "#999999"),
            linestyle="-", linewidth=lw, label=source,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Eigenvalue Spectrum Decay")
    ax.set_xlim(1, x_max)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, "eigenvalue_decay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    logger = setup_logging()
    start_time = time.time()

    # Load embeddings
    embeddings = load_embeddings(logger)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # PCA on each individual source
    results = []
    for source_name, matrix in embeddings.items():
        logger.info("Running PCA on %s...", source_name)
        stats = compute_pca_stats(matrix, source_name, logger)
        results.append(stats)

    # Build all_llm combined matrix and run PCA
    all_llm_matrix = build_all_llm_matrix(embeddings, logger)
    if all_llm_matrix is not None:
        logger.info("Running PCA on all_llm (combined)...")
        all_llm_stats = compute_pca_stats(all_llm_matrix, "all_llm", logger)
        results.append(all_llm_stats)

    # Kernel PCA robustness check
    kernel_pca_results: dict[str, float | None] = {}
    all_matrices: dict[str, np.ndarray] = dict(embeddings)
    if all_llm_matrix is not None:
        all_matrices["all_llm"] = all_llm_matrix
    for source_name, matrix in all_matrices.items():
        logger.info("Running Kernel PCA on %s...", source_name)
        kpr = compute_kernel_pca_pr(matrix, source_name, logger)
        kernel_pca_results[source_name] = kpr

    # Save structured outputs
    save_summary_csv(results, kernel_pca_results, OUTPUT_DIR, logger)
    save_eigenvalue_spectra(results, OUTPUT_DIR, logger)

    # Generate plots
    plot_participation_ratio(results, kernel_pca_results, OUTPUT_DIR, logger)
    plot_cumulative_variance(results, OUTPUT_DIR, logger)
    plot_eigenvalue_decay(results, OUTPUT_DIR, logger)

    elapsed = time.time() - start_time
    logger.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
