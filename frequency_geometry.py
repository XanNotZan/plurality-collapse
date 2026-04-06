"""Continuous frequency analysis: measure LLM density around human rationales
in PCA-reduced Kaleido embedding space, without categorical value labels."""

import csv
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

try:
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── Constants ────────────────────────────────────────────────────────────────

EMBEDDINGS_DIR = "data/embeddings"
OUTPUT_DIR = "data/analysis"

ALL_SOURCES = ["human", "gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
LLM_SOURCE_ORDER = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]

PCA_COMPONENTS = 50
K_NEIGHBORS = 50
RANDOM_SEED = 42

WINDOW_SIZE = 200
WINDOW_STEP = 50

SOURCE_COLORS = {
    "human": "#2ca02c",
    "all_llm": "#e377c2",
}


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("frequency_geometry")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger = setup_logging()
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load embeddings
    logger.info("Loading embeddings...")
    human = np.load(os.path.join(EMBEDDINGS_DIR, "human.npy"))
    llm_parts = [np.load(os.path.join(EMBEDDINGS_DIR, f"{s}.npy")) for s in LLM_SOURCE_ORDER]
    all_llm = np.vstack(llm_parts)
    del llm_parts
    logger.info("Human: %s, All LLM: %s", human.shape, all_llm.shape)

    # PCA reduce (fit on combined)
    logger.info("PCA reducing to %d components...", PCA_COMPONENTS)
    combined = np.vstack([human, all_llm])
    pca = PCA(n_components=PCA_COMPONENTS)
    pca.fit(combined)
    logger.info("Variance explained: %.1f%%", 100 * pca.explained_variance_ratio_.sum())

    human_r = pca.transform(human)
    llm_r = pca.transform(all_llm)
    del combined

    # Compute reconstruction error (from Layer 2: PCA on all_llm, project human)
    logger.info("Computing reconstruction errors...")
    pca_llm = PCA()
    pca_llm.fit(all_llm)
    cum = np.cumsum(pca_llm.explained_variance_ratio_)
    k = int(np.searchsorted(cum, 0.90) + 1)
    k = min(k, len(cum))
    comp_k = pca_llm.components_[:k]
    centered = human - pca_llm.mean_
    projected = centered @ comp_k.T
    reconstructed = projected @ comp_k
    recon_errors = ((centered - reconstructed) ** 2).sum(axis=1)
    logger.info("Reconstruction errors: mean=%.4f, k=%d", recon_errors.mean(), k)

    # KNN: LLM neighbors of each human rationale
    logger.info("Fitting KNN on all_llm (%d points, %d dims)...", llm_r.shape[0], llm_r.shape[1])
    nn_llm = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm="ball_tree")
    nn_llm.fit(llm_r)
    llm_dists, _ = nn_llm.kneighbors(human_r)
    llm_mean_dist = llm_dists.mean(axis=1)
    logger.info("LLM KNN done. Mean dist: %.4f", llm_mean_dist.mean())

    # KNN: human neighbors of each human rationale (self-density baseline)
    logger.info("Fitting KNN on human (%d points)...", human_r.shape[0])
    nn_human = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm="ball_tree")
    nn_human.fit(human_r)
    human_dists, _ = nn_human.kneighbors(human_r)
    human_mean_dist = human_dists.mean(axis=1)
    logger.info("Human KNN done. Mean dist: %.4f", human_mean_dist.mean())

    # Density ratio: higher = more isolated from LLMs relative to own distribution
    density_ratio = llm_mean_dist / human_mean_dist
    logger.info("Density ratio: mean=%.3f, median=%.3f", density_ratio.mean(), np.median(density_ratio))

    # Radius-based count: LLM neighbors within fixed radius
    radius = np.median(llm_dists[:, -1])  # median of 50th-neighbor distance
    logger.info("Radius for count-based analysis: %.4f", radius)
    llm_count = nn_llm.radius_neighbors(human_r, radius=radius, return_distance=False)
    llm_neighbor_counts = np.array([len(idx) for idx in llm_count])
    human_count = nn_human.radius_neighbors(human_r, radius=radius, return_distance=False)
    human_neighbor_counts = np.array([len(idx) for idx in human_count])
    logger.info("Radius counts: LLM mean=%.1f, Human mean=%.1f",
                llm_neighbor_counts.mean(), human_neighbor_counts.mean())

    # Correlations
    if HAS_SCIPY:
        r_dist, p_dist = pearsonr(recon_errors, llm_mean_dist)
        r_ratio, p_ratio = pearsonr(recon_errors, density_ratio)
        r_count, p_count = pearsonr(recon_errors, llm_neighbor_counts)
    else:
        r_dist = float(np.corrcoef(recon_errors, llm_mean_dist)[0, 1])
        r_ratio = float(np.corrcoef(recon_errors, density_ratio)[0, 1])
        r_count = float(np.corrcoef(recon_errors, llm_neighbor_counts)[0, 1])
        p_dist = p_ratio = p_count = float("nan")

    logger.info("Pearson r (recon_error vs llm_mean_dist): %.3f (p=%s)",
                r_dist, f"{p_dist:.2e}" if p_dist == p_dist else "N/A")
    logger.info("Pearson r (recon_error vs density_ratio): %.3f (p=%s)",
                r_ratio, f"{p_ratio:.2e}" if p_ratio == p_ratio else "N/A")
    logger.info("Pearson r (recon_error vs llm_neighbor_count): %.3f (p=%s)",
                r_count, f"{p_count:.2e}" if p_count == p_count else "N/A")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "frequency_geometry.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "reconstruction_error", "llm_knn_dist", "human_knn_dist",
            "density_ratio", "llm_neighbor_count", "human_neighbor_count"])
        writer.writeheader()
        for i in range(len(human)):
            writer.writerow({
                "index": i,
                "reconstruction_error": f"{recon_errors[i]:.6f}",
                "llm_knn_dist": f"{llm_mean_dist[i]:.6f}",
                "human_knn_dist": f"{human_mean_dist[i]:.6f}",
                "density_ratio": f"{density_ratio[i]:.4f}",
                "llm_neighbor_count": int(llm_neighbor_counts[i]),
                "human_neighbor_count": int(human_neighbor_counts[i]),
            })
    logger.info("Saved %s", csv_path)

    # Plot 1: Scatter of density ratio vs reconstruction error
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(recon_errors, density_ratio, s=3, alpha=0.2, color=SOURCE_COLORS["human"])
    z = np.polyfit(recon_errors, density_ratio, 1)
    x_line = np.linspace(recon_errors.min(), recon_errors.max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), color="black", linewidth=2, linestyle="--")
    p_str = f", p={p_ratio:.2e}" if p_ratio == p_ratio else ""
    ax.set_xlabel("Reconstruction Error (under LLM principal components)")
    ax.set_ylabel("LLM-to-Human Distance Ratio")
    ax.set_title(f"Frequency Isolation: Human Rationales in Kaleido Space\n"
                 f"(Pearson r={r_ratio:.3f}{p_str})")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "frequency_geometry_correlation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)

    # Plot 2: Binned gradient (sliding window) matching Layer 3 format
    sorted_idx = np.argsort(recon_errors)
    sorted_errors = recon_errors[sorted_idx]
    sorted_ratio = density_ratio[sorted_idx]
    sorted_llm_dist = llm_mean_dist[sorted_idx]
    sorted_llm_count = llm_neighbor_counts[sorted_idx]

    bins = []
    for start in range(0, len(sorted_errors) - WINDOW_SIZE + 1, WINDOW_STEP):
        w_err = sorted_errors[start:start + WINDOW_SIZE]
        w_ratio = sorted_ratio[start:start + WINDOW_SIZE]
        w_dist = sorted_llm_dist[start:start + WINDOW_SIZE]
        w_count = sorted_llm_count[start:start + WINDOW_SIZE]
        bins.append({
            "mean_error": float(w_err.mean()),
            "mean_ratio": float(w_ratio.mean()),
            "mean_llm_dist": float(w_dist.mean()),
            "mean_llm_count": float(w_count.mean()),
        })

    bin_errors = np.array([b["mean_error"] for b in bins])
    bin_ratios = np.array([b["mean_ratio"] for b in bins])
    bin_dists = np.array([b["mean_llm_dist"] for b in bins])
    bin_counts = np.array([b["mean_llm_count"] for b in bins])

    if HAS_SCIPY:
        r_bin_ratio, p_bin_ratio = pearsonr(bin_errors, bin_ratios)
        r_bin_dist, p_bin_dist = pearsonr(bin_errors, bin_dists)
        r_bin_count, p_bin_count = pearsonr(bin_errors, bin_counts)
    else:
        r_bin_ratio = float(np.corrcoef(bin_errors, bin_ratios)[0, 1])
        r_bin_dist = float(np.corrcoef(bin_errors, bin_dists)[0, 1])
        r_bin_count = float(np.corrcoef(bin_errors, bin_counts)[0, 1])
        p_bin_ratio = p_bin_dist = p_bin_count = float("nan")

    logger.info("Binned Pearson r (error vs density_ratio): %.3f", r_bin_ratio)
    logger.info("Binned Pearson r (error vs llm_dist): %.3f", r_bin_dist)
    logger.info("Binned Pearson r (error vs llm_count): %.3f", r_bin_count)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = SOURCE_COLORS["human"]
    ax1.scatter(bin_errors, bin_dists, color=color1, alpha=0.6, s=20, zorder=3)
    ax1.plot(bin_errors, bin_dists, color=color1, alpha=0.4, linewidth=1)
    ax1.set_xlabel("Mean Reconstruction Error")
    ax1.set_ylabel("Mean Distance to 50 Nearest LLM Neighbors", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    z1 = np.polyfit(bin_errors, bin_dists, 1)
    ax1.plot(bin_errors, np.poly1d(z1)(bin_errors), color=color1, linestyle="--",
             linewidth=1.5, alpha=0.7)

    ax2 = ax1.twinx()
    color2 = "#ff7f0e"
    ax2.scatter(bin_errors, bin_counts, color=color2, alpha=0.6, s=20, zorder=3)
    ax2.plot(bin_errors, bin_counts, color=color2, alpha=0.4, linewidth=1)
    ax2.set_ylabel("Mean LLM Neighbors Within Radius", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    z2 = np.polyfit(bin_errors, bin_counts, 1)
    ax2.plot(bin_errors, np.poly1d(z2)(bin_errors), color=color2, linestyle="--",
             linewidth=1.5, alpha=0.7)

    p_str1 = f", p<0.0001" if p_bin_dist < 0.0001 else f", p={p_bin_dist:.4f}"
    ax1.set_title(f"LLM Density Around Human Rationales vs Reconstruction Error\n"
                  f"(Distance r={r_bin_dist:.3f}{p_str1}; "
                  f"Count r={r_bin_count:.3f})")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "frequency_geometry_binned.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)

    elapsed = time.time() - start_time
    logger.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
