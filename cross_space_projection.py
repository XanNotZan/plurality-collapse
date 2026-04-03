"""Cross-space projection analysis: do LLMs occupy a subspace of human moral reasoning?"""

import csv
import json
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# ── Constants ──────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = "data/embeddings"
OUTPUT_DIR = "data/analysis"
HIDDEN_DIM = 2048
MAX_K = 500

ALL_SOURCES = ["human", "gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]

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
    logger = logging.getLogger("cross_space_projection")
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


# ── PCA Fitting ──────────────────────────────────────────────────────────────
def fit_all_pcas(
    embeddings: dict[str, np.ndarray], logger: logging.Logger,
) -> dict[str, dict]:
    """Fit PCA on each source. Returns {source: {pca, components_90, mean}}."""
    fitted = {}
    for source, matrix in embeddings.items():
        pca = PCA()
        pca.fit(matrix)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        comp90 = int(np.searchsorted(cumulative, 0.90) + 1)
        comp90 = min(comp90, len(cumulative))
        fitted[source] = {
            "pca": pca,
            "comp90": comp90,
            "mean": pca.mean_,
            "total_variance": float(pca.explained_variance_.sum()),
        }
        logger.info("%s: PCA fit, 90%% at k=%d, total_var=%.1f",
                    source, comp90, fitted[source]["total_variance"])
    return fitted


# ── Cross-Projection ────────────────────────────────────────────────────────
def variance_captured_at_k(
    basis_pca: PCA, target_data: np.ndarray, k: int,
) -> float:
    """Fraction of target_data's variance captured by basis's top-k components.

    Projects target_data (centered by basis mean) onto basis's top-k PCs,
    then measures what fraction of target's total variance is explained.
    """
    # Center target data using basis mean
    centered = target_data - basis_pca.mean_
    total_var = float(np.var(centered, axis=0).sum())
    if total_var < 1e-12:
        return 1.0

    # Project onto top-k components
    components_k = basis_pca.components_[:k]  # [k, d]
    projections = centered @ components_k.T    # [n, k]
    captured_var = float(np.var(projections, axis=0).sum())

    return captured_var / total_var


def compute_all_cross_projections(
    embeddings: dict[str, np.ndarray],
    fitted: dict[str, dict],
    logger: logging.Logger,
) -> list[dict]:
    """Compute cross-projection variance for all directed pairs at k=90%."""
    results = []
    sources = [s for s in PLOT_ORDER if s in embeddings]

    for basis_name in sources:
        basis_pca = fitted[basis_name]["pca"]
        k = fitted[basis_name]["comp90"]
        for target_name in sources:
            vc = variance_captured_at_k(basis_pca, embeddings[target_name], k)

            # Self-variance at same k (for reference)
            self_vc = variance_captured_at_k(basis_pca, embeddings[basis_name], k)

            results.append({
                "basis_source": basis_name,
                "projected_source": target_name,
                "k": k,
                "variance_captured": vc,
                "self_variance_at_k": self_vc,
            })

            if basis_name != target_name:
                logger.info("%s PCs (k=%d) → %s: %.4f variance captured",
                            basis_name, k, target_name, vc)

    return results


# ── Variance Curves (human ↔ all_llm) ───────────────────────────────────────
def compute_variance_curves(
    embeddings: dict[str, np.ndarray],
    fitted: dict[str, dict],
    logger: logging.Logger,
) -> dict:
    """Sweep k=1..MAX_K for human↔all_llm cross-projection curves."""
    if "human" not in fitted or "all_llm" not in fitted:
        logger.warning("Need both human and all_llm for variance curves.")
        return {}

    human_pca = fitted["human"]["pca"]
    allllm_pca = fitted["all_llm"]["pca"]
    human_data = embeddings["human"]
    allllm_data = embeddings["all_llm"]

    max_k_human = min(MAX_K, human_pca.components_.shape[0])
    max_k_allllm = min(MAX_K, allllm_pca.components_.shape[0])
    k_range = max(max_k_human, max_k_allllm)

    curves = {
        "human_self": [],
        "human_to_allllm": [],
        "allllm_self": [],
        "allllm_to_human": [],
        "k_values": [],
    }

    for k in range(1, k_range + 1):
        curves["k_values"].append(k)

        if k <= max_k_human:
            curves["human_self"].append(
                variance_captured_at_k(human_pca, human_data, k))
            curves["human_to_allllm"].append(
                variance_captured_at_k(human_pca, allllm_data, k))
        else:
            curves["human_self"].append(curves["human_self"][-1])
            curves["human_to_allllm"].append(curves["human_to_allllm"][-1])

        if k <= max_k_allllm:
            curves["allllm_self"].append(
                variance_captured_at_k(allllm_pca, allllm_data, k))
            curves["allllm_to_human"].append(
                variance_captured_at_k(allllm_pca, human_data, k))
        else:
            curves["allllm_self"].append(curves["allllm_self"][-1])
            curves["allllm_to_human"].append(curves["allllm_to_human"][-1])

        if k % 100 == 0:
            logger.info("Variance curves: k=%d/%d", k, k_range)

    logger.info("Variance curves computed (k=1..%d)", k_range)
    return curves


# ── Outputs ──────────────────────────────────────────────────────────────────
def save_summary_csv(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Write cross_projection_summary.csv."""
    path = os.path.join(output_dir, "cross_projection_summary.csv")
    fieldnames = ["basis_source", "projected_source", "k",
                  "variance_captured", "self_variance_at_k"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "basis_source": r["basis_source"],
                "projected_source": r["projected_source"],
                "k": r["k"],
                "variance_captured": f"{r['variance_captured']:.4f}",
                "self_variance_at_k": f"{r['self_variance_at_k']:.4f}",
            })
    logger.info("Saved %s", path)


def save_curves_json(
    curves: dict, output_dir: str, logger: logging.Logger,
) -> None:
    """Write cross_projection_curves.json."""
    path = os.path.join(output_dir, "cross_projection_curves.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)
    logger.info("Saved %s", path)


def plot_curves(
    curves: dict, fitted: dict, output_dir: str, logger: logging.Logger,
) -> None:
    """Line plot of 4 variance-captured curves (human↔all_llm)."""
    if not curves:
        logger.warning("No curves to plot.")
        return

    k_vals = curves["k_values"]
    k_max_plot = min(MAX_K, len(k_vals))
    k_plot = k_vals[:k_max_plot]

    # Compute asymmetry at 90% thresholds
    human_k90 = fitted["human"]["comp90"]
    allllm_k90 = fitted["all_llm"]["comp90"]

    human_to_llm_at_k = curves["human_to_allllm"][human_k90 - 1]
    llm_to_human_at_k = curves["allllm_to_human"][allllm_k90 - 1]
    asymmetry = human_to_llm_at_k - llm_to_human_at_k

    fig, ax = plt.subplots(figsize=(10, 6))

    # Self curves (solid)
    ax.plot(k_plot, curves["human_self"][:k_max_plot],
            color=SOURCE_COLORS["human"], linestyle="-", linewidth=2.5,
            label="human PCs \u2192 human (self)")
    ax.plot(k_plot, curves["allllm_self"][:k_max_plot],
            color=SOURCE_COLORS["all_llm"], linestyle="-", linewidth=2.5,
            label="all_llm PCs \u2192 all_llm (self)")

    # Cross curves (dashed)
    ax.plot(k_plot, curves["human_to_allllm"][:k_max_plot],
            color=SOURCE_COLORS["human"], linestyle="--", linewidth=2,
            label=f"human PCs \u2192 all_llm ({human_to_llm_at_k:.3f} at k={human_k90})")
    ax.plot(k_plot, curves["allllm_to_human"][:k_max_plot],
            color=SOURCE_COLORS["all_llm"], linestyle="--", linewidth=2,
            label=f"all_llm PCs \u2192 human ({llm_to_human_at_k:.3f} at k={allllm_k90})")

    # Annotate the gap at human's k90
    ax.annotate(
        f"\u0394={asymmetry:+.3f}",
        xy=(human_k90, (human_to_llm_at_k + llm_to_human_at_k) / 2),
        fontsize=11, fontweight="bold", ha="left",
        xytext=(human_k90 + 20, (human_to_llm_at_k + llm_to_human_at_k) / 2),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
    )

    # Vertical reference lines at k90 thresholds
    ax.axvline(human_k90, color=SOURCE_COLORS["human"], linestyle=":",
               linewidth=0.8, alpha=0.5)
    ax.axvline(allllm_k90, color=SOURCE_COLORS["all_llm"], linestyle=":",
               linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Number of Principal Components (k)")
    ax.set_ylabel("Fraction of Variance Captured")
    ax.set_title(f"Cross-Space Projection: Human \u2194 All-LLM (asymmetry={asymmetry:+.3f})")
    ax.set_xlim(1, k_max_plot)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, "cross_projection_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s (asymmetry=%+.4f)", path, asymmetry)


def plot_heatmap(
    results: list[dict],
    embeddings: dict[str, np.ndarray],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Heatmap of variance captured at each basis's k90 for all pairs."""
    sources = [s for s in PLOT_ORDER if s in embeddings]
    n = len(sources)

    # Build matrix
    lookup = {}
    for r in results:
        lookup[(r["basis_source"], r["projected_source"])] = r["variance_captured"]

    matrix = np.zeros((n, n))
    for i, basis in enumerate(sources):
        for j, target in enumerate(sources):
            matrix[i, j] = lookup.get((basis, target), 0.0)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.5, vmax=1.0, aspect="auto")

    # Labels
    ax.set_xticks(range(n))
    ax.set_xticklabels(sources, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sources, fontsize=9)
    ax.set_xlabel("Projected Source (data)")
    ax.set_ylabel("Basis Source (PCs)")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val > 0.85 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title("Cross-Projection Variance Captured (at basis 90% k)")
    fig.colorbar(im, ax=ax, label="Fraction of variance captured", shrink=0.8)
    fig.tight_layout()
    path = os.path.join(output_dir, "cross_projection_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    logger = setup_logging()
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load embeddings
    embeddings = load_embeddings(logger)

    # Build all_llm
    all_llm = build_all_llm_matrix(embeddings, logger)
    if all_llm is not None:
        embeddings["all_llm"] = all_llm

    # Fit PCA on each source
    logger.info("Fitting PCA on all sources...")
    fitted = fit_all_pcas(embeddings, logger)

    # Cross-projection at k=90% for all pairs
    logger.info("Computing cross-projections (all pairs)...")
    cross_results = compute_all_cross_projections(embeddings, fitted, logger)
    save_summary_csv(cross_results, OUTPUT_DIR, logger)

    # Focused comparison: human ↔ all_llm
    if "human" in fitted and "all_llm" in fitted:
        human_k90 = fitted["human"]["comp90"]
        allllm_k90 = fitted["all_llm"]["comp90"]

        vc_human_to_llm = variance_captured_at_k(
            fitted["human"]["pca"], embeddings["all_llm"], human_k90)
        vc_llm_to_human = variance_captured_at_k(
            fitted["all_llm"]["pca"], embeddings["human"], allllm_k90)
        asymmetry = vc_human_to_llm - vc_llm_to_human

        logger.info("=== Key Result: Human ↔ All-LLM ===")
        logger.info("Human PCs (k=%d) capture %.4f of all_llm variance",
                    human_k90, vc_human_to_llm)
        logger.info("All-LLM PCs (k=%d) capture %.4f of human variance",
                    allllm_k90, vc_llm_to_human)
        logger.info("Asymmetry: %+.4f (positive = human PCs better at capturing LLMs)",
                    asymmetry)

    # Variance curves (k=1..500)
    logger.info("Computing variance curves (k=1..%d)...", MAX_K)
    curves = compute_variance_curves(embeddings, fitted, logger)
    if curves:
        save_curves_json(curves, OUTPUT_DIR, logger)
        plot_curves(curves, fitted, OUTPUT_DIR, logger)

    # Heatmap
    plot_heatmap(cross_results, embeddings, OUTPUT_DIR, logger)

    elapsed = time.time() - start_time
    logger.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
