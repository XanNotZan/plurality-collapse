"""Replicate the plurality-collapse dimensionality analysis using Sentence-BERT
(all-mpnet-base-v2, 768-dim) instead of Kaleido-XL (2048-dim).

Answers: does the human-vs-LLM dimensionality gap persist in a general-purpose
encoder, or is it specific to the moral-reasoning embedding space?
"""

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
from sklearn.decomposition import PCA, KernelPCA

try:
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── Constants ────────────────────────────────────────────────────────────────

DATASET_NAME = "ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas"
DATASET_SPLIT = "test"
SBERT_MODEL_NAME = "all-mpnet-base-v2"
HIDDEN_DIM = 768
BATCH_SIZE = 64

EMBEDDINGS_DIR = "data/embeddings_sbert"
ANALYSIS_DIR = "data/analysis_sbert"
KALEIDO_ANALYSIS_DIR = "data/analysis"
KALEIDO_EMBEDDINGS_DIR = "data/embeddings"

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
LLM_SOURCE_LIST = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
PLOT_ORDER = ["human"] + sorted(LLM_SOURCE_LIST) + ["all_llm"]

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

KERNEL_PCA_MAX_SAMPLES = 5000
RANDOM_SEED = 42
SUBSAMPLE_SEEDS = [42, 43, 44, 45, 46]
WINDOW_SIZE = 200
WINDOW_STEP = 50

HUMAN_VALUES_PATH = "data/analysis/human_values_all.json"


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("alternative_encoder")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Extract SBERT embeddings
# ═══════════════════════════════════════════════════════════════════════════════

def collect_rationales(ds, logger: logging.Logger) -> dict[str, list[dict]]:
    """Collect rationales per source. Returns {source: [{submission_id, column, text}, ...]}."""
    sources = {}

    # Human rationales
    human = []
    for row in ds:
        text = row.get("top_comment")
        if text and isinstance(text, str) and text.strip():
            human.append({
                "submission_id": row["submission_id"],
                "column": "top_comment",
                "text": text.strip(),
            })
    sources["human"] = human
    logger.info("human: %d rationales", len(human))

    # LLM rationales
    for source_name, columns in LLM_SOURCES.items():
        entries = []
        for row in ds:
            for col in columns:
                text = row.get(col)
                if text and isinstance(text, str) and text.strip():
                    entries.append({
                        "submission_id": row["submission_id"],
                        "column": col,
                        "text": text.strip(),
                    })
        sources[source_name] = entries
        logger.info("%s: %d rationales", source_name, len(entries))

    return sources


def step1_extract_embeddings(logger: logging.Logger) -> None:
    """Load dataset, encode all sources with SBERT, save .npy + _meta.json."""
    all_present = all(
        os.path.exists(os.path.join(EMBEDDINGS_DIR, f"{s}.npy"))
        for s in ALL_SOURCES
    )
    if all_present:
        logger.info("All SBERT embeddings already exist, skipping Step 1.")
        return

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("SBERT device: %s", device)

    model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    logger.info("Loaded SBERT model: %s (%d-dim)", SBERT_MODEL_NAME, HIDDEN_DIM)

    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    logger.info("Dataset: %d rows", len(ds))

    sources = collect_rationales(ds, logger)

    for source_name in ALL_SOURCES:
        entries = sources.get(source_name, [])
        if not entries:
            logger.warning("Skipping %s: no rationales", source_name)
            continue

        npy_path = os.path.join(EMBEDDINGS_DIR, f"{source_name}.npy")
        if os.path.exists(npy_path):
            logger.info("%s: already exists, skipping", source_name)
            continue

        texts = [e["text"] for e in entries]
        logger.info("%s: encoding %d rationales...", source_name, len(texts))

        matrix = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        metadata = [
            {"index": idx, "submission_id": e["submission_id"], "column": e["column"]}
            for idx, e in enumerate(entries)
        ]

        np.save(npy_path, matrix)
        meta_path = os.path.join(EMBEDDINGS_DIR, f"{source_name}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        logger.info("%s: saved shape %s", source_name, matrix.shape)

    del model
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: PCA dimensionality analysis
# ═══════════════════════════════════════════════════════════════════════════════

def load_sbert_embeddings(logger: logging.Logger) -> dict[str, np.ndarray]:
    """Load SBERT embedding matrices from disk."""
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
    return embeddings


def build_all_llm_matrix(
    embeddings: dict[str, np.ndarray], logger: logging.Logger,
) -> np.ndarray | None:
    matrices = [embeddings[s] for s in LLM_SOURCE_LIST if s in embeddings]
    if not matrices:
        return None
    all_llm = np.vstack(matrices)
    logger.info("all_llm matrix: shape %s", all_llm.shape)
    return all_llm


def compute_pca_stats(
    matrix: np.ndarray, source_name: str, logger: logging.Logger,
) -> dict:
    """Run full PCA on an embedding matrix and return dimensionality statistics."""
    pca = PCA()
    pca.fit(matrix)

    eigenvalues = pca.explained_variance_
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    pr = float((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum())

    components_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    components_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    components_90 = min(components_90, len(cumulative))
    components_95 = min(components_95, len(cumulative))

    logger.info("%s: n=%d, PR=%.2f, comp90=%d, comp95=%d",
                source_name, matrix.shape[0], pr, components_90, components_95)

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
        logger.info("%s: Kernel PCA PR=%.2f", source_name, pr)
        return pr
    except Exception:
        logger.exception("%s: Kernel PCA failed", source_name)
        return None


def plot_cumulative_variance(
    results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Line plot of cumulative explained variance curves."""
    by_source = {r["source"]: r for r in results}

    x_max = 500
    if "human" in by_source:
        cum = np.array(by_source["human"]["cumulative_variance"])
        idx_99 = int(np.searchsorted(cum, 0.99) + 1)
        x_max = min(max(idx_99, 50), 500)

    fig, ax = plt.subplots(figsize=(10, 6))

    for source in sorted(LLM_SOURCE_LIST):
        if source not in by_source:
            continue
        cum = by_source[source]["cumulative_variance"]
        n = min(len(cum), x_max)
        ax.plot(
            range(1, n + 1), cum[:n],
            color=SOURCE_COLORS.get(source, "#999999"),
            linestyle="--", linewidth=1, alpha=0.5, label=source,
        )

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

    ax.axhline(0.90, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(x_max * 0.98, 0.90, "90%", ha="right", va="bottom", fontsize=8, color="gray")
    ax.text(x_max * 0.98, 0.95, "95%", ha="right", va="bottom", fontsize=8, color="gray")

    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title("Cumulative Explained Variance (SBERT)")
    ax.set_xlim(1, x_max)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, "cumulative_variance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def step2_pca_analysis(logger: logging.Logger) -> None:
    """Run PCA + Kernel PCA on all sources, save summary_table.csv + plots."""
    embeddings = load_sbert_embeddings(logger)
    all_llm = build_all_llm_matrix(embeddings, logger)
    if all_llm is not None:
        embeddings["all_llm"] = all_llm

    results = []
    for source in PLOT_ORDER:
        if source not in embeddings:
            continue
        results.append(compute_pca_stats(embeddings[source], source, logger))

    kernel_pca_results = {}
    for source in PLOT_ORDER:
        if source not in embeddings:
            continue
        kernel_pca_results[source] = compute_kernel_pca_pr(
            embeddings[source], source, logger)

    # Save summary CSV
    path = os.path.join(ANALYSIS_DIR, "summary_table.csv")
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

    plot_cumulative_variance(results, ANALYSIS_DIR, logger)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Cross-space projection
# ═══════════════════════════════════════════════════════════════════════════════

def variance_captured_at_k(
    basis_pca: PCA, target_data: np.ndarray, k: int,
) -> float:
    """Fraction of target_data's variance captured by basis's top-k components."""
    centered = target_data - basis_pca.mean_
    total_var = float(np.var(centered, axis=0).sum())
    if total_var < 1e-12:
        return 1.0

    components_k = basis_pca.components_[:k]  # [k, d]
    projections = centered @ components_k.T    # [n, k]
    captured_var = float(np.var(projections, axis=0).sum())

    return captured_var / total_var


def step3_cross_projection(logger: logging.Logger) -> None:
    """Fit PCA per source, compute all directed pairs, save CSV + heatmap."""
    embeddings = load_sbert_embeddings(logger)
    all_llm = build_all_llm_matrix(embeddings, logger)
    if all_llm is not None:
        embeddings["all_llm"] = all_llm

    # Fit PCA per source
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
        logger.info("%s: PCA fitted, k90=%d", source, comp90)

    # Compute all directed pairs
    sources = [s for s in PLOT_ORDER if s in embeddings]
    results = []
    for basis_name in sources:
        basis_pca = fitted[basis_name]["pca"]
        k = fitted[basis_name]["comp90"]
        for target_name in sources:
            vc = variance_captured_at_k(basis_pca, embeddings[target_name], k)
            self_vc = variance_captured_at_k(basis_pca, embeddings[basis_name], k)
            results.append({
                "basis_source": basis_name,
                "projected_source": target_name,
                "k": k,
                "variance_captured": vc,
                "self_variance_at_k": self_vc,
            })

    # Save CSV
    path = os.path.join(ANALYSIS_DIR, "cross_projection_summary.csv")
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

    # Log asymmetry
    if "human" in fitted and "all_llm" in fitted:
        human_k90 = fitted["human"]["comp90"]
        allllm_k90 = fitted["all_llm"]["comp90"]
        vc_h2l = variance_captured_at_k(
            fitted["human"]["pca"], embeddings["all_llm"], human_k90)
        vc_l2h = variance_captured_at_k(
            fitted["all_llm"]["pca"], embeddings["human"], allllm_k90)
        asymmetry = vc_h2l - vc_l2h
        logger.info("SBERT asymmetry: human->all_llm=%.4f, all_llm->human=%.4f, diff=%+.4f",
                     vc_h2l, vc_l2h, asymmetry)

    # Heatmap
    n = len(sources)
    lookup = {}
    for r in results:
        lookup[(r["basis_source"], r["projected_source"])] = r["variance_captured"]

    heatmap_matrix = np.zeros((n, n))
    for i, basis in enumerate(sources):
        for j, target in enumerate(sources):
            heatmap_matrix[i, j] = lookup.get((basis, target), 0.0)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(heatmap_matrix, cmap="YlOrRd", vmin=0.5, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(sources, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sources, fontsize=9)
    ax.set_xlabel("Projected Source (data)")
    ax.set_ylabel("Basis Source (PCs)")

    for i in range(n):
        for j in range(n):
            val = heatmap_matrix[i, j]
            color = "white" if val > 0.85 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title("Cross-Projection Variance Captured — SBERT (at basis 90% k)")
    fig.colorbar(im, ax=ax, label="Fraction of variance captured", shrink=0.8)
    fig.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "cross_projection_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Subsampling robustness
# ═══════════════════════════════════════════════════════════════════════════════

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


def step4_subsampling_robustness(logger: logging.Logger) -> None:
    """Subsample LLMs to human sample size and recompute PCA."""
    embeddings = load_sbert_embeddings(logger)

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
            logger.info("%s: n=%d (no subsampling needed), PR=%.2f, comp90=%d",
                        source, n_rows, metrics["pr"], metrics["comp90"])
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
            logger.info("%s: subsampled to %d, PR=%.2f+/-%.2f, comp90=%.1f+/-%.1f",
                        source, human_n,
                        np.mean(pr_vals), np.std(pr_vals),
                        np.mean(comp90_vals), np.std(comp90_vals))

    # Save CSV
    path = os.path.join(ANALYSIS_DIR, "subsample_table.csv")
    fieldnames = ["source", "n_samples", "pr_mean", "pr_std",
                  "comp90_mean", "comp90_std", "comp95_mean", "comp95_std"]
    with open(path, "w", newline="", encoding="utf-8") as f:
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
    logger.info("Saved %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Value diversity gradient
# ═══════════════════════════════════════════════════════════════════════════════

def compute_shannon_entropy(values: list[str]) -> float:
    """Compute Shannon entropy H = -sum(p * log2(p)) for value frequency distribution."""
    if not values:
        return 0.0
    freq: dict[str, int] = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    total = len(values)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def _bin_window(records: list[dict], bin_index: int) -> dict:
    """Compute metrics for a single bin/window of records."""
    errors = [r["reconstruction_error"] for r in records]
    values = [r["generated_values"] for r in records]
    return {
        "bin_index": bin_index,
        "n_rationales": len(records),
        "mean_error": float(np.mean(errors)),
        "min_error": float(min(errors)),
        "max_error": float(max(errors)),
        "n_unique_values": len(set(values)),
        "shannon_entropy": compute_shannon_entropy(values),
    }


def compute_nonoverlapping_bins(
    sorted_records: list[dict], window_size: int, logger: logging.Logger,
) -> list[dict]:
    bins = []
    for i in range(0, len(sorted_records), window_size):
        window = sorted_records[i:i + window_size]
        bins.append(_bin_window(window, len(bins)))
    logger.info("Non-overlapping bins: %d (window=%d)", len(bins), window_size)
    return bins


def compute_sliding_bins(
    sorted_records: list[dict], window_size: int, step: int, logger: logging.Logger,
) -> list[dict]:
    bins = []
    for start in range(0, len(sorted_records) - window_size + 1, step):
        window = sorted_records[start:start + window_size]
        bins.append(_bin_window(window, len(bins)))
    logger.info("Sliding bins: %d (window=%d, step=%d)", len(bins), window_size, step)
    return bins


def save_gradient_csv(
    bins: list[dict], filename: str, output_dir: str, logger: logging.Logger,
) -> None:
    path = os.path.join(output_dir, filename)
    fieldnames = ["bin_index", "n_rationales", "mean_error", "min_error", "max_error",
                  "n_unique_values", "shannon_entropy"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for b in bins:
            writer.writerow({
                "bin_index": b["bin_index"],
                "n_rationales": b["n_rationales"],
                "mean_error": f"{b['mean_error']:.6f}",
                "min_error": f"{b['min_error']:.6f}",
                "max_error": f"{b['max_error']:.6f}",
                "n_unique_values": b["n_unique_values"],
                "shannon_entropy": f"{b['shannon_entropy']:.4f}",
            })
    logger.info("Saved %s", path)


def _compute_pearson(bins: list[dict], label: str, logger: logging.Logger) -> tuple[float, float]:
    mean_errors = np.array([b["mean_error"] for b in bins])
    n_unique = np.array([b["n_unique_values"] for b in bins])
    if HAS_SCIPY:
        r, p = pearsonr(mean_errors, n_unique)
    else:
        r = float(np.corrcoef(mean_errors, n_unique)[0, 1])
        p = float("nan")
    logger.info("%s: Pearson r=%.3f, p=%s", label,
                r, f"{p:.4f}" if not math.isnan(p) else "N/A")
    return float(r), float(p)


def plot_diversity_gradient(
    bins: list[dict], filename: str, title_suffix: str,
    output_dir: str, logger: logging.Logger,
) -> None:
    """Dual y-axis plot: unique values + Shannon entropy vs reconstruction error."""
    mean_errors = np.array([b["mean_error"] for b in bins])
    n_unique = np.array([b["n_unique_values"] for b in bins])
    entropy = np.array([b["shannon_entropy"] for b in bins])

    if HAS_SCIPY:
        r, p = pearsonr(mean_errors, n_unique)
    else:
        r = float(np.corrcoef(mean_errors, n_unique)[0, 1])
        p = float("nan")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = SOURCE_COLORS["human"]
    ax1.scatter(mean_errors, n_unique, color=color1, alpha=0.6, s=20, zorder=3)
    ax1.plot(mean_errors, n_unique, color=color1, alpha=0.4, linewidth=1)
    ax1.set_xlabel("Mean Reconstruction Error (SBERT)")
    ax1.set_ylabel("Unique Values per Bin", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    z = np.polyfit(mean_errors, n_unique, 1)
    trend = np.poly1d(z)
    ax1.plot(mean_errors, trend(mean_errors), color=color1, linestyle="--",
             linewidth=1.5, alpha=0.7)

    ax2 = ax1.twinx()
    color2 = "#ff7f0e"
    ax2.scatter(mean_errors, entropy, color=color2, alpha=0.6, s=20, zorder=3)
    ax2.plot(mean_errors, entropy, color=color2, alpha=0.4, linewidth=1)
    ax2.set_ylabel("Shannon Entropy (bits)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    z2 = np.polyfit(mean_errors, entropy, 1)
    trend2 = np.poly1d(z2)
    ax2.plot(mean_errors, trend2(mean_errors), color=color2, linestyle="--",
             linewidth=1.5, alpha=0.7)

    p_str = f", p={p:.4f}" if not math.isnan(p) else ""
    ax1.set_title(f"Value Diversity vs Reconstruction Error (SBERT) \u2014 {title_suffix}\n"
                  f"(Pearson r={r:.3f}{p_str})")
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def step5_value_diversity_gradient(logger: logging.Logger) -> None:
    """Fit PCA on all_llm SBERT, compute reconstruction error per human rationale,
    load human_values_all.json, bin by error, plot + save CSV."""
    embeddings = load_sbert_embeddings(logger)
    all_llm = build_all_llm_matrix(embeddings, logger)
    if all_llm is None:
        raise SystemExit("No LLM embeddings found.")

    # Fit PCA on all_llm
    pca = PCA()
    pca.fit(all_llm)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumulative, 0.90) + 1)
    k = min(k, len(cumulative))
    logger.info("all_llm SBERT PCA: k=%d for 90%% variance", k)

    # Compute reconstruction error for each human rationale
    human = embeddings["human"]
    components_k = pca.components_[:k]
    centered = human - pca.mean_
    projected = centered @ components_k.T
    reconstructed = projected @ components_k
    residuals = centered - reconstructed
    errors = (residuals ** 2).sum(axis=1)

    logger.info("Human reconstruction: mean=%.4f, median=%.4f, max=%.4f",
                errors.mean(), np.median(errors), errors.max())

    # Load SBERT human metadata for alignment check
    meta_path = os.path.join(EMBEDDINGS_DIR, "human_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        sbert_meta = json.load(f)

    # Load existing Kaleido-generated value records
    with open(HUMAN_VALUES_PATH, encoding="utf-8") as f:
        value_records = json.load(f)

    # Hard alignment assertion
    if len(sbert_meta) != len(value_records):
        raise ValueError(
            f"Length mismatch: {len(sbert_meta)} SBERT embeddings vs "
            f"{len(value_records)} value records"
        )
    for i, (sm, vr) in enumerate(zip(sbert_meta, value_records)):
        if sm["submission_id"] != vr["submission_id"]:
            raise ValueError(
                f"Index {i}: SBERT sid={sm['submission_id']} != "
                f"values sid={vr['submission_id']}"
            )
    logger.info("Alignment verified: %d records match by submission_id", len(sbert_meta))

    # Build records with SBERT errors and Kaleido values
    records = []
    for i, (vr, e) in enumerate(zip(value_records, errors)):
        records.append({
            "index": i,
            "submission_id": vr["submission_id"],
            "rationale_text": vr.get("rationale_text", ""),
            "reconstruction_error": float(e),
            "generated_values": vr["generated_values"],
        })

    # Filter and sort
    valid_records = [r for r in records if r["rationale_text"] and r["generated_values"]]
    logger.info("Valid records: %d / %d", len(valid_records), len(records))
    sorted_records = sorted(valid_records, key=lambda r: r["reconstruction_error"])

    # Non-overlapping bins
    nonoverlap_bins = compute_nonoverlapping_bins(sorted_records, WINDOW_SIZE, logger)

    # Sliding window bins
    sliding_bins = compute_sliding_bins(sorted_records, WINDOW_SIZE, WINDOW_STEP, logger)

    # Compute Pearson r for both bin types
    nonoverlap_r, nonoverlap_p = _compute_pearson(nonoverlap_bins, "non-overlapping", logger)
    sliding_r, sliding_p = _compute_pearson(sliding_bins, "sliding", logger)

    # Save CSVs
    save_gradient_csv(nonoverlap_bins, "value_diversity_gradient_sbert.csv", ANALYSIS_DIR, logger)
    save_gradient_csv(sliding_bins, "value_diversity_sliding_sbert.csv", ANALYSIS_DIR, logger)

    # Save correlation JSON (both bin types)
    corr_path = os.path.join(ANALYSIS_DIR, "value_diversity_correlation.json")
    with open(corr_path, "w", encoding="utf-8") as f:
        json.dump({
            "nonoverlap_r": nonoverlap_r,
            "nonoverlap_p": nonoverlap_p if not math.isnan(nonoverlap_p) else None,
            "sliding_r": sliding_r,
            "sliding_p": sliding_p if not math.isnan(sliding_p) else None,
        }, f, indent=2)
    logger.info("Saved %s", corr_path)

    # Plot (sliding window)
    plot_diversity_gradient(
        sliding_bins, "value_diversity_gradient_sbert.png",
        f"Sliding Window (w={WINDOW_SIZE}, step={WINDOW_STEP})",
        ANALYSIS_DIR, logger)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Kaleido vs SBERT comparison
# ═══════════════════════════════════════════════════════════════════════════════

def step6_kaleido_vs_sbert(logger: logging.Logger) -> None:
    """Load both summary tables, compute side-by-side comparison CSV."""
    kaleido_summary_path = os.path.join(KALEIDO_ANALYSIS_DIR, "summary_table.csv")
    sbert_summary_path = os.path.join(ANALYSIS_DIR, "summary_table.csv")

    # Load summary tables
    kaleido_data = {}
    with open(kaleido_summary_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kaleido_data[row["source"]] = row

    sbert_data = {}
    with open(sbert_summary_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sbert_data[row["source"]] = row

    # Load cross-projection summaries
    def get_cross_proj(csv_path: str) -> dict:
        lookup = {}
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lookup[(row["basis_source"], row["projected_source"])] = row
        return lookup

    kaleido_xproj = get_cross_proj(
        os.path.join(KALEIDO_ANALYSIS_DIR, "cross_projection_summary.csv"))
    sbert_xproj = get_cross_proj(
        os.path.join(ANALYSIS_DIR, "cross_projection_summary.csv"))

    def get_asymmetry(lookup: dict) -> tuple[float, float, float]:
        h2l = float(lookup[("human", "all_llm")]["variance_captured"])
        l2h = float(lookup[("all_llm", "human")]["variance_captured"])
        return h2l, l2h, h2l - l2h

    kaleido_h2l, kaleido_l2h, kaleido_asym = get_asymmetry(kaleido_xproj)
    sbert_h2l, sbert_l2h, sbert_asym = get_asymmetry(sbert_xproj)

    # Load value diversity Pearson r
    # Kaleido: recompute from non-overlapping bins CSV
    kaleido_gradient_path = os.path.join(KALEIDO_ANALYSIS_DIR, "value_diversity_gradient.csv")
    kaleido_bins = []
    with open(kaleido_gradient_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kaleido_bins.append({
                "mean_error": float(row["mean_error"]),
                "n_unique_values": int(row["n_unique_values"]),
            })

    kaleido_me = np.array([b["mean_error"] for b in kaleido_bins])
    kaleido_nu = np.array([b["n_unique_values"] for b in kaleido_bins])
    if HAS_SCIPY:
        kaleido_gradient_r, _ = pearsonr(kaleido_me, kaleido_nu)
    else:
        kaleido_gradient_r = float(np.corrcoef(kaleido_me, kaleido_nu)[0, 1])

    # SBERT: load from JSON (use non-overlapping r for apples-to-apples)
    corr_path = os.path.join(ANALYSIS_DIR, "value_diversity_correlation.json")
    with open(corr_path, encoding="utf-8") as f:
        sbert_corr = json.load(f)
    sbert_gradient_r = sbert_corr["nonoverlap_r"]

    # Build comparison CSV
    sources_to_compare = [s for s in PLOT_ORDER if s in kaleido_data and s in sbert_data]

    output_path = os.path.join(KALEIDO_ANALYSIS_DIR, "kaleido_vs_sbert_comparison.csv")
    fieldnames = ["source", "kaleido_pr", "kaleido_comp90", "sbert_pr", "sbert_comp90"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for source in sources_to_compare:
            writer.writerow({
                "source": source,
                "kaleido_pr": kaleido_data[source]["participation_ratio"],
                "kaleido_comp90": kaleido_data[source]["components_90pct"],
                "sbert_pr": sbert_data[source]["participation_ratio"],
                "sbert_comp90": sbert_data[source]["components_90pct"],
            })

        # Summary rows
        writer.writerow({
            "source": "_asymmetry",
            "kaleido_pr": f"{kaleido_asym:+.4f}",
            "kaleido_comp90": f"h2l={kaleido_h2l:.4f} l2h={kaleido_l2h:.4f}",
            "sbert_pr": f"{sbert_asym:+.4f}",
            "sbert_comp90": f"h2l={sbert_h2l:.4f} l2h={sbert_l2h:.4f}",
        })

        writer.writerow({
            "source": "_value_gradient_r_nonoverlap",
            "kaleido_pr": f"{kaleido_gradient_r:.3f}",
            "kaleido_comp90": "",
            "sbert_pr": f"{sbert_gradient_r:.3f}",
            "sbert_comp90": "",
        })

    logger.info("Saved %s", output_path)

    # Print summary
    logger.info("=== Kaleido vs SBERT Comparison ===")
    if "human" in kaleido_data and "human" in sbert_data:
        logger.info("Human comp90: Kaleido=%s, SBERT=%s",
                     kaleido_data["human"]["components_90pct"],
                     sbert_data["human"]["components_90pct"])
    if "all_llm" in kaleido_data and "all_llm" in sbert_data:
        logger.info("All-LLM comp90: Kaleido=%s, SBERT=%s",
                     kaleido_data["all_llm"]["components_90pct"],
                     sbert_data["all_llm"]["components_90pct"])
    logger.info("Cross-projection asymmetry: Kaleido=%+.4f, SBERT=%+.4f",
                kaleido_asym, sbert_asym)
    logger.info("Value gradient Pearson r (non-overlapping): Kaleido=%.3f, SBERT=%.3f",
                kaleido_gradient_r, sbert_gradient_r)

    # Note on dimensional constraint
    logger.info("NOTE: SBERT space is 768-dim vs Kaleido 2048-dim. "
                "Cross-projection variance is mechanically higher in lower-dim spaces. "
                "Compare asymmetry direction and relative magnitude, not absolute values.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger = setup_logging()
    start_time = time.time()

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    logger.info("=== Step 1: Extract SBERT embeddings ===")
    step1_extract_embeddings(logger)

    logger.info("=== Step 2: PCA dimensionality analysis ===")
    step2_pca_analysis(logger)

    logger.info("=== Step 3: Cross-space projection ===")
    step3_cross_projection(logger)

    logger.info("=== Step 4: Subsampling robustness ===")
    step4_subsampling_robustness(logger)

    logger.info("=== Step 5: Value diversity gradient ===")
    step5_value_diversity_gradient(logger)

    logger.info("=== Step 6: Kaleido vs SBERT comparison ===")
    step6_kaleido_vs_sbert(logger)

    elapsed = time.time() - start_time
    logger.info("All steps complete in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
