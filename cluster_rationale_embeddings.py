"""K-means clustering of Kaleido encoder embeddings to evaluate the categorical
approach to measuring moral diversity (direct comparison to Russo et al.)."""

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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ── Constants ────────────────────────────────────────────────────────────────

EMBEDDINGS_DIR = "data/embeddings"
OUTPUT_DIR = "data/analysis"
HIDDEN_DIM = 2048
PCA_COMPONENTS = 100
RANDOM_SEED = 42

HUMAN_VALUES_PATH = "data/analysis/human_values_all.json"

ALL_SOURCES = ["human", "gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
LLM_SOURCE_ORDER = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
PLOT_ORDER = ["human"] + sorted(LLM_SOURCE_ORDER) + ["all_llm"]

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

K_SWEEP = [10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200]
K_GAP = [20, 40, 60, 80, 100, 150]
K_RUSSO = 60

SWEEP_N_INIT = 5
FINAL_N_INIT = 10
N_NULL_DATASETS = 5
SILHOUETTE_SAMPLE_SIZE = 10_000
EFFECTIVE_THRESHOLD = 0.01


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("cluster_rationale_embeddings")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_embeddings(logger: logging.Logger) -> dict[str, np.ndarray]:
    """Load all source embedding .npy files."""
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
    return embeddings


def load_value_records(logger: logging.Logger) -> dict[str, list[dict]]:
    """Load value label JSON files for cluster characterization."""
    records: dict[str, list[dict]] = {}

    if os.path.exists(HUMAN_VALUES_PATH):
        with open(HUMAN_VALUES_PATH, encoding="utf-8") as f:
            records["human"] = json.load(f)
        logger.info("Loaded human value records: %d", len(records["human"]))
    else:
        logger.warning("Missing %s", HUMAN_VALUES_PATH)

    for source in LLM_SOURCE_ORDER:
        path = os.path.join(OUTPUT_DIR, f"llm_values_{source}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                records[source] = json.load(f)
            logger.info("Loaded %s value records: %d", source, len(records[source]))
        else:
            logger.warning("Missing %s", path)

    return records


def build_combined_matrix(
    embeddings: dict[str, np.ndarray],
    logger: logging.Logger,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Stack human + LLM embeddings. Returns (combined, source_indices)."""
    parts = []
    source_indices: dict[str, tuple[int, int]] = {}
    offset = 0

    # Human first
    if "human" in embeddings:
        n = embeddings["human"].shape[0]
        source_indices["human"] = (offset, offset + n)
        parts.append(embeddings["human"])
        offset += n

    # LLMs in order
    llm_start = offset
    for source in LLM_SOURCE_ORDER:
        if source not in embeddings:
            continue
        n = embeddings[source].shape[0]
        source_indices[source] = (offset, offset + n)
        parts.append(embeddings[source])
        offset += n
    source_indices["all_llm"] = (llm_start, offset)

    combined = np.vstack(parts)
    logger.info("Combined matrix: shape %s", combined.shape)
    for source, (start, end) in source_indices.items():
        logger.info("  %s: rows %d-%d (%d)", source, start, end, end - start)

    return combined, source_indices


# ── Helpers ──────────────────────────────────────────────────────────────────

def _compute_wk(data: np.ndarray, labels: np.ndarray) -> float:
    """Within-cluster sum of squared distances to centroids."""
    wk = 0.0
    for k in np.unique(labels):
        cluster_data = data[labels == k]
        centroid = cluster_data.mean(axis=0)
        wk += ((cluster_data - centroid) ** 2).sum()
    return wk


def compute_shannon_entropy(freq: np.ndarray) -> float:
    """H = -sum(p * log2(p)) for p > 0."""
    freq = freq[freq > 0]
    if len(freq) == 0:
        return 0.0
    return -float(np.sum(freq * np.log2(freq)))


def compute_source_metrics_at_k(
    labels: np.ndarray,
    source_indices: dict[str, tuple[int, int]],
    k: int,
) -> list[dict]:
    """Compute per-source diversity metrics at a given k."""
    results = []
    for source in PLOT_ORDER:
        if source not in source_indices:
            continue
        start, end = source_indices[source]
        source_labels = labels[start:end]
        n = len(source_labels)
        if n == 0:
            continue

        # Cluster frequency distribution
        counts = np.bincount(source_labels, minlength=k)
        freq = counts / n

        # Effective clusters (1% threshold)
        effective = int(np.sum(freq >= EFFECTIVE_THRESHOLD))

        # Shannon entropy
        entropy = compute_shannon_entropy(freq)

        # Top-5 concentration
        sorted_freq = np.sort(freq)[::-1]
        top5 = float(sorted_freq[:5].sum())

        results.append({
            "source": source,
            "n_rationales": n,
            "effective_clusters_1pct": effective,
            "shannon_entropy": entropy,
            "top5_concentration": top5,
        })

    return results


# ── Gap Statistic ────────────────────────────────────────────────────────────

def compute_gap_statistic(
    reduced: np.ndarray,
    eigenvalues: np.ndarray,
    k_values: list[int],
    sweep_labels: dict[int, np.ndarray],
    n_null: int,
    seed: int,
    logger: logging.Logger,
) -> list[dict]:
    """Gap statistic with PCA-preserving null."""
    n_samples, n_comp = reduced.shape
    results = []

    for k in k_values:
        t0 = time.time()

        # Real data — reuse labels from sweep
        if k in sweep_labels:
            labels = sweep_labels[k]
        else:
            km = KMeans(n_clusters=k, random_state=seed, n_init=SWEEP_N_INIT)
            labels = km.fit_predict(reduced)

        w_k = _compute_wk(reduced, labels)
        log_wk = np.log(w_k) if w_k > 0 else 0.0

        # Null datasets
        log_wk_nulls = []
        for b in range(n_null):
            rng = np.random.RandomState(seed + b + 1)
            null_data = rng.randn(n_samples, n_comp) * np.sqrt(eigenvalues)
            km_null = KMeans(n_clusters=k, random_state=seed, n_init=SWEEP_N_INIT)
            null_labels = km_null.fit_predict(null_data)
            wk_null = _compute_wk(null_data, null_labels)
            log_wk_nulls.append(np.log(wk_null) if wk_null > 0 else 0.0)

        gap = float(np.mean(log_wk_nulls) - log_wk)
        gap_se = float(np.std(log_wk_nulls) * np.sqrt(1 + 1 / n_null))

        results.append({"k": k, "gap": gap, "gap_se": gap_se})

        elapsed = time.time() - t0
        logger.info("Gap k=%d: gap=%.4f\u00b1%.4f (%.1fs)", k, gap, gap_se, elapsed)

    return results


def select_optimal_k(
    gap_results: list[dict], logger: logging.Logger,
) -> int:
    """Select first k where gap(k) >= gap(k+1) - se(k+1). Fallback to max-gap k."""
    for i in range(len(gap_results) - 1):
        curr = gap_results[i]
        nxt = gap_results[i + 1]
        if curr["gap"] >= nxt["gap"] - nxt["gap_se"]:
            logger.info("Gap criterion met at k=%d (gap=%.4f)", curr["k"], curr["gap"])
            return curr["k"]

    best = max(gap_results, key=lambda r: r["gap"])
    logger.warning("Gap criterion not met; falling back to max-gap k=%d (gap=%.4f)",
                   best["k"], best["gap"])
    return best["k"]


# ── Cluster Characterization ─────────────────────────────────────────────────

def characterize_clusters(
    labels: np.ndarray,
    source_indices: dict[str, tuple[int, int]],
    value_records: dict[str, list[dict]],
    k: int,
    logger: logging.Logger,
) -> list[dict]:
    """For each cluster, determine top-3 value labels and per-source frequencies."""
    n_total = len(labels)
    n_human = source_indices["human"][1] - source_indices["human"][0] if "human" in source_indices else 0
    n_allllm = source_indices["all_llm"][1] - source_indices["all_llm"][0] if "all_llm" in source_indices else 0

    # Build per-source label arrays for efficient lookup
    source_label_slices = {}
    for source in ["human"] + LLM_SOURCE_ORDER:
        if source not in source_indices:
            continue
        start, end = source_indices[source]
        source_label_slices[source] = labels[start:end]

    # Per-source totals
    source_totals = {}
    for source in PLOT_ORDER:
        if source in source_indices:
            start, end = source_indices[source]
            source_totals[source] = end - start

    results = []
    for cid in range(k):
        # Value label frequencies from all rationales in this cluster
        value_freq: dict[str, int] = {}
        cluster_source_counts: dict[str, int] = {}

        for source in ["human"] + LLM_SOURCE_ORDER:
            if source not in source_label_slices or source not in value_records:
                continue
            s_labels = source_label_slices[source]
            mask = s_labels == cid
            count = int(mask.sum())
            cluster_source_counts[source] = count

            # Look up value labels for rationales in this cluster
            indices_in_cluster = np.where(mask)[0]
            for idx in indices_in_cluster:
                rec = value_records[source][idx]
                v = rec.get("generated_values", "")
                if v:
                    value_freq[v] = value_freq.get(v, 0) + 1

        # all_llm count
        allllm_count = sum(cluster_source_counts.get(s, 0) for s in LLM_SOURCE_ORDER)
        cluster_source_counts["all_llm"] = allllm_count

        # Top 3 value labels
        sorted_values = sorted(value_freq.items(), key=lambda x: (-x[1], x[0]))
        top3 = sorted_values[:3]
        cluster_label = " / ".join(v for v, _ in top3) if top3 else "unknown"

        # Per-source frequencies
        human_count = cluster_source_counts.get("human", 0)
        human_freq = human_count / n_human if n_human > 0 else 0.0
        allllm_freq = allllm_count / n_allllm if n_allllm > 0 else 0.0

        if allllm_freq > 0:
            freq_ratio = human_freq / allllm_freq
        elif human_freq > 0:
            freq_ratio = float("inf")
        else:
            freq_ratio = 0.0

        row = {
            "cluster_id": cid,
            "cluster_label": cluster_label,
            "size": int((labels == cid).sum()),
            "human_freq": human_freq,
            "allllm_freq": allllm_freq,
            "freq_ratio": freq_ratio,
        }
        for source in LLM_SOURCE_ORDER:
            total = source_totals.get(source, 1)
            count = cluster_source_counts.get(source, 0)
            row[f"{source}_freq"] = count / total if total > 0 else 0.0

        results.append(row)

    logger.info("Characterized %d clusters at k=%d", k, k)
    return results


# ── CSV Outputs ──────────────────────────────────────────────────────────────

def save_sweep_csv(
    all_metrics: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    path = os.path.join(output_dir, "rationale_cluster_sweep.csv")
    fieldnames = ["k", "source", "n_rationales", "effective_clusters_1pct",
                  "shannon_entropy", "top5_concentration"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_metrics:
            writer.writerow({
                "k": r["k"],
                "source": r["source"],
                "n_rationales": r["n_rationales"],
                "effective_clusters_1pct": r["effective_clusters_1pct"],
                "shannon_entropy": f"{r['shannon_entropy']:.4f}",
                "top5_concentration": f"{r['top5_concentration']:.4f}",
            })
    logger.info("Saved %s (%d rows)", path, len(all_metrics))


def save_silhouette_csv(
    sil_results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    path = os.path.join(output_dir, "rationale_cluster_silhouette.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "silhouette_score"])
        writer.writeheader()
        for r in sil_results:
            writer.writerow({
                "k": r["k"],
                "silhouette_score": f"{r['silhouette_score']:.4f}",
            })
    logger.info("Saved %s (%d rows)", path, len(sil_results))


def save_gap_csv(
    gap_results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    path = os.path.join(output_dir, "rationale_cluster_gap.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "gap", "gap_se"])
        writer.writeheader()
        for r in gap_results:
            writer.writerow({
                "k": r["k"],
                "gap": f"{r['gap']:.4f}",
                "gap_se": f"{r['gap_se']:.4f}",
            })
    logger.info("Saved %s (%d rows)", path, len(gap_results))


def save_frequencies_csv(
    char_data: list[dict], k: int, suffix: str,
    output_dir: str, logger: logging.Logger,
) -> None:
    filename = f"rationale_cluster_frequencies{suffix}.csv"
    path = os.path.join(output_dir, filename)
    fieldnames = ["cluster_id", "cluster_label", "size",
                  "human_freq", "allllm_freq", "freq_ratio"]
    for source in LLM_SOURCE_ORDER:
        fieldnames.append(f"{source}_freq")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(char_data, key=lambda x: -x["size"]):
            row = {
                "cluster_id": r["cluster_id"],
                "cluster_label": r["cluster_label"],
                "size": r["size"],
                "human_freq": f"{r['human_freq']:.6f}",
                "allllm_freq": f"{r['allllm_freq']:.6f}",
                "freq_ratio": f"{r['freq_ratio']:.4f}" if not math.isinf(r["freq_ratio"]) else "inf",
            }
            for source in LLM_SOURCE_ORDER:
                row[f"{source}_freq"] = f"{r[f'{source}_freq']:.6f}"
            writer.writerow(row)
    logger.info("Saved %s (%d clusters at k=%d)", path, len(char_data), k)


# ── Plots ────────────────────────────────────────────────────────────────────

def _plot_source_lines(
    ax, all_metrics: list[dict], k_values: list[int], metric_key: str,
) -> None:
    """Helper: draw one line per source on ax for a given metric."""
    for source in PLOT_ORDER:
        data = [(r["k"], r[metric_key]) for r in all_metrics if r["source"] == source]
        if not data:
            continue
        ks, vals = zip(*sorted(data))
        if source in ("human", "all_llm"):
            ax.plot(ks, vals, color=SOURCE_COLORS[source], linewidth=2.5,
                    linestyle="-", label=source, marker="o", markersize=4)
        else:
            ax.plot(ks, vals, color=SOURCE_COLORS.get(source, "#999999"),
                    linewidth=1, linestyle="--", alpha=0.5, label=source,
                    marker="o", markersize=3)


def plot_sweep(all_metrics: list[dict], output_dir: str, logger: logging.Logger) -> None:
    k_values = sorted(set(r["k"] for r in all_metrics))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, k_values, color="gray", linestyle="--", linewidth=0.8,
            alpha=0.5, label="y=k")
    _plot_source_lines(ax, all_metrics, k_values, "effective_clusters_1pct")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Effective Clusters (>1% threshold)")
    ax.set_title("Effective Cluster Count per Source vs k (Kaleido Embeddings)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, "rationale_cluster_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_silhouette(
    sil_results: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    ks = [r["k"] for r in sil_results]
    sils = [r["silhouette_score"] for r in sil_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, sils, color="#1f77b4", linewidth=2, marker="o", markersize=5)
    ax.axhline(0.09, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(ks[-1], 0.09 + 0.002, "Russo (0.09)",
            ha="right", va="bottom", fontsize=9, color="gray")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs k (Kaleido Embeddings)\n"
                 f"(computed on {SILHOUETTE_SAMPLE_SIZE:,} subsample)")
    fig.tight_layout()
    path = os.path.join(output_dir, "rationale_cluster_silhouette.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_entropy(all_metrics: list[dict], output_dir: str, logger: logging.Logger) -> None:
    k_values = sorted(set(r["k"] for r in all_metrics))

    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_source_lines(ax, all_metrics, k_values, "shannon_entropy")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title("Cluster Distribution Entropy per Source vs k (Kaleido Embeddings)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, "rationale_cluster_entropy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_concentration(all_metrics: list[dict], output_dir: str, logger: logging.Logger) -> None:
    k_values = sorted(set(r["k"] for r in all_metrics))

    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_source_lines(ax, all_metrics, k_values, "top5_concentration")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Top-5 Cluster Concentration")
    ax.set_title("Top-5 Cluster Concentration per Source vs k (Kaleido Embeddings)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, "rationale_cluster_concentration.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger = setup_logging()
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load embeddings
    logger.info("=== Step 1: Load embeddings ===")
    embeddings = load_embeddings(logger)
    if "human" not in embeddings:
        raise SystemExit("Human embeddings not found.")

    # Step 2: Build combined matrix
    logger.info("=== Step 2: Build combined matrix ===")
    combined, source_indices = build_combined_matrix(embeddings, logger)
    del embeddings  # free individual matrices

    # Step 3: PCA reduce
    logger.info("=== Step 3: PCA reduction (2048 -> %d) ===", PCA_COMPONENTS)
    n_comp = min(PCA_COMPONENTS, combined.shape[0], combined.shape[1])
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(combined)
    eigenvalues = pca.explained_variance_
    logger.info("PCA: %d components, %.1f%% variance explained",
                n_comp, 100 * pca.explained_variance_ratio_.sum())
    del combined  # free ~1.8 GB

    # Step 4: K-means sweep
    logger.info("=== Step 4: K-means sweep (%d k values) ===", len(K_SWEEP))
    all_metrics = []
    sil_results = []
    sweep_labels: dict[int, np.ndarray] = {}

    for k in K_SWEEP:
        t0 = time.time()
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=SWEEP_N_INIT)
        labels = km.fit_predict(reduced)
        sweep_labels[k] = labels

        sil = float(silhouette_score(
            reduced, labels,
            sample_size=SILHOUETTE_SAMPLE_SIZE,
            random_state=RANDOM_SEED,
        ))
        sil_results.append({"k": k, "silhouette_score": sil})

        metrics = compute_source_metrics_at_k(labels, source_indices, k)
        for m in metrics:
            m["k"] = k
        all_metrics.extend(metrics)

        elapsed = time.time() - t0
        human_eff = next((m["effective_clusters_1pct"] for m in metrics
                          if m["source"] == "human"), 0)
        allllm_eff = next((m["effective_clusters_1pct"] for m in metrics
                           if m["source"] == "all_llm"), 0)
        logger.info("k=%d: sil=%.4f, human_eff=%d, allllm_eff=%d (%.1fs)",
                     k, sil, human_eff, allllm_eff, elapsed)

    # Step 5: Gap statistic
    logger.info("=== Step 5: Gap statistic (%d k values, %d nulls) ===",
                len(K_GAP), N_NULL_DATASETS)
    gap_results = compute_gap_statistic(
        reduced, eigenvalues, K_GAP, sweep_labels, N_NULL_DATASETS, RANDOM_SEED, logger)
    selected_k = select_optimal_k(gap_results, logger)
    logger.info("Selected k=%d", selected_k)

    # Step 6: Final clustering at selected_k and k=60
    logger.info("=== Step 6: Final clustering ===")
    if selected_k in sweep_labels:
        # Refit with higher n_init for quality
        logger.info("Refitting k=%d with n_init=%d", selected_k, FINAL_N_INIT)
    km_selected = KMeans(n_clusters=selected_k, random_state=RANDOM_SEED, n_init=FINAL_N_INIT)
    labels_selected = km_selected.fit_predict(reduced)

    if selected_k != K_RUSSO:
        logger.info("Fitting k=%d (Russo) with n_init=%d", K_RUSSO, FINAL_N_INIT)
        km_60 = KMeans(n_clusters=K_RUSSO, random_state=RANDOM_SEED, n_init=FINAL_N_INIT)
        labels_60 = km_60.fit_predict(reduced)
    else:
        labels_60 = labels_selected

    # Per-source metrics at final k values
    metrics_selected = compute_source_metrics_at_k(labels_selected, source_indices, selected_k)
    metrics_60 = compute_source_metrics_at_k(labels_60, source_indices, K_RUSSO)

    logger.info("Final metrics at k=%d:", selected_k)
    for m in metrics_selected:
        logger.info("  %s: eff=%d, entropy=%.3f, top5=%.3f",
                     m["source"], m["effective_clusters_1pct"],
                     m["shannon_entropy"], m["top5_concentration"])

    logger.info("Final metrics at k=%d (Russo):", K_RUSSO)
    for m in metrics_60:
        logger.info("  %s: eff=%d, entropy=%.3f, top5=%.3f",
                     m["source"], m["effective_clusters_1pct"],
                     m["shannon_entropy"], m["top5_concentration"])

    # Step 7: Cluster characterization
    logger.info("=== Step 7: Cluster characterization ===")
    value_records = load_value_records(logger)

    char_selected = characterize_clusters(
        labels_selected, source_indices, value_records, selected_k, logger)
    char_60 = characterize_clusters(
        labels_60, source_indices, value_records, K_RUSSO, logger)

    # Step 8: Save outputs
    logger.info("=== Step 8: Save outputs ===")
    save_sweep_csv(all_metrics, OUTPUT_DIR, logger)
    save_silhouette_csv(sil_results, OUTPUT_DIR, logger)
    save_gap_csv(gap_results, OUTPUT_DIR, logger)

    save_frequencies_csv(char_selected, selected_k, "", OUTPUT_DIR, logger)
    if selected_k != K_RUSSO:
        save_frequencies_csv(char_60, K_RUSSO, "_k60", OUTPUT_DIR, logger)

    # Step 9: Plots
    logger.info("=== Step 9: Plots ===")
    plot_sweep(all_metrics, OUTPUT_DIR, logger)
    plot_silhouette(sil_results, OUTPUT_DIR, logger)
    plot_entropy(all_metrics, OUTPUT_DIR, logger)
    plot_concentration(all_metrics, OUTPUT_DIR, logger)

    elapsed = time.time() - start_time
    logger.info("All steps complete in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
