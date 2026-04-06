"""Cluster Kaleido-extracted value labels to deduplicate semantically similar expressions."""

import csv
import json
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── Constants ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = "data/analysis"
HUMAN_VALUES_PATH = "data/analysis/human_values_all.json"

LLM_SOURCE_ORDER = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
ALL_SOURCES_ORDER = ["human"] + LLM_SOURCE_ORDER
PLOT_ORDER = ["human"] + sorted(LLM_SOURCE_ORDER) + ["all_llm"]

SENTENCE_MODEL = "all-MiniLM-L6-v2"
PCA_COMPONENTS = 50
K_RANGE = list(range(10, 201, 5))  # 10, 15, 20, ..., 200
K_RUSSO = 60
N_NULL_DATASETS = 10
GAP_SEED = 42

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
    logger = logging.getLogger("cluster_value_labels")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_all_values(
    logger: logging.Logger,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load human and per-source LLM value records."""
    if not os.path.exists(HUMAN_VALUES_PATH):
        raise SystemExit(f"{HUMAN_VALUES_PATH} not found. Run value_diversity_gradient.py first.")

    with open(HUMAN_VALUES_PATH, encoding="utf-8") as f:
        human_records = json.load(f)
    logger.info("Loaded human values: %d records", len(human_records))

    per_source_records: dict[str, list[dict]] = {}
    for source in LLM_SOURCE_ORDER:
        path = os.path.join(OUTPUT_DIR, f"llm_values_{source}.json")
        if not os.path.exists(path):
            logger.warning("Missing %s, skipping", path)
            continue
        with open(path, encoding="utf-8") as f:
            per_source_records[source] = json.load(f)
        logger.info("Loaded %s values: %d records", source, len(per_source_records[source]))

    return human_records, per_source_records


def collect_unique_values(
    human_records: list[dict],
    per_source_records: dict[str, list[dict]],
    logger: logging.Logger,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Collect unique value strings and per-source occurrence counts.

    Returns (sorted universe list, {source: {value: count}}).
    """
    source_counts: dict[str, dict[str, int]] = {}

    # Human
    h_counts: dict[str, int] = {}
    for r in human_records:
        v = r.get("generated_values", "")
        if v:
            h_counts[v] = h_counts.get(v, 0) + 1
    source_counts["human"] = h_counts
    logger.info("human: %d unique values, %d total", len(h_counts), sum(h_counts.values()))

    # LLMs
    allllm_counts: dict[str, int] = {}
    for source, records in per_source_records.items():
        s_counts: dict[str, int] = {}
        for r in records:
            v = r.get("generated_values", "")
            if v:
                s_counts[v] = s_counts.get(v, 0) + 1
                allllm_counts[v] = allllm_counts.get(v, 0) + 1
        source_counts[source] = s_counts
        logger.info("%s: %d unique values, %d total", source, len(s_counts), sum(s_counts.values()))
    source_counts["all_llm"] = allllm_counts
    logger.info("all_llm: %d unique values, %d total",
                len(allllm_counts), sum(allllm_counts.values()))

    # Universe
    all_values: set[str] = set()
    for counts in source_counts.values():
        all_values.update(counts.keys())
    universe = sorted(all_values)
    logger.info("Universe: %d unique value strings", len(universe))

    return universe, source_counts


# ── Embedding ──────────────────────────────────────────────────────────────────
def embed_value_strings(
    universe: list[str], logger: logging.Logger,
) -> np.ndarray:
    """Embed value strings using Sentence-BERT."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence model: %s", SENTENCE_MODEL)
    model = SentenceTransformer(SENTENCE_MODEL)
    logger.info("Encoding %d value strings...", len(universe))
    embeddings = model.encode(universe, show_progress_bar=False)
    logger.info("Embeddings shape: %s", embeddings.shape)
    return np.array(embeddings)


# ── Gap Statistic ──────────────────────────────────────────────────────────────
def _compute_wk(data: np.ndarray, labels: np.ndarray) -> float:
    """Within-cluster sum of squared distances to centroids."""
    wk = 0.0
    for k in np.unique(labels):
        cluster_data = data[labels == k]
        centroid = cluster_data.mean(axis=0)
        wk += ((cluster_data - centroid) ** 2).sum()
    return wk


def compute_gap_statistic(
    embeddings: np.ndarray,
    k_range: list[int],
    n_null: int,
    seed: int,
    logger: logging.Logger,
) -> list[dict]:
    """Gap statistic with PCA-preserving null for k selection.

    Returns list of {k, w_k, gap, gap_se, silhouette} per k.
    """
    # PCA-reduce
    n_comp = min(PCA_COMPONENTS, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(embeddings)
    eigenvalues = pca.explained_variance_
    n_samples = reduced.shape[0]
    logger.info("PCA-reduced to %d components (%.1f%% variance)",
                n_comp, 100 * pca.explained_variance_ratio_.sum())

    results = []
    for k in k_range:
        if k >= n_samples:
            logger.warning("k=%d >= n_samples=%d, skipping", k, n_samples)
            break

        # Real data
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(reduced)
        w_k = _compute_wk(reduced, labels)
        log_wk = np.log(w_k) if w_k > 0 else 0.0

        # Silhouette
        sil = float(silhouette_score(reduced, labels)) if k > 1 else 0.0

        # Null datasets
        log_wk_nulls = []
        for b in range(n_null):
            rng = np.random.RandomState(seed + b + 1)
            null_data = rng.randn(n_samples, n_comp) * np.sqrt(eigenvalues)
            km_null = KMeans(n_clusters=k, random_state=seed, n_init=10)
            null_labels = km_null.fit_predict(null_data)
            wk_null = _compute_wk(null_data, null_labels)
            log_wk_nulls.append(np.log(wk_null) if wk_null > 0 else 0.0)

        gap = float(np.mean(log_wk_nulls) - log_wk)
        gap_se = float(np.std(log_wk_nulls) * np.sqrt(1 + 1 / n_null))

        results.append({
            "k": k,
            "w_k": w_k,
            "gap": gap,
            "gap_se": gap_se,
            "silhouette": sil,
        })

        if k % 50 == 0 or k == k_range[0]:
            logger.info("k=%d: gap=%.4f\u00b1%.4f, silhouette=%.4f", k, gap, gap_se, sil)

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

    # Fallback: max gap
    best = max(gap_results, key=lambda r: r["gap"])
    logger.warning("Gap criterion not met; falling back to max-gap k=%d (gap=%.4f)",
                   best["k"], best["gap"])
    return best["k"]


# ── Clustering ─────────────────────────────────────────────────────────────────
def cluster_at_k(
    reduced: np.ndarray,
    universe: list[str],
    total_counts: dict[str, int],
    k: int,
    seed: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict[int, dict]]:
    """Cluster at given k. Returns (labels, cluster_info).

    cluster_info: {cluster_id: {"label": str, "members": list[str]}}.
    """
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(reduced)

    cluster_info: dict[int, dict] = {}
    for cid in range(k):
        member_indices = np.where(labels == cid)[0]
        members = [universe[i] for i in member_indices]

        # Label by most frequent member (total occurrences across all sources)
        member_counts = [(m, total_counts.get(m, 0)) for m in members]
        member_counts.sort(key=lambda x: (-x[1], x[0]))  # highest count, then alphabetical
        label = member_counts[0][0] if member_counts else f"cluster_{cid}"

        cluster_info[cid] = {"label": label, "members": members}

    logger.info("Clustered %d values into %d clusters", len(universe), k)
    return labels, cluster_info


# ── Mapping ────────────────────────────────────────────────────────────────────
def map_values_to_clusters(
    source_counts: dict[str, dict[str, int]],
    universe: list[str],
    labels: np.ndarray,
    logger: logging.Logger,
) -> dict[str, int]:
    """Map each source's values to clusters, count effective clusters per source.

    Returns {source: effective_cluster_count}.
    """
    value_to_cluster = {universe[i]: int(labels[i]) for i in range(len(universe))}

    effective: dict[str, int] = {}
    for source in PLOT_ORDER:
        if source not in source_counts:
            continue
        clusters_used = set()
        for v, count in source_counts[source].items():
            if v in value_to_cluster and count > 0:
                clusters_used.add(value_to_cluster[v])
        effective[source] = len(clusters_used)
        logger.info("%s: %d raw unique values -> %d effective clusters",
                    source, len(source_counts[source]), len(clusters_used))

    return effective


# ── UMAP ───────────────────────────────────────────────────────────────────────
def compute_umap(
    embeddings: np.ndarray, logger: logging.Logger,
) -> np.ndarray:
    """UMAP projection to 2D."""
    import umap

    logger.info("Computing UMAP (n=%d)...", embeddings.shape[0])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords_2d = reducer.fit_transform(embeddings)
    logger.info("UMAP done: shape %s", coords_2d.shape)
    return coords_2d


# ── Outputs ────────────────────────────────────────────────────────────────────
def save_cluster_assignments(
    universe: list[str],
    labels: np.ndarray,
    cluster_info: dict[int, dict],
    source_counts: dict[str, dict[str, int]],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Write value_label_clusters.csv."""
    path = os.path.join(output_dir, "value_label_clusters.csv")
    fieldnames = ["value", "cluster_id", "cluster_label",
                  "n_occurrences_human", "n_occurrences_allllm"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, value in enumerate(universe):
            cid = int(labels[i])
            writer.writerow({
                "value": value,
                "cluster_id": cid,
                "cluster_label": cluster_info[cid]["label"],
                "n_occurrences_human": source_counts.get("human", {}).get(value, 0),
                "n_occurrences_allllm": source_counts.get("all_llm", {}).get(value, 0),
            })
    logger.info("Saved %s (%d values)", path, len(universe))


def save_cluster_frequency_comparison(
    cluster_info: dict[int, dict],
    source_counts: dict[str, dict[str, int]],
    universe: list[str],
    labels: np.ndarray,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Write cluster_frequency_comparison.csv."""
    value_to_cluster = {universe[i]: int(labels[i]) for i in range(len(universe))}

    # Compute per-source totals
    source_totals = {s: sum(c.values()) for s, c in source_counts.items()}

    # Aggregate occurrences per cluster per source
    cluster_source_counts: dict[int, dict[str, int]] = {}
    for cid in cluster_info:
        cluster_source_counts[cid] = {}
        for source in PLOT_ORDER:
            if source not in source_counts:
                continue
            count = 0
            for member in cluster_info[cid]["members"]:
                count += source_counts[source].get(member, 0)
            cluster_source_counts[cid][source] = count

    path = os.path.join(output_dir, "cluster_frequency_comparison.csv")
    fieldnames = ["cluster_id", "cluster_label", "n_members", "human_freq", "all_llm_freq"]
    for source in LLM_SOURCE_ORDER:
        fieldnames.append(f"{source}_freq")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cid in sorted(cluster_info.keys()):
            info = cluster_info[cid]
            row = {
                "cluster_id": cid,
                "cluster_label": info["label"],
                "n_members": len(info["members"]),
            }
            for source in ["human", "all_llm"] + LLM_SOURCE_ORDER:
                total = source_totals.get(source, 1)
                count = cluster_source_counts[cid].get(source, 0)
                freq = count / total if total > 0 else 0.0
                col = f"{source}_freq" if source in LLM_SOURCE_ORDER else f"{source}_freq"
                row[col] = f"{freq:.6f}"
            writer.writerow(row)
    logger.info("Saved %s (%d clusters)", path, len(cluster_info))


def save_gap_silhouette(
    gap_results: list[dict],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Write cluster_silhouette.csv."""
    path = os.path.join(output_dir, "cluster_silhouette.csv")
    fieldnames = ["k", "silhouette_score", "gap_statistic", "gap_se"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in gap_results:
            writer.writerow({
                "k": r["k"],
                "silhouette_score": f"{r['silhouette']:.4f}",
                "gap_statistic": f"{r['gap']:.4f}",
                "gap_se": f"{r['gap_se']:.4f}",
            })
    logger.info("Saved %s (%d entries)", path, len(gap_results))


def plot_cluster_scatter(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    universe: list[str],
    cluster_info: dict[int, dict],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """UMAP scatter plot colored by cluster."""
    n_clusters = len(cluster_info)
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(min(n_clusters, 20))

    fig, ax = plt.subplots(figsize=(10, 8))

    for cid in range(n_clusters):
        mask = labels == cid
        color = cmap(cid % 20)
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   color=color, s=20, alpha=0.7, edgecolors="none")

    # Annotate cluster labels for larger clusters (>= 5 members)
    for cid, info in cluster_info.items():
        if len(info["members"]) >= 5:
            mask = labels == cid
            cx = coords_2d[mask, 0].mean()
            cy = coords_2d[mask, 1].mean()
            ax.annotate(info["label"], (cx, cy), fontsize=6, alpha=0.8,
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Value Label Clusters (k={n_clusters}, UMAP projection)")
    fig.tight_layout()
    path = os.path.join(output_dir, "value_label_cluster_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_effective_clusters(
    effective: dict[str, int],
    k: int,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Bar chart of effective cluster count per source."""
    sources = [s for s in PLOT_ORDER if s in effective]
    counts = [effective[s] for s in sources]
    colors = [SOURCE_COLORS.get(s, "#999999") for s in sources]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(sources, counts, color=colors,
                  edgecolor="black", linewidth=0.5)

    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(c), ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Source")
    ax.set_ylabel("Effective Clusters Used")
    ax.set_title(f"Effective Cluster Count per Source (k={k})")
    fig.tight_layout()
    path = os.path.join(output_dir, "effective_clusters_by_source.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    logger = setup_logging()
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Collect unique values
    logger.info("=== Step 1: Collect unique values ===")
    human_records, per_source_records = load_all_values(logger)
    universe, source_counts = collect_unique_values(human_records, per_source_records, logger)

    # Step 2: Embed value strings
    logger.info("=== Step 2: Embed value strings ===")
    value_embeddings = embed_value_strings(universe, logger)

    # Step 3: PCA-reduce for clustering
    logger.info("=== Step 3: PCA reduction ===")
    n_comp = min(PCA_COMPONENTS, value_embeddings.shape[0], value_embeddings.shape[1])
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(value_embeddings)
    logger.info("Reduced to %d components (%.1f%% variance)",
                n_comp, 100 * pca.explained_variance_ratio_.sum())

    # Step 4: Gap statistic
    logger.info("=== Step 4: Gap statistic ===")
    gap_results = compute_gap_statistic(value_embeddings, K_RANGE, N_NULL_DATASETS, GAP_SEED, logger)
    optimal_k = select_optimal_k(gap_results, logger)

    # Total counts for cluster labeling
    total_counts: dict[str, int] = {}
    for counts in source_counts.values():
        for v, c in counts.items():
            total_counts[v] = total_counts.get(v, 0) + c

    # Step 5: Cluster at optimal k
    logger.info("=== Step 5: Cluster at optimal k=%d ===", optimal_k)
    labels_opt, cluster_info_opt = cluster_at_k(
        reduced, universe, total_counts, optimal_k, GAP_SEED, logger)

    # Step 6: Cluster at k=60 (Russo comparability)
    logger.info("=== Step 6: Cluster at k=%d (Russo) ===", K_RUSSO)
    labels_60, cluster_info_60 = cluster_at_k(
        reduced, universe, total_counts, K_RUSSO, GAP_SEED, logger)

    # Report silhouette at both k values
    sil_opt = float(silhouette_score(reduced, labels_opt))
    sil_60 = float(silhouette_score(reduced, labels_60))
    logger.info("Silhouette at k=%d (optimal): %.4f", optimal_k, sil_opt)
    logger.info("Silhouette at k=%d (Russo):   %.4f", K_RUSSO, sil_60)

    # Step 7: Map values to clusters
    logger.info("=== Step 7: Map values to clusters ===")
    effective_opt = map_values_to_clusters(source_counts, universe, labels_opt, logger)
    effective_60 = map_values_to_clusters(source_counts, universe, labels_60, logger)

    # Step 8: UMAP
    logger.info("=== Step 8: UMAP visualization ===")
    coords_2d = compute_umap(value_embeddings, logger)
    plot_cluster_scatter(coords_2d, labels_opt, universe, cluster_info_opt, OUTPUT_DIR, logger)

    # Step 9: Save outputs
    logger.info("=== Step 9: Save outputs ===")
    save_cluster_assignments(universe, labels_opt, cluster_info_opt, source_counts, OUTPUT_DIR, logger)
    save_cluster_frequency_comparison(
        cluster_info_opt, source_counts, universe, labels_opt, OUTPUT_DIR, logger)
    save_gap_silhouette(gap_results, OUTPUT_DIR, logger)
    plot_effective_clusters(effective_opt, optimal_k, OUTPUT_DIR, logger)

    elapsed = time.time() - start_time
    logger.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
