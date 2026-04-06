"""Sweep KMeans k values and report effective cluster counts + silhouette scores per source."""

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


# ── Constants ────────────────────────────────────────────────────────────────

OUTPUT_DIR = "data/analysis"
HUMAN_VALUES_PATH = "data/analysis/human_values_all.json"

LLM_SOURCE_ORDER = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
PLOT_ORDER = ["human"] + sorted(LLM_SOURCE_ORDER) + ["all_llm"]

SENTENCE_MODEL = "all-MiniLM-L6-v2"
PCA_COMPONENTS = 50
SEED = 42

K_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200]
RUSSO_SILHOUETTE = 0.09

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


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("effective_clusters_sweep")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger


# ── Data Loading ─────────────────────────────────────────────────────────────

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
    """Collect unique value strings and per-source occurrence counts."""
    source_counts: dict[str, dict[str, int]] = {}

    h_counts: dict[str, int] = {}
    for r in human_records:
        v = r.get("generated_values", "")
        if v:
            h_counts[v] = h_counts.get(v, 0) + 1
    source_counts["human"] = h_counts
    logger.info("human: %d unique values, %d total", len(h_counts), sum(h_counts.values()))

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

    all_values: set[str] = set()
    for counts in source_counts.values():
        all_values.update(counts.keys())
    universe = sorted(all_values)
    logger.info("Universe: %d unique value strings", len(universe))

    return universe, source_counts


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


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger = setup_logging()
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and embed
    logger.info("=== Loading values ===")
    human_records, per_source_records = load_all_values(logger)
    universe, source_counts = collect_unique_values(human_records, per_source_records, logger)
    value_embeddings = embed_value_strings(universe, logger)

    # PCA reduce
    n_comp = min(PCA_COMPONENTS, value_embeddings.shape[0], value_embeddings.shape[1])
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(value_embeddings)
    logger.info("PCA-reduced to %d components (%.1f%% variance)",
                n_comp, 100 * pca.explained_variance_ratio_.sum())

    # Raw unique counts per source (for CSV)
    raw_unique = {}
    for source in PLOT_ORDER:
        if source in source_counts:
            raw_unique[source] = len(source_counts[source])

    # Sweep
    logger.info("=== K-sweep: %s ===", K_VALUES)
    effective_rows = []
    silhouette_rows = []

    for k in K_VALUES:
        if k >= reduced.shape[0]:
            logger.warning("k=%d >= n_values=%d, skipping", k, reduced.shape[0])
            break

        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(reduced)

        sil = float(silhouette_score(reduced, labels))
        silhouette_rows.append({"k": k, "silhouette_score": sil})

        value_to_cluster = {universe[i]: int(labels[i]) for i in range(len(universe))}

        for source in PLOT_ORDER:
            if source not in source_counts:
                continue
            clusters_used = set()
            for v, count in source_counts[source].items():
                if v in value_to_cluster and count > 0:
                    clusters_used.add(value_to_cluster[v])
            eff = len(clusters_used)
            coverage = eff / k * 100
            effective_rows.append({
                "k": k,
                "source": source,
                "raw_unique_values": raw_unique.get(source, 0),
                "effective_clusters": eff,
                "coverage_pct": coverage,
            })

        logger.info("k=%d: silhouette=%.4f, human_eff=%d, all_llm_eff=%d",
                     k, sil,
                     next((r["effective_clusters"] for r in effective_rows
                           if r["k"] == k and r["source"] == "human"), 0),
                     next((r["effective_clusters"] for r in effective_rows
                           if r["k"] == k and r["source"] == "all_llm"), 0))

    # Save effective_clusters_by_k.csv
    path = os.path.join(OUTPUT_DIR, "effective_clusters_by_k.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "k", "source", "raw_unique_values", "effective_clusters", "coverage_pct"])
        writer.writeheader()
        for r in effective_rows:
            writer.writerow({
                "k": r["k"],
                "source": r["source"],
                "raw_unique_values": r["raw_unique_values"],
                "effective_clusters": r["effective_clusters"],
                "coverage_pct": f"{r['coverage_pct']:.1f}",
            })
    logger.info("Saved %s (%d rows)", path, len(effective_rows))

    # Save silhouette_by_k.csv
    path = os.path.join(OUTPUT_DIR, "silhouette_by_k.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "silhouette_score"])
        writer.writeheader()
        for r in silhouette_rows:
            writer.writerow({
                "k": r["k"],
                "silhouette_score": f"{r['silhouette_score']:.4f}",
            })
    logger.info("Saved %s (%d rows)", path, len(silhouette_rows))

    # Plot effective clusters by k
    logger.info("=== Plotting ===")
    k_vals = sorted(set(r["k"] for r in effective_rows))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Reference line y=k
    ax.plot(k_vals, k_vals, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="y=k")

    for source in PLOT_ORDER:
        source_data = [(r["k"], r["effective_clusters"]) for r in effective_rows
                       if r["source"] == source]
        if not source_data:
            continue
        ks, effs = zip(*sorted(source_data))

        if source in ("human", "all_llm"):
            ax.plot(ks, effs, color=SOURCE_COLORS[source], linewidth=2.5,
                    linestyle="-", label=source, marker="o", markersize=4)
        else:
            ax.plot(ks, effs, color=SOURCE_COLORS.get(source, "#999999"),
                    linewidth=1, linestyle="--", alpha=0.5, label=source,
                    marker="o", markersize=3)

    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Effective Clusters Used")
    ax.set_title("Effective Cluster Count per Source vs k")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "effective_clusters_by_k.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)

    # Plot silhouette by k
    sil_ks = [r["k"] for r in silhouette_rows]
    sil_vals = [r["silhouette_score"] for r in silhouette_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sil_ks, sil_vals, color="#1f77b4", linewidth=2, marker="o", markersize=5)
    ax.axhline(RUSSO_SILHOUETTE, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(sil_ks[-1], RUSSO_SILHOUETTE + 0.002, f"Russo ({RUSSO_SILHOUETTE})",
            ha="right", va="bottom", fontsize=9, color="gray")

    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs k")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "silhouette_by_k.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)

    elapsed = time.time() - start_time
    logger.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
