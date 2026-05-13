"""2D UMAP scatter of Kaleido-XL embeddings: shows LLMs cluster together, humans spread out.

Reads data/embeddings/{source}.npy, subsamples, runs UMAP, saves two-panel figure:
  Panel A — all sources individually coloured
  Panel B — human vs all LLM (clean version for slides)

Output: data/analysis/embedding_scatter.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

EMBEDDINGS_DIR = "data/embeddings"
OUTPUT_PATH = "data/analysis/embedding_scatter.png"
HIDDEN_DIM = 2048
SAMPLES_PER_SOURCE = 500
RANDOM_SEED = 42

SOURCES = ["human", "gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]

SOURCE_COLORS = {
    "human":   "#2ca02c",
    "gpt3.5":  "#1f77b4",
    "gpt4":    "#aec7e8",
    "claude":  "#ff7f0e",
    "bison":   "#ffbb78",
    "gemma":   "#d62728",
    "mistral": "#9467bd",
    "llama":   "#8c564b",
}

SOURCE_LABELS = {
    "human":   "Human",
    "gpt3.5":  "GPT-3.5",
    "gpt4":    "GPT-4",
    "claude":  "Claude",
    "bison":   "Bison",
    "gemma":   "Gemma",
    "mistral": "Mistral",
    "llama":   "LLaMA",
}


def load_and_subsample(rng):
    embeddings, labels = [], []
    for source in SOURCES:
        path = os.path.join(EMBEDDINGS_DIR, f"{source}.npy")
        if not os.path.exists(path):
            print(f"  missing: {path}, skipping")
            continue
        mat = np.load(path)
        if mat.ndim != 2 or mat.shape[1] != HIDDEN_DIM:
            print(f"  unexpected shape for {source}: {mat.shape}, skipping")
            continue
        n = min(SAMPLES_PER_SOURCE, mat.shape[0])
        idx = rng.choice(mat.shape[0], n, replace=False)
        embeddings.append(mat[idx])
        labels.extend([source] * n)
        print(f"  {source}: {mat.shape[0]} total, using {n}")
    if not embeddings:
        raise SystemExit(f"No embedding files found in {EMBEDDINGS_DIR}/")
    return np.vstack(embeddings), labels


def run_umap(matrix):
    # PCA pre-reduction to 50 dims speeds up UMAP on 2048-dim vectors
    print("Pre-reducing to 50 dims with PCA...")
    pca = PCA(n_components=50, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(matrix)
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric="cosine", random_state=RANDOM_SEED)
    return reducer.fit_transform(reduced)


def plot(coords, labels, sources_present):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: all sources individually
    ax = axes[0]
    for source in sources_present:
        mask = np.array(labels) == source
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=SOURCE_COLORS[source], label=SOURCE_LABELS[source],
                   s=8, alpha=0.5, linewidths=0)
    ax.set_title("A. Embeddings by source", fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=2.5, fontsize=8, framealpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel B: human (green) vs all LLMs (pink), cleaner for slides
    ax = axes[1]
    llm_mask = np.array([l != "human" for l in labels])
    human_mask = ~llm_mask
    ax.scatter(coords[llm_mask, 0], coords[llm_mask, 1],
               c="#e377c2", s=6, alpha=0.3, linewidths=0, label="LLMs (all)")
    ax.scatter(coords[human_mask, 0], coords[human_mask, 1],
               c="#2ca02c", s=10, alpha=0.6, linewidths=0, label="Human")
    ax.set_title("B. Human vs. all LLMs", fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=2.5, fontsize=10, framealpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle(
        "Kaleido-XL moral value embeddings — LLMs converge, humans spread out",
        fontsize=13, y=1.01
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")


def main():
    rng = np.random.RandomState(RANDOM_SEED)
    print("Loading embeddings...")
    matrix, labels = load_and_subsample(rng)
    print(f"Total points: {matrix.shape[0]}")
    coords = run_umap(matrix)
    sources_present = [s for s in SOURCES if s in labels]
    plot(coords, labels, sources_present)


if __name__ == "__main__":
    main()
