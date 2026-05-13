"""Figures for milestone 3 validation analyses (cluster preservation, ArcticShift within-dilemma)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS = Path("data/analysis")


def fig_cluster_preservation():
    p = ANALYSIS / "milestone3_paraphrase_clusters.json"
    if not p.exists():
        print(f"missing {p}"); return
    d = json.loads(p.read_text())

    fig, ax = plt.subplots(figsize=(6, 3))
    labels = ["Human (orig->formal)", "LLM (orig->casual)"]
    same_value = [d["human_orig_vs_formal"]["same_value_pct"], d["llm_orig_vs_casual"]["same_value_pct"]]
    same_cluster = [d["human_orig_vs_formal"]["same_cluster_pct"], d["llm_orig_vs_casual"]["same_cluster_pct"]]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, same_value, w, color="#4c72b0", edgecolor="black", linewidth=0.5, label="Same exact value")
    ax.bar(x + w/2, same_cluster, w, color="#55a868", edgecolor="black", linewidth=0.5, label="Same cluster (k=60)")
    for i, (sv, sc) in enumerate(zip(same_value, same_cluster)):
        ax.text(i - w/2, sv + 1, f"{sv:.1f}%", ha="center", fontsize=9)
        ax.text(i + w/2, sc + 1, f"{sc:.1f}%", ha="center", fontsize=9)
    ax.axhline(100, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.axhline(50, color="red", linestyle="--", linewidth=0.7, alpha=0.5, label="50% (50/50 random)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("% of pairs preserving Kaleido value/cluster")
    ax.set_ylim(0, 100)
    ax.set_title("Paraphrase content preservation (Kaleido-decoded value labels)")
    ax.legend(loc="upper center", fontsize=8)
    fig.tight_layout()
    out = ANALYSIS / "milestone3_paraphrase_clusters.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_arctic_within_dilemma():
    p = ANALYSIS / "milestone3_arctic_within_dilemma.json"
    if not p.exists():
        print(f"missing {p}"); return
    d = json.loads(p.read_text())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    labels = ["Within ArcticShift\nhuman (per dilemma)", "Within Sachdeva\nLLM (per dilemma)", "Human-to-LLM\ncentroid (per dilemma)"]
    means = [
        d["within_human_pairwise_cosine_dist"]["mean"],
        d["within_llm_pairwise_cosine_dist"]["mean"],
        d["human_to_llm_centroid_dist"]["mean"],
    ]
    stds = [
        d["within_human_pairwise_cosine_dist"]["std"],
        d["within_llm_pairwise_cosine_dist"]["std"],
        d["human_to_llm_centroid_dist"]["std"],
    ]
    colors = ["#2ca02c", "#e377c2", "#1f77b4"]
    bars = ax.bar(labels, means, yerr=stds, color=colors, edgecolor="black", linewidth=0.5, capsize=6)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Cosine distance (per-dilemma mean)")
    ax.set_title(f"ArcticShift validation: within-dilemma diversity\n(n_dilemmas={d['within_human_pairwise_cosine_dist']['n_dilemmas']}, n_arctic_comments={d['n_arctic_comments']})")
    fig.tight_layout()
    out = ANALYSIS / "milestone3_arctic_within_dilemma.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    fig_cluster_preservation()
    fig_arctic_within_dilemma()


if __name__ == "__main__":
    main()
