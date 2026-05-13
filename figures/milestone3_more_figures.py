"""Figures for milestone 3 supplementary analyses (within-dilemma, verdict, modernity, paraphrase)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS = Path("data/analysis")
EMBEDDINGS = Path("data/embeddings")


def fig_within_dilemma():
    """Histogram: within-LLM dist vs human-to-LLM-centroid dist across dilemmas."""
    # Need per-dilemma metrics - recompute quickly from embeddings
    import json as _json
    from collections import defaultdict
    LLM_SOURCES = ["gpt3.5","gpt4","claude","bison","gemma","mistral","llama"]
    metas = {s: _json.loads((EMBEDDINGS / f"{s}_meta.json").read_text()) for s in LLM_SOURCES + ["human"]}
    embs = {s: np.load(EMBEDDINGS / f"{s}.npy") for s in LLM_SOURCES + ["human"]}

    per_dilemma = defaultdict(lambda: {"human": [], "llm": []})
    for src in LLM_SOURCES + ["human"]:
        E = embs[src]
        for m in metas[src]:
            tag = "human" if src == "human" else "llm"
            per_dilemma[m["submission_id"]][tag].append(E[m["index"]])

    within_llm = []
    h_to_centroid = []
    for sid, srcs in per_dilemma.items():
        if not srcs["human"] or len(srcs["llm"]) < 2:
            continue
        h = srcs["human"][0]
        L = np.stack(srcs["llm"])
        n = len(L)
        # Within-LLM mean pairwise cosine distance
        d_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                ca = L[i] / (np.linalg.norm(L[i]) + 1e-12)
                cb = L[j] / (np.linalg.norm(L[j]) + 1e-12)
                d_pairs.append(1 - ca @ cb)
        within_llm.append(float(np.mean(d_pairs)))
        # Human-to-LLM-centroid
        Lc = L.mean(axis=0)
        a = h / (np.linalg.norm(h) + 1e-12)
        b = Lc / (np.linalg.norm(Lc) + 1e-12)
        h_to_centroid.append(float(1 - a @ b))

    within_llm = np.array(within_llm)
    h_to_centroid = np.array(h_to_centroid)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.linspace(0, 0.6, 60)
    ax.hist(within_llm, bins=bins, alpha=0.65, color="#1f77b4", label=f"Within-LLM pairwise (mean={within_llm.mean():.3f})")
    ax.hist(h_to_centroid, bins=bins, alpha=0.65, color="#2ca02c", label=f"Human-to-LLM-centroid (mean={h_to_centroid.mean():.3f})")
    ax.set_xlabel("Per-dilemma cosine distance")
    ax.set_ylabel("Number of dilemmas")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Per-dilemma: LLMs cluster tightly; humans live outside the cluster")
    fig.tight_layout()
    out = ANALYSIS / "milestone3_within_dilemma.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_verdict_gap():
    d = json.loads((ANALYSIS / "milestone3_verdict_gap.json").read_text())
    verdicts = ["NTA","YTA","ESH","NAH"]
    h = [d["verdicts"][v]["human_comp90"] for v in verdicts]
    l = [d["verdicts"][v]["llm_comp90"] for v in verdicts]
    n = [d["verdicts"][v]["matched_n"] for v in verdicts]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(verdicts))
    w = 0.35
    ax.bar(x - w/2, h, w, color="#2ca02c", edgecolor="black", linewidth=0.5, label="Human comp90")
    ax.bar(x + w/2, l, w, color="#e377c2", edgecolor="black", linewidth=0.5, label="LLM comp90")
    for i, (a, b, ni) in enumerate(zip(h, l, n)):
        ax.text(i - w/2, a + 4, str(a), ha="center", fontsize=8)
        ax.text(i + w/2, b + 4, str(b), ha="center", fontsize=8)
        ax.text(i, max(a, b) + 25, f"n={ni}\n+{a-b}", ha="center", fontsize=8, color="black")
    ax.set_xticks(x); ax.set_xticklabels(verdicts)
    ax.set_xlabel("Dominant community verdict")
    ax.set_ylabel("comp90")
    ax.set_title("Dimensionality gap by verdict (matched n)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = ANALYSIS / "milestone3_verdict_gap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_modernity():
    p = ANALYSIS / "milestone3_modernity.json"
    if not p.exists():
        print(f"missing {p}, skipping modernity figure")
        return
    d = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = ["Human (matched n)", "Sachdeva all-LLM (matched n)", "Qwen 2.5 3B (modern)"]
    vals = [d["human_matched_n_comp90_mean"], d["alllm_matched_n_comp90_mean"], d["qwen_comp90"]]
    colors = ["#2ca02c", "#e377c2", "#d62728"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 4, f"{v:.0f}", ha="center", fontsize=10)
    ax.set_ylabel("comp90")
    ax.set_title(f"Modernity test: Qwen 2.5 3B vs Sachdeva LLMs (n={d['qwen_n']})")
    fig.tight_layout()
    out = ANALYSIS / "milestone3_modernity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_register_paraphrase():
    p = ANALYSIS / "milestone3_register_paraphrase.json"
    if not p.exists():
        print(f"missing {p}, skipping paraphrase figure")
        return
    d = json.loads(p.read_text())
    cells = d["cells"]
    labels = [c["label"].replace("_", "\n") for c in cells]
    gaps = [c["gap"] for c in cells]
    cdists = [c["centroid_dist"] for c in cells]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    bars = ax1.bar(range(len(labels)), gaps, color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("comp90 gap (human - LLM)")
    ax1.set_title("Dimensionality gap, 4 register configs")
    for b, g in zip(bars, gaps):
        ax1.text(b.get_x()+b.get_width()/2, g + 0.5, str(g), ha="center", fontsize=9)

    bars2 = ax2.bar(range(len(labels)), cdists, color="#9467bd", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("Centroid cosine distance")
    ax2.set_title("Centroid separation, 4 register configs")
    for b, c in zip(bars2, cdists):
        ax2.text(b.get_x()+b.get_width()/2, c + 0.005, f"{c:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    out = ANALYSIS / "milestone3_register_paraphrase.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    if (ANALYSIS / "milestone3_verdict_gap.json").exists():
        fig_verdict_gap()
    fig_within_dilemma()
    fig_modernity()
    fig_register_paraphrase()


if __name__ == "__main__":
    main()
