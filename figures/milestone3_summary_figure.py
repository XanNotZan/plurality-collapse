"""Build a single summary figure for milestone 3:
Two panels:
  (left) Bar chart: corpus-level gap measures (baseline, INLP T20, TF-IDF residual, register paraphrase, modernity, perm null)
  (right) Within-dilemma diversity: humans vs LLMs across stratifications
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS = Path("data/analysis")


def main():
    # Load all relevant JSONs
    inlp = json.loads((ANALYSIS / "milestone3_inlp.json").read_text())
    inlp20 = json.loads((ANALYSIS / "milestone3_inlp_T20.json").read_text())
    tfidf = json.loads((ANALYSIS / "milestone3_tfidf_residualized.json").read_text())
    perm = json.loads((ANALYSIS / "milestone3_permutation.json").read_text())
    modern = json.loads((ANALYSIS / "milestone3_modernity.json").read_text())
    paraphrase = json.loads((ANALYSIS / "milestone3_register_paraphrase_clean.json").read_text())
    arctic_strat = json.loads((ANALYSIS / "milestone3_arctic_stratified.json").read_text())
    arctic_within = json.loads((ANALYSIS / "milestone3_arctic_within_dilemma.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.0, 1.1]})

    # === Left panel: corpus-level gap measures ===
    measures = []
    measures.append(("Baseline\n(matched-n)", inlp["baseline"]["gap_human_minus_llm"], "#888888"))
    measures.append(("After 5 INLP\n(linear axis)", inlp["post_inlp"]["gap_human_minus_llm"], "#1f77b4"))
    measures.append(("After 20 INLP\n(20 dirs)", inlp20["comp90_track"]["20"]["gap"], "#1f77b4"))
    measures.append(("TF-IDF Ridge\nresidualized", tfidf["residualized"]["gap"], "#d62728"))
    measures.append(("Register-matched\n(paraphrase, n=289)", paraphrase["cells"][3]["gap"], "#d62728"))
    measures.append(("Permutation null\nmean (sampling)", round(perm["null_distribution"]["gap_mean"], 1), "#cccccc"))

    labels = [m[0] for m in measures]
    vals = [m[1] for m in measures]
    colors = [m[2] for m in measures]
    x = np.arange(len(labels))
    bars = ax1.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 v + (1 if v > 0 else -3), str(v), ha="center", fontsize=9)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("comp90 gap (human − LLM)")
    ax1.set_title("Corpus-level gap shrinks under register/surface controls")
    ax1.set_ylim(-10, max(vals) + 30)
    # Annotate region
    ax1.axvspan(-0.5, 0.5, alpha=0.05, color="gray")
    ax1.text(0, -8, "(reference)", ha="center", fontsize=7, color="gray", style="italic")

    # === Right panel: ArcticShift within-dilemma diversity, all stratifications ===
    # Pool baseline + consensus + verdict + length
    bars_data = []
    # overall
    bars_data.append(("Overall\n(n=1991)", arctic_within["within_human_pairwise_cosine_dist"]["mean"], arctic_within["within_llm_pairwise_cosine_dist"]["mean"], 1.0))
    # consensus
    for level in ["low", "medium", "high"]:
        s = arctic_strat["by_consensus"][level]
        bars_data.append((f"Cons.\n{level}\n(n={s['n_dilemmas']})", s["mean_within_human"], s["mean_within_llm"], s["frac_dilemmas_h_more_diverse"]))
    # verdict
    for v in ["NTA", "YTA", "ESH", "NAH"]:
        s = arctic_strat["by_verdict"].get(v)
        if s is None: continue
        bars_data.append((f"{v}\n(n={s['n_dilemmas']})", s["mean_within_human"], s["mean_within_llm"], s["frac_dilemmas_h_more_diverse"]))
    # length
    for label_q in ["Q1 (<25th)", "Q2 (25-50th)", "Q3 (50-75th)", "Q4 (>75th)"]:
        s = arctic_strat["by_human_length_quartile"].get(label_q)
        if s is None: continue
        q_short = label_q.split()[0]
        bars_data.append((f"len {q_short}\n(n={s['n_dilemmas']})", s["mean_within_human"], s["mean_within_llm"], s["frac_dilemmas_h_more_diverse"]))

    n = len(bars_data)
    x = np.arange(n)
    w = 0.4
    h_vals = [b[1] for b in bars_data]
    l_vals = [b[2] for b in bars_data]
    fracs = [b[3] for b in bars_data]
    labels = [b[0] for b in bars_data]
    ax2.bar(x - w/2, h_vals, w, color="#2ca02c", edgecolor="black", linewidth=0.5, label="Within-human pairwise cos dist")
    ax2.bar(x + w/2, l_vals, w, color="#e377c2", edgecolor="black", linewidth=0.5, label="Within-LLM pairwise cos dist")
    # Mark frac=1.0 above each pair
    for i, f in enumerate(fracs):
        ax2.text(i, max(h_vals[i], l_vals[i]) + 0.01, f"{f*100:.0f}%", ha="center", fontsize=7, color="darkred")

    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("Per-dilemma mean cosine distance")
    ax2.set_title("ArcticShift: per-dilemma humans 2x LLM diversity, 100% of dilemmas\n(red %: fraction of dilemmas with human > LLM)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(0, max(h_vals) + 0.05)
    # Vertical separators between groups
    ax2.axvline(0.5, color="gray", linestyle=":", linewidth=0.5)
    ax2.axvline(3.5, color="gray", linestyle=":", linewidth=0.5)
    ax2.axvline(7.5, color="gray", linestyle=":", linewidth=0.5)
    # Group labels
    ax2.text(0, -0.04, "overall", fontsize=7, ha="center", style="italic", transform=ax2.get_xaxis_transform())
    ax2.text(2, -0.04, "by consensus", fontsize=7, ha="center", style="italic", transform=ax2.get_xaxis_transform())
    ax2.text(5.5, -0.04, "by verdict", fontsize=7, ha="center", style="italic", transform=ax2.get_xaxis_transform())
    ax2.text(9.5, -0.04, "by length quartile", fontsize=7, ha="center", style="italic", transform=ax2.get_xaxis_transform())

    fig.tight_layout()
    out = ANALYSIS / "milestone3_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
