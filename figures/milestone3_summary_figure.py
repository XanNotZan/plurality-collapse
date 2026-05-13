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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.0, 1.35]})

    # === Left panel: corpus-level gap measures ===
    measures = []
    measures.append(("Baseline (matched-n)", inlp["baseline"]["gap_human_minus_llm"], "#888888"))
    measures.append(("INLP T=5 (linear)", inlp["post_inlp"]["gap_human_minus_llm"], "#1f77b4"))
    measures.append(("INLP T=20 (linear)", inlp20["comp90_track"]["20"]["gap"], "#1f77b4"))
    measures.append(("TF-IDF Ridge residual.", tfidf["residualized"]["gap"], "#d62728"))
    measures.append(("Register paraphrase (n=289)", paraphrase["cells"][3]["gap"], "#d62728"))
    measures.append(("Permutation null (sampling)", round(perm["null_distribution"]["gap_mean"], 1), "#cccccc"))

    labels = [m[0] for m in measures]
    vals = [m[1] for m in measures]
    colors = [m[2] for m in measures]
    x = np.arange(len(labels))
    bars = ax1.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        if v >= 0:
            ax1.text(bar.get_x() + bar.get_width()/2, v + 3, str(v), ha="center", fontsize=10, fontweight="bold")
        else:
            ax1.text(bar.get_x() + bar.get_width()/2, v - 6, str(v), ha="center", fontsize=10, fontweight="bold")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9, rotation=35, ha="right")
    ax1.set_ylabel("comp90 gap (human − LLM)", fontsize=10)
    ax1.set_title("Corpus-level gap shrinks under register/surface controls", fontsize=11)
    ax1.set_ylim(-20, max(vals) + 25)
    ax1.axvspan(-0.5, 0.5, alpha=0.05, color="gray")

    # === Right panel: ArcticShift within-dilemma diversity, all stratifications ===
    # Pool baseline + consensus + verdict + length
    bars_data = []
    # overall
    bars_data.append((f"Overall (n={arctic_within['within_human_pairwise_cosine_dist']['n_dilemmas']})", arctic_within["within_human_pairwise_cosine_dist"]["mean"], arctic_within["within_llm_pairwise_cosine_dist"]["mean"], 1.0))
    # consensus
    for level in ["low", "medium", "high"]:
        s = arctic_strat["by_consensus"][level]
        bars_data.append((f"Cons. {level} (n={s['n_dilemmas']})", s["mean_within_human"], s["mean_within_llm"], s["frac_dilemmas_h_more_diverse"]))
    # verdict
    for v in ["NTA", "YTA", "ESH", "NAH"]:
        s = arctic_strat["by_verdict"].get(v)
        if s is None: continue
        bars_data.append((f"{v} (n={s['n_dilemmas']})", s["mean_within_human"], s["mean_within_llm"], s["frac_dilemmas_h_more_diverse"]))
    # length
    for label_q in ["Q1 (<25th)", "Q2 (25-50th)", "Q3 (50-75th)", "Q4 (>75th)"]:
        s = arctic_strat["by_human_length_quartile"].get(label_q)
        if s is None: continue
        q_short = label_q.split()[0]
        bars_data.append((f"len {q_short} (n={s['n_dilemmas']})", s["mean_within_human"], s["mean_within_llm"], s["frac_dilemmas_h_more_diverse"]))

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
        ax2.text(i, max(h_vals[i], l_vals[i]) + 0.008, f"{f*100:.0f}%", ha="center", fontsize=8, color="darkred", fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9, rotation=35, ha="right")
    ax2.set_ylabel("Per-dilemma mean cosine distance", fontsize=10)
    ax2.set_title("ArcticShift: per-dilemma humans 2× LLM diversity, 100% of dilemmas\n(red %: fraction of dilemmas with human > LLM)", fontsize=11)
    ax2.set_ylim(0, max(h_vals) + 0.08)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=10, ncol=2, framealpha=0.95)
    # Vertical separators between groups
    for sep in [0.5, 3.5, 7.5]:
        ax2.axvline(sep, color="gray", linestyle=":", linewidth=0.6)

    fig.tight_layout()
    out = ANALYSIS / "milestone3_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
