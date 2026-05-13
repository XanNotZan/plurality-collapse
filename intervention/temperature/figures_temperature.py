"""Plot per-temperature cosine + cluster diversity results.

Uses matched-K cosine (qwen and human) and the human K-sweep cluster baseline,
so reference lines and panels are apples-to-apples.
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS = Path("data/analysis/intervention/temperature")
TEMPERATURES = [0.3, 0.7, 1.0, 1.3]
K_PER_TEMP = {0.3: 3, 0.7: 5, 1.0: 8, 1.3: 12}

# Sachdeva LLM baseline from milestone 3 (full per-dilemma, not K-controlled)
SACHDEVA_LLM_COSINE = 0.147
SACHDEVA_LLM_CLUSTERS_K10 = 3.39

logger = logging.getLogger("fig_temp")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg); sys.stdout.flush()


def plot_cosine():
    data = json.loads((ANALYSIS / "per_dilemma_cosine.json").read_text())
    matched = json.loads((ANALYSIS / "matched_k_cosine.json").read_text())
    H_full = data["baselines"]["human_arctic_within_dilemma_cosine"]
    H_K3 = matched["human"]["by_K"]["K_3"]["mean_cosine_distance"]
    L = SACHDEVA_LLM_COSINE

    rows = [data["per_temperature"][f"T_{T:.1f}"] for T in TEMPERATURES]
    means = [r["mean_cosine_distance"] for r in rows]
    ci_lo = [r["bootstrap_ci_95"][0] for r in rows]
    ci_hi = [r["bootstrap_ci_95"][1] for r in rows]
    err_lo = [m - lo for m, lo in zip(means, ci_lo)]
    err_hi = [hi - m for m, hi in zip(means, ci_hi)]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})

    x = np.arange(len(TEMPERATURES))
    axA.bar(x, means, yerr=[err_lo, err_hi], capsize=4, color="#1f77b4", alpha=0.85,
            edgecolor="black", linewidth=0.6)
    axA.axhline(H_full, color="tab:green", linestyle="--", linewidth=1.2,
                label=f"Human ArcticShift full-K ({H_full:.3f})")
    axA.axhline(H_K3, color="tab:olive", linestyle=":", linewidth=1.2,
                label=f"Human matched K=3 ({H_K3:.3f})")
    axA.axhline(L, color="tab:red", linestyle="--", linewidth=1.2,
                label=f"Sachdeva LLMs ({L:.3f})")
    axA.set_xticks(x); axA.set_xticklabels([f"T={T}\nK={K_PER_TEMP[T]}" for T in TEMPERATURES])
    axA.set_ylabel("Mean per-dilemma cosine distance")
    axA.set_title("A. Within-dilemma diversity vs temperature")
    axA.legend(loc="lower right", fontsize=8)
    axA.grid(True, axis="y", alpha=0.3)

    # Ratio at full-K human (the headline number)
    ratios = [H_full / m for m in means]
    axB.bar(x, ratios, color="#ff7f0e", alpha=0.85, edgecolor="black", linewidth=0.6)
    axB.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Parity (human = Qwen)")
    axB.set_xticks(x); axB.set_xticklabels([f"T={T}" for T in TEMPERATURES])
    axB.set_ylabel("Ratio: human / Qwen mean cosine")
    axB.set_title("B. Diversity gap closure vs temperature")
    axB.legend(loc="upper right", fontsize=9)
    axB.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    p = ANALYSIS / "temperature_per_dilemma_cosine.png"
    plt.savefig(p, dpi=140)
    plt.close(fig)
    _flush(f"saved {p}")


def plot_cluster_diversity():
    data = json.loads((ANALYSIS / "cluster_diversity.json").read_text())
    human_sweep = json.loads((ANALYSIS / "human_cluster_diversity_sweep.json").read_text())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.5, 1]})

    # Panel A: full K-sweep, one line per temperature + human line
    colors = {0.3: "#1f77b4", 0.7: "#2ca02c", 1.0: "#ff7f0e", 1.3: "#d62728"}
    for T in TEMPERATURES:
        per_T = data["per_temperature"][f"T_{T:.1f}"]
        Ks = sorted(int(k.split("_")[1]) for k in per_T["by_K"].keys())
        means = [per_T["by_K"][f"K_{k}"]["mean_distinct_clusters"] for k in Ks]
        ci_lo = [per_T["by_K"][f"K_{k}"]["bootstrap_ci_95"][0] for k in Ks]
        ci_hi = [per_T["by_K"][f"K_{k}"]["bootstrap_ci_95"][1] for k in Ks]
        axA.plot(Ks, means, marker="o", color=colors[T], label=f"Qwen T={T}", linewidth=1.6)
        axA.fill_between(Ks, ci_lo, ci_hi, alpha=0.18, color=colors[T])

    # Human K-sweep line
    h_Ks = sorted(int(k.split("_")[1]) for k in human_sweep["by_K"].keys())
    h_means = [human_sweep["by_K"][f"K_{k}"]["mean_distinct_clusters"] for k in h_Ks]
    h_lo = [human_sweep["by_K"][f"K_{k}"]["bootstrap_ci_95"][0] for k in h_Ks]
    h_hi = [human_sweep["by_K"][f"K_{k}"]["bootstrap_ci_95"][1] for k in h_Ks]
    axA.plot(h_Ks, h_means, marker="s", color="black", label="Human ArcticShift",
             linewidth=2.0, linestyle="--")
    axA.fill_between(h_Ks, h_lo, h_hi, alpha=0.15, color="black")

    axA.set_xlabel("K (samples per dilemma)")
    axA.set_ylabel("Mean distinct 60-clusters per dilemma")
    axA.set_title("A. Kaleido cluster diversity vs K, by source")
    axA.legend(fontsize=8, loc="lower right")
    axA.grid(True, alpha=0.3)

    # Panel B: matched-K=3 across temperatures + human K=3 reference
    x = np.arange(len(TEMPERATURES))
    means3, lo3, hi3 = [], [], []
    for T in TEMPERATURES:
        per_T = data["per_temperature"][f"T_{T:.1f}"]
        rec = per_T["by_K"].get("K_3")
        if rec is None:
            means3.append(np.nan); lo3.append(np.nan); hi3.append(np.nan)
        else:
            means3.append(rec["mean_distinct_clusters"])
            lo3.append(rec["bootstrap_ci_95"][0])
            hi3.append(rec["bootstrap_ci_95"][1])
    err_lo = [m - lo for m, lo in zip(means3, lo3)]
    err_hi = [hi - m for m, hi in zip(means3, hi3)]
    axB.bar(x, means3, yerr=[err_lo, err_hi], capsize=4, color="#1f77b4",
            alpha=0.85, edgecolor="black", linewidth=0.6, label="Qwen")
    H_K3 = human_sweep["by_K"]["K_3"]["mean_distinct_clusters"]
    axB.axhline(H_K3, color="tab:green", linestyle="--", linewidth=1.4,
                label=f"Human K=3 ({H_K3:.2f})")
    axB.set_xticks(x); axB.set_xticklabels([f"T={T}" for T in TEMPERATURES])
    axB.set_ylabel("Mean distinct clusters per dilemma (K=3)")
    axB.set_title("B. Matched K=3 cluster diversity")
    axB.legend(fontsize=9, loc="lower right")
    axB.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    p = ANALYSIS / "temperature_cluster_diversity.png"
    plt.savefig(p, dpi=140)
    plt.close(fig)
    _flush(f"saved {p}")


def main():
    plot_cosine()
    plot_cluster_diversity()


if __name__ == "__main__":
    main()
