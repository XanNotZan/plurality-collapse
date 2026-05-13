"""Figures for milestone 3 extra diagnostics."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS = Path("data/analysis")


def fig_per_pc_attribution():
    d = json.loads((ANALYSIS / "milestone3_per_pc_attribution.json").read_text())
    var_h = np.array(d["var_human"])
    var_l = np.array(d["var_llm"])
    ratio = np.array(d["ratio_h_over_l"])
    n = len(ratio)
    pcs = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(pcs, ratio, color="#444444", linewidth=0.8, alpha=0.7)
    # Smooth via moving average for clarity
    win = 10
    if n >= win:
        kernel = np.ones(win) / win
        ratio_smooth = np.convolve(ratio, kernel, mode="same")
        ax.plot(pcs, ratio_smooth, color="#d62728", linewidth=2,
                label=f"Moving avg (w={win})")
    ax.axhline(1.0, color="black", linestyle="-", linewidth=0.7, alpha=0.6)
    ax.axhline(1.5, color="green", linestyle="--", linewidth=0.7, alpha=0.6,
               label="ratio = 1.5")
    ax.set_xlabel("Principal component index (pooled basis)")
    ax.set_ylabel("var(human) / var(LLM)")
    ax.set_title("Human variance dominates LLM variance across nearly all PCs")
    ax.set_xlim(1, n)
    # Y limits: clip extreme outliers
    p99 = np.percentile(ratio, 99)
    ax.set_ylim(0.5, max(p99 * 1.1, 3))
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = ANALYSIS / "milestone3_per_pc_ratio.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_inlp_t20():
    d = json.loads((ANALYSIS / "milestone3_inlp_T20.json").read_text())
    accs = d["classifier_acc_per_iter"]
    track = d["comp90_track"]
    iters = list(range(len(accs)))
    track_iters = sorted(int(k) for k in track.keys())
    track_h = [track[str(k)]["human_comp90"] for k in track_iters]
    track_l = [track[str(k)]["llm_comp90"] for k in track_iters]
    track_gap = [track[str(k)]["gap"] for k in track_iters]

    fig, ax1 = plt.subplots(figsize=(7.5, 3.5))
    ax1.plot(iters, accs, "o-", color="#d62728", linewidth=1.5, markersize=4,
             label="Classifier accuracy")
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                label="Chance (0.5)")
    ax1.set_xlabel("INLP iteration (linear directions removed)")
    ax1.set_ylabel("Classifier accuracy", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_ylim(0.4, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(track_iters, track_h, "s--", color="#2ca02c", linewidth=2,
             markersize=8, label="Human comp90")
    ax2.plot(track_iters, track_l, "s--", color="#e377c2", linewidth=2,
             markersize=8, label="LLM comp90")
    ax2.set_ylabel("PCA components for 90% variance")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)

    ax1.set_title(f"Higher-T INLP: gap stable at all directions removed up to T={d['n_iter']}")
    fig.tight_layout()
    out = ANALYSIS / "milestone3_inlp_T20.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_mfd_gradient():
    d = json.loads((ANALYSIS / "milestone3_mfd_gradient.json").read_text())
    bins = d["bins"]
    err = np.array([b["mean_recon_err"] for b in bins])
    uf = np.array([b["used_foundations"] for b in bins])
    ent = np.array([b["bin_level_entropy"] for b in bins])
    valid = ~np.isnan(ent)

    fig, ax1 = plt.subplots(figsize=(7.5, 3.5))
    ax1.plot(err, uf, "o-", color="#2ca02c", linewidth=1.5, markersize=3,
             label=f"Foundations used (r={d['bin_correlation_used_foundations']:.3f})")
    ax1.set_xlabel("Reconstruction error (under all-LLM PCs, k=448)")
    ax1.set_ylabel("Foundations used (>=1 hit)", color="#2ca02c")
    ax1.tick_params(axis="y", labelcolor="#2ca02c")

    ax2 = ax1.twinx()
    ax2.plot(err[valid], ent[valid], "s-", color="#ff7f0e", linewidth=1.5, markersize=3,
             label=f"Bin entropy (r={d['bin_correlation_bin_entropy']:.3f})")
    ax2.set_ylabel("MFD foundation entropy (bits)", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)
    ax1.set_title("Decoder-independent diversity gradient (MFD lemma counts)")
    fig.tight_layout()
    out = ANALYSIS / "milestone3_mfd_gradient.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    if (ANALYSIS / "milestone3_per_pc_attribution.json").exists():
        fig_per_pc_attribution()
    if (ANALYSIS / "milestone3_inlp_T20.json").exists():
        fig_inlp_t20()
    if (ANALYSIS / "milestone3_mfd_gradient.json").exists():
        fig_mfd_gradient()


if __name__ == "__main__":
    main()
