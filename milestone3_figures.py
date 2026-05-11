"""Generate milestone 3 figures from diagnostic JSON outputs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS = Path("data/analysis")


def fig_inlp_curve():
    data = json.loads((ANALYSIS / "milestone3_inlp.json").read_text())
    accs = data["classifier_acc_per_iter"]
    iters = list(range(len(accs)))  # iter 0 = baseline classifier on raw, ..., last = post-projection

    base_h = data["baseline"]["human_comp90"]
    post_h = data["post_inlp"]["human_comp90"]
    base_l = data["baseline"]["llm_matched_comp90"]
    post_l = data["post_inlp"]["llm_matched_comp90"]

    fig, ax1 = plt.subplots(figsize=(7.5, 3.7))

    color_acc = "#d62728"
    ax1.plot(iters, accs, "o-", color=color_acc, linewidth=2, markersize=7,
             label="Classifier accuracy")
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                label="Chance (0.5)")
    ax1.set_xlabel("INLP iteration (linear directions removed)")
    ax1.set_ylabel("Logreg classifier accuracy", color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(0.4, 1.05)
    ax1.set_xticks(iters)

    ax2 = ax1.twinx()
    color_h = "#2ca02c"
    color_l = "#e377c2"
    # Endpoints only — line connects baseline → post-INLP (at last iter)
    ax2.plot([0, iters[-1]], [base_h, post_h], "s--", color=color_h, linewidth=2,
             markersize=8, label=f"Human comp90 ({base_h}$\\to${post_h})")
    ax2.plot([0, iters[-1]], [base_l, post_l], "s--", color=color_l, linewidth=2,
             markersize=8, label=f"LLM comp90 ({base_l}$\\to${post_l})")
    ax2.set_ylabel("PCA components for 90% variance")
    ax2.set_ylim(min(base_l, post_l) - 30, max(base_h, post_h) + 30)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)

    ax1.set_title("INLP: classifier accuracy collapses, dimensionality gap persists")
    fig.tight_layout()
    out = ANALYSIS / "milestone3_inlp_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_permutation_null():
    data = json.loads((ANALYSIS / "milestone3_permutation.json").read_text())
    null_gap = np.array(data["raw_null_gap"])
    obs = data["observed"]["gap"]
    z = data["test_statistic"]["z_score"]

    fig, ax = plt.subplots(figsize=(7.5, 3.7))
    ax.hist(null_gap, bins=30, color="#999999", edgecolor="black", alpha=0.85,
            label="Null distribution (100 label shuffles)")
    ax.axvline(obs, color="#d62728", linestyle="-", linewidth=2.5,
               label=f"Observed gap = {obs} ($z={z:.1f}$)")
    ax.axvline(0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("comp90(group A) - comp90(group B)")
    ax.set_ylabel("Permutations")
    ax.set_title("Permutation null: observed gap is far outside chance")
    # Use symlog or split-axis since the null is ~0 and observed is ~120
    null_max = float(max(np.abs(null_gap).max(), 5))
    ax.set_xlim(-null_max - 5, obs + 10)
    ax.legend(loc="upper center", fontsize=9)
    fig.tight_layout()
    out = ANALYSIS / "milestone3_permutation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    fig_inlp_curve()
    fig_permutation_null()


if __name__ == "__main__":
    main()
