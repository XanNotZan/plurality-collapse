"""Synthesis: assemble lexical-feature + diff-PCA + qualitative-probe results into
publishable figures and a single LaTeX-ready summary JSON.

Reads (all optional - skip what's missing):
  data/analysis/milestone3_lexical_summary.csv
  data/analysis/milestone3_lexical_pairdiff.csv
  data/analysis/milestone3_lexical_lengthctrl.csv
  data/analysis/milestone3_lexical_bootstrap.json
  data/analysis/milestone3_diff_pca.json
  data/analysis/milestone3_diff_pca_exemplars.json
  data/analysis/milestone3_diff_pca_lexcorr.csv
  data/analysis/milestone3_qualitative_probe_clusters.json

Writes:
  data/analysis/milestone3_framing_figure.png      (panelled summary)
  data/analysis/milestone3_framing_summary.json    (paper-ready numbers)
  data/analysis/milestone3_framing_table.csv       (top feature differences)
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = Path("data/analysis")

logger = logging.getLogger("synth")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg); sys.stdout.flush()


FEAT_LABELS = {
    "hedge_density": "Hedges (might/maybe/I think)",
    "booster_density": "Boosters (definitely/clearly)",
    "p1_density": "1st-person pronouns (I/my/we)",
    "p2_density": "2nd-person pronouns (you/your)",
    "anecdote_density": "Anecdote markers (my friend, I was)",
    "cf_density": "Counterfactuals (if/would have)",
    "causal_density": "Causal markers (because/so)",
    "concessive_density": "Concessives (although/yet)",
    "noun_density": "Noun density",
    "verb_density": "Verb density",
    "adj_density": "Adjective density",
    "adv_density": "Adverb density",
    "pron_density": "Pronoun density (spaCy POS)",
    "ner_person_density": "NER: PERSON",
    "ner_org_density": "NER: ORG",
    "ner_gpe_density": "NER: GPE",
    "ner_date_density": "NER: DATE",
    "ner_cardinal_density": "NER: NUM/ORDINAL",
    "ner_total_density": "NER: total",
    "mean_word_len": "Mean word length (chars)",
    "type_token_ratio": "Type-token ratio",
    "mean_sent_len": "Mean sentence length (tokens)",
}


def main():
    summary = {}

    # --- Lexical pair-diff ------------------------------------------------
    pd_path = ANALYSIS / "milestone3_lexical_pairdiff.csv"
    lex_lc = ANALYSIS / "milestone3_lexical_lengthctrl.csv"
    lex_sum = ANALYSIS / "milestone3_lexical_summary.csv"
    has_lex = pd_path.exists()
    if has_lex:
        dfd = pd.read_csv(pd_path)
        dfl = pd.read_csv(lex_lc) if lex_lc.exists() else None
        dfs = pd.read_csv(lex_sum) if lex_sum.exists() else None

        # rank features by |cohen_d|
        dfd_sorted = dfd.copy()
        dfd_sorted["abs_d"] = dfd_sorted["cohen_d"].abs()
        dfd_sorted = dfd_sorted.sort_values("abs_d", ascending=False)
        _flush(f"top 10 features by |Cohen's d|:")
        for _, r in dfd_sorted.head(10).iterrows():
            _flush(f"  {r['feature']:30s}  d={r['cohen_d']:+.3f}  diff(h-l)={r['h_minus_l_mean']:+.3f}  "
                   f"CI=[{r['h_minus_l_ci_lo']:+.3f},{r['h_minus_l_ci_hi']:+.3f}]  p={r['perm_pvalue']:.4f}")

        # join with length-controlled (raw_diff, resid_diff, attenuation)
        if dfl is not None:
            merged = dfd_sorted.merge(dfl, on="feature", how="left", suffixes=("", "_lc"))
            merged.to_csv(ANALYSIS / "milestone3_framing_table.csv", index=False)
            _flush("saved milestone3_framing_table.csv")
        else:
            merged = dfd_sorted
            merged.to_csv(ANALYSIS / "milestone3_framing_table.csv", index=False)

        summary["lexical"] = {
            "n_pair_cells": int(dfd["n_cells"].max()),
            "top10_features_by_abs_cohend": merged.head(10).to_dict(orient="records"),
        }

    # --- Diff-PCA --------------------------------------------------------
    pca_path = ANALYSIS / "milestone3_diff_pca.json"
    if pca_path.exists():
        pca = json.loads(pca_path.read_text())
        summary["diff_pca"] = {
            "n_pairs": pca["n_pairs"],
            "n_cells": pca["n_cells"],
            "mean_offset_norm": pca["mean_offset_norm"],
            "raw_pc1_cos_to_mean_offset": pca["raw_pc1_cos_to_mean_offset"],
            "raw_explained_var_pcs1to5": pca["raw_explained_var"][:5],
            "centered_explained_var_pcs1to5": pca["centered_explained_var"][:5],
            "scrambled_explained_var_pcs1to5": pca["scrambled_explained_var"][:5],
            "scrambled_pc_cosines_vs_paired_pcs1to5": pca["scrambled_pc_cosines_vs_paired"][:5],
            "bootstrap_pc_stability_mean_pcs1to5": pca["bootstrap_stability_centered"]["pc_mean_cosine_to_full"][:5],
        }

    lcorr_path = ANALYSIS / "milestone3_diff_pca_lexcorr.csv"
    if lcorr_path.exists():
        lc = pd.read_csv(lcorr_path)
        # per-PC top features by |pearson_r|
        per_pc_top = {}
        for pc in sorted(lc["pc"].unique(), key=lambda s: int(s[2:])):
            sub = lc[lc["pc"] == pc].copy()
            sub["abs_r"] = sub["pearson_r"].abs()
            sub = sub.sort_values("abs_r", ascending=False).head(5)
            per_pc_top[pc] = sub[["feature", "pearson_r"]].to_dict(orient="records")
        summary["diff_pca_lexcorr_top5_per_pc"] = per_pc_top

    # --- Qualitative probe -----------------------------------------------
    qp_path = ANALYSIS / "milestone3_qualitative_probe_clusters.json"
    if qp_path.exists():
        qp = json.loads(qp_path.read_text())
        # use k=8 as canonical
        canonical = qp["paired_clusters"].get("k_8")
        if canonical:
            summary["qualitative_probe"] = {
                "n_paired": qp["n_paired"],
                "n_scrambled": qp["n_scrambled"],
                "model": qp["model_name"],
                "k8_clusters": {str(c): {"n": v["n"], "top_terms": v["top_terms"][:6]}
                                for c, v in canonical["clusters"].items()},
            }

    # --- Figure ----------------------------------------------------------
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.4, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # Panel A: top features by |Cohen's d|, bars
    if has_lex:
        top = merged.head(14).iloc[::-1]
        labels = [FEAT_LABELS.get(f, f) for f in top["feature"]]
        cd = top["cohen_d"].values
        colors = ["tab:blue" if d > 0 else "tab:red" for d in cd]
        y = np.arange(len(top))
        ax_a.barh(y, cd, color=colors, alpha=0.85)
        ax_a.axvline(0, color="black", linewidth=0.7)
        ax_a.set_yticks(y); ax_a.set_yticklabels(labels, fontsize=8)
        ax_a.set_xlabel("Cohen's d (positive = humans higher)")
        ax_a.set_title("A.  Top within-cluster lexical/pragmatic differences")
        ax_a.grid(True, axis="x", alpha=0.3)

    # Panel B: PCA explained variance + scrambled
    if pca_path.exists():
        pcs = list(range(1, 1 + len(pca["centered_explained_var"][:10])))
        ax_b.bar([p - 0.18 for p in pcs], pca["centered_explained_var"][:10], width=0.36,
                 label="paired", color="tab:blue", alpha=0.85)
        ax_b.bar([p + 0.18 for p in pcs], pca["scrambled_explained_var"][:10], width=0.36,
                 label="scrambled", color="tab:gray", alpha=0.75)
        ax_b.set_xticks(pcs)
        ax_b.set_xlabel("PC index"); ax_b.set_ylabel("Var ratio")
        ax_b.set_title("B.  Diff-PCA explained variance")
        ax_b.legend(fontsize=8); ax_b.grid(True, axis="y", alpha=0.3)

    # Panel C: PC-feature correlation heatmap
    if lcorr_path.exists():
        lc = pd.read_csv(lcorr_path)
        pcs_keep = [f"PC{i}" for i in range(1, 6)]
        lc = lc[lc["pc"].isin(pcs_keep)]
        pivot = lc.pivot(index="feature", columns="pc", values="pearson_r")
        pivot = pivot[pcs_keep]
        # order features by |max r| across top PCs
        pivot["abs_max"] = pivot.abs().max(axis=1)
        pivot = pivot.sort_values("abs_max", ascending=False).drop(columns="abs_max")
        # restrict to top 14 features
        pivot = pivot.head(14)
        # human-readable labels
        labels_c = [FEAT_LABELS.get(f, f) for f in pivot.index]
        im = ax_c.imshow(pivot.values, cmap="RdBu_r", vmin=-0.85, vmax=0.85, aspect="auto")
        ax_c.set_xticks(range(len(pcs_keep))); ax_c.set_xticklabels(pcs_keep)
        ax_c.set_yticks(range(len(pivot))); ax_c.set_yticklabels(labels_c, fontsize=8)
        ax_c.set_title("C.  Lexical-feature correlation with diff-PCA axes (Pearson r)")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if abs(v) > 0.15:
                    ax_c.text(j, i, f"{v:+.2f}", ha="center", va="center",
                              color="white" if abs(v) > 0.55 else "black", fontsize=7)
        plt.colorbar(im, ax=ax_c, fraction=0.025, label="Pearson r")

    plt.tight_layout()
    fig_path = ANALYSIS / "milestone3_framing_figure.png"
    plt.savefig(fig_path, dpi=140)
    plt.close(fig)
    _flush(f"saved figure: {fig_path}")

    (ANALYSIS / "milestone3_framing_summary.json").write_text(json.dumps(summary, indent=2))
    _flush("saved milestone3_framing_summary.json")


if __name__ == "__main__":
    main()
