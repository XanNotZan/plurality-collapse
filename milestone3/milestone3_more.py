"""Milestone 3 supplementary CPU analyses: within-dilemma LLM clustering, verdict-conditional gap, MFD length-stratified."""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

EMBEDDINGS_DIR = Path("data/embeddings")
OUTPUT_DIR = Path("data/analysis")
HIDDEN_DIM = 2048
ALL_SOURCES = ["human", "gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
RANDOM_SEED = 42

# MFD same as milestone3_extra (compact)
MFD = {
    "care_virtue": ["safe","peace","compassion","empath","kind","care","caring","protect","shelter","save","guard","defend","shield","nurtur","support","help","rescue","secur","comfort","sympath","mercy","gentle","tender","warmth","love","loving"],
    "care_vice":   ["harm","hurt","suffer","pain","kill","murder","abuse","abus","cruel","brutal","violen","wound","damage","destroy","attack","assault","bully","torment","torture","victim","agony","mistreat","punish","neglect"],
    "fairness_virtue": ["fair","fairness","equal","equit","justice","just","righteous","honest","balanced","impartial","evenhanded","reciproc","deserv","merit","lawful","rights"],
    "fairness_vice":   ["unfair","unjust","biased","bias","discriminat","prejudice","inequal","inequit","cheat","scam","exploit","rob","steal","stole","deceiv","deceit","dishonest","fraud","wrong"],
    "loyalty_virtue":  ["loyal","loyalty","family","families","ally","allies","patriot","together","unite","united","communit","team","comrade","brother","sister","fellow","kin","tribe","nation","homeland","solidar"],
    "loyalty_vice":    ["betray","traitor","disloyal","foreign","foreigner","enemy","enemies","outsider","deserter","rebel","renegade","abandon","desert","treason"],
    "authority_virtue":["obey","obedien","respect","tradition","hierarchy","leader","rank","authority","authorit","duty","honor","honour","superior","command","comply","law","rule"],
    "authority_vice":  ["defy","disobey","rebel","rebellion","insubordinat","disrespect","subver","anarchy","betray","dissent"],
    "sanctity_virtue": ["pure","purity","holy","sacred","saint","virtue","virtu","modest","decent","chaste","innocent","clean","wholesome"],
    "sanctity_vice":   ["dirty","filth","contamin","disgust","gross","vile","sin","sinful","obscen","perver","promiscu","debauch","profane","depraved","vulgar","impure"],
}
MFD_PAIRS = [(stem, fnd) for fnd, stems in MFD.items() for stem in stems]

logger = logging.getLogger("milestone3_more")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg)
    sys.stdout.flush()


def load_embeddings():
    out = {}
    for src in ALL_SOURCES:
        out[src] = np.load(EMBEDDINGS_DIR / f"{src}.npy")
        _flush(f"loaded {src} {out[src].shape}")
    return out


def load_meta(src):
    return json.loads((EMBEDDINGS_DIR / f"{src}_meta.json").read_text())


def comp90(X):
    pca = PCA(svd_solver="full")
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    eig = pca.explained_variance_
    c = int(np.searchsorted(cum, 0.90) + 1)
    pr = float((eig.sum() ** 2) / (eig ** 2).sum())
    return c, pr


def comp90_random(X, k=500):
    k = min(k, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=RANDOM_SEED)
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    if cum[-1] < 0.90:
        return k, float((pca.explained_variance_.sum()**2) / (pca.explained_variance_**2).sum())
    return int(np.searchsorted(cum, 0.90) + 1), float((pca.explained_variance_.sum()**2) / (pca.explained_variance_**2).sum())


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_dist(a, b):
    return 1.0 - cosine(a, b)


# --- 1. Within-dilemma LLM diversity vs human-LLM separation -------------------
def within_dilemma_diversity(embeddings):
    _flush("=== A. Within-dilemma LLM clustering ===")

    # Build per-dilemma index: submission_id -> {source: [embeddings]}
    per_dilemma = defaultdict(lambda: {"human": [], "llm": []})
    for src in ALL_SOURCES:
        meta = load_meta(src)
        E = embeddings[src]
        for m in meta:
            sub_id = m["submission_id"]
            tag = "human" if src == "human" else "llm"
            per_dilemma[sub_id][tag].append(E[m["index"]])
    _flush(f"per-dilemma index: {len(per_dilemma)} submissions")

    # Per-dilemma metrics
    metrics = []
    for sub_id, srcs in per_dilemma.items():
        if len(srcs["human"]) == 0 or len(srcs["llm"]) < 2:
            continue
        human = srcs["human"][0]  # 1 per dilemma
        llms = np.stack(srcs["llm"])  # (k, 2048), k = 14-21
        llm_centroid = llms.mean(axis=0)

        # Within-LLM pairwise cosine distance
        n = len(llms)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append(cosine_dist(llms[i], llms[j]))
        mean_within_llm = float(np.mean(pairs)) if pairs else float("nan")

        # Human-to-LLM-centroid distance
        h_to_llm = cosine_dist(human, llm_centroid)

        # Human-to-each-LLM distance, mean
        mean_h_to_each_llm = float(np.mean([cosine_dist(human, l) for l in llms]))

        metrics.append({
            "sub_id": sub_id,
            "n_llms": n,
            "mean_within_llm_dist": mean_within_llm,
            "human_to_llm_centroid_dist": float(h_to_llm),
            "mean_human_to_each_llm_dist": mean_h_to_each_llm,
        })

    # Summary
    arr_within = np.array([m["mean_within_llm_dist"] for m in metrics])
    arr_h2c = np.array([m["human_to_llm_centroid_dist"] for m in metrics])
    arr_h2e = np.array([m["mean_human_to_each_llm_dist"] for m in metrics])
    n = len(metrics)
    frac_human_outside = float(np.mean(arr_h2c > arr_within))

    _flush(f"n dilemmas analyzed: {n}")
    _flush(f"mean within-LLM cosine dist: {arr_within.mean():.4f} ± {arr_within.std():.4f}")
    _flush(f"mean human-to-LLM-centroid dist: {arr_h2c.mean():.4f} ± {arr_h2c.std():.4f}")
    _flush(f"mean human-to-each-LLM dist: {arr_h2e.mean():.4f} ± {arr_h2e.std():.4f}")
    _flush(f"fraction of dilemmas where human is farther from LLM centroid than within-LLM mean: {frac_human_outside:.3f}")

    out = {
        "n_dilemmas": int(n),
        "mean_within_llm_pairwise_cosine_dist": float(arr_within.mean()),
        "std_within_llm_pairwise_cosine_dist": float(arr_within.std()),
        "mean_human_to_llm_centroid_dist": float(arr_h2c.mean()),
        "std_human_to_llm_centroid_dist": float(arr_h2c.std()),
        "mean_human_to_each_llm_dist": float(arr_h2e.mean()),
        "fraction_human_outside_llm_cluster": frac_human_outside,
        "ratio_h2c_over_within_mean": float(arr_h2c.mean() / arr_within.mean()),
    }
    path = OUTPUT_DIR / "milestone3_within_dilemma.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")
    return out


# --- 2. Verdict-conditional gap ------------------------------------------------
def verdict_stratified_gap(embeddings):
    _flush("=== B. Verdict-conditional gap ===")
    from datasets import load_dataset
    ds = load_dataset(
        "ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas", split="test"
    )

    # Build sub_id -> dominant verdict (NTA/YTA/ESH/NAH)
    cols = ["comments_nta_agreement_weighted", "comments_yta_agreement_weighted",
            "comments_esh_agreement_weighted", "comments_nah_agreement_weighted"]
    labels = ["NTA", "YTA", "ESH", "NAH"]
    sub_to_verdict = {}
    sub_ids_col = ds["submission_id"]
    cols_data = [ds[c] for c in cols]
    for i, sub_id in enumerate(sub_ids_col):
        scores = [cols_data[k][i] for k in range(len(cols))]
        scores = [(0.0 if s is None else float(s)) for s in scores]
        idx = int(np.argmax(scores))
        sub_to_verdict[sub_id] = labels[idx]

    # Group human + LLM embeddings by verdict
    human_meta = load_meta("human")
    X_human = embeddings["human"]
    by_verdict_human = defaultdict(list)
    for m in human_meta:
        v = sub_to_verdict.get(m["submission_id"])
        if v:
            by_verdict_human[v].append(X_human[m["index"]])

    # all-LLM pool per verdict
    by_verdict_llm = defaultdict(list)
    for src in LLM_SOURCES:
        meta = load_meta(src)
        E = embeddings[src]
        for m in meta:
            v = sub_to_verdict.get(m["submission_id"])
            if v:
                by_verdict_llm[v].append(E[m["index"]])

    # Compute comp90 per verdict at matched-n
    rng = np.random.RandomState(RANDOM_SEED)
    out_per_verdict = {}
    for v in labels:
        Xh = np.stack(by_verdict_human[v]) if by_verdict_human[v] else None
        Xl = np.stack(by_verdict_llm[v]) if by_verdict_llm[v] else None
        if Xh is None or Xl is None:
            continue
        nh, nl = Xh.shape[0], Xl.shape[0]
        n_match = min(nh, nl, 5000)  # cap for tractability
        h_idx = rng.choice(nh, n_match, replace=False)
        l_idx = rng.choice(nl, n_match, replace=False)
        Xh_s = Xh[h_idx]
        Xl_s = Xl[l_idx]
        ch, prh = comp90(Xh_s) if n_match <= 2000 else comp90_random(Xh_s)
        cl, prl = comp90(Xl_s) if n_match <= 2000 else comp90_random(Xl_s)
        # also: human centroid vs LLM centroid distance
        h_cent = Xh_s.mean(axis=0)
        l_cent = Xl_s.mean(axis=0)
        cent_dist = cosine_dist(h_cent, l_cent)
        out_per_verdict[v] = {
            "n_human_total": int(nh),
            "n_llm_total": int(nl),
            "matched_n": int(n_match),
            "human_comp90": int(ch),
            "human_pr": float(prh),
            "llm_comp90": int(cl),
            "llm_pr": float(prl),
            "gap": int(ch - cl),
            "human_llm_centroid_cosine_dist": float(cent_dist),
        }
        _flush(f"  verdict={v}: n_h={nh}, n_l={nl}, matched_n={n_match}, human_comp90={ch}, llm_comp90={cl}, gap={ch-cl}, cent_dist={cent_dist:.3f}")

    out = {"verdicts": out_per_verdict}
    path = OUTPUT_DIR / "milestone3_verdict_gap.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")
    return out


# --- 3. MFD length-stratified --------------------------------------------------
def mfd_length_stratified(embeddings):
    _flush("=== C. MFD length-stratified ===")
    from datasets import load_dataset
    ds = load_dataset(
        "ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas", split="test"
    )

    # Reload texts for human only (need length)
    sub_ids = ds["submission_id"]
    top_comments = ds["top_comment"]
    sub_to_text = {sub_ids[i]: top_comments[i] for i in range(len(sub_ids))}

    human_meta = load_meta("human")
    X_human = embeddings["human"]
    texts = []
    for m in human_meta:
        t = sub_to_text.get(m["submission_id"], "") or ""
        texts.append(t)
    n = len(texts)
    word_counts = np.array([len(t.split()) for t in texts])
    _flush(f"n={n}, word counts: min={word_counts.min()}, "
           f"q25={np.percentile(word_counts, 25):.0f}, "
           f"q50={np.percentile(word_counts, 50):.0f}, "
           f"q75={np.percentile(word_counts, 75):.0f}, "
           f"max={word_counts.max()}")

    # Compute reconstruction error under all-LLM PCs at k=448 (milestone-2 setup, randomized for speed)
    all_llm = np.vstack([embeddings[s] for s in LLM_SOURCES])
    _flush(f"fitting PCA k=448 on all-LLM (n={all_llm.shape[0]}, randomized)")
    pca_llm = PCA(n_components=448, svd_solver="randomized", random_state=RANDOM_SEED)
    pca_llm.fit(all_llm)
    proj = pca_llm.transform(X_human)
    recon = pca_llm.inverse_transform(proj)
    err = np.linalg.norm(X_human - recon, axis=1) ** 2

    # MFD counts per rationale
    foundations = list(MFD.keys())
    f_to_idx = {f: i for i, f in enumerate(foundations)}
    counts = np.zeros((n, len(foundations)), dtype=np.int32)
    for i, t in enumerate(texts):
        text_lc = t.lower()
        toks = re.findall(r"\b[a-z]+\b", text_lc)
        for tok in toks:
            for stem, fnd in MFD_PAIRS:
                if tok.startswith(stem):
                    counts[i, f_to_idx[fnd]] += 1
                    break
    _flush("MFD counts done")

    # Stratify by length quartile
    edges = [np.percentile(word_counts, q) for q in (25, 50, 75)]
    strata_labels = ["Q1 (<25th)", "Q2 (25-50th)", "Q3 (50-75th)", "Q4 (>75th)"]

    out_strata = {}
    for q in range(4):
        if q == 0:
            mask = word_counts < edges[0]
        elif q == 1:
            mask = (word_counts >= edges[0]) & (word_counts < edges[1])
        elif q == 2:
            mask = (word_counts >= edges[1]) & (word_counts < edges[2])
        else:
            mask = word_counts >= edges[2]
        idx = np.where(mask)[0]
        if len(idx) < 200:
            continue
        err_q = err[idx]
        counts_q = counts[idx]
        # Sort by error within stratum, sliding window
        order = np.argsort(err_q)
        err_sorted = err_q[order]
        counts_sorted = counts_q[order]
        w = 100
        step = 25
        bin_uf = []
        bin_err = []
        bin_ent = []
        for start in range(0, len(idx) - w + 1, step):
            end = start + w
            bin_err.append(float(err_sorted[start:end].mean()))
            cnt_bin = counts_sorted[start:end]
            any_match = (cnt_bin > 0)
            used = (any_match.sum(axis=0) > 0).sum()
            bin_uf.append(int(used))
            tot = cnt_bin.sum(axis=0)
            tot_sum = tot.sum()
            if tot_sum > 0:
                p = tot / tot_sum
                bin_ent.append(float(-np.sum(p * np.log(p + 1e-12))))
            else:
                bin_ent.append(float("nan"))

        bin_err_arr = np.array(bin_err)
        bin_uf_arr = np.array(bin_uf, dtype=float)
        bin_ent_arr = np.array(bin_ent)

        if len(bin_err_arr) >= 3:
            corr_uf = float(np.corrcoef(bin_err_arr, bin_uf_arr)[0, 1])
            valid = ~np.isnan(bin_ent_arr)
            if valid.sum() >= 3:
                corr_ent = float(np.corrcoef(bin_err_arr[valid], bin_ent_arr[valid])[0, 1])
            else:
                corr_ent = float("nan")
        else:
            corr_uf = float("nan")
            corr_ent = float("nan")

        out_strata[strata_labels[q]] = {
            "n_rationales": int(len(idx)),
            "n_with_match": int((counts_q.sum(axis=1) > 0).sum()),
            "mean_word_count": float(word_counts[idx].mean()),
            "corr_used_foundations": corr_uf,
            "corr_bin_entropy": corr_ent,
            "n_bins": int(len(bin_err_arr)),
        }
        _flush(f"  {strata_labels[q]}: n={len(idx)}, mean_words={word_counts[idx].mean():.1f}, "
               f"corr_uf={corr_uf:.3f}, corr_ent={corr_ent:.3f}")

    # Also: MFD count per rationale (sum across foundations) vs reconstruction error
    total_mfd = counts.sum(axis=1)
    rho_count_err = float(np.corrcoef(total_mfd, err)[0, 1])
    rho_count_len = float(np.corrcoef(total_mfd, word_counts)[0, 1])
    rho_err_len = float(np.corrcoef(err, word_counts)[0, 1])
    _flush(f"corr(total_mfd_count, recon_err) = {rho_count_err:.3f}")
    _flush(f"corr(total_mfd_count, word_count) = {rho_count_len:.3f}")
    _flush(f"corr(recon_err, word_count) = {rho_err_len:.3f}")

    out = {
        "raw_rationale_correlations": {
            "total_mfd_vs_recon_err": rho_count_err,
            "total_mfd_vs_word_count": rho_count_len,
            "recon_err_vs_word_count": rho_err_len,
        },
        "by_length_quartile": out_strata,
    }
    path = OUTPUT_DIR / "milestone3_mfd_length_stratified.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all")
    args = ap.parse_args()
    steps = (
        {"within", "verdict", "mfdlen"}
        if args.steps == "all"
        else set(args.steps.split(","))
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = load_embeddings()
    if "within" in steps:
        within_dilemma_diversity(embeddings)
    if "verdict" in steps:
        verdict_stratified_gap(embeddings)
    if "mfdlen" in steps:
        mfd_length_stratified(embeddings)
    _flush("ALL DONE")


if __name__ == "__main__":
    main()
