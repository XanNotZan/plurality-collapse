"""Milestone 3 extra: TF-IDF residualization, MFD gradient, T=20 INLP, per-PC attribution."""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score

EMBEDDINGS_DIR = Path("data/embeddings")
OUTPUT_DIR = Path("data/analysis")
HIDDEN_DIM = 2048
ALL_SOURCES = ["human", "gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
MATCHED_N = 10826
RANDOM_SEED = 42

# -- Moral Foundations Dictionary (compact, stem-based) -------------------------
# Adapted from Graham, Haidt & Nosek (2009) MFD; entries are word stems matched as prefixes.
MFD = {
    "care_virtue": [
        "safe", "peace", "compassion", "empath", "kind", "care", "caring",
        "protect", "shelter", "save", "guard", "defend", "shield", "nurtur",
        "support", "help", "rescue", "secur", "comfort", "sympath", "mercy",
        "gentle", "tender", "warmth", "love", "loving",
    ],
    "care_vice": [
        "harm", "hurt", "suffer", "pain", "kill", "murder", "abuse", "abus",
        "cruel", "brutal", "violen", "wound", "damage", "destroy", "attack",
        "assault", "bully", "torment", "torture", "victim", "agony",
        "mistreat", "punish", "neglect",
    ],
    "fairness_virtue": [
        "fair", "fairness", "equal", "equit", "justice", "just",
        "righteous", "honest", "balanced", "impartial", "evenhanded",
        "reciproc", "deserv", "merit", "lawful", "rights",
    ],
    "fairness_vice": [
        "unfair", "unjust", "biased", "bias", "discriminat", "prejudice",
        "inequal", "inequit", "cheat", "scam", "exploit", "rob",
        "steal", "stole", "deceiv", "deceit", "dishonest", "fraud", "wrong",
    ],
    "loyalty_virtue": [
        "loyal", "loyalty", "family", "families", "ally", "allies",
        "patriot", "together", "unite", "united", "communit", "team",
        "comrade", "brother", "sister", "fellow", "kin", "tribe",
        "nation", "homeland", "solidar",
    ],
    "loyalty_vice": [
        "betray", "traitor", "disloyal", "foreign", "foreigner",
        "enemy", "enemies", "outsider", "deserter", "rebel", "renegade",
        "abandon", "desert", "treason",
    ],
    "authority_virtue": [
        "obey", "obedien", "respect", "tradition", "hierarchy", "leader",
        "rank", "authority", "authorit", "duty", "honor", "honour",
        "superior", "command", "comply", "law", "rule",
    ],
    "authority_vice": [
        "defy", "disobey", "rebel", "rebellion", "insubordinat",
        "disrespect", "subver", "anarchy", "betray", "dissent",
    ],
    "sanctity_virtue": [
        "pure", "purity", "holy", "sacred", "saint", "virtue", "virtu",
        "modest", "decent", "chaste", "innocent", "clean", "wholesome",
    ],
    "sanctity_vice": [
        "dirty", "filth", "contamin", "disgust", "gross", "vile",
        "sin", "sinful", "obscen", "perver", "promiscu", "debauch",
        "profane", "depraved", "vulgar", "impure",
    ],
}
# Flattened: (stem, foundation_name)
MFD_PAIRS = [(stem, fnd) for fnd, stems in MFD.items() for stem in stems]


logger = logging.getLogger("milestone3_extra")
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
        _flush(f"loaded {src} shape {out[src].shape}")
    return out


def load_meta(src):
    return json.loads((EMBEDDINGS_DIR / f"{src}_meta.json").read_text())


def matched_llm_pool(embeddings, n=MATCHED_N, seed=RANDOM_SEED):
    parts = []
    src_tags = []
    row_indices = []
    for src in LLM_SOURCES:
        E = embeddings[src]
        parts.append(E)
        src_tags.extend([src] * E.shape[0])
        row_indices.extend(list(range(E.shape[0])))
    pool = np.vstack(parts)
    rng = np.random.RandomState(seed)
    idx = rng.choice(pool.shape[0], n, replace=False)
    return pool[idx], idx, np.array(src_tags), np.array(row_indices)


def load_texts():
    from datasets import load_dataset
    _flush("loading HF dataset")
    ds = load_dataset(
        "ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas", split="test"
    )
    needed_cols = {"top_comment"}
    llm_cols_per_src = {
        "gpt3.5":  ["gpt3.5_reason_1", "gpt3.5_reason_2", "gpt3.5_reason_3"],
        "gpt4":    ["gpt4_reason_1", "gpt4_reason_2"],
        "claude":  ["claude_reason_1", "claude_reason_2", "claude_reason_3"],
        "bison":   ["bison_reason_1", "bison_reason_2", "bison_reason_3"],
        "gemma":   ["gemma_reason_1", "gemma_reason_2", "gemma_reason_3"],
        "mistral": ["mistral_reason_1", "mistral_reason_2", "mistral_reason_3"],
        "llama":   ["llama_reason_1", "llama_reason_2", "llama_reason_3"],
    }
    for cols in llm_cols_per_src.values():
        needed_cols.update(cols)
    sub_ids_col = ds["submission_id"]
    col_data = {c: ds[c] for c in needed_cols if c in ds.column_names}
    sub_to_cols = {}
    for i, sub_id in enumerate(sub_ids_col):
        sub_to_cols[sub_id] = {c: col_data[c][i] for c in col_data}
    _flush(f"HF: {len(sub_to_cols)} ids indexed")
    texts = {}
    for src in ALL_SOURCES:
        meta = load_meta(src)
        out = []
        for m in meta:
            row = sub_to_cols[m["submission_id"]]
            text = row.get(m["column"], "")
            out.append(text if isinstance(text, str) else "")
        texts[src] = out
        _flush(f"texts {src}: {len(out)}")
    return texts


def matched_llm_text_pool(texts, idx, src_tags, row_indices):
    out = []
    for i in idx:
        s = src_tags[i]
        r = row_indices[i]
        out.append(texts[s][r])
    return out


def comp90_pr(X):
    pca = PCA(svd_solver="full")
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    eig = pca.explained_variance_
    comp90 = int(np.searchsorted(cum, 0.90) + 1)
    pr = float((eig.sum() ** 2) / (eig ** 2).sum())
    return comp90, pr, eig, cum


# --- A. TF-IDF + length residualized gap ----------------------------------------
def tfidf_residualize(embeddings, texts):
    _flush("=== A. TF-IDF + length residualized gap ===")
    X_human = embeddings["human"]
    X_llm_matched, llm_idx, llm_src_tags, llm_row_idx = matched_llm_pool(embeddings)
    texts_human = texts["human"]
    texts_llm = matched_llm_text_pool(texts, llm_idx, llm_src_tags, llm_row_idx)

    X = np.vstack([X_human, X_llm_matched])
    txt = list(texts_human) + list(texts_llm)
    n_h = len(texts_human)
    _flush(f"X shape {X.shape}, n_human {n_h}")

    # Build dense design matrix: top-2K TF-IDF + length features
    # (full 10K TF-IDF + sparse_cg Ridge with 2048 outputs was too slow on this hardware)
    vec = TfidfVectorizer(
        max_features=2000, ngram_range=(1, 2), min_df=5,
        sublinear_tf=True, token_pattern=r"(?u)\b\w+\b",
    )
    F_tfidf = vec.fit_transform(txt).toarray().astype(np.float64)
    log_len = np.log1p(np.array([[len(t.split()), len(t)] for t in txt], dtype=float))
    F = np.hstack([F_tfidf, log_len])
    _flush(f"design matrix F shape {F.shape} (dense)")

    # Center F columns (so the residual respects column-mean offsets in X)
    F_mean = F.mean(axis=0)
    F = F - F_mean

    # Solve closed-form Ridge: W = (F^T F + αI)^-1 F^T X
    _flush("fitting Ridge via Cholesky")
    t0 = time.time()
    alpha = 10.0
    p = F.shape[1]
    G = F.T @ F + alpha * np.eye(p)
    rhs = F.T @ X
    W = np.linalg.solve(G, rhs)
    _flush(f"Ridge fit in {time.time()-t0:.1f}s, W shape {W.shape}")

    X_pred = F @ W
    X_resid = X - X_pred
    _flush(f"residualized: X_resid shape {X_resid.shape}, "
           f"frac variance preserved = {X_resid.var() / X.var():.4f}")

    # Residual classifier: how well can a logreg on RESIDUALIZED embeddings still discriminate?
    y = np.concatenate([np.zeros(n_h), np.ones(X.shape[0] - n_h)])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_resid, y)
    acc_resid = accuracy_score(y, clf.predict(X_resid))
    clf2 = LogisticRegression(max_iter=2000, C=1.0)
    clf2.fit(X, y)
    acc_orig = accuracy_score(y, clf2.predict(X))
    _flush(f"classifier acc: original={acc_orig:.4f} residualized={acc_resid:.4f}")

    # Recompute comp90 + PR on residualized human and LLM separately
    _flush("PCA on residualized human")
    Xh_resid = X_resid[:n_h]
    Xl_resid = X_resid[n_h:]
    ch, prh, _, _ = comp90_pr(Xh_resid)
    cl, prl, _, _ = comp90_pr(Xl_resid)

    # Baseline (no residualization) for comparison
    cb_h, prb_h, _, _ = comp90_pr(X_human)
    cb_l, prb_l, _, _ = comp90_pr(X_llm_matched)

    # Frac of variance removed per source
    frac_var_human_removed = 1 - Xh_resid.var() / X_human.var()
    frac_var_llm_removed = 1 - Xl_resid.var() / X_llm_matched.var()

    out = {
        "matched_n": int(MATCHED_N),
        "n_features": int(F.shape[1]),
        "ridge_alpha": 10.0,
        "baseline": {
            "human_comp90": int(cb_h), "human_pr": float(prb_h),
            "llm_comp90": int(cb_l), "llm_pr": float(prb_l),
            "gap": int(cb_h - cb_l),
            "classifier_acc": float(acc_orig),
        },
        "residualized": {
            "human_comp90": int(ch), "human_pr": float(prh),
            "llm_comp90": int(cl), "llm_pr": float(prl),
            "gap": int(ch - cl),
            "classifier_acc": float(acc_resid),
            "frac_var_human_removed": float(frac_var_human_removed),
            "frac_var_llm_removed": float(frac_var_llm_removed),
        },
    }
    path = OUTPUT_DIR / "milestone3_tfidf_residualized.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")
    return out


# --- B. MFD-based moral diversity gradient -------------------------------------
def mfd_count_per_text(text):
    """Return dict: foundation_name -> match count."""
    text_lc = text.lower() if isinstance(text, str) else ""
    counts = {fnd: 0 for fnd in MFD}
    if not text_lc:
        return counts
    # Tokenize by simple word boundaries
    tokens = re.findall(r"\b[a-z]+\b", text_lc)
    for tok in tokens:
        for stem, fnd in MFD_PAIRS:
            if tok.startswith(stem):
                counts[fnd] += 1
                break  # one stem per token (avoid double-count)
    return counts


def mfd_diversity_gradient(embeddings, texts):
    _flush("=== B. MFD diversity gradient ===")
    # Need: per-rationale reconstruction error for human under LLM PCs
    X_human = embeddings["human"]
    all_llm = np.vstack([embeddings[s] for s in LLM_SOURCES])

    # Fit PCA on all-LLM at k=448 to match milestone 2 (matches Layer 2 setup)
    k = 448
    _flush(f"fitting PCA on all-LLM (k={k}, n={all_llm.shape[0]}) - randomized")
    pca_llm = PCA(n_components=k, svd_solver="randomized", random_state=RANDOM_SEED)
    pca_llm.fit(all_llm)
    _flush("computing reconstruction error for human")
    proj = pca_llm.transform(X_human)
    recon = pca_llm.inverse_transform(proj)
    err = np.linalg.norm(X_human - recon, axis=1) ** 2
    _flush(f"recon error: mean={err.mean():.2f} std={err.std():.2f}")

    # MFD counts per rationale
    _flush("computing MFD counts for human texts")
    texts_human = texts["human"]
    # Vectorize counts
    count_arr = np.zeros((len(texts_human), len(MFD)), dtype=np.int32)
    foundations = list(MFD.keys())
    f_to_idx = {f: i for i, f in enumerate(foundations)}
    for i, t in enumerate(texts_human):
        c = mfd_count_per_text(t)
        for f, v in c.items():
            count_arr[i, f_to_idx[f]] = v
        if (i + 1) % 2000 == 0:
            _flush(f"MFD: {i+1}/{len(texts_human)}")

    # Foundation diversity per rationale: Shannon entropy
    sums = count_arr.sum(axis=1)
    rationales_with_any = (sums > 0).sum()
    _flush(f"rationales with >=1 MFD match: {rationales_with_any}/{len(texts_human)}")

    # Compute Shannon entropy per rationale (over foundation distribution)
    eps = 1e-12
    p = count_arr / (sums[:, None] + eps)
    ent = -np.sum(p * np.log(p + eps), axis=1)
    ent = np.where(sums > 0, ent, np.nan)

    # Sliding-window correlation: sort by reconstruction error, bin
    order = np.argsort(err)
    err_sorted = err[order]
    counts_sorted = count_arr[order]
    ent_sorted = ent[order]
    sums_sorted = sums[order]

    w = 200
    step = 50
    n = len(order)
    bins = []
    for start in range(0, n - w + 1, step):
        end = start + w
        bin_err = err_sorted[start:end].mean()
        # Foundation diversity at bin level: count distinct foundations used (>=1 hit) AND mean per-rationale entropy
        bin_counts = counts_sorted[start:end]  # (w, K)
        any_match = (bin_counts > 0)
        used_foundations = (any_match.sum(axis=0) > 0).sum()
        mean_ent = float(np.nanmean(ent_sorted[start:end]))
        # Bin-level foundation distribution
        bin_total_per_fnd = bin_counts.sum(axis=0)
        bin_total = bin_total_per_fnd.sum()
        if bin_total > 0:
            p_bin = bin_total_per_fnd / bin_total
            bin_ent = float(-np.sum(p_bin * np.log(p_bin + eps)))
        else:
            bin_ent = float("nan")
        bins.append({
            "bin_start": start,
            "mean_recon_err": float(bin_err),
            "used_foundations": int(used_foundations),
            "mean_per_rationale_entropy": mean_ent,
            "bin_level_entropy": bin_ent,
            "rationales_with_match": int(any_match.any(axis=1).sum()),
        })

    # Correlations
    bin_err = np.array([b["mean_recon_err"] for b in bins])
    bin_uf = np.array([b["used_foundations"] for b in bins], dtype=float)
    bin_ent_vec = np.array([b["bin_level_entropy"] for b in bins])
    valid = ~np.isnan(bin_ent_vec)

    corr_uf = float(np.corrcoef(bin_err, bin_uf)[0, 1])
    corr_ent = float(np.corrcoef(bin_err[valid], bin_ent_vec[valid])[0, 1])
    _flush(f"MFD bin correlations: used_foundations r={corr_uf:.3f}, "
           f"bin_entropy r={corr_ent:.3f}")

    # Per-foundation frequency in low-error vs high-error halves
    half = n // 2
    low_err_fnd = counts_sorted[:half].sum(axis=0)
    high_err_fnd = counts_sorted[half:].sum(axis=0)
    foundation_shift = {
        f: {
            "low_err_count": int(low_err_fnd[f_to_idx[f]]),
            "high_err_count": int(high_err_fnd[f_to_idx[f]]),
        }
        for f in foundations
    }

    out = {
        "n_rationales": int(len(texts_human)),
        "rationales_with_any_match": int(rationales_with_any),
        "k_pca": k,
        "window": w, "step": step,
        "bin_correlation_used_foundations": corr_uf,
        "bin_correlation_bin_entropy": corr_ent,
        "foundation_shift_low_vs_high_err": foundation_shift,
        "bins": bins,
    }
    path = OUTPUT_DIR / "milestone3_mfd_gradient.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")
    return out


# --- C. Higher-T INLP ----------------------------------------------------------
def higher_t_inlp(embeddings, n_iter=20):
    _flush(f"=== C. Higher-T INLP (T={n_iter}) ===")
    X_human = embeddings["human"]
    X_llm_matched, _, _, _ = matched_llm_pool(embeddings)
    n_h = X_human.shape[0]
    X = np.vstack([X_human, X_llm_matched])
    y = np.concatenate([np.zeros(n_h), np.ones(X_llm_matched.shape[0])])

    Xc = X.copy().astype(np.float64)
    accs = []
    P_total = np.eye(X.shape[1])

    # Track comp90 every 5 iters
    sample_iters = list(range(0, n_iter + 1, 5))
    comp_track = {}

    for it in range(n_iter):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(Xc, y)
        acc = accuracy_score(y, clf.predict(Xc))
        accs.append(float(acc))
        if it in sample_iters:
            # Track comp90 on residualized full sources
            Xh_resid = X_human @ P_total
            Xl_resid = X_llm_matched @ P_total
            ch, _, _, _ = comp90_pr(Xh_resid)
            cl, _, _, _ = comp90_pr(Xl_resid)
            comp_track[it] = {"human_comp90": int(ch), "llm_comp90": int(cl), "gap": int(ch - cl)}
            _flush(f"[INLP T={n_iter}] iter {it}: acc={acc:.4f} h={ch} l={cl} gap={ch-cl}")
        else:
            _flush(f"[INLP T={n_iter}] iter {it}: acc={acc:.4f}")

        w = clf.coef_[0]
        wn = np.linalg.norm(w)
        if wn < 1e-10:
            _flush("zero coef - stopping")
            break
        wh = w / wn
        proj = Xc @ wh
        Xc = Xc - np.outer(proj, wh)
        P_iter = np.eye(X.shape[1]) - np.outer(wh, wh)
        P_total = P_total @ P_iter

    # Final
    Xh_resid = X_human @ P_total
    Xl_resid = X_llm_matched @ P_total
    ch, _, _, _ = comp90_pr(Xh_resid)
    cl, _, _, _ = comp90_pr(Xl_resid)
    comp_track[n_iter] = {"human_comp90": int(ch), "llm_comp90": int(cl), "gap": int(ch - cl)}
    _flush(f"[INLP T={n_iter}] final h={ch} l={cl} gap={ch-cl}")

    out = {
        "n_iter": n_iter,
        "classifier_acc_per_iter": accs,
        "comp90_track": {str(k): v for k, v in comp_track.items()},
    }
    path = OUTPUT_DIR / "milestone3_inlp_T20.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")
    return out


# --- D. Per-PC source-variance attribution -------------------------------------
def per_pc_attribution(embeddings, n_components=500):
    _flush("=== D. Per-PC source-variance attribution ===")
    X_human = embeddings["human"]
    X_llm_matched, _, _, _ = matched_llm_pool(embeddings)
    X_pool = np.vstack([X_human, X_llm_matched])
    n_h = X_human.shape[0]
    n_l = X_llm_matched.shape[0]

    # Fit PCA on pooled (matched-n) - equal-weight basis
    _flush(f"fitting pooled PCA k={n_components}")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=RANDOM_SEED)
    pca.fit(X_pool)

    proj_h = pca.transform(X_human)  # (n_h, k)
    proj_l = pca.transform(X_llm_matched)  # (n_l, k)

    # Per-PC variance per source
    var_h = proj_h.var(axis=0)
    var_l = proj_l.var(axis=0)
    ratio = var_h / (var_l + 1e-12)
    diff = var_h - var_l

    # Cumulative variance for each source along pooled basis
    cum_h_norm = np.cumsum(var_h) / var_h.sum()
    cum_l_norm = np.cumsum(var_l) / var_l.sum()
    comp90_h = int(np.searchsorted(cum_h_norm, 0.90) + 1)
    comp90_l = int(np.searchsorted(cum_l_norm, 0.90) + 1)

    # Identify "human-spread" PCs (where ratio > 1.5) and their position
    human_spread_pcs = np.where(ratio > 1.5)[0]
    llm_spread_pcs = np.where(ratio < 0.67)[0]
    _flush(f"PCs where human var > 1.5x LLM: {len(human_spread_pcs)} (out of {n_components})")
    _flush(f"PCs where LLM var > 1.5x human: {len(llm_spread_pcs)}")
    if len(human_spread_pcs) > 0:
        _flush(f"  first such PC index: {human_spread_pcs[0]}")
        _flush(f"  last such PC index: {human_spread_pcs[-1]}")

    out = {
        "n_components": n_components,
        "comp90_along_pooled_basis": {"human": comp90_h, "llm": comp90_l, "gap": comp90_h - comp90_l},
        "var_human": var_h.tolist(),
        "var_llm": var_l.tolist(),
        "ratio_h_over_l": ratio.tolist(),
        "n_pcs_human_dominated_1p5x": int(len(human_spread_pcs)),
        "n_pcs_llm_dominated_1p5x": int(len(llm_spread_pcs)),
        "first_human_dominated_pc": int(human_spread_pcs[0]) if len(human_spread_pcs) else -1,
        "last_human_dominated_pc": int(human_spread_pcs[-1]) if len(human_spread_pcs) else -1,
        "median_ratio_top50": float(np.median(ratio[:50])),
        "median_ratio_pcs_50_200": float(np.median(ratio[50:200])),
        "median_ratio_pcs_200_500": float(np.median(ratio[200:500])),
    }
    path = OUTPUT_DIR / "milestone3_per_pc_attribution.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")
    return out


# --- Main ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all", help="comma-separated: tfidf,mfd,inlp_t20,perpc,all")
    args = ap.parse_args()
    steps = (
        {"tfidf", "mfd", "inlp_t20", "perpc"}
        if args.steps == "all"
        else set(args.steps.split(","))
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = load_embeddings()
    texts = None
    if "tfidf" in steps or "mfd" in steps:
        texts = load_texts()

    if "tfidf" in steps:
        tfidf_residualize(embeddings, texts)
    if "mfd" in steps:
        mfd_diversity_gradient(embeddings, texts)
    if "inlp_t20" in steps:
        higher_t_inlp(embeddings, n_iter=20)
    if "perpc" in steps:
        per_pc_attribution(embeddings, n_components=500)

    _flush("ALL DONE")


if __name__ == "__main__":
    main()
