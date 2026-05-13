"""Milestone 3 diagnostics: style classifier, INLP residualized gap, permutation null."""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

EMBEDDINGS_DIR = Path("data/embeddings")
OUTPUT_DIR = Path("data/analysis")
HIDDEN_DIM = 2048

ALL_SOURCES = ["human", "gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
MATCHED_N = 10826
RANDOM_SEED = 42
N_PERMUTATIONS = 100
INLP_ITERATIONS = 5

logger = logging.getLogger("milestone3")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    """Print + flush so monitor sees lines immediately."""
    logger.info(msg)
    sys.stdout.flush()


def load_embeddings():
    out = {}
    for src in ALL_SOURCES:
        path = EMBEDDINGS_DIR / f"{src}.npy"
        out[src] = np.load(path)
        _flush(f"loaded {src} shape {out[src].shape}")
    return out


def load_meta(src):
    return json.loads((EMBEDDINGS_DIR / f"{src}_meta.json").read_text())


def load_texts():
    """Load raw rationale text per source via HF dataset, aligned to embedding row order."""
    from datasets import load_dataset
    _flush("loading HF dataset")
    ds = load_dataset(
        "ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas", split="test"
    )

    # Determine which columns we need
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

    # Build sub_id -> {col_name: text} dict in one pass over ds
    _flush("indexing HF dataset rows")
    sub_to_cols = {}
    sub_ids_col = ds["submission_id"]
    col_data = {c: ds[c] for c in needed_cols if c in ds.column_names}
    for i, sub_id in enumerate(sub_ids_col):
        sub_to_cols[sub_id] = {c: col_data[c][i] for c in col_data}
    _flush(f"HF dataset: {len(sub_to_cols)} unique submission_ids indexed")

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


def matched_llm_pool(embeddings, n=MATCHED_N, seed=RANDOM_SEED):
    """Concatenate all LLM embeddings then subsample n rows. Returns (X, idx_map)."""
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


def matched_llm_text_pool(texts, idx, src_tags, row_indices):
    """Recover text in same order as the matched LLM pool."""
    out = []
    for i in idx:
        s = src_tags[i]
        r = row_indices[i]
        out.append(texts[s][r])
    return out


def comp90_pr(X):
    """Return (comp90, pr) using full PCA."""
    pca = PCA(svd_solver="full")
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    eig = pca.explained_variance_
    comp90 = int(np.searchsorted(cum, 0.90) + 1)
    pr = float((eig.sum() ** 2) / (eig ** 2).sum())
    return comp90, pr


def comp90_pr_randomized(X, n_components=500):
    """Faster comp90 via randomized PCA. Returns (comp90, pr-truncated)."""
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=RANDOM_SEED)
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    eig = pca.explained_variance_
    if cum[-1] < 0.90:
        # didn't capture 90% - return n_components as a lower bound
        comp90 = n_components
    else:
        comp90 = int(np.searchsorted(cum, 0.90) + 1)
    pr = float((eig.sum() ** 2) / (eig ** 2).sum())
    return comp90, pr


# -- Diagnostic 1: Style classifier ---------------------------------------------
def style_classifier(texts_human, texts_llm, label="all_llm"):
    """TF-IDF + length features, logreg classifier, 5-fold CV."""
    _flush(f"[style {label}] vectorizing TF-IDF")
    n_h = len(texts_human)
    n_l = len(texts_llm)
    all_text = list(texts_human) + list(texts_llm)
    y = np.concatenate([np.zeros(n_h), np.ones(n_l)])

    # Length features
    word_counts = np.array([[len(t.split()), len(t)] for t in all_text], dtype=float)
    log_len = np.log1p(word_counts)

    vec = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        token_pattern=r"(?u)\b\w+\b",
    )
    X_tfidf = vec.fit_transform(all_text)

    # Combine via hstack: sparse + dense numeric
    from scipy.sparse import hstack, csr_matrix
    X_full = hstack([X_tfidf, csr_matrix(log_len)]).tocsr()
    _flush(f"[style {label}] X shape {X_full.shape}, y balance {y.mean():.3f}")

    # 5-fold CV with logreg
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    accs = []
    aucs = []
    for fold, (tr, te) in enumerate(skf.split(X_full, y)):
        clf = LogisticRegression(
            max_iter=1000, C=1.0, n_jobs=-1, solver="liblinear"
        )
        clf.fit(X_full[tr], y[tr])
        p = clf.predict_proba(X_full[te])[:, 1]
        pred = (p >= 0.5).astype(int)
        accs.append(accuracy_score(y[te], pred))
        aucs.append(roc_auc_score(y[te], p))
        _flush(f"[style {label}] fold {fold+1}/5 acc={accs[-1]:.3f} auc={aucs[-1]:.3f}")

    # Length-only baseline
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    accs_len = []
    aucs_len = []
    for tr, te in skf2.split(log_len, y):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(log_len[tr], y[tr])
        p = clf.predict_proba(log_len[te])[:, 1]
        accs_len.append(accuracy_score(y[te], (p >= 0.5).astype(int)))
        aucs_len.append(roc_auc_score(y[te], p))

    # Top features (full classifier on full data)
    final = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")
    final.fit(X_full, y)
    feature_names = list(vec.get_feature_names_out()) + ["log_word_count", "log_char_count"]
    coefs = final.coef_[0]
    order = np.argsort(coefs)
    top_human = [(feature_names[i], float(coefs[i])) for i in order[:25]]
    top_llm = [(feature_names[i], float(coefs[i])) for i in order[-25:][::-1]]

    return {
        "label": label,
        "n_human": n_h,
        "n_llm": n_l,
        "tfidf_plus_length": {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "fold_accuracies": [float(a) for a in accs],
            "fold_aucs": [float(a) for a in aucs],
        },
        "length_only": {
            "accuracy_mean": float(np.mean(accs_len)),
            "auc_mean": float(np.mean(aucs_len)),
        },
        "top_human_features": top_human,
        "top_llm_features": top_llm,
    }


# -- Diagnostic 2: INLP residualized gap ----------------------------------------
def inlp_residualize(X, y, n_iter=INLP_ITERATIONS):
    """Iteratively project out logreg discriminator directions.

    Returns: (X_resid, accuracies_per_iter, projection_matrix)
    """
    Xc = X.copy().astype(np.float64)
    accs = []
    P_total = np.eye(X.shape[1])  # cumulative null-space projection
    directions = []

    for it in range(n_iter):
        clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, solver="liblinear")
        clf.fit(Xc, y)
        acc = accuracy_score(y, clf.predict(Xc))
        accs.append(float(acc))
        _flush(f"[INLP] iter {it+1}/{n_iter} train acc={acc:.4f}")

        w = clf.coef_[0]
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-10:
            _flush(f"[INLP] iter {it+1}: zero coef, stopping")
            break
        w_hat = w / w_norm
        directions.append(w_hat.copy())

        # Project onto null space: x_new = x - (x . w_hat) w_hat
        proj = Xc @ w_hat
        Xc = Xc - np.outer(proj, w_hat)

        # Update cumulative projection matrix
        P_iter = np.eye(X.shape[1]) - np.outer(w_hat, w_hat)
        P_total = P_total @ P_iter

    # Final accuracy after projection
    clf_final = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, solver="liblinear")
    clf_final.fit(Xc, y)
    final_acc = accuracy_score(y, clf_final.predict(Xc))
    accs.append(float(final_acc))
    _flush(f"[INLP] post-projection acc={final_acc:.4f}")

    return Xc, accs, P_total, np.array(directions)


# -- Diagnostic 3: Permutation null ---------------------------------------------
def permutation_null(X_pooled, n_human, n_perms=N_PERMUTATIONS, seed=RANDOM_SEED):
    """Shuffle labels in pooled (human + matched-LLM) data, recompute comp90 on each side."""
    rng = np.random.RandomState(seed)
    null_human = []
    null_llm = []
    null_gap = []
    n = X_pooled.shape[0]
    indices = np.arange(n)
    for p in range(n_perms):
        rng.shuffle(indices)
        h_idx = indices[:n_human]
        l_idx = indices[n_human:]
        c_h, _ = comp90_pr_randomized(X_pooled[h_idx])
        c_l, _ = comp90_pr_randomized(X_pooled[l_idx])
        null_human.append(c_h)
        null_llm.append(c_l)
        null_gap.append(c_h - c_l)
        if (p + 1) % 5 == 0 or p == 0:
            _flush(f"[perm] {p+1}/{n_perms}: human={c_h} llm={c_l} gap={c_h-c_l}")
    return np.array(null_human), np.array(null_llm), np.array(null_gap)


# -- Main -----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all", help="comma-separated: style,inlp,perm,all")
    ap.add_argument("--n_perms", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--n_inlp_iter", type=int, default=INLP_ITERATIONS)
    args = ap.parse_args()

    steps = set(args.steps.split(",")) if args.steps != "all" else {"style", "inlp", "perm"}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = load_embeddings()

    # Build matched-n samples
    rng = np.random.RandomState(RANDOM_SEED)
    X_human = embeddings["human"]
    n_human = X_human.shape[0]
    _flush(f"human n={n_human}")

    # Sample MATCHED_N LLM rows from concatenated pool
    X_llm_matched, llm_idx, llm_src_tags, llm_row_idx = matched_llm_pool(
        embeddings, n=MATCHED_N, seed=RANDOM_SEED
    )
    _flush(f"LLM matched pool: {X_llm_matched.shape}")

    # -- Diagnostic 1: style classifier -----------------------------------------
    if "style" in steps:
        _flush("=== DIAGNOSTIC 1: style classifier ===")
        texts = load_texts()
        texts_human = texts["human"]
        texts_llm_matched = matched_llm_text_pool(texts, llm_idx, llm_src_tags, llm_row_idx)
        all_llm_result = style_classifier(texts_human, texts_llm_matched, label="all_llm_matched")

        per_llm = {}
        for src in LLM_SOURCES:
            t_llm = texts[src]
            # Sample MATCHED_N to keep balanced (or all if fewer)
            n_l = min(len(t_llm), MATCHED_N)
            idx = rng.choice(len(t_llm), n_l, replace=False)
            t_llm_s = [t_llm[i] for i in idx]
            res = style_classifier(texts_human, t_llm_s, label=src)
            per_llm[src] = res

        out = {"all_llm": all_llm_result, "per_llm": per_llm}
        path = OUTPUT_DIR / "milestone3_style_classifier.json"
        path.write_text(json.dumps(out, indent=2))
        _flush(f"saved {path}")

    # -- Diagnostic 2: INLP residualized gap -------------------------------------
    if "inlp" in steps:
        _flush("=== DIAGNOSTIC 2: INLP residualized gap ===")
        # Pool: human (10826) + matched LLM (10826)
        X_pool = np.vstack([X_human, X_llm_matched])
        y_pool = np.concatenate([np.zeros(MATCHED_N), np.ones(MATCHED_N)])

        # Baseline comp90 (matches milestone 2 numbers expected)
        _flush("[INLP] baseline comp90 on full embeddings")
        baseline_h, baseline_h_pr = comp90_pr(X_human)
        baseline_l, baseline_l_pr = comp90_pr(X_llm_matched)
        _flush(f"[INLP] baseline human comp90={baseline_h} pr={baseline_h_pr:.2f}")
        _flush(f"[INLP] baseline llm comp90={baseline_l} pr={baseline_l_pr:.2f}")

        # Run INLP on pooled data
        X_pool_resid, accs, P_total, directions = inlp_residualize(
            X_pool, y_pool, n_iter=args.n_inlp_iter
        )

        # Apply same projection to ALL human and ALL all-llm pool, then recompute comp90
        # Note: P_total acts on the right: x_new = x @ P_total
        _flush("[INLP] applying projection to full source matrices for comp90")
        X_human_resid = X_human @ P_total
        X_llm_matched_resid = X_llm_matched @ P_total

        post_h, post_h_pr = comp90_pr(X_human_resid)
        post_l, post_l_pr = comp90_pr(X_llm_matched_resid)

        _flush(f"[INLP] post-INLP human comp90={post_h} pr={post_h_pr:.2f}")
        _flush(f"[INLP] post-INLP llm comp90={post_l} pr={post_l_pr:.2f}")

        # Apply to full all-LLM pool too (216K)
        all_llm_pool = np.vstack([embeddings[s] for s in LLM_SOURCES])
        all_llm_pool_resid = all_llm_pool @ P_total
        _flush("[INLP] computing all-LLM-pool comp90 (post-INLP)")
        post_alllm, post_alllm_pr = comp90_pr_randomized(all_llm_pool_resid, n_components=500)
        _flush(f"[INLP] post-INLP all-LLM comp90={post_alllm} pr={post_alllm_pr:.2f}")
        # Baseline comp90 for all-LLM (random subsample to match milestone 2 metric)
        all_llm_baseline_idx = np.random.RandomState(RANDOM_SEED).choice(
            all_llm_pool.shape[0], MATCHED_N, replace=False
        )
        base_alllm_matched, base_alllm_matched_pr = comp90_pr(all_llm_pool[all_llm_baseline_idx])
        post_alllm_matched, post_alllm_matched_pr = comp90_pr(all_llm_pool_resid[all_llm_baseline_idx])
        _flush(f"[INLP] baseline all-LLM matched comp90={base_alllm_matched}")
        _flush(f"[INLP] post-INLP all-LLM matched comp90={post_alllm_matched}")

        out = {
            "matched_n": MATCHED_N,
            "n_iter": args.n_inlp_iter,
            "classifier_acc_per_iter": accs,
            "baseline": {
                "human_comp90": baseline_h,
                "human_pr": baseline_h_pr,
                "llm_matched_comp90": baseline_l,
                "llm_matched_pr": baseline_l_pr,
                "all_llm_matched_comp90": int(base_alllm_matched),
                "all_llm_matched_pr": float(base_alllm_matched_pr),
                "gap_human_minus_llm": baseline_h - baseline_l,
                "gap_human_minus_alllm_matched": baseline_h - int(base_alllm_matched),
            },
            "post_inlp": {
                "human_comp90": post_h,
                "human_pr": post_h_pr,
                "llm_matched_comp90": post_l,
                "llm_matched_pr": post_l_pr,
                "all_llm_matched_comp90": int(post_alllm_matched),
                "all_llm_matched_pr": float(post_alllm_matched_pr),
                "gap_human_minus_llm": post_h - post_l,
                "gap_human_minus_alllm_matched": post_h - int(post_alllm_matched),
            },
            "directions_removed": int(directions.shape[0]),
        }
        path = OUTPUT_DIR / "milestone3_inlp.json"
        path.write_text(json.dumps(out, indent=2))
        _flush(f"saved {path}")

    # -- Diagnostic 3: permutation null ------------------------------------------
    if "perm" in steps:
        _flush("=== DIAGNOSTIC 3: permutation null ===")
        X_pool = np.vstack([X_human, X_llm_matched])
        # Baseline (no shuffle): use comp90_pr_randomized for fair comparison
        baseline_h, _ = comp90_pr_randomized(X_human)
        baseline_l, _ = comp90_pr_randomized(X_llm_matched)
        observed_gap = baseline_h - baseline_l
        _flush(f"[perm] observed (randomized PCA) human={baseline_h} llm={baseline_l} gap={observed_gap}")

        null_h, null_l, null_g = permutation_null(
            X_pool, n_human=MATCHED_N, n_perms=args.n_perms, seed=RANDOM_SEED
        )
        z = (observed_gap - null_g.mean()) / (null_g.std() + 1e-9)
        p_two = float(np.mean(np.abs(null_g - null_g.mean()) >= abs(observed_gap - null_g.mean())))
        out = {
            "n_perms": args.n_perms,
            "observed": {
                "human_comp90": int(baseline_h),
                "llm_comp90": int(baseline_l),
                "gap": int(observed_gap),
            },
            "null_distribution": {
                "human_comp90_mean": float(null_h.mean()),
                "human_comp90_std": float(null_h.std()),
                "llm_comp90_mean": float(null_l.mean()),
                "llm_comp90_std": float(null_l.std()),
                "gap_mean": float(null_g.mean()),
                "gap_std": float(null_g.std()),
                "gap_min": int(null_g.min()),
                "gap_max": int(null_g.max()),
                "gap_q25": float(np.percentile(null_g, 25)),
                "gap_q75": float(np.percentile(null_g, 75)),
            },
            "test_statistic": {
                "z_score": float(z),
                "two_sided_pvalue_empirical": p_two,
            },
            "raw_null_human": null_h.tolist(),
            "raw_null_llm": null_l.tolist(),
            "raw_null_gap": null_g.tolist(),
        }
        path = OUTPUT_DIR / "milestone3_permutation.json"
        path.write_text(json.dumps(out, indent=2))
        _flush(f"saved {path}")

    _flush("ALL DONE")


if __name__ == "__main__":
    main()
