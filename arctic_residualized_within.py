"""TF-IDF residualize ALL comments (humans + LLMs), recompute within-dilemma diversity.
Tests whether the per-dilemma 2x ratio is register-driven or content-driven.
"""

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

EMBEDDINGS_DIR = Path("data/embeddings")
ANALYSIS = Path("data/analysis")
ARCTIC_DIR = Path("data/arcticshift")
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]

logger = logging.getLogger("arctic_resid")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg); sys.stdout.flush()


def cosine_pairwise_mean(E):
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    sim = En @ En.T
    n = sim.shape[0]
    if n < 2:
        return float("nan")
    triu = sim[np.triu_indices(n, k=1)]
    return float(1 - triu.mean())


def main():
    # 1. Load arctic embeddings + texts
    H_arctic = np.load(EMBEDDINGS_DIR / "human_arctic.npy")
    arctic_meta = json.loads((EMBEDDINGS_DIR / "human_arctic_meta.json").read_text())
    _flush(f"arctic embeddings: {H_arctic.shape}")

    arctic_texts = {}
    for line in (ARCTIC_DIR / "filtered_comments.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("_empty") or not j.get("body"):
            continue
        arctic_texts[(j["submission_id"], j.get("comment_id"))] = j["body"]

    # Build text array aligned to arctic_meta
    H_texts = []
    H_subs = []
    for m in arctic_meta:
        sid = m["submission_id"]
        cid = m.get("comment_id")
        H_texts.append(arctic_texts.get((sid, cid), ""))
        H_subs.append(sid)
    _flush(f"arctic texts: {len(H_texts)}, missing: {sum(1 for t in H_texts if not t)}")

    # 2. Load LLM embeddings + texts (matched to dilemmas in arctic set)
    sub_set = set(H_subs)
    LLM_texts = []
    LLM_embs_list = []
    LLM_subs = []
    LLM_meta_full = {s: json.loads((EMBEDDINGS_DIR / f"{s}_meta.json").read_text()) for s in LLM_SOURCES}
    LLM_emb_full = {s: np.load(EMBEDDINGS_DIR / f"{s}.npy") for s in LLM_SOURCES}

    # Need text for LLM rationales — fetch from HF dataset
    _flush("loading HF dataset for LLM text")
    ds = load_dataset("ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas", split="test")
    sub_to_row = {ds[i]["submission_id"]: i for i in range(len(ds))}
    needed_cols = set()
    for s in LLM_SOURCES:
        for c in [f"{s}_reason_1", f"{s}_reason_2", f"{s}_reason_3"]:
            if c in ds.column_names:
                needed_cols.add(c)
    col_data = {c: ds[c] for c in needed_cols}

    for s in LLM_SOURCES:
        meta = LLM_meta_full[s]
        E = LLM_emb_full[s]
        for m in meta:
            sid = m["submission_id"]
            if sid not in sub_set:
                continue
            col = m["column"]
            text = col_data[col][sub_to_row[sid]] if col in col_data else ""
            text = text if isinstance(text, str) else ""
            LLM_texts.append(text)
            LLM_embs_list.append(E[m["index"]])
            LLM_subs.append(sid)
    L_arctic = np.stack(LLM_embs_list)
    _flush(f"LLM matched to arctic dilemmas: n={len(LLM_texts)}, embeddings shape={L_arctic.shape}")

    # 3. Pool all texts + embeddings, fit Ridge
    all_texts = H_texts + LLM_texts
    all_embs = np.vstack([H_arctic, L_arctic])
    n_h = H_arctic.shape[0]
    _flush(f"pooled: texts={len(all_texts)}, embeddings={all_embs.shape}, n_h={n_h}")

    # TF-IDF + length features (same as corpus residualization)
    vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=5,
                         sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    F_tfidf = vec.fit_transform(all_texts).toarray().astype(np.float32)
    log_len = np.log1p(np.array([[len(t.split()), len(t)] for t in all_texts], dtype=np.float32))
    F = np.hstack([F_tfidf, log_len])
    F_mean = F.mean(axis=0)
    F = F - F_mean
    _flush(f"design matrix F: {F.shape}")

    # Ridge regression: closed form W = (F^T F + αI)^-1 F^T X
    alpha = 10.0
    p = F.shape[1]
    _flush("solving Ridge")
    t0 = time.time()
    G = (F.T @ F).astype(np.float64) + alpha * np.eye(p, dtype=np.float64)
    rhs = (F.T @ all_embs).astype(np.float64)
    W = np.linalg.solve(G, rhs).astype(np.float32)
    _flush(f"Ridge solved in {time.time()-t0:.1f}s, W shape {W.shape}")
    X_pred = F @ W
    X_resid = (all_embs - X_pred).astype(np.float32)
    frac_var_kept = float(X_resid.var() / all_embs.var())
    _flush(f"residualized: var preserved = {frac_var_kept:.3f}")

    # 4. Recompute per-dilemma within-source cosine on residualized
    H_resid = X_resid[:n_h]
    L_resid = X_resid[n_h:]

    h_by_sub = defaultdict(list)
    l_by_sub = defaultdict(list)
    for i, sid in enumerate(H_subs):
        h_by_sub[sid].append(i)
    for i, sid in enumerate(LLM_subs):
        l_by_sub[sid].append(i)

    # Per-dilemma metrics on BASELINE and RESIDUALIZED
    per_dilemma = []
    qualified = [s for s in h_by_sub if len(h_by_sub[s]) >= 5 and len(l_by_sub.get(s, [])) >= 2]
    _flush(f"qualified dilemmas: {len(qualified)}")
    for sid in qualified:
        h_idx = h_by_sub[sid]
        l_idx = l_by_sub[sid]
        # baseline
        wh_base = cosine_pairwise_mean(H_arctic[h_idx])
        wl_base = cosine_pairwise_mean(L_arctic[l_idx])
        # residualized
        wh_res = cosine_pairwise_mean(H_resid[h_idx])
        wl_res = cosine_pairwise_mean(L_resid[l_idx])
        per_dilemma.append({
            "sub_id": sid,
            "n_h": len(h_idx), "n_l": len(l_idx),
            "wh_base": wh_base, "wl_base": wl_base,
            "wh_resid": wh_res, "wl_resid": wl_res,
        })

    wh_b = np.array([d["wh_base"] for d in per_dilemma])
    wl_b = np.array([d["wl_base"] for d in per_dilemma])
    wh_r = np.array([d["wh_resid"] for d in per_dilemma])
    wl_r = np.array([d["wl_resid"] for d in per_dilemma])

    out = {
        "n_dilemmas": len(per_dilemma),
        "frac_var_preserved_after_ridge": frac_var_kept,
        "baseline": {
            "mean_within_human": float(wh_b.mean()),
            "mean_within_llm": float(wl_b.mean()),
            "ratio_h_over_l": float(wh_b.mean() / wl_b.mean()),
            "frac_dilemmas_h_more_diverse": float(np.mean(wh_b > wl_b)),
        },
        "residualized": {
            "mean_within_human": float(wh_r.mean()),
            "mean_within_llm": float(wl_r.mean()),
            "ratio_h_over_l": float(wh_r.mean() / (wl_r.mean() + 1e-12)),
            "frac_dilemmas_h_more_diverse": float(np.mean(wh_r > wl_r)),
        },
    }
    _flush(f"BASELINE: human={wh_b.mean():.4f}, llm={wl_b.mean():.4f}, ratio={wh_b.mean()/wl_b.mean():.3f}, frac_h>l={np.mean(wh_b>wl_b):.4f}")
    _flush(f"RESIDUALIZED: human={wh_r.mean():.4f}, llm={wl_r.mean():.4f}, ratio={wh_r.mean()/wl_r.mean():.3f}, frac_h>l={np.mean(wh_r>wl_r):.4f}")
    path = ANALYSIS / "milestone3_arctic_residualized.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")


if __name__ == "__main__":
    main()
