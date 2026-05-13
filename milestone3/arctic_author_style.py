"""Author-style residualization test.

For each comment: subtract author-mean (humans) or model-mean (LLMs) from embedding.
Recompute per-dilemma within-source cosine on residualized embeddings.
Tests whether the 2x per-dilemma ratio is author-identity vs genuine moral expression.

Restrict to authors with >= 2 comments (singletons have author-mean = self, residualize to 0).
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

EMBEDDINGS_DIR = Path("data/embeddings")
ANALYSIS = Path("data/analysis")
ARCTIC_DIR = Path("data/arcticshift")
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
RNG = np.random.RandomState(42)

logger = logging.getLogger("author_style")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg); sys.stdout.flush()


def cosine_pairwise_mean(E):
    if E.shape[0] < 2:
        return float("nan")
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    sim = En @ En.T
    n = sim.shape[0]
    return float(1 - sim[np.triu_indices(n, k=1)].mean())


def main():
    # --- Load arctic data ------------------------------------------------------
    H_emb = np.load(EMBEDDINGS_DIR / "human_arctic.npy")
    arctic_meta = json.loads((EMBEDDINGS_DIR / "human_arctic_meta.json").read_text())
    arctic_recs = {}
    for line in (ARCTIC_DIR / "filtered_comments.jsonl").read_text().splitlines():
        if not line.strip(): continue
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("_empty") or not j.get("body"): continue
        arctic_recs[(j["submission_id"], j.get("comment_id"))] = j
    _flush(f"arctic embeddings: {H_emb.shape}")

    # Build idx -> author
    idx_to_author = {}
    idx_to_sub = {}
    for m in arctic_meta:
        rec = arctic_recs.get((m["submission_id"], m.get("comment_id")))
        if rec:
            idx_to_author[m["index"]] = rec.get("author", "[unknown]")
            idx_to_sub[m["index"]] = m["submission_id"]

    # Author -> list of indices
    author_to_idxs = defaultdict(list)
    for i, a in idx_to_author.items():
        if a and a not in ("[deleted]", "[unknown]"):
            author_to_idxs[a].append(i)
    _flush(f"total authors: {len(author_to_idxs)}")
    multi_authors = {a: idxs for a, idxs in author_to_idxs.items() if len(idxs) >= 2}
    _flush(f"multi-commenter authors (>=2 comments): {len(multi_authors)}")

    # Compute author-mean for multi-commenters
    author_means = {}
    for a, idxs in multi_authors.items():
        author_means[a] = H_emb[idxs].mean(axis=0)

    # Author-residualized embeddings (for multi-commenters only)
    H_resid = H_emb.copy().astype(np.float32)
    n_residualized = 0
    multi_indices = set()
    for a, mean_v in author_means.items():
        for i in multi_authors[a]:
            H_resid[i] = H_emb[i] - mean_v
            multi_indices.add(i)
            n_residualized += 1
    _flush(f"residualized {n_residualized} comments by author-mean")
    _flush(f"frac var preserved (residualized subset): {H_resid[list(multi_indices)].var() / H_emb[list(multi_indices)].var():.3f}")

    # --- Load LLM embeddings + compute model-mean residualization --------------
    _flush("loading LLM embeddings + texts")
    ds = load_dataset("ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas", split="test")

    LLM_metas = {s: json.loads((EMBEDDINGS_DIR / f"{s}_meta.json").read_text()) for s in LLM_SOURCES}
    LLM_embs = {s: np.load(EMBEDDINGS_DIR / f"{s}.npy") for s in LLM_SOURCES}

    # Model-mean for each LLM source
    model_means = {s: LLM_embs[s].mean(axis=0) for s in LLM_SOURCES}
    _flush(f"computed model-means for {len(LLM_SOURCES)} LLM sources")

    # Restrict to ArcticShift dilemmas
    arctic_subs = set(idx_to_sub.values())
    l_by_sub = defaultdict(list)
    l_by_sub_resid = defaultdict(list)
    for s in LLM_SOURCES:
        for m in LLM_metas[s]:
            sid = m["submission_id"]
            if sid not in arctic_subs:
                continue
            E_orig = LLM_embs[s][m["index"]]
            E_resid = E_orig - model_means[s]
            l_by_sub[sid].append({"source": s, "emb": E_orig})
            l_by_sub_resid[sid].append({"source": s, "emb": E_resid})

    # --- Per-dilemma analysis (multi-commenter humans only) --------------------
    # Group multi-commenter humans by submission
    h_by_sub_multi_idxs = defaultdict(list)
    for i in multi_indices:
        sid = idx_to_sub.get(i)
        if sid:
            h_by_sub_multi_idxs[sid].append(i)

    qualified = sorted([s for s in h_by_sub_multi_idxs if len(h_by_sub_multi_idxs[s]) >= 5 and len(l_by_sub.get(s, [])) >= 2])
    _flush(f"qualified dilemmas (>=5 multi-commenter humans + >=2 LLMs): {len(qualified)}")

    # Baseline: original embeddings, multi-commenter subset
    baseline_h, baseline_l = [], []
    for sid in qualified:
        h_idxs = h_by_sub_multi_idxs[sid]
        H_dil = H_emb[h_idxs]
        L_dil = np.stack([it["emb"] for it in l_by_sub[sid]])
        baseline_h.append(cosine_pairwise_mean(H_dil))
        baseline_l.append(cosine_pairwise_mean(L_dil))
    baseline_h = np.array(baseline_h)
    baseline_l = np.array(baseline_l)
    base_ratio = float(baseline_h.mean() / baseline_l.mean())
    base_frac = float(np.mean(baseline_h > baseline_l))
    _flush(f"BASELINE (multi-comm subset, NO residualization):")
    _flush(f"  within-human: {baseline_h.mean():.4f}, within-llm: {baseline_l.mean():.4f}, "
           f"ratio: {base_ratio:.3f}, frac_h>l: {base_frac:.4f}")

    # Author-residualized: humans residualized, LLMs UNRESIDUALIZED (asymmetric, less fair)
    h_resid_only_h, h_resid_only_l = [], []
    for sid in qualified:
        h_idxs = h_by_sub_multi_idxs[sid]
        H_dil = H_resid[h_idxs]
        L_dil = np.stack([it["emb"] for it in l_by_sub[sid]])
        h_resid_only_h.append(cosine_pairwise_mean(H_dil))
        h_resid_only_l.append(cosine_pairwise_mean(L_dil))
    h_resid_only_h = np.array(h_resid_only_h)
    h_resid_only_l = np.array(h_resid_only_l)
    h_only_ratio = float(h_resid_only_h.mean() / h_resid_only_l.mean())
    h_only_frac = float(np.mean(h_resid_only_h > h_resid_only_l))
    _flush(f"HUMAN-ONLY-RESID (humans author-residualized, LLMs untouched):")
    _flush(f"  within-human: {h_resid_only_h.mean():.4f}, within-llm: {h_resid_only_l.mean():.4f}, "
           f"ratio: {h_only_ratio:.3f}, frac_h>l: {h_only_frac:.4f}")

    # Both residualized (fair): humans by author-mean, LLMs by model-mean
    both_h, both_l = [], []
    for sid in qualified:
        h_idxs = h_by_sub_multi_idxs[sid]
        H_dil = H_resid[h_idxs]
        L_dil = np.stack([it["emb"] for it in l_by_sub_resid[sid]])
        both_h.append(cosine_pairwise_mean(H_dil))
        both_l.append(cosine_pairwise_mean(L_dil))
    both_h = np.array(both_h)
    both_l = np.array(both_l)
    both_ratio = float(both_h.mean() / both_l.mean())
    both_frac = float(np.mean(both_h > both_l))
    _flush(f"BOTH-RESID (humans author-residualized, LLMs model-residualized):")
    _flush(f"  within-human: {both_h.mean():.4f}, within-llm: {both_l.mean():.4f}, "
           f"ratio: {both_ratio:.3f}, frac_h>l: {both_frac:.4f}")

    # --- Strict author-id-only test: SAME-author cross-dilemma cosine ---------
    # For each multi-author with comments in multiple dilemmas, average pairwise cosine
    # of their own comments. This is "intra-author style consistency".
    intra_author_cos = []
    for a, idxs in multi_authors.items():
        if len(idxs) < 2:
            continue
        E_a = H_emb[idxs]
        intra_author_cos.append(cosine_pairwise_mean(E_a))
    intra_author_cos = np.array([x for x in intra_author_cos if not np.isnan(x)])
    _flush(f"intra-author cosine across own comments (style consistency): "
           f"mean={intra_author_cos.mean():.4f}, n_authors={len(intra_author_cos)}")
    _flush(f"  -> distance baseline if 2x ratio were purely author-style: humans pair-distance would be ~{intra_author_cos.mean():.3f}")

    # Per-dilemma within-LLM, multi-LLM unfiltered (full set):
    # already computed in baseline_l above.

    out = {
        "n_dilemmas_qualified": int(len(qualified)),
        "n_multi_commenter_authors": int(len(multi_authors)),
        "n_human_comments_residualized": int(n_residualized),
        "baseline": {
            "within_human": float(baseline_h.mean()), "within_llm": float(baseline_l.mean()),
            "ratio": base_ratio, "frac_h>l": base_frac,
        },
        "human_only_residualized": {
            "within_human": float(h_resid_only_h.mean()), "within_llm": float(h_resid_only_l.mean()),
            "ratio": h_only_ratio, "frac_h>l": h_only_frac,
        },
        "both_residualized": {
            "within_human": float(both_h.mean()), "within_llm": float(both_l.mean()),
            "ratio": both_ratio, "frac_h>l": both_frac,
        },
        "intra_author_cosine_mean": float(intra_author_cos.mean()) if len(intra_author_cos) else None,
        "intra_author_n_authors": int(len(intra_author_cos)),
    }
    path = ANALYSIS / "milestone3_author_style.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")


if __name__ == "__main__":
    main()
