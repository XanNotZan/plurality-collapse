"""Stratified ArcticShift within-dilemma analysis: consensus, verdict, length."""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EMBEDDINGS_DIR = Path("data/embeddings")
ANALYSIS = Path("data/analysis")
ARCTIC_DIR = Path("data/arcticshift")
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]

logger = logging.getLogger("arctic_strat")
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


def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(1 - np.dot(a, b) / (na * nb))


def main():
    # Load embeddings + meta
    H_arctic = np.load(EMBEDDINGS_DIR / "human_arctic.npy")
    arctic_meta = json.loads((EMBEDDINGS_DIR / "human_arctic_meta.json").read_text())
    _flush(f"loaded arctic: {H_arctic.shape}, meta {len(arctic_meta)}")

    # Build per-dilemma index of arctic comments
    arctic_by_sub = defaultdict(list)
    for m in arctic_meta:
        arctic_by_sub[m["submission_id"]].append(m["index"])

    # Per-dilemma LLM embeddings
    LLM_metas = {s: json.loads((EMBEDDINGS_DIR / f"{s}_meta.json").read_text()) for s in LLM_SOURCES}
    LLM_embs = {s: np.load(EMBEDDINGS_DIR / f"{s}.npy") for s in LLM_SOURCES}
    llm_by_sub = defaultdict(list)
    for s in LLM_SOURCES:
        E = LLM_embs[s]
        for m in LLM_metas[s]:
            llm_by_sub[m["submission_id"]].append(E[m["index"]])

    # Need text length for human comments to do length stratification
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

    # Need verdicts and consensus
    from datasets import load_dataset
    _flush("loading HF dataset for verdicts/consensus")
    ds = load_dataset("ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas", split="test")
    sub_ids = ds["submission_id"]
    cols = ["comments_nta_agreement_weighted", "comments_yta_agreement_weighted",
            "comments_esh_agreement_weighted", "comments_nah_agreement_weighted"]
    labels = ["NTA", "YTA", "ESH", "NAH"]
    cols_data = [ds[c] for c in cols]
    sub_to_verdict = {}
    sub_to_consensus = {}
    for i, sid in enumerate(sub_ids):
        scores = [(0.0 if cols_data[k][i] is None else float(cols_data[k][i])) for k in range(len(cols))]
        idx = int(np.argmax(scores))
        sub_to_verdict[sid] = labels[idx]
        sub_to_consensus[sid] = float(scores[idx])

    # Compute per-dilemma metrics for qualified dilemmas (both humans + LLMs available)
    per_dilemma = []
    for sid in sorted(arctic_by_sub.keys()):
        h_idxs = arctic_by_sub[sid]
        if len(h_idxs) < 5:
            continue
        if sid not in llm_by_sub or len(llm_by_sub[sid]) < 2:
            continue
        H = H_arctic[h_idxs]
        L = np.stack(llm_by_sub[sid])
        within_h = cosine_pairwise_mean(H)
        within_l = cosine_pairwise_mean(L)
        h_centroid = H.mean(axis=0)
        l_centroid = L.mean(axis=0)
        cd = cosine_dist(h_centroid, l_centroid)
        # Mean human comment length
        comment_ids = []
        for m_idx in h_idxs:
            for m in arctic_meta:
                if m["index"] == m_idx:
                    comment_ids.append(m.get("comment_id"))
                    break
        # Skip the slow lookup; fetch via meta directly:
        # Better: precompute index→cid map
        per_dilemma.append({
            "sub_id": sid,
            "n_human": len(h_idxs),
            "n_llm": L.shape[0],
            "within_human": within_h,
            "within_llm": within_l,
            "centroid_dist": cd,
            "verdict": sub_to_verdict.get(sid),
            "consensus": sub_to_consensus.get(sid, 0.0),
        })
    _flush(f"per-dilemma metrics: n={len(per_dilemma)}")

    # Stratify by consensus (low <0.5, med 0.5-0.8, high >0.8)
    by_cons = {"low": [], "medium": [], "high": []}
    for d in per_dilemma:
        c = d["consensus"]
        if c < 0.5:
            by_cons["low"].append(d)
        elif c < 0.8:
            by_cons["medium"].append(d)
        else:
            by_cons["high"].append(d)
    cons_stats = {}
    for level, items in by_cons.items():
        if not items:
            cons_stats[level] = None; continue
        wh = np.array([d["within_human"] for d in items])
        wl = np.array([d["within_llm"] for d in items])
        cd = np.array([d["centroid_dist"] for d in items])
        frac = float(np.mean(wh > wl))
        cons_stats[level] = {
            "n_dilemmas": len(items),
            "mean_within_human": float(wh.mean()), "std_within_human": float(wh.std()),
            "mean_within_llm": float(wl.mean()), "std_within_llm": float(wl.std()),
            "mean_centroid_dist": float(cd.mean()),
            "ratio_h_over_l": float(wh.mean() / wl.mean()),
            "frac_dilemmas_h_more_diverse": frac,
        }
        _flush(f"consensus={level} (n={len(items)}): h={wh.mean():.4f}, l={wl.mean():.4f}, "
               f"ratio={wh.mean()/wl.mean():.2f}, frac_h>l={frac:.3f}")

    # Stratify by verdict
    by_verdict = defaultdict(list)
    for d in per_dilemma:
        by_verdict[d["verdict"]].append(d)
    verdict_stats = {}
    for v, items in by_verdict.items():
        if not items:
            continue
        wh = np.array([d["within_human"] for d in items])
        wl = np.array([d["within_llm"] for d in items])
        cd = np.array([d["centroid_dist"] for d in items])
        frac = float(np.mean(wh > wl))
        verdict_stats[v] = {
            "n_dilemmas": len(items),
            "mean_within_human": float(wh.mean()),
            "mean_within_llm": float(wl.mean()),
            "mean_centroid_dist": float(cd.mean()),
            "ratio_h_over_l": float(wh.mean() / wl.mean()),
            "frac_dilemmas_h_more_diverse": frac,
        }
        _flush(f"verdict={v} (n={len(items)}): h={wh.mean():.4f}, l={wl.mean():.4f}, "
               f"ratio={wh.mean()/wl.mean():.2f}, frac_h>l={frac:.3f}")

    # Length-stratified: per-dilemma mean human comment length quartile
    # Build cid→length lookup
    cid_to_len = {(sid, cid): len(text.split()) for (sid, cid), text in arctic_texts.items()}
    # Per-dilemma mean comment length
    cid_by_idx = {m["index"]: (m["submission_id"], m.get("comment_id")) for m in arctic_meta}
    for d in per_dilemma:
        sid = d["sub_id"]
        h_idxs = arctic_by_sub[sid]
        lens = [cid_to_len.get(cid_by_idx[i], 0) for i in h_idxs]
        d["mean_human_comment_words"] = float(np.mean(lens)) if lens else 0.0

    lens = np.array([d["mean_human_comment_words"] for d in per_dilemma])
    edges = [np.percentile(lens, q) for q in (25, 50, 75)]
    length_stats = {}
    for q, label in enumerate(["Q1 (<25th)", "Q2 (25-50th)", "Q3 (50-75th)", "Q4 (>75th)"]):
        if q == 0: mask = lens < edges[0]
        elif q == 1: mask = (lens >= edges[0]) & (lens < edges[1])
        elif q == 2: mask = (lens >= edges[1]) & (lens < edges[2])
        else: mask = lens >= edges[2]
        items = [d for d, m in zip(per_dilemma, mask) if m]
        if not items:
            continue
        wh = np.array([d["within_human"] for d in items])
        wl = np.array([d["within_llm"] for d in items])
        frac = float(np.mean(wh > wl))
        length_stats[label] = {
            "n_dilemmas": len(items),
            "mean_words": float(np.mean([d["mean_human_comment_words"] for d in items])),
            "mean_within_human": float(wh.mean()),
            "mean_within_llm": float(wl.mean()),
            "ratio_h_over_l": float(wh.mean() / wl.mean()),
            "frac_dilemmas_h_more_diverse": frac,
        }
        _flush(f"length {label} (n={len(items)}): mean_words={np.mean([d['mean_human_comment_words'] for d in items]):.1f}, "
               f"h={wh.mean():.4f}, l={wl.mean():.4f}, ratio={wh.mean()/wl.mean():.2f}, frac_h>l={frac:.3f}")

    out = {
        "n_dilemmas_total": len(per_dilemma),
        "by_consensus": cons_stats,
        "by_verdict": verdict_stats,
        "by_human_length_quartile": length_stats,
    }
    path = ANALYSIS / "milestone3_arctic_stratified.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")


if __name__ == "__main__":
    main()
