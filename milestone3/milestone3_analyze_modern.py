"""Analyse modern-LLM-generated rationales and register-paraphrased rationales.

Inputs (computed by milestone3_modern_gen.py):
  data/embeddings/qwen25_3b_modern.{npy,_meta.json}
  data/embeddings/human_to_formal.{npy,_meta.json}
  data/embeddings/llm_to_casual.{npy,_meta.json}

Outputs:
  data/analysis/milestone3_modernity.json     -- comp90, PR, gap vs human; MFD gradient on modern.
  data/analysis/milestone3_register_paraphrase.json -- 4-cell gap matrix.
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

EMBEDDINGS_DIR = Path("data/embeddings")
OUTPUT_DIR = Path("data/analysis")
GEN_DIR = Path("data/generated")
RANDOM_SEED = 42
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]

# MFD same as before (compact)
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


logger = logging.getLogger("analyze_modern")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg)
    sys.stdout.flush()


def comp90_pr(X):
    pca = PCA(svd_solver="full")
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    eig = pca.explained_variance_
    pr = float((eig.sum() ** 2) / (eig ** 2).sum())
    if cum[-1] < 0.90:
        return int(len(cum)), pr
    return int(np.searchsorted(cum, 0.90) + 1), pr


def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def mfd_count(text):
    counts = {f: 0 for f in MFD}
    if not text:
        return counts
    for tok in re.findall(r"\b[a-z]+\b", text.lower()):
        for stem, fnd in MFD_PAIRS:
            if tok.startswith(stem):
                counts[fnd] += 1
                break
    return counts


def modernity_analysis():
    _flush("=== Modernity test ===")
    Q = np.load(EMBEDDINGS_DIR / "qwen25_3b_modern.npy")
    qwen_meta = json.loads((EMBEDDINGS_DIR / "qwen25_3b_modern_meta.json").read_text())
    H = np.load(EMBEDDINGS_DIR / "human.npy")
    all_llm = np.vstack([np.load(EMBEDDINGS_DIR / f"{s}.npy") for s in LLM_SOURCES])
    n_qwen = Q.shape[0]
    _flush(f"Qwen 2.5 3B: shape {Q.shape}; human {H.shape}; all-LLM {all_llm.shape}")

    # Modern model: comp90 + PR on full set
    cQ, prQ = comp90_pr(Q)

    # Compare against:
    # 1. all-LLM matched-n (n_qwen) at multiple seeds
    # 2. human matched-n at multiple seeds
    rng = np.random.RandomState(RANDOM_SEED)
    n_seeds = 5
    comp90_alllm_seeds = []
    pr_alllm_seeds = []
    comp90_human_seeds = []
    pr_human_seeds = []
    for s in range(n_seeds):
        idx = rng.choice(all_llm.shape[0], n_qwen, replace=False)
        c, p = comp90_pr(all_llm[idx])
        comp90_alllm_seeds.append(c); pr_alllm_seeds.append(p)
        idx_h = rng.choice(H.shape[0], min(n_qwen, H.shape[0]), replace=False)
        ch, ph = comp90_pr(H[idx_h])
        comp90_human_seeds.append(ch); pr_human_seeds.append(ph)

    _flush(f"Qwen modern comp90={cQ} PR={prQ:.2f}")
    _flush(f"all-LLM matched-n={n_qwen}: comp90={np.mean(comp90_alllm_seeds):.1f}±{np.std(comp90_alllm_seeds):.1f} PR={np.mean(pr_alllm_seeds):.2f}")
    _flush(f"human matched-n: comp90={np.mean(comp90_human_seeds):.1f}±{np.std(comp90_human_seeds):.1f} PR={np.mean(pr_human_seeds):.2f}")

    # Centroid distances
    h_cent = H.mean(axis=0)
    q_cent = Q.mean(axis=0)
    l_cent = all_llm.mean(axis=0)
    d_q_h = cosine_dist(q_cent, h_cent)
    d_q_l = cosine_dist(q_cent, l_cent)
    d_h_l = cosine_dist(h_cent, l_cent)
    _flush(f"centroid distances: human-Sachdeva-LLM={d_h_l:.4f}, "
           f"Qwen-human={d_q_h:.4f}, Qwen-Sachdeva-LLM={d_q_l:.4f}")

    # MFD gradient on Qwen rationales using all-LLM PCs (k=448)
    # Need text - load from JSONL
    rationales = []
    for line in (GEN_DIR / "qwen25_3b_rationales.jsonl").read_text().splitlines():
        rationales.append(json.loads(line)["rationale"])

    pca_llm = PCA(n_components=448, svd_solver="randomized", random_state=RANDOM_SEED)
    pca_llm.fit(all_llm)
    proj = pca_llm.transform(Q)
    recon = pca_llm.inverse_transform(proj)
    err = np.linalg.norm(Q - recon, axis=1) ** 2
    counts_arr = np.zeros((n_qwen, len(MFD)), dtype=np.int32)
    fnd_idx = {f: i for i, f in enumerate(MFD.keys())}
    for i, t in enumerate(rationales):
        c = mfd_count(t)
        for f, v in c.items():
            counts_arr[i, fnd_idx[f]] = v

    # Bin-level MFD gradient, sliding window 100, step 25
    order = np.argsort(err)
    err_sorted = err[order]
    counts_sorted = counts_arr[order]
    w, step = 100, 25
    bin_err, bin_uf, bin_ent = [], [], []
    for s in range(0, n_qwen - w + 1, step):
        e = err_sorted[s:s + w]
        c = counts_sorted[s:s + w]
        any_match = (c > 0)
        used = (any_match.sum(axis=0) > 0).sum()
        bin_err.append(float(e.mean()))
        bin_uf.append(int(used))
        tot = c.sum(axis=0)
        if tot.sum() > 0:
            p = tot / tot.sum()
            bin_ent.append(float(-np.sum(p * np.log(p + 1e-12))))
        else:
            bin_ent.append(float("nan"))
    bin_err_a = np.array(bin_err)
    bin_uf_a = np.array(bin_uf, dtype=float)
    bin_ent_a = np.array(bin_ent)
    if len(bin_err_a) >= 3:
        rho_uf = float(np.corrcoef(bin_err_a, bin_uf_a)[0, 1])
        valid = ~np.isnan(bin_ent_a)
        rho_ent = float(np.corrcoef(bin_err_a[valid], bin_ent_a[valid])[0, 1]) if valid.sum() >= 3 else float("nan")
    else:
        rho_uf, rho_ent = float("nan"), float("nan")
    _flush(f"Qwen MFD gradient: corr(used_foundations, recon_err)={rho_uf:.3f}, corr(bin_ent, recon_err)={rho_ent:.3f}")

    out = {
        "qwen_n": n_qwen,
        "qwen_comp90": cQ, "qwen_pr": prQ,
        "alllm_matched_n_comp90_mean": float(np.mean(comp90_alllm_seeds)),
        "alllm_matched_n_comp90_std": float(np.std(comp90_alllm_seeds)),
        "alllm_matched_n_pr_mean": float(np.mean(pr_alllm_seeds)),
        "human_matched_n_comp90_mean": float(np.mean(comp90_human_seeds)),
        "human_matched_n_pr_mean": float(np.mean(pr_human_seeds)),
        "centroid_distances": {
            "human_to_sachdeva_llm": d_h_l,
            "qwen_to_human": d_q_h,
            "qwen_to_sachdeva_llm": d_q_l,
        },
        "qwen_mfd_gradient": {
            "corr_used_foundations": rho_uf,
            "corr_bin_entropy": rho_ent,
            "n_bins": int(len(bin_err_a)),
        },
        "qwen_gap_vs_human_matched_n": float(np.mean(comp90_human_seeds)) - cQ,
        "qwen_gap_vs_alllm_matched_n": cQ - float(np.mean(comp90_alllm_seeds)),
    }
    path = OUTPUT_DIR / "milestone3_modernity.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")


def register_paraphrase_analysis(clean=True):
    _flush(f"=== Register paraphrase test (clean={clean}) ===")
    H = np.load(EMBEDDINGS_DIR / "human.npy")
    suffix = "_clean" if clean else ""
    Hf = np.load(EMBEDDINGS_DIR / f"human_to_formal{suffix}.npy")
    L_pool = np.vstack([np.load(EMBEDDINGS_DIR / f"{s}.npy") for s in LLM_SOURCES])
    Lc = np.load(EMBEDDINGS_DIR / f"llm_to_casual{suffix}.npy")
    n_h_para = Hf.shape[0]
    n_l_para = Lc.shape[0]
    _flush(f"sizes: H={H.shape}, Hf={Hf.shape}, L_pool={L_pool.shape}, Lc={Lc.shape}")

    # Match original human / LLM samples that were paraphrased
    # Human paraphrase meta: orig_index in human_meta
    h_meta_para = json.loads((EMBEDDINGS_DIR / f"human_to_formal{suffix}_meta.json").read_text())
    l_meta_para = json.loads((EMBEDDINGS_DIR / f"llm_to_casual{suffix}_meta.json").read_text())
    h_jsonl_path = GEN_DIR / ("human_to_formal_clean.jsonl" if clean else "human_to_formal.jsonl")
    l_jsonl_path = GEN_DIR / ("llm_to_casual_clean.jsonl" if clean else "llm_to_casual.jsonl")
    para_h_meta_jsonl = [json.loads(line) for line in h_jsonl_path.read_text().splitlines()]
    para_l_meta_jsonl = [json.loads(line) for line in l_jsonl_path.read_text().splitlines()]

    # Original human embeddings (matched to paraphrase set)
    h_indices_orig = [item["orig_index"] for item in para_h_meta_jsonl]
    H_orig_subset = H[h_indices_orig]

    # Original LLM embeddings (matched to paraphrase set)
    # For each item, find the right LLM source's embedding by orig_column + orig_index
    src_to_emb = {s: np.load(EMBEDDINGS_DIR / f"{s}.npy") for s in LLM_SOURCES}
    src_to_meta = {s: json.loads((EMBEDDINGS_DIR / f"{s}_meta.json").read_text()) for s in LLM_SOURCES}
    src_to_lookup = {}
    for s in LLM_SOURCES:
        lk = {}
        for m in src_to_meta[s]:
            lk[(m["submission_id"], m["column"])] = m["index"]
        src_to_lookup[s] = lk

    L_orig_rows = []
    for item in para_l_meta_jsonl:
        s = item["source"]
        sid = item["submission_id"]
        col = item["orig_column"]
        idx = src_to_lookup[s].get((sid, col))
        if idx is None:
            L_orig_rows.append(None)
        else:
            L_orig_rows.append(src_to_emb[s][idx])
    L_orig_rows = [r for r in L_orig_rows if r is not None]
    L_orig_subset = np.stack(L_orig_rows) if L_orig_rows else None
    _flush(f"H_orig_subset={H_orig_subset.shape}, L_orig_subset={L_orig_subset.shape if L_orig_subset is not None else None}")

    # 4-cell gap matrix:
    # (human_orig vs llm_orig), (human_formal vs llm_orig), (human_orig vs llm_casual), (human_formal vs llm_casual)
    def cell_metrics(Xh, Xl, label):
        nh, nl = Xh.shape[0], Xl.shape[0]
        n_match = min(nh, nl)
        ch, prh = comp90_pr(Xh[:n_match])
        cl, prl = comp90_pr(Xl[:n_match])
        h_cent = Xh.mean(axis=0)
        l_cent = Xl.mean(axis=0)
        cdist = cosine_dist(h_cent, l_cent)
        return {"label": label, "n": n_match,
                "human_comp90": ch, "llm_comp90": cl, "gap": ch - cl,
                "human_pr": prh, "llm_pr": prl,
                "centroid_dist": cdist}

    cells = []
    cells.append(cell_metrics(H_orig_subset, L_orig_subset, "human_orig_vs_llm_orig"))
    cells.append(cell_metrics(Hf, L_orig_subset, "human_formal_vs_llm_orig"))
    cells.append(cell_metrics(H_orig_subset, Lc, "human_orig_vs_llm_casual"))
    cells.append(cell_metrics(Hf, Lc, "human_formal_vs_llm_casual"))
    for c in cells:
        _flush(f"  {c['label']}: n={c['n']}, h={c['human_comp90']}, l={c['llm_comp90']}, "
               f"gap={c['gap']}, cent_dist={c['centroid_dist']:.4f}")

    # Per-pair distance: orig vs paraphrased of same rationale
    # Human: orig embedding vs formal-paraphrased embedding (paired)
    pair_h = float(np.mean([cosine_dist(H_orig_subset[i], Hf[i]) for i in range(min(H_orig_subset.shape[0], Hf.shape[0]))]))
    pair_l = float(np.mean([cosine_dist(L_orig_subset[i], Lc[i]) for i in range(min(L_orig_subset.shape[0], Lc.shape[0]))]))
    _flush(f"paired distance: human(orig->formal)={pair_h:.4f}, llm(orig->casual)={pair_l:.4f}")

    out = {
        "n_human_paraphrased": int(n_h_para),
        "n_llm_paraphrased": int(n_l_para),
        "cells": cells,
        "paired_paraphrase_distance": {
            "human_orig_to_formal": pair_h,
            "llm_orig_to_casual": pair_l,
        },
    }
    path_name = "milestone3_register_paraphrase_clean.json" if clean else "milestone3_register_paraphrase.json"
    path = OUTPUT_DIR / path_name
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all", help="comma-separated: modern,paraphrase,all")
    args = ap.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    steps = {"modern", "paraphrase"} if args.steps == "all" else set(args.steps.split(","))
    if "modern" in steps:
        modernity_analysis()
    if "paraphrase" in steps:
        register_paraphrase_analysis()
    _flush("ALL DONE")


if __name__ == "__main__":
    main()
