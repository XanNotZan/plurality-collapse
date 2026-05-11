"""A2: diff-vector PCA on (human, LLM) pairs within same Kaleido 60-cluster + same dilemma.

For each (submission_id, cluster_id) cell with ≥1 human + ≥1 LLM comment, form pairs.
Sample-cap per cell to avoid large-cell dominance.

Compute diff = h_emb_norm − l_emb_norm.  PCA on stacked diffs.

Robustness:
  - Two PCAs: raw (PC1 = global human-LLM offset) and centered (variation around offset)
  - Scrambled baseline: pair (h, l) from different dilemmas + clusters
  - Bootstrap PC stability (N=100 boots), measure cosine similarity of top-K PCs to full-data PCs
  - Sample-size sweep (25/50/75/100%) of cells
  - Sweep across max_pairs_per_cell ∈ {1, 5, 10, 20}
  - Top-PC exemplars: most positive and most negative projection comments
  - Lexical correlation: regress each lexical-feature delta (h − l) on each PC projection

Outputs:
  data/analysis/milestone3_diff_pca.json   (PC eigenvalues, stability, baseline ratios)
  data/analysis/milestone3_diff_pca_exemplars.json   (top-PC exemplar comments)
  data/analysis/milestone3_diff_pca_lexcorr.csv      (PC-feature correlations)
  data/analysis/milestone3_diff_pca_projection.npz   (PC matrix, projections)
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

EMBEDDINGS_DIR = Path("data/embeddings")
ANALYSIS = Path("data/analysis")
ARCTIC_DIR = Path("data/arcticshift")
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
RNG = np.random.RandomState(42)

MAX_PAIRS_PER_CELL = 10
N_PCS = 10
N_BOOT_STABILITY = 100
EXEMPLAR_K = 8

logger = logging.getLogger("diff_pca")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg); sys.stdout.flush()


def l2_normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + 1e-12)


def load_cluster_map():
    df = pd.read_csv(ANALYSIS / "value_label_clusters.csv")
    return dict(zip(df["value"], df["cluster_id"])), dict(zip(df["cluster_id"], df["cluster_label"]))


def build_pair_index(cluster_map):
    """Return dict[(sub, cluster_id)] -> {"h": [arctic_idx,...], "l": [(idx, source), ...]}"""
    cells = defaultdict(lambda: {"h": [], "l": []})

    # ─ humans
    arctic_meta = json.loads((EMBEDDINGS_DIR / "human_arctic_meta.json").read_text())
    decoded = json.loads((ANALYSIS / "milestone3_arctic_decoded_values.json").read_text())
    n_h = 0
    for m in arctic_meta:
        idx = m["index"]
        val = decoded.get(str(idx))
        if not val: continue
        cid = cluster_map.get(val)
        if cid is None: continue
        cells[(m["submission_id"], int(cid))]["h"].append(idx)
        n_h += 1
    _flush(f"human comments with decoded value + cluster: {n_h}")

    # ─ LLMs
    n_l = 0
    arctic_subs = {m["submission_id"] for m in arctic_meta}
    for s in LLM_SOURCES:
        data = json.loads((ANALYSIS / f"llm_values_{s}.json").read_text())
        for r in data:
            if r["submission_id"] not in arctic_subs: continue
            val = r.get("generated_values")
            if not val: continue
            cid = cluster_map.get(val)
            if cid is None: continue
            cells[(r["submission_id"], int(cid))]["l"].append((r["index"], s))
            n_l += 1
    _flush(f"LLM rationales with decoded value + cluster + arctic submission: {n_l}")
    _flush(f"total cells: {len(cells)}")

    shared = {k: v for k, v in cells.items() if v["h"] and v["l"]}
    _flush(f"cells with both human + LLM: {len(shared)}")
    return shared


def sample_pairs(cells, max_pairs_per_cell=MAX_PAIRS_PER_CELL, rng=RNG):
    pairs = []
    for (sid, cid), v in cells.items():
        h_idxs = v["h"]; l_items = v["l"]
        all_pairs = [(h, li, ls) for h in h_idxs for (li, ls) in l_items]
        if len(all_pairs) > max_pairs_per_cell:
            sel = rng.choice(len(all_pairs), size=max_pairs_per_cell, replace=False)
            all_pairs = [all_pairs[i] for i in sel]
        for h, li, ls in all_pairs:
            pairs.append({"submission_id": sid, "cluster_id": cid,
                          "h_idx": h, "l_idx": li, "l_source": ls})
    return pairs


def build_diff_matrix(pairs, H_emb, llm_embs):
    n = len(pairs)
    D = np.zeros((n, H_emb.shape[1]), dtype=np.float32)
    Hn = l2_normalize(H_emb.astype(np.float32))
    Ln = {s: l2_normalize(llm_embs[s].astype(np.float32)) for s in LLM_SOURCES}
    for i, p in enumerate(pairs):
        D[i] = Hn[p["h_idx"]] - Ln[p["l_source"]][p["l_idx"]]
    return D


def pca_summary(D, n_pcs=N_PCS, center=False):
    if center:
        D_in = D - D.mean(axis=0, keepdims=True)
    else:
        D_in = D
    pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=42)
    proj = pca.fit_transform(D_in)
    return pca, proj


def cosine_match(A, B):
    """Return greedy max-cosine matching between rows of A and rows of B."""
    sims = np.abs(A @ B.T) / (np.linalg.norm(A, axis=1, keepdims=True) * np.linalg.norm(B, axis=1)[None, :] + 1e-12)
    out = []
    A_used = set(); B_used = set()
    flat = [(sims[i, j], i, j) for i in range(sims.shape[0]) for j in range(sims.shape[1])]
    flat.sort(reverse=True)
    for s, i, j in flat:
        if i in A_used or j in B_used: continue
        A_used.add(i); B_used.add(j)
        out.append({"pc_a": int(i), "pc_b": int(j), "cos": float(s)})
        if len(out) == min(sims.shape):
            break
    return out


def bootstrap_stability(D, ref_pca, n_pcs, n_boot=N_BOOT_STABILITY, center=False, rng=RNG):
    n = D.shape[0]
    ref_comp = ref_pca.components_[:n_pcs]
    matches = []
    for b in range(n_boot):
        sel = rng.choice(n, size=n, replace=True)
        Db = D[sel]
        if center:
            Db = Db - Db.mean(axis=0, keepdims=True)
        pca_b = PCA(n_components=n_pcs, svd_solver="randomized", random_state=b)
        pca_b.fit(Db)
        m = cosine_match(ref_comp, pca_b.components_)
        # PC-level: cosine to nearest match
        per_pc = sorted(m, key=lambda x: x["pc_a"])
        matches.append([x["cos"] for x in per_pc])
    M = np.array(matches)  # [n_boot, n_pcs]
    return {
        "pc_mean_cosine_to_full": [float(M[:, k].mean()) for k in range(n_pcs)],
        "pc_p05_cosine_to_full": [float(np.percentile(M[:, k], 5)) for k in range(n_pcs)],
    }


def scrambled_baseline(cells, H_emb, llm_embs, max_pairs=MAX_PAIRS_PER_CELL, rng=RNG):
    """Pair humans and LLMs from RANDOM different dilemmas (no shared cluster).
    Same total pair count as paired-PCA for comparability.
    """
    h_pool = [(idx, sid, cid) for (sid, cid), v in cells.items() for idx in v["h"]]
    l_pool = [(idx, src, sid, cid) for (sid, cid), v in cells.items() for (idx, src) in v["l"]]
    n_target = sum(min(len(v["h"]) * len(v["l"]), max_pairs) for v in cells.values())
    pairs = []
    rng.shuffle(h_pool); rng.shuffle(l_pool)
    i_h = 0; i_l = 0
    while len(pairs) < n_target and i_h < len(h_pool) and i_l < len(l_pool):
        h_idx, h_sid, h_cid = h_pool[i_h]
        l_idx, l_src, l_sid, l_cid = l_pool[i_l]
        if h_sid != l_sid:
            pairs.append({"h_idx": h_idx, "l_idx": l_idx, "l_source": l_src,
                          "h_cell": (h_sid, h_cid), "l_cell": (l_sid, l_cid)})
            i_h += 1; i_l += 1
        else:
            i_l += 1
    Hn = l2_normalize(H_emb.astype(np.float32))
    Ln = {s: l2_normalize(llm_embs[s].astype(np.float32)) for s in LLM_SOURCES}
    D = np.array([Hn[p["h_idx"]] - Ln[p["l_source"]][p["l_idx"]] for p in pairs], dtype=np.float32)
    return D, pairs


def load_texts(pairs):
    """Load h and l text for given pairs from sources."""
    arctic_recs = {}
    with open(ARCTIC_DIR / "filtered_comments.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: j = json.loads(line)
            except Exception: continue
            if j.get("_empty") or not j.get("body"): continue
            arctic_recs[(j["submission_id"], j.get("comment_id"))] = j["body"]
    arctic_meta = json.loads((EMBEDDINGS_DIR / "human_arctic_meta.json").read_text())
    h_idx_to_text = {}
    for m in arctic_meta:
        body = arctic_recs.get((m["submission_id"], m.get("comment_id")))
        if body: h_idx_to_text[m["index"]] = body

    llm_idx_to_text = {s: {} for s in LLM_SOURCES}
    for s in LLM_SOURCES:
        data = json.loads((ANALYSIS / f"llm_values_{s}.json").read_text())
        for r in data:
            llm_idx_to_text[s][r["index"]] = r["rationale_text"]

    return h_idx_to_text, llm_idx_to_text


def extract_exemplars(pairs, proj, h_text_map, l_text_map, n_pcs=N_PCS, k=EXEMPLAR_K):
    exemplars = {}
    for pc_i in range(n_pcs):
        scores = proj[:, pc_i]
        order_pos = np.argsort(-scores)
        order_neg = np.argsort(scores)
        pos_ex = []
        neg_ex = []
        for j in order_pos[:k]:
            p = pairs[j]
            pos_ex.append({
                "score": float(scores[j]),
                "submission_id": p["submission_id"],
                "cluster_id": p["cluster_id"],
                "l_source": p["l_source"],
                "human_text": h_text_map.get(p["h_idx"], "")[:600],
                "llm_text": l_text_map.get(p["l_source"], {}).get(p["l_idx"], "")[:600],
            })
        for j in order_neg[:k]:
            p = pairs[j]
            neg_ex.append({
                "score": float(scores[j]),
                "submission_id": p["submission_id"],
                "cluster_id": p["cluster_id"],
                "l_source": p["l_source"],
                "human_text": h_text_map.get(p["h_idx"], "")[:600],
                "llm_text": l_text_map.get(p["l_source"], {}).get(p["l_idx"], "")[:600],
            })
        exemplars[f"PC{pc_i+1}"] = {"positive": pos_ex, "negative": neg_ex}
    return exemplars


def lexical_correlate(pairs, proj, lex_path, n_pcs=N_PCS):
    """For each lexical feature, compute correlation between (h_feat - l_feat) and PC projection."""
    if not lex_path.exists():
        _flush(f"lexical features not yet available at {lex_path}; skip correlation")
        return None
    df = pd.read_csv(lex_path)
    # build per-(source, embed_idx) lookup
    df_h = df[df["source"] == "human"].set_index("embed_idx")
    feature_cols = [c for c in df.columns if c.endswith("_density") or c in ("mean_word_len", "type_token_ratio", "mean_sent_len", "token_count")]
    h_feat = {}
    for fc in feature_cols:
        h_feat[fc] = df_h[fc].to_dict()
    l_feat = {s: {} for s in LLM_SOURCES}
    for s in LLM_SOURCES:
        sub = df[df["source"] == s].set_index("embed_idx")
        for fc in feature_cols:
            l_feat[s][fc] = sub[fc].to_dict()

    rows = []
    n = len(pairs)
    deltas = {fc: np.zeros(n, dtype=np.float64) for fc in feature_cols}
    valid = np.ones(n, dtype=bool)
    for i, p in enumerate(pairs):
        for fc in feature_cols:
            h_v = h_feat[fc].get(p["h_idx"])
            l_v = l_feat[p["l_source"]][fc].get(p["l_idx"])
            if h_v is None or l_v is None or not np.isfinite(h_v) or not np.isfinite(l_v):
                valid[i] = False
                continue
            deltas[fc][i] = h_v - l_v
    for fc in feature_cols:
        d = deltas[fc][valid]
        for pc_i in range(n_pcs):
            pc_v = proj[valid, pc_i]
            if d.std() > 0 and pc_v.std() > 0:
                r = float(np.corrcoef(d, pc_v)[0, 1])
            else:
                r = float("nan")
            rows.append({"feature": fc, "pc": f"PC{pc_i+1}", "pearson_r": r, "n": int(len(d))})
    return pd.DataFrame(rows)


def main():
    cluster_map, _cluster_labels = load_cluster_map()
    cells = build_pair_index(cluster_map)

    _flush("loading human + LLM embeddings")
    H_emb = np.load(EMBEDDINGS_DIR / "human_arctic.npy")
    llm_embs = {s: np.load(EMBEDDINGS_DIR / f"{s}.npy") for s in LLM_SOURCES}
    _flush(f"  H_emb: {H_emb.shape}, LLM shapes: {[llm_embs[s].shape for s in LLM_SOURCES]}")

    pairs = sample_pairs(cells, max_pairs_per_cell=MAX_PAIRS_PER_CELL)
    _flush(f"paired (≤{MAX_PAIRS_PER_CELL}/cell): n_pairs={len(pairs)}")

    D = build_diff_matrix(pairs, H_emb, llm_embs)
    _flush(f"diff matrix: {D.shape}")

    # Raw PCA (mean offset preserved)
    pca_raw, proj_raw = pca_summary(D, n_pcs=N_PCS, center=False)
    _flush(f"raw PCA explained var: {[float(x) for x in pca_raw.explained_variance_ratio_]}")

    # Centered PCA (offset removed)
    pca_cen, proj_cen = pca_summary(D, n_pcs=N_PCS, center=True)
    _flush(f"centered PCA explained var: {[float(x) for x in pca_cen.explained_variance_ratio_]}")

    # PC1 of raw is global offset → its direction is mean(D) / |mean(D)|
    mean_D = D.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_D))
    pc1_offset_cos = float(np.abs(pca_raw.components_[0] @ (mean_D / (mean_norm + 1e-12))))
    _flush(f"raw PC1 cosine to mean-offset direction: {pc1_offset_cos:.4f}, mean-norm={mean_norm:.4f}")

    # Bootstrap stability for centered PCA
    _flush("bootstrap stability (centered)")
    stab = bootstrap_stability(D, pca_cen, n_pcs=N_PCS, center=True)
    _flush(f"  PC stability cosines (mean): {stab['pc_mean_cosine_to_full']}")

    # Sample-size sweep
    sweep = {}
    for frac in [0.25, 0.5, 0.75]:
        n_sel = int(frac * D.shape[0])
        sel = RNG.choice(D.shape[0], size=n_sel, replace=False)
        Dsub = D[sel]
        pca_sub, _ = pca_summary(Dsub, n_pcs=N_PCS, center=True)
        m = cosine_match(pca_cen.components_, pca_sub.components_)
        m = sorted(m, key=lambda x: x["pc_a"])
        sweep[f"frac_{int(frac*100)}"] = [float(x["cos"]) for x in m]
    _flush(f"sample-size sweep (cosines to full PCs): {sweep}")

    # Max-pairs sweep
    pairs_sweep = {}
    for mpc in [1, 5, 20]:
        psweep = sample_pairs(cells, max_pairs_per_cell=mpc, rng=np.random.RandomState(7))
        Dsw = build_diff_matrix(psweep, H_emb, llm_embs)
        pca_sw, _ = pca_summary(Dsw, n_pcs=N_PCS, center=True)
        m = cosine_match(pca_cen.components_, pca_sw.components_)
        m = sorted(m, key=lambda x: x["pc_a"])
        pairs_sweep[f"max_pairs_{mpc}"] = {
            "n_pairs": int(len(psweep)),
            "pc_cosines": [float(x["cos"]) for x in m],
            "explained_var": [float(v) for v in pca_sw.explained_variance_ratio_],
        }
    _flush(f"max-pairs sweep: {[(k, v['n_pairs']) for k, v in pairs_sweep.items()]}")

    # Scrambled baseline
    _flush("scrambled baseline (cross-dilemma pairing)")
    D_scr, scr_pairs = scrambled_baseline(cells, H_emb, llm_embs, max_pairs=MAX_PAIRS_PER_CELL)
    _flush(f"  scrambled diff matrix: {D_scr.shape}")
    pca_scr, _ = pca_summary(D_scr, n_pcs=N_PCS, center=True)
    _flush(f"  scrambled explained var: {[float(x) for x in pca_scr.explained_variance_ratio_]}")
    # Cosine of scrambled top-PCs to paired top-PCs
    m_scr = cosine_match(pca_cen.components_, pca_scr.components_)
    m_scr = sorted(m_scr, key=lambda x: x["pc_a"])
    scr_cosines = [float(x["cos"]) for x in m_scr]
    _flush(f"  cosines (paired vs scrambled): {scr_cosines}")

    # Texts + exemplars
    _flush("loading texts for exemplars")
    h_text_map, l_text_map = load_texts(pairs)
    exemplars = extract_exemplars(pairs, proj_cen, h_text_map, l_text_map, n_pcs=N_PCS, k=EXEMPLAR_K)
    (ANALYSIS / "milestone3_diff_pca_exemplars.json").write_text(json.dumps(exemplars, indent=2))
    _flush("saved exemplars")

    # Save PC components + projections (small files)
    np.savez_compressed(ANALYSIS / "milestone3_diff_pca_projection.npz",
                        components_raw=pca_raw.components_.astype(np.float32),
                        components_cen=pca_cen.components_.astype(np.float32),
                        proj_raw=proj_raw.astype(np.float32),
                        proj_cen=proj_cen.astype(np.float32),
                        mean_D=mean_D.astype(np.float32))

    out = {
        "n_cells": len(cells),
        "n_pairs": len(pairs),
        "max_pairs_per_cell": MAX_PAIRS_PER_CELL,
        "mean_offset_norm": mean_norm,
        "raw_pc1_cos_to_mean_offset": pc1_offset_cos,
        "raw_explained_var": [float(v) for v in pca_raw.explained_variance_ratio_],
        "centered_explained_var": [float(v) for v in pca_cen.explained_variance_ratio_],
        "scrambled_explained_var": [float(v) for v in pca_scr.explained_variance_ratio_],
        "scrambled_pc_cosines_vs_paired": scr_cosines,
        "bootstrap_stability_centered": stab,
        "sample_size_sweep_cosines": sweep,
        "max_pairs_sweep": pairs_sweep,
    }
    (ANALYSIS / "milestone3_diff_pca.json").write_text(json.dumps(out, indent=2))
    _flush("saved milestone3_diff_pca.json")

    # Lexical correlation (only if lexical CSV available)
    lex_path = ANALYSIS / "milestone3_lexical_features.csv"
    corr = lexical_correlate(pairs, proj_cen, lex_path, n_pcs=N_PCS)
    if corr is not None:
        corr.to_csv(ANALYSIS / "milestone3_diff_pca_lexcorr.csv", index=False)
        _flush(f"saved lexical-PC correlations  rows={len(corr)}")


if __name__ == "__main__":
    main()
