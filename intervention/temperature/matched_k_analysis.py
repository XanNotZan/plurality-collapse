"""Matched-K control analyses.

Addresses two K-confounds in the temperature intervention:
  1. Qwen cosine across T was measured at variable K (3/5/8/12). Higher T = more
     pairs = better estimator. Recompute Qwen cosine at matched K=3 (and K=2)
     across all temperatures.
  2. Human ArcticShift cosine was averaged over all pairs per dilemma (median 34).
     Recompute human cosine at K=3 to match Qwen.

Also K-sweeps human cluster diversity (K=2..12) so the cluster_diversity panel
has an apples-to-apples human reference at each K.

Outputs:
  data/analysis/intervention/temperature/matched_k_cosine.json
  data/analysis/intervention/temperature/human_cluster_diversity_sweep.json
"""

import csv
import gc
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

EMB_DIR_INT = Path("data/embeddings/intervention/temperature")
EMB_DIR = Path("data/embeddings")
ANALYSIS = Path("data/analysis/intervention/temperature")
ANALYSIS_BASE = Path("data/analysis")
ANALYSIS.mkdir(parents=True, exist_ok=True)

TEMPERATURES = [0.3, 0.7, 1.0, 1.3]
K_PER_TEMP = {0.3: 3, 0.7: 5, 1.0: 8, 1.3: 12}
MATCHED_K_LIST = [2, 3]
HUMAN_K_SWEEP = list(range(2, 13))
N_SEEDS = 50
BOOTSTRAP = 500
RNG_SEED = 42

logger = logging.getLogger("matched_k")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg); sys.stdout.flush()


def normalize_value(v):
    return re.sub(r"\s+", " ", v.strip().lower())


def load_value_cluster_map():
    val_to_cluster = {}
    with open(ANALYSIS_BASE / "value_label_clusters.csv", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            val_to_cluster[normalize_value(row["value"])] = int(row["cluster_id"])
    return val_to_cluster


def pairwise_cos_dist(E_subset):
    """Mean (1 - cosine) over all pairs in the rows of E_subset."""
    norms = np.linalg.norm(E_subset, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    En = E_subset / norms
    cos = En @ En.T
    iu = np.triu_indices_from(cos, k=1)
    return float((1.0 - cos[iu]).mean())


# ---------------------------------------------------------------------------
# Matched-K cosine for Qwen
# ---------------------------------------------------------------------------
def matched_k_cosine_qwen():
    rng = np.random.RandomState(RNG_SEED)
    out_T = {}
    for T in TEMPERATURES:
        emb_path = EMB_DIR_INT / f"qwen25_3b_T{T:.1f}.npy"
        meta_path = EMB_DIR_INT / f"qwen25_3b_T{T:.1f}_meta.json"
        if not emb_path.exists():
            _flush(f"missing {emb_path}, skip"); continue
        E = np.load(emb_path)
        meta = json.loads(meta_path.read_text())
        by_sub = defaultdict(list)
        for m in meta:
            by_sub[m["submission_id"]].append(m["index"])
        K_max = K_PER_TEMP[T]
        per_T = {"K_max": K_max, "by_K": {}}
        for K in MATCHED_K_LIST:
            if K > K_max:
                continue
            per_dilemma = []
            for sid, idxs in by_sub.items():
                if len(idxs) < K:
                    continue
                if len(idxs) == K:
                    per_dilemma.append(pairwise_cos_dist(E[idxs]))
                else:
                    seed_res = []
                    for _ in range(N_SEEDS):
                        choice = rng.choice(len(idxs), size=K, replace=False)
                        seed_res.append(pairwise_cos_dist(E[[idxs[i] for i in choice]]))
                    per_dilemma.append(float(np.mean(seed_res)))
            arr = np.array(per_dilemma)
            boot = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                             for _ in range(BOOTSTRAP)])
            per_T["by_K"][f"K_{K}"] = {
                "mean_cosine_distance": float(arr.mean()),
                "std": float(arr.std()),
                "bootstrap_ci_95": (float(np.percentile(boot, 2.5)),
                                    float(np.percentile(boot, 97.5))),
                "n_dilemmas": int(len(arr)),
            }
            _flush(f"Qwen T={T} matched K={K}: mean cos = {arr.mean():.4f} "
                   f"(95% CI {per_T['by_K'][f'K_{K}']['bootstrap_ci_95'][0]:.4f}"
                   f"-{per_T['by_K'][f'K_{K}']['bootstrap_ci_95'][1]:.4f})")
        out_T[f"T_{T:.1f}"] = per_T
    return out_T


# ---------------------------------------------------------------------------
# Matched-K cosine for human ArcticShift
# ---------------------------------------------------------------------------
def matched_k_cosine_human():
    rng = np.random.RandomState(RNG_SEED + 1)
    emb_path = EMB_DIR / "human_arctic.npy"
    meta_path = EMB_DIR / "human_arctic_meta.json"
    E = np.load(emb_path)
    meta = json.loads(meta_path.read_text())
    by_sub = defaultdict(list)
    for m in meta:
        by_sub[m["submission_id"]].append(m["index"])
    out = {"by_K": {}}
    for K in MATCHED_K_LIST:
        per_dilemma = []
        for sid, idxs in by_sub.items():
            if len(idxs) < K:
                continue
            if len(idxs) == K:
                per_dilemma.append(pairwise_cos_dist(E[idxs]))
            else:
                seed_res = []
                for _ in range(N_SEEDS):
                    choice = rng.choice(len(idxs), size=K, replace=False)
                    seed_res.append(pairwise_cos_dist(E[[idxs[i] for i in choice]]))
                per_dilemma.append(float(np.mean(seed_res)))
        arr = np.array(per_dilemma)
        boot = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                         for _ in range(BOOTSTRAP)])
        out["by_K"][f"K_{K}"] = {
            "mean_cosine_distance": float(arr.mean()),
            "std": float(arr.std()),
            "bootstrap_ci_95": (float(np.percentile(boot, 2.5)),
                                float(np.percentile(boot, 97.5))),
            "n_dilemmas": int(len(arr)),
        }
        _flush(f"Human matched K={K}: mean cos = {arr.mean():.4f} "
               f"(95% CI {out['by_K'][f'K_{K}']['bootstrap_ci_95'][0]:.4f}"
               f"-{out['by_K'][f'K_{K}']['bootstrap_ci_95'][1]:.4f}) n={len(arr)}")
    return out


# ---------------------------------------------------------------------------
# Human cluster diversity K-sweep
# ---------------------------------------------------------------------------
def human_cluster_diversity_sweep():
    val_to_cluster = load_value_cluster_map()
    decoded = json.loads((ANALYSIS_BASE / "milestone3_arctic_decoded_values.json").read_text())
    arctic_meta = json.loads((EMB_DIR / "human_arctic_meta.json").read_text())
    # decoded is keyed by string index into arctic_meta (the m3 numbering).
    # Build {submission_id: [cluster_id, ...]} for each comment with a decoded value.
    grouped = defaultdict(list)
    n_unmapped = 0
    for i, m in enumerate(arctic_meta):
        v = decoded.get(str(i))
        if not v:
            continue
        cid = val_to_cluster.get(normalize_value(v), -1)
        if cid < 0:
            n_unmapped += 1
        grouped[m["submission_id"]].append(cid)
    _flush(f"human comments with decoded values: {sum(len(v) for v in grouped.values())} "
           f"across {len(grouped)} dilemmas (unmapped values: {n_unmapped})")
    rng = np.random.RandomState(RNG_SEED + 2)
    out = {"by_K": {}}
    for K in HUMAN_K_SWEEP:
        per_dilemma = []
        for sid, cids in grouped.items():
            if len(cids) < K:
                continue
            if len(cids) == K:
                per_dilemma.append(len(set(cids)))
            else:
                seed_res = []
                for _ in range(N_SEEDS):
                    choice = rng.choice(len(cids), size=K, replace=False)
                    seed_res.append(len(set(cids[i] for i in choice)))
                per_dilemma.append(float(np.mean(seed_res)))
        arr = np.array(per_dilemma)
        boot = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                         for _ in range(BOOTSTRAP)])
        out["by_K"][f"K_{K}"] = {
            "mean_distinct_clusters": float(arr.mean()),
            "std": float(arr.std()),
            "bootstrap_ci_95": (float(np.percentile(boot, 2.5)),
                                float(np.percentile(boot, 97.5))),
            "n_dilemmas": int(len(arr)),
        }
        _flush(f"Human cluster K={K}: mean = {arr.mean():.3f} "
               f"(95% CI {out['by_K'][f'K_{K}']['bootstrap_ci_95'][0]:.3f}"
               f"-{out['by_K'][f'K_{K}']['bootstrap_ci_95'][1]:.3f}) n={len(arr)}")
    return out


def main():
    _flush("=== Matched-K cosine: Qwen ===")
    q = matched_k_cosine_qwen()
    _flush("=== Matched-K cosine: Human ===")
    h = matched_k_cosine_human()
    out = {"qwen_per_temperature": q, "human": h}
    (ANALYSIS / "matched_k_cosine.json").write_text(json.dumps(out, indent=2))
    _flush(f"saved {ANALYSIS / 'matched_k_cosine.json'}")

    _flush("=== Human cluster diversity K-sweep ===")
    hc = human_cluster_diversity_sweep()
    (ANALYSIS / "human_cluster_diversity_sweep.json").write_text(json.dumps(hc, indent=2))
    _flush(f"saved {ANALYSIS / 'human_cluster_diversity_sweep.json'}")

    _flush("MATCHED-K ANALYSIS DONE")


if __name__ == "__main__":
    main()
