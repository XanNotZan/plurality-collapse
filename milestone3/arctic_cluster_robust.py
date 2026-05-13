"""Cluster-level robustness suite for sub-foundation diversity claim.

Tests:
  1) Bootstrap CIs on cluster-diversity ratio + Jaccard
  2) k-size sensitivity (k=5, 10, 15, 20)
  3) Cluster ratio stratified by consensus level
  4) Cluster ratio stratified by verdict
  5) Within-same-cluster cosine (pairs of humans in cluster X vs LLM pairs in cluster X)
  6) Cluster-granularity sweep (k_clusters=20, 40, 100, 200) via SBERT k-means on value labels
  7) Cluster diversity entropy-weighted (not just count)
"""

import csv
import gc
import json
import logging
import re
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

EMBEDDINGS_DIR = Path("data/embeddings")
ANALYSIS = Path("data/analysis")
ARCTIC_DIR = Path("data/arcticshift")
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_BOOTSTRAP = 1000
RNG = np.random.RandomState(42)

logger = logging.getLogger("cluster_robust")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg); sys.stdout.flush()


def normalize_value(v):
    return re.sub(r"\s+", " ", v.strip().lower())


def cosine_pairwise_mean(E):
    if E.shape[0] < 2:
        return float("nan")
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    sim = En @ En.T
    n = sim.shape[0]
    return float(1 - sim[np.triu_indices(n, k=1)].mean())


def shannon_ent(counts_arr):
    s = counts_arr.sum()
    if s == 0:
        return 0.0
    p = counts_arr / s
    return float(-np.sum(p[p > 0] * np.log(p[p > 0])))


def main():
    # Load existing decode output for humans (saved during cluster_diversity run)
    # The previous script decoded values for the unique humans; we have arctic embeddings + texts already.
    # Re-decode is in arctic_cluster_diversity.py; here we reuse via the saved freq tables won't work.
    # We need to re-run minimal decode OR redo from scratch.
    # Easier: load decoded values from a cached file if exists, else fail.

    cache_path = ANALYSIS / "milestone3_arctic_decoded_values.json"
    if cache_path.exists():
        _flush(f"loading cached decoded values: {cache_path}")
        cache = json.loads(cache_path.read_text())
        arctic_idx_to_value = {int(k): v for k, v in cache.items()}
    else:
        # Need to re-decode. Skip if cache not built. Build it here.
        _flush("decoded values cache not found - script requires arctic_cluster_diversity.py run first")
        _flush("attempting to rebuild from milestone3_per_dilemma_cluster_diversity.json frequency tables (not exact)")
        # Cannot reconstruct per-comment values from frequency aggregates. Need full decode.
        # Re-decode here.
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _flush("loading Kaleido for re-decode")
        tok = AutoTokenizer.from_pretrained("allenai/kaleido-xl")
        model = AutoModelForSeq2SeqLM.from_pretrained("allenai/kaleido-xl", dtype=torch.float16).to("cuda").eval()
        try:
            template = model.config.task_specific_params["generate"]["template"]
        except Exception:
            template = "[Generate]:\tAction: ACTION"

        # Load arctic data
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

        idx_to_text = {}
        for m in arctic_meta:
            rec = arctic_recs.get((m["submission_id"], m.get("comment_id")))
            if rec:
                idx_to_text[m["index"]] = rec["body"]

        # Determine which indices to decode: same plan as cluster_diversity script
        h_by_sub = defaultdict(list)
        for m in arctic_meta:
            sid = m["submission_id"]
            if m["index"] in idx_to_text:
                h_by_sub[sid].append(m["index"])

        # Use existing LLM values to identify qualified dilemmas
        # We need >= 10 per side
        from collections import defaultdict as dd
        l_count = dd(int)
        for s in LLM_SOURCES:
            data = json.loads((ANALYSIS / f"llm_values_{s}.json").read_text())
            for e in data:
                l_count[e["submission_id"]] += 1
        qualified = sorted([s for s in h_by_sub if len(h_by_sub[s]) >= 10 and l_count.get(s, 0) >= 10])
        _flush(f"qualified dilemmas: {len(qualified)}")

        # Decode unique 50-per-dilemma humans
        K_SAMPLE = 10
        N_SAMPLE_SEEDS = 5
        to_decode = set()
        for sid in qualified:
            humans = h_by_sub[sid]
            n = len(humans)
            for seed in range(N_SAMPLE_SEEDS):
                r = np.random.RandomState(seed)
                idx = r.choice(n, K_SAMPLE, replace=False)
                for i in idx:
                    to_decode.add(humans[i])
        decode_list = sorted(to_decode)
        texts = [idx_to_text[i] for i in decode_list]
        _flush(f"decoding {len(texts)} human comments")
        out = []
        for s in range(0, len(texts), 24):
            b = texts[s:s+24]
            formatted = [template.replace("ACTION", t if t else "(empty)") for t in b]
            inputs = tok(formatted, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=24)
            decoded = tok.batch_decode(outputs, skip_special_tokens=True)
            for d in decoded:
                d = d.strip()
                if d.lower().startswith("value:"):
                    d = d[len("value:"):].strip()
                out.append(d)
            if (s // 24) % 100 == 0:
                _flush(f"decode {s+len(b)}/{len(texts)}")
        arctic_idx_to_value = dict(zip(decode_list, out))
        cache_path.write_text(json.dumps({str(k): v for k, v in arctic_idx_to_value.items()}))
        del model, tok
        gc.collect(); torch.cuda.empty_cache()
        _flush(f"saved cache to {cache_path}")

    # Load arctic metadata + records for cluster-level analysis
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

    # Load arctic embeddings (for within-cluster cosine)
    H_emb = np.load(EMBEDDINGS_DIR / "human_arctic.npy")

    # Build h_by_sub with idx + body
    h_by_sub = defaultdict(list)
    for m in arctic_meta:
        rec = arctic_recs.get((m["submission_id"], m.get("comment_id")))
        if rec:
            h_by_sub[m["submission_id"]].append({"idx": m["index"], "body": rec["body"]})

    # Load LLM values + texts + embeddings
    _flush("loading LLM values + texts + embeddings")
    ds = load_dataset("ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas", split="test")
    sub_to_row = {ds[i]["submission_id"]: i for i in range(len(ds))}
    needed = set()
    for s in LLM_SOURCES:
        for c in [f"{s}_reason_1", f"{s}_reason_2", f"{s}_reason_3"]:
            if c in ds.column_names:
                needed.add(c)
    col_data = {c: ds[c] for c in needed}

    LLM_metas = {s: json.loads((EMBEDDINGS_DIR / f"{s}_meta.json").read_text()) for s in LLM_SOURCES}
    LLM_embs = {s: np.load(EMBEDDINGS_DIR / f"{s}.npy") for s in LLM_SOURCES}

    # Reload LLM decoded values
    llm_values_by_subcol = {}  # (sid, col) -> value
    for s in LLM_SOURCES:
        data = json.loads((ANALYSIS / f"llm_values_{s}.json").read_text())
        for e in data:
            llm_values_by_subcol[(e["submission_id"], e["column"])] = e["generated_values"]

    l_by_sub = defaultdict(list)
    for s in LLM_SOURCES:
        for m in LLM_metas[s]:
            sid = m["submission_id"]
            if sid not in h_by_sub:
                continue
            col = m["column"]
            v = llm_values_by_subcol.get((sid, col), "")
            E = LLM_embs[s][m["index"]]
            l_by_sub[sid].append({"source": s, "value": v, "emb": E, "col": col})

    qualified = sorted([s for s in h_by_sub if len(h_by_sub[s]) >= 10 and len(l_by_sub.get(s, [])) >= 10])
    _flush(f"qualified dilemmas: {len(qualified)}")

    # Existing 60-cluster mapping
    val_to_cluster_60 = {}
    with open(ANALYSIS / "value_label_clusters.csv", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            val_to_cluster_60[normalize_value(row["value"])] = int(row["cluster_id"])

    # SBERT NN for unmapped values
    all_values_seen = set()
    for v in arctic_idx_to_value.values():
        all_values_seen.add(v)
    for items in l_by_sub.values():
        for it in items:
            all_values_seen.add(it["value"])
    unmapped = [v for v in all_values_seen if normalize_value(v) not in val_to_cluster_60]
    if unmapped:
        _flush(f"SBERT NN for {len(unmapped)} unmapped values")
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer(SBERT_MODEL, device="cuda")
        known = list(val_to_cluster_60.keys())
        known_emb = sbert.encode(known, convert_to_numpy=True, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        unmapped_emb = sbert.encode(unmapped, convert_to_numpy=True, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        for i, v in enumerate(unmapped):
            sims = known_emb @ unmapped_emb[i]
            j = int(np.argmax(sims))
            val_to_cluster_60[normalize_value(v)] = val_to_cluster_60[known[j]]
        del sbert; gc.collect(); torch.cuda.empty_cache()

    def cluster60(v):
        return val_to_cluster_60[normalize_value(v)]

    K_SAMPLE = 10
    N_SEEDS = 5

    def per_dilemma_cluster_diversity(qualified_set, cluster_fn, k=K_SAMPLE, n_seeds=N_SEEDS):
        """Return per-dilemma (mean_h_clusters, mean_l_clusters) over n_seeds."""
        rows = []
        for sid in qualified_set:
            humans = h_by_sub[sid]
            l_items = l_by_sub[sid]
            if len(humans) < k or len(l_items) < k:
                continue
            h_counts = []; l_counts = []
            h_jaccards = []
            for seed in range(n_seeds):
                rh = np.random.RandomState(seed)
                rl = np.random.RandomState(seed + 1000)
                h_idx = rh.choice(len(humans), k, replace=False)
                l_idx = rl.choice(len(l_items), k, replace=False)
                h_vals = [arctic_idx_to_value.get(humans[i]["idx"]) for i in h_idx]
                l_vals = [l_items[i]["value"] for i in l_idx]
                hc = {cluster_fn(v) for v in h_vals if v}
                lc = {cluster_fn(v) for v in l_vals if v}
                h_counts.append(len(hc))
                l_counts.append(len(lc))
                if hc and lc:
                    h_jaccards.append(len(hc & lc) / len(hc | lc))
            rows.append({
                "sid": sid,
                "h_mean": float(np.mean(h_counts)),
                "l_mean": float(np.mean(l_counts)),
                "jaccard_mean": float(np.mean(h_jaccards)) if h_jaccards else float("nan"),
            })
        return rows

    out = {}

    # --- 1. Bootstrap CIs on 60-cluster ratio + Jaccard ------------------------
    _flush("=== 1. Bootstrap CIs on 60-cluster ratio + Jaccard ===")
    rows = per_dilemma_cluster_diversity(qualified, cluster60)
    h_arr = np.array([r["h_mean"] for r in rows])
    l_arr = np.array([r["l_mean"] for r in rows])
    jaccard_arr = np.array([r["jaccard_mean"] for r in rows if not np.isnan(r["jaccard_mean"])])
    boot_ratios = []
    boot_fracs = []
    boot_jacc = []
    for _ in range(N_BOOTSTRAP):
        idx = RNG.randint(0, len(rows), size=len(rows))
        boot_ratios.append(float(h_arr[idx].mean() / l_arr[idx].mean()))
        boot_fracs.append(float(np.mean(h_arr[idx] > l_arr[idx])))
        jidx = RNG.randint(0, len(jaccard_arr), size=len(jaccard_arr))
        boot_jacc.append(float(jaccard_arr[jidx].mean()))
    rq = np.percentile(boot_ratios, [2.5, 50, 97.5])
    fq = np.percentile(boot_fracs, [2.5, 50, 97.5])
    jq = np.percentile(boot_jacc, [2.5, 50, 97.5])
    _flush(f"bootstrap ratio: {rq[1]:.4f} [{rq[0]:.4f}, {rq[2]:.4f}]")
    _flush(f"bootstrap frac_h>l: {fq[1]:.4f} [{fq[0]:.4f}, {fq[2]:.4f}]")
    _flush(f"bootstrap Jaccard: {jq[1]:.4f} [{jq[0]:.4f}, {jq[2]:.4f}]")
    out["bootstrap_60cluster"] = {
        "n_dilemmas": int(len(rows)),
        "ratio_2.5_50_97.5": list(rq),
        "frac_h>l_2.5_50_97.5": list(fq),
        "jaccard_2.5_50_97.5": list(jq),
    }

    # --- 2. k-size sensitivity -------------------------------------------------
    _flush("=== 2. k-size sensitivity ===")
    k_results = {}
    for k in [5, 10, 15, 20]:
        rows_k = per_dilemma_cluster_diversity(qualified, cluster60, k=k)
        if not rows_k:
            continue
        ha = np.array([r["h_mean"] for r in rows_k])
        la = np.array([r["l_mean"] for r in rows_k])
        k_results[k] = {
            "n_dilemmas": len(rows_k),
            "h_mean": float(ha.mean()),
            "l_mean": float(la.mean()),
            "ratio": float(ha.mean() / la.mean()),
            "frac_h>l": float(np.mean(ha > la)),
        }
        _flush(f"k={k}: n={len(rows_k)}, h={ha.mean():.3f}, l={la.mean():.3f}, "
               f"ratio={ha.mean()/la.mean():.3f}, frac_h>l={float(np.mean(ha>la)):.4f}")
    out["k_sensitivity"] = k_results

    # --- 3. Stratified by consensus --------------------------------------------
    _flush("=== 3. Stratified by consensus ===")
    sub_to_consensus = {}
    cols = ["comments_nta_agreement_weighted", "comments_yta_agreement_weighted",
            "comments_esh_agreement_weighted", "comments_nah_agreement_weighted"]
    labels = ["NTA", "YTA", "ESH", "NAH"]
    cols_data_strat = [ds[c] for c in cols]
    sub_to_verdict = {}
    for i, sid in enumerate(ds["submission_id"]):
        scores = [(0.0 if cols_data_strat[k][i] is None else float(cols_data_strat[k][i])) for k in range(len(cols))]
        sub_to_consensus[sid] = float(np.max(scores))
        sub_to_verdict[sid] = labels[int(np.argmax(scores))]
    cons_results = {}
    for level, lo, hi in [("low", 0.0, 0.5), ("medium", 0.5, 0.8), ("high", 0.8, 1.01)]:
        sel = [s for s in qualified if lo <= sub_to_consensus.get(s, 0.0) < hi]
        if len(sel) < 5: continue
        rows_s = per_dilemma_cluster_diversity(sel, cluster60)
        ha = np.array([r["h_mean"] for r in rows_s])
        la = np.array([r["l_mean"] for r in rows_s])
        cons_results[level] = {
            "n_dilemmas": len(rows_s),
            "h_mean": float(ha.mean()), "l_mean": float(la.mean()),
            "ratio": float(ha.mean()/la.mean()), "frac_h>l": float(np.mean(ha>la)),
        }
        _flush(f"consensus={level} (n={len(rows_s)}): h={ha.mean():.3f}, l={la.mean():.3f}, "
               f"ratio={ha.mean()/la.mean():.3f}, frac_h>l={float(np.mean(ha>la)):.4f}")
    out["by_consensus"] = cons_results

    # --- 4. Stratified by verdict ----------------------------------------------
    _flush("=== 4. Stratified by verdict ===")
    verdict_results = {}
    for v in labels:
        sel = [s for s in qualified if sub_to_verdict.get(s) == v]
        if len(sel) < 5: continue
        rows_v = per_dilemma_cluster_diversity(sel, cluster60)
        ha = np.array([r["h_mean"] for r in rows_v])
        la = np.array([r["l_mean"] for r in rows_v])
        verdict_results[v] = {
            "n_dilemmas": len(rows_v),
            "h_mean": float(ha.mean()), "l_mean": float(la.mean()),
            "ratio": float(ha.mean()/la.mean()), "frac_h>l": float(np.mean(ha>la)),
        }
        _flush(f"verdict={v} (n={len(rows_v)}): h={ha.mean():.3f}, l={la.mean():.3f}, "
               f"ratio={ha.mean()/la.mean():.3f}, frac_h>l={float(np.mean(ha>la)):.4f}")
    out["by_verdict"] = verdict_results

    # --- 5. Within-same-cluster cosine -----------------------------------------
    _flush("=== 5. Within-same-cluster cosine ===")
    # For each dilemma, group human/LLM comments by their decoded cluster.
    # For each cluster present in both human and LLM sets for that dilemma:
    #   compute pairwise mean cosine among humans in cluster, among LLMs in cluster.
    # Aggregate across dilemmas. Sample-matched: take min count per side per cluster.
    same_cluster_h_diversity = []
    same_cluster_l_diversity = []
    n_same_cluster_dilemmas = 0
    for sid in qualified:
        humans = h_by_sub[sid]
        l_items = l_by_sub[sid]
        # Group by cluster
        h_by_c = defaultdict(list)
        for h in humans:
            v = arctic_idx_to_value.get(h["idx"])
            if v:
                h_by_c[cluster60(v)].append(h["idx"])
        l_by_c = defaultdict(list)
        for li, it in enumerate(l_items):
            v = it["value"]
            if v:
                l_by_c[cluster60(v)].append(li)
        # For clusters with >=2 in both
        ok_clusters = [c for c in h_by_c if c in l_by_c and len(h_by_c[c]) >= 2 and len(l_by_c[c]) >= 2]
        if not ok_clusters:
            continue
        n_same_cluster_dilemmas += 1
        for c in ok_clusters:
            h_idxs = h_by_c[c]
            l_idxs = l_by_c[c]
            k_match = min(len(h_idxs), len(l_idxs), 5)
            r1 = np.random.RandomState(42 + c)
            r2 = np.random.RandomState(43 + c)
            h_sel = r1.choice(len(h_idxs), k_match, replace=False)
            l_sel = r2.choice(len(l_idxs), k_match, replace=False)
            Hs = H_emb[[h_idxs[i] for i in h_sel]]
            Ls = np.stack([l_items[l_idxs[i]]["emb"] for i in l_sel])
            same_cluster_h_diversity.append(cosine_pairwise_mean(Hs))
            same_cluster_l_diversity.append(cosine_pairwise_mean(Ls))
    h_within = np.array([x for x in same_cluster_h_diversity if not np.isnan(x)])
    l_within = np.array([x for x in same_cluster_l_diversity if not np.isnan(x)])
    if len(h_within) and len(l_within):
        _flush(f"within-same-cluster cosine: humans={h_within.mean():.4f}, llms={l_within.mean():.4f}, "
               f"ratio={h_within.mean()/l_within.mean():.3f}, frac_h>l={float(np.mean(h_within>l_within)):.4f}, "
               f"n_pairs={len(h_within)}")
    out["within_same_cluster_cosine"] = {
        "n_dilemmas_with_same_cluster_pairs": int(n_same_cluster_dilemmas),
        "n_pair_obs": int(len(h_within)),
        "h_mean": float(h_within.mean()) if len(h_within) else None,
        "l_mean": float(l_within.mean()) if len(l_within) else None,
        "ratio": float(h_within.mean()/l_within.mean()) if len(l_within) and l_within.mean() > 0 else None,
        "frac_h>l": float(np.mean(h_within > l_within)) if len(h_within) and len(l_within) else None,
    }

    # --- 6. Cluster granularity sweep ------------------------------------------
    _flush("=== 6. Cluster granularity sweep ===")
    # Use SBERT to embed all 324 known value labels, k-means at multiple k.
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(SBERT_MODEL, device="cuda")
    all_known_values = list(set(val_to_cluster_60.keys()))
    label_emb = sbert.encode(all_known_values, convert_to_numpy=True, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    del sbert; gc.collect(); torch.cuda.empty_cache()
    # PCA to 50 dims for k-means stability
    pca = PCA(n_components=50, random_state=42)
    label_emb_pca = pca.fit_transform(label_emb)
    granularity_results = {}
    for k_c in [20, 40, 100, 200]:
        if k_c > len(all_known_values): continue
        km = KMeans(n_clusters=k_c, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(label_emb_pca)
        val_to_c_k = {all_known_values[i]: int(cluster_ids[i]) for i in range(len(all_known_values))}
        def cluster_k(v):
            nv = normalize_value(v)
            return val_to_c_k.get(nv, val_to_c_k.get(list(val_to_c_k.keys())[0], 0))
        rows_g = per_dilemma_cluster_diversity(qualified, cluster_k)
        ha = np.array([r["h_mean"] for r in rows_g])
        la = np.array([r["l_mean"] for r in rows_g])
        granularity_results[k_c] = {
            "n_dilemmas": len(rows_g),
            "h_mean": float(ha.mean()), "l_mean": float(la.mean()),
            "ratio": float(ha.mean()/la.mean()), "frac_h>l": float(np.mean(ha>la)),
        }
        _flush(f"k_clusters={k_c}: h={ha.mean():.3f}, l={la.mean():.3f}, "
               f"ratio={ha.mean()/la.mean():.3f}, frac_h>l={float(np.mean(ha>la)):.4f}")
    out["cluster_granularity_sweep"] = granularity_results

    # --- 7. Cluster entropy (not count) ----------------------------------------
    _flush("=== 7. Cluster distribution entropy per dilemma ===")
    h_ent_arr, l_ent_arr = [], []
    for sid in qualified:
        humans = h_by_sub[sid]
        l_items = l_by_sub[sid]
        if len(humans) < 10 or len(l_items) < 10:
            continue
        # Use all available (no subsample) for entropy
        h_cnt = Counter()
        l_cnt = Counter()
        for h in humans:
            v = arctic_idx_to_value.get(h["idx"])
            if v:
                h_cnt[cluster60(v)] += 1
        for it in l_items:
            v = it["value"]
            if v:
                l_cnt[cluster60(v)] += 1
        h_counts_arr = np.array(list(h_cnt.values()), dtype=float)
        l_counts_arr = np.array(list(l_cnt.values()), dtype=float)
        h_ent_arr.append(shannon_ent(h_counts_arr))
        l_ent_arr.append(shannon_ent(l_counts_arr))
    h_ent_arr = np.array(h_ent_arr); l_ent_arr = np.array(l_ent_arr)
    _flush(f"entropy per dilemma (all samples): h={h_ent_arr.mean():.4f}, l={l_ent_arr.mean():.4f}, "
           f"ratio={h_ent_arr.mean()/l_ent_arr.mean():.3f}, frac_h>l={float(np.mean(h_ent_arr>l_ent_arr)):.4f}")
    out["cluster_entropy_all_samples"] = {
        "n_dilemmas": int(len(h_ent_arr)),
        "h_mean": float(h_ent_arr.mean()), "l_mean": float(l_ent_arr.mean()),
        "ratio": float(h_ent_arr.mean()/l_ent_arr.mean()),
        "frac_h>l": float(np.mean(h_ent_arr > l_ent_arr)),
    }

    path = ANALYSIS / "milestone3_cluster_robustness.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")


if __name__ == "__main__":
    main()
