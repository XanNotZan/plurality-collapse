"""Per-dilemma cluster-level value diversity test.

For each of 1,991 ArcticShift dilemmas:
  - Sample k=10 human comments + k=10 LLM rationales (sample-size matched)
  - Decode Kaleido top-1 value per comment
  - Map to milestone-2's 60-cluster value-label clustering
  - Count distinct clusters in each k-sample per source
  - Compare diversity

Tests sub-foundation moral-expressive variation hypothesis:
  If humans use MORE distinct 60-clusters per dilemma than LLMs → diversity at sub-foundation cluster level.
  If similar → diversity is even finer than 60 Kaleido clusters (framing, perspective, affect).
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
import torch

EMBEDDINGS_DIR = Path("data/embeddings")
ANALYSIS = Path("data/analysis")
ARCTIC_DIR = Path("data/arcticshift")
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
K_SAMPLE = 10
N_SAMPLE_SEEDS = 5
KALEIDO_MODEL = "allenai/kaleido-xl"
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RNG = np.random.RandomState(42)

logger = logging.getLogger("cluster_diversity")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg); sys.stdout.flush()


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def normalize_value(v):
    return re.sub(r"\s+", " ", v.strip().lower())


def load_arctic_data():
    arctic_meta = json.loads((EMBEDDINGS_DIR / "human_arctic_meta.json").read_text())
    arctic_recs = {}
    for line in (ARCTIC_DIR / "filtered_comments.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("_empty") or not j.get("body"):
            continue
        arctic_recs[(j["submission_id"], j.get("comment_id"))] = j

    h_by_sub = defaultdict(list)  # sid -> [(arctic_idx, body)]
    for m in arctic_meta:
        rec = arctic_recs.get((m["submission_id"], m.get("comment_id")))
        if rec:
            h_by_sub[m["submission_id"]].append({"arctic_idx": m["index"], "body": rec["body"]})
    return h_by_sub


def load_llm_values():
    """Load existing decoded values for LLMs (milestone 2)."""
    l_by_sub = defaultdict(list)
    for s in LLM_SOURCES:
        path = ANALYSIS / f"llm_values_{s}.json"
        if not path.exists():
            _flush(f"missing {path}, skipping {s}"); continue
        data = json.loads(path.read_text())
        for entry in data:
            sid = entry["submission_id"]
            val = entry.get("generated_values") or ""
            l_by_sub[sid].append({"source": s, "value": val, "column": entry.get("column")})
        _flush(f"loaded {s}: {len(data)} entries")
    return l_by_sub


def load_value_cluster_map():
    val_to_cluster = {}
    cluster_label_by_id = {}
    with open(ANALYSIS / "value_label_clusters.csv", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            val_to_cluster[normalize_value(row["value"])] = int(row["cluster_id"])
            cluster_label_by_id[int(row["cluster_id"])] = row["cluster_label"]
    return val_to_cluster, cluster_label_by_id


def load_kaleido():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    _flush("loading Kaleido")
    tok = AutoTokenizer.from_pretrained(KALEIDO_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(KALEIDO_MODEL, dtype=torch.float16).to("cuda").eval()
    try:
        template = model.config.task_specific_params["generate"]["template"]
    except Exception:
        template = "[Generate]:\tAction: ACTION"
    return tok, model, template


def decode_values(tok, model, template, texts, batch=24, max_new=24):
    out = []
    for s in range(0, len(texts), batch):
        b = texts[s:s + batch]
        formatted = [template.replace("ACTION", t if t else "(empty)") for t in b]
        inputs = tok(formatted, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=max_new)
        decoded = tok.batch_decode(outputs, skip_special_tokens=True)
        for d in decoded:
            d = d.strip()
            if d.lower().startswith("value:"):
                d = d[len("value:"):].strip()
            out.append(d)
        if (s // batch) % 100 == 0:
            _flush(f"decode {s + len(b)}/{len(texts)}")
    return out


def main():
    val_to_cluster, cluster_label_by_id = load_value_cluster_map()
    _flush(f"value clusters: {len(set(val_to_cluster.values()))} unique cluster_ids")

    h_by_sub = load_arctic_data()
    l_by_sub = load_llm_values()
    _flush(f"arctic dilemmas with humans: {len(h_by_sub)}")
    _flush(f"llm dilemmas with values: {len(l_by_sub)}")

    # Qualified: ≥ K_SAMPLE humans + ≥ K_SAMPLE LLMs
    qualified = sorted([s for s in h_by_sub if len(h_by_sub[s]) >= K_SAMPLE and len(l_by_sub.get(s, [])) >= K_SAMPLE])
    _flush(f"qualified dilemmas (>={K_SAMPLE} each): {len(qualified)}")

    # Collect human comments to decode (subsample 10 per dilemma with multiple seeds)
    # Strategy: For each dilemma, subsample K_SAMPLE * N_SAMPLE_SEEDS unique humans (or all if fewer).
    # To save decode time, decode each UNIQUE human comment only once and reuse across seeds.
    # Build per-dilemma sample plans first.

    human_to_decode = set()  # set of arctic_idx
    sample_plans = {}  # sid -> list of (seed, [arctic_idx, ...])
    rng = np.random.RandomState(42)
    for sid in qualified:
        humans = h_by_sub[sid]
        n = len(humans)
        plans = []
        # For each seed, sample K_SAMPLE without replacement
        for seed in range(N_SAMPLE_SEEDS):
            r = np.random.RandomState(seed)
            idx = r.choice(n, K_SAMPLE, replace=False)
            picked = [humans[i]["arctic_idx"] for i in idx]
            plans.append(picked)
            human_to_decode.update(picked)
        sample_plans[sid] = plans
    _flush(f"unique human comments to decode: {len(human_to_decode)}")

    # Build text array (one per arctic_idx)
    arctic_idx_to_text = {}
    for sid, items in h_by_sub.items():
        for it in items:
            arctic_idx_to_text[it["arctic_idx"]] = it["body"]
    h_decode_ids = sorted(human_to_decode)
    h_decode_texts = [arctic_idx_to_text[i] for i in h_decode_ids]
    _flush(f"decoding {len(h_decode_texts)} human comments")

    # Decode
    free_gpu()
    tok, model, template = load_kaleido()
    t0 = time.time()
    h_decoded_vals = decode_values(tok, model, template, h_decode_texts, batch=24)
    _flush(f"human decode done in {time.time()-t0:.1f}s")
    del model, tok
    free_gpu()

    arctic_idx_to_value = dict(zip(h_decode_ids, h_decoded_vals))

    # Map values to clusters
    # For values not in val_to_cluster, use SBERT NN
    unmapped = set()
    for v in h_decoded_vals:
        if normalize_value(v) not in val_to_cluster:
            unmapped.add(v)
    for sid, items in l_by_sub.items():
        for it in items:
            if normalize_value(it["value"]) not in val_to_cluster:
                unmapped.add(it["value"])
    _flush(f"unmapped values needing SBERT NN: {len(unmapped)}")

    if unmapped:
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer(SBERT_MODEL, device="cuda")
        known = list(val_to_cluster.keys())
        known_emb = sbert.encode(known, convert_to_numpy=True, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        unmapped_list = list(unmapped)
        unmapped_emb = sbert.encode(unmapped_list, convert_to_numpy=True, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        # NN
        for i, v in enumerate(unmapped_list):
            sims = known_emb @ unmapped_emb[i]
            j = int(np.argmax(sims))
            val_to_cluster[normalize_value(v)] = val_to_cluster[known[j]]
        del sbert
        free_gpu()
        _flush("unmapped values resolved via SBERT NN")

    def to_cluster(v):
        return val_to_cluster[normalize_value(v)]

    # Per-dilemma cluster diversity
    results = []
    for sid in qualified:
        # Humans across seeds
        h_per_seed_n_clusters = []
        for plan in sample_plans[sid]:
            vals = [arctic_idx_to_value[i] for i in plan]
            clusters = {to_cluster(v) for v in vals if v}
            h_per_seed_n_clusters.append(len(clusters))
        # LLMs: sample K_SAMPLE per seed
        l_items = l_by_sub[sid]
        l_per_seed_n_clusters = []
        for seed in range(N_SAMPLE_SEEDS):
            r = np.random.RandomState(seed + 1000)
            idx = r.choice(len(l_items), K_SAMPLE, replace=False)
            vals = [l_items[i]["value"] for i in idx]
            clusters = {to_cluster(v) for v in vals if v}
            l_per_seed_n_clusters.append(len(clusters))
        results.append({
            "sid": sid,
            "h_clusters_per_seed": h_per_seed_n_clusters,
            "l_clusters_per_seed": l_per_seed_n_clusters,
            "h_mean_clusters": float(np.mean(h_per_seed_n_clusters)),
            "l_mean_clusters": float(np.mean(l_per_seed_n_clusters)),
        })

    h_arr = np.array([r["h_mean_clusters"] for r in results])
    l_arr = np.array([r["l_mean_clusters"] for r in results])
    ratio = float(h_arr.mean() / l_arr.mean())
    frac_h_more = float(np.mean(h_arr > l_arr))
    frac_h_strict_more = float(np.mean(h_arr > l_arr + 0.5))  # strict > by at least 0.5 cluster (robust to ties)
    _flush(f"per-dilemma mean clusters used (k={K_SAMPLE}, avg over {N_SAMPLE_SEEDS} seeds):")
    _flush(f"  human: {h_arr.mean():.3f} ± {h_arr.std():.3f}")
    _flush(f"  llm:   {l_arr.mean():.3f} ± {l_arr.std():.3f}")
    _flush(f"  ratio (h/l): {ratio:.3f}")
    _flush(f"  frac dilemmas h > l: {frac_h_more:.4f}")
    _flush(f"  frac dilemmas h > l + 0.5: {frac_h_strict_more:.4f}")

    # Also: cluster overlap analysis. Average cluster intersection / union per dilemma (Jaccard)
    jaccard_per_dilemma = []
    for r_d in results:
        # For each seed: cluster set h vs l
        # Need actual cluster sets, not counts. Recompute here using seed 0
        plan_h = sample_plans[r_d["sid"]][0]
        h_clusters = {to_cluster(arctic_idx_to_value[i]) for i in plan_h if arctic_idx_to_value[i]}
        l_items_sd = l_by_sub[r_d["sid"]]
        rr = np.random.RandomState(1000)
        l_pick = rr.choice(len(l_items_sd), K_SAMPLE, replace=False)
        l_clusters = {to_cluster(l_items_sd[i]["value"]) for i in l_pick if l_items_sd[i]["value"]}
        if h_clusters and l_clusters:
            jaccard = len(h_clusters & l_clusters) / len(h_clusters | l_clusters)
            jaccard_per_dilemma.append(jaccard)
    jacc_arr = np.array(jaccard_per_dilemma)
    _flush(f"  Jaccard(h_clusters, l_clusters) per dilemma: mean={jacc_arr.mean():.3f}, median={np.median(jacc_arr):.3f}")

    # Save
    out = {
        "n_dilemmas": len(qualified),
        "k_sample": K_SAMPLE,
        "n_seeds": N_SAMPLE_SEEDS,
        "human_mean_distinct_clusters": float(h_arr.mean()),
        "human_std": float(h_arr.std()),
        "llm_mean_distinct_clusters": float(l_arr.mean()),
        "llm_std": float(l_arr.std()),
        "ratio_h_over_l": ratio,
        "frac_dilemmas_h_more": frac_h_more,
        "frac_dilemmas_h_strict_more": frac_h_strict_more,
        "mean_jaccard_overlap": float(jacc_arr.mean()),
        "median_jaccard_overlap": float(np.median(jacc_arr)),
    }
    path = ANALYSIS / "milestone3_per_dilemma_cluster_diversity.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")

    # Bonus: dump value-cluster frequency tables per source
    h_cluster_freq = defaultdict(int)
    l_cluster_freq = defaultdict(int)
    for sid in qualified:
        for plan in sample_plans[sid]:
            for i in plan:
                v = arctic_idx_to_value[i]
                if v:
                    h_cluster_freq[to_cluster(v)] += 1
        for seed in range(N_SAMPLE_SEEDS):
            r = np.random.RandomState(seed + 1000)
            idx = r.choice(len(l_by_sub[sid]), K_SAMPLE, replace=False)
            for i in idx:
                v = l_by_sub[sid][i]["value"]
                if v:
                    l_cluster_freq[to_cluster(v)] += 1
    out["cluster_frequency_human"] = {cluster_label_by_id.get(k, str(k)): v for k, v in sorted(h_cluster_freq.items(), key=lambda x: -x[1])}
    out["cluster_frequency_llm"] = {cluster_label_by_id.get(k, str(k)): v for k, v in sorted(l_cluster_freq.items(), key=lambda x: -x[1])}
    path.write_text(json.dumps(out, indent=2))
    _flush(f"resaved with cluster freqs {path}")


if __name__ == "__main__":
    main()
