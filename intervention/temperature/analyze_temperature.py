"""Per-dilemma analysis of Qwen rationales across temperatures.

Two passes:
  1. Per-dilemma pairwise cosine within each (submission_id, T) cell.
  2. Kaleido 60-cluster diversity per (submission_id, T) cell. Decode each
     rationale to a value label via Kaleido, map to milestone-2 60-cluster
     taxonomy, count distinct clusters at every feasible K from 2 to K_temp.

Outputs:
  data/analysis/intervention/temperature/per_dilemma_cosine.json
  data/analysis/intervention/temperature/cluster_diversity.json
  data/analysis/intervention/temperature/decoded_values_T{T}.json   (intermediate cache)
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

EMB_DIR = Path("data/embeddings/intervention/temperature")
GEN_DIR = Path("data/generated/intervention/temperature")
ANALYSIS_BASE = Path("data/analysis")
ANALYSIS = ANALYSIS_BASE / "intervention" / "temperature"
ANALYSIS.mkdir(parents=True, exist_ok=True)

TEMPERATURES = [0.3, 0.7, 1.0, 1.3]
K_PER_TEMP = {0.3: 3, 0.7: 5, 1.0: 8, 1.3: 12}
KALEIDO_MODEL = "allenai/kaleido-xl"
DECODE_BATCH = 32
N_SUBSAMPLE_SEEDS = 50
BOOTSTRAP = 500

# Reference baselines from milestone 3
HUMAN_ARCTIC_COSINE = 0.297
SACHDEVA_LLM_COSINE = 0.147
HUMAN_CLUSTER_DIV = 4.93
LLM_CLUSTER_DIV = 3.39

logger = logging.getLogger("analyze_temp")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg)
    sys.stdout.flush()


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def normalize_value(v):
    return re.sub(r"\s+", " ", v.strip().lower())


def load_value_cluster_map():
    val_to_cluster = {}
    cluster_label_by_id = {}
    with open(ANALYSIS_BASE / "value_label_clusters.csv", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            val_to_cluster[normalize_value(row["value"])] = int(row["cluster_id"])
            cluster_label_by_id[int(row["cluster_id"])] = row["cluster_label"]
    return val_to_cluster, cluster_label_by_id


def load_temperature_data(T):
    """Return (E, meta) where E is the (N, 2048) embedding matrix and
    meta is the list of {submission_id, sample_idx, ...}.
    """
    emb_path = EMB_DIR / f"qwen25_3b_T{T:.1f}.npy"
    meta_path = EMB_DIR / f"qwen25_3b_T{T:.1f}_meta.json"
    if not emb_path.exists() or not meta_path.exists():
        _flush(f"missing embeddings for T={T}")
        return None, None
    E = np.load(emb_path)
    meta = json.loads(meta_path.read_text())
    return E, meta


def per_dilemma_cosine(E, meta):
    """Mean pairwise (1 - cos) within each (submission_id) cell at this T."""
    by_sub = defaultdict(list)
    for m in meta:
        by_sub[m["submission_id"]].append(m["index"])
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    En = E / norms

    distances = []
    for sid, idxs in by_sub.items():
        if len(idxs) < 2:
            continue
        sub = En[idxs]
        # pairwise cos
        cos = sub @ sub.T
        iu = np.triu_indices_from(cos, k=1)
        dist = (1.0 - cos[iu]).mean()
        distances.append(float(dist))
    return distances


def cosine_pass():
    out = {
        "baselines": {
            "human_arctic_within_dilemma_cosine": HUMAN_ARCTIC_COSINE,
            "sachdeva_llm_within_dilemma_cosine": SACHDEVA_LLM_COSINE,
        },
        "per_temperature": {},
    }
    for T in TEMPERATURES:
        E, meta = load_temperature_data(T)
        if E is None:
            continue
        dists = per_dilemma_cosine(E, meta)
        arr = np.array(dists)
        ratio_human = HUMAN_ARCTIC_COSINE / arr.mean() if arr.mean() > 0 else None
        # bootstrap CI for the mean
        rng = np.random.RandomState(42)
        boot = []
        for _ in range(BOOTSTRAP):
            sample = rng.choice(arr, size=len(arr), replace=True)
            boot.append(sample.mean())
        boot = np.array(boot)
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        out["per_temperature"][f"T_{T:.1f}"] = {
            "K": K_PER_TEMP[T],
            "n_dilemmas": int(len(arr)),
            "mean_cosine_distance": float(arr.mean()),
            "std_cosine_distance": float(arr.std()),
            "bootstrap_ci_95": ci,
            "ratio_human_to_qwen": float(ratio_human) if ratio_human else None,
        }
        _flush(f"T={T} mean per-dilemma cos-dist = {arr.mean():.4f} "
               f"(95% CI {ci[0]:.4f}-{ci[1]:.4f}) "
               f"ratio human/qwen = {ratio_human:.3f}")
    return out


# ---------------------------------------------------------------------------
# Cluster diversity pass
# ---------------------------------------------------------------------------
def load_rationales(T):
    jsonl = GEN_DIR / f"qwen25_3b_T{T:.1f}.jsonl"
    items = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            continue
    return items


def load_kaleido():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    _flush(f"loading {KALEIDO_MODEL} (fp16)")
    tok = AutoTokenizer.from_pretrained(KALEIDO_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(KALEIDO_MODEL, dtype=torch.float16).to("cuda").eval()
    try:
        template = model.config.task_specific_params["generate"]["template"]
    except Exception:
        template = "[Generate]:\tAction: ACTION"
    _flush(f"vram: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return tok, model, template


def decode_values(tok, model, template, texts, max_new=24):
    out = []
    t0 = time.time()
    for s in range(0, len(texts), DECODE_BATCH):
        b = texts[s : s + DECODE_BATCH]
        formatted = [template.replace("ACTION", t if t else "(empty)") for t in b]
        inputs = tok(formatted, return_tensors="pt", padding=True, truncation=True,
                     max_length=512).to("cuda")
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs["input_ids"],
                                     attention_mask=inputs["attention_mask"],
                                     max_new_tokens=max_new)
        decoded = tok.batch_decode(outputs, skip_special_tokens=True)
        for d in decoded:
            d = d.strip()
            if d.lower().startswith("value:"):
                d = d[len("value:"):].strip()
            out.append(d)
        if (s // DECODE_BATCH) % 20 == 0:
            elapsed = time.time() - t0
            rate = (s + len(b)) / elapsed if elapsed > 0 else 0
            _flush(f"  decode {s + len(b)}/{len(texts)} ({rate:.0f}/s)")
    return out


def cluster_diversity_pass():
    val_to_cluster, _ = load_value_cluster_map()
    n_clusters_total = len(set(val_to_cluster.values()))
    _flush(f"{n_clusters_total} cluster ids in milestone-2 60-cluster taxonomy")

    tok, model, template = load_kaleido()

    # Decode each rationale once per T (cached)
    decoded_by_T = {}
    for T in TEMPERATURES:
        cache = ANALYSIS / f"decoded_values_T{T:.1f}.json"
        if cache.exists():
            _flush(f"loading cached decodes for T={T}")
            decoded_by_T[T] = json.loads(cache.read_text())
            continue
        items = load_rationales(T)
        if not items:
            _flush(f"no rationales for T={T}")
            decoded_by_T[T] = []
            continue
        texts = [it["rationale"] for it in items]
        _flush(f"=== T={T} decoding {len(texts)} rationales")
        t0 = time.time()
        values = decode_values(tok, model, template, texts)
        _flush(f"T={T} decoded in {(time.time() - t0):.0f}s")
        rec = [{"submission_id": items[i]["submission_id"],
                "sample_idx": items[i]["sample_idx"],
                "value": values[i]} for i in range(len(items))]
        cache.write_text(json.dumps(rec))
        decoded_by_T[T] = rec
    del model
    free_gpu()

    # Group by (T, sid) -> list of cluster ids
    grouped = {T: defaultdict(list) for T in TEMPERATURES}
    unknown_value_count = defaultdict(int)
    for T, rec in decoded_by_T.items():
        for r in rec:
            v = normalize_value(r["value"])
            cid = val_to_cluster.get(v, -1)
            if cid < 0:
                unknown_value_count[T] += 1
            grouped[T][r["submission_id"]].append(cid)
    for T, c in unknown_value_count.items():
        _flush(f"T={T}: {c} values not in 60-cluster map (assigned cluster=-1)")

    # Full K-sweep cluster diversity table
    out = {
        "baselines": {
            "human_arctic_within_dilemma_clusters_K_eq_3": HUMAN_CLUSTER_DIV,
            "sachdeva_llm_within_dilemma_clusters_K_eq_3": LLM_CLUSTER_DIV,
        },
        "per_temperature": {},
    }
    for T in TEMPERATURES:
        K_max = K_PER_TEMP[T]
        per_T = {"K_max": K_max, "by_K": {}}
        cells = grouped[T]
        rng = np.random.RandomState(123)

        for K in range(2, K_max + 1):
            # Subsample at K within each dilemma
            per_dilemma = []
            for sid, cids in cells.items():
                if len(cids) < K:
                    continue
                if len(cids) == K:
                    per_dilemma.append(len(set(cids)))
                else:
                    # subsample N_SUBSAMPLE_SEEDS times, mean of distinct
                    seed_results = []
                    for _ in range(N_SUBSAMPLE_SEEDS):
                        idx = rng.choice(len(cids), size=K, replace=False)
                        seed_results.append(len(set(cids[i] for i in idx)))
                    per_dilemma.append(float(np.mean(seed_results)))
            arr = np.array(per_dilemma)
            # bootstrap CI for the mean over dilemmas
            boot = []
            for _ in range(BOOTSTRAP):
                samp = rng.choice(arr, size=len(arr), replace=True)
                boot.append(samp.mean())
            boot = np.array(boot)
            ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
            per_T["by_K"][f"K_{K}"] = {
                "mean_distinct_clusters": float(arr.mean()),
                "std": float(arr.std()),
                "bootstrap_ci_95": ci,
                "n_dilemmas": int(len(arr)),
            }
            _flush(f"T={T} K={K}: distinct clusters mean={arr.mean():.3f} "
                   f"(95% CI {ci[0]:.3f}-{ci[1]:.3f}) n={len(arr)}")
        out["per_temperature"][f"T_{T:.1f}"] = per_T
    return out


def main():
    _flush("=== Per-dilemma cosine pass ===")
    cos_out = cosine_pass()
    (ANALYSIS / "per_dilemma_cosine.json").write_text(json.dumps(cos_out, indent=2))
    _flush(f"saved {ANALYSIS / 'per_dilemma_cosine.json'}")

    _flush("=== Kaleido 60-cluster diversity pass ===")
    clus_out = cluster_diversity_pass()
    (ANALYSIS / "cluster_diversity.json").write_text(json.dumps(clus_out, indent=2))
    _flush(f"saved {ANALYSIS / 'cluster_diversity.json'}")

    _flush("ANALYSIS DONE")


if __name__ == "__main__":
    main()
