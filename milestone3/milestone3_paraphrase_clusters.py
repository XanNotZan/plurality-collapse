"""Test if Qwen paraphrases preserve Kaleido moral-value clusters.

For each (orig, paraphrase) pair: decode top-1 Kaleido value for both texts.
Compare value labels via (a) string equality (b) SBERT cosine sim (c) shared cluster
in the existing 60-cluster value-label clustering from milestone 2.
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
GEN_DIR = Path("data/generated")
ANALYSIS = Path("data/analysis")
KALEIDO_MODEL = "allenai/kaleido-xl"
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

logger = logging.getLogger("paraphrase_clusters")
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
        torch.cuda.ipc_collect()


def kaleido_load():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    _flush("loading Kaleido")
    tok = AutoTokenizer.from_pretrained(KALEIDO_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(KALEIDO_MODEL, dtype=torch.float16).to("cuda").eval()
    try:
        template = model.config.task_specific_params["generate"]["template"]
    except Exception:
        template = "[Generate]:\tAction: ACTION"
    return tok, model, template


def decode_values(tok, model, template, texts, batch=8, max_new=24):
    out = []
    for s in range(0, len(texts), batch):
        b = texts[s:s + batch]
        formatted = [template.replace("ACTION", t if t else "(empty)") for t in b]
        inputs = tok(formatted, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
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
        if s == 0:
            _flush(f"first batch decoded: {out[:3]}")
        if (s // batch) % 10 == 0:
            _flush(f"decode {s + len(b)}/{len(texts)}")
    return out


def normalize_value(v):
    return re.sub(r"\s+", " ", v.strip().lower())


def main():
    # 1. Load paraphrased pairs
    h_items = [json.loads(l) for l in (GEN_DIR / "human_to_formal_clean.jsonl").read_text().splitlines()]
    l_items = [json.loads(l) for l in (GEN_DIR / "llm_to_casual_clean.jsonl").read_text().splitlines()]
    _flush(f"human pairs: {len(h_items)}, llm pairs: {len(l_items)}")

    # Build text lists: orig + paraphrase, ordered
    h_orig = [it["orig_text"] for it in h_items]
    h_para = [it["paraphrased_clean"] for it in h_items]
    l_orig = [it["orig_text"] for it in l_items]
    l_para = [it["paraphrased_clean"] for it in l_items]

    # 2. Decode Kaleido values
    free_gpu()
    tok, model, template = kaleido_load()
    _flush("decoding human originals")
    h_orig_vals = decode_values(tok, model, template, h_orig)
    _flush("decoding human formals")
    h_para_vals = decode_values(tok, model, template, h_para)
    _flush("decoding LLM originals")
    l_orig_vals = decode_values(tok, model, template, l_orig)
    _flush("decoding LLM casuals")
    l_para_vals = decode_values(tok, model, template, l_para)

    # Free Kaleido
    del model, tok
    free_gpu()

    # 3. Load existing value-label clustering (cluster_id per value)
    val_to_cluster = {}
    cluster_label_by_id = {}
    with open(ANALYSIS / "value_label_clusters.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_to_cluster[normalize_value(row["value"])] = int(row["cluster_id"])
            cluster_label_by_id[int(row["cluster_id"])] = row["cluster_label"]
    _flush(f"clustered values: {len(val_to_cluster)}, clusters: {len(cluster_label_by_id)}")

    # 4. SBERT for unmapped values
    from sentence_transformers import SentenceTransformer
    _flush(f"loading SBERT {SBERT_MODEL}")
    sbert = SentenceTransformer(SBERT_MODEL, device="cuda")

    all_known = list(val_to_cluster.keys())
    known_emb = sbert.encode(all_known, convert_to_numpy=True, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    def to_cluster(value):
        nv = normalize_value(value)
        if nv in val_to_cluster:
            return val_to_cluster[nv], "known"
        # SBERT NN to nearest known value
        emb = sbert.encode([value], convert_to_numpy=True, normalize_embeddings=True)
        sims = known_emb @ emb[0]
        i = int(np.argmax(sims))
        return val_to_cluster[all_known[i]], f"NN_sim={sims[i]:.3f}"

    # 5. Compute cluster preservation per pair
    def cell_stats(orig_vals, para_vals, label):
        n = len(orig_vals)
        same_value = 0
        same_cluster = 0
        nn_used_orig = 0
        nn_used_para = 0
        details = []
        for ov, pv in zip(orig_vals, para_vals):
            ov_clean = normalize_value(ov)
            pv_clean = normalize_value(pv)
            if ov_clean == pv_clean:
                same_value += 1
            oc, om = to_cluster(ov)
            pc, pm = to_cluster(pv)
            if om != "known":
                nn_used_orig += 1
            if pm != "known":
                nn_used_para += 1
            if oc == pc:
                same_cluster += 1
            details.append({
                "orig_value": ov, "para_value": pv,
                "orig_cluster_id": oc, "orig_cluster_label": cluster_label_by_id.get(oc, "?"),
                "para_cluster_id": pc, "para_cluster_label": cluster_label_by_id.get(pc, "?"),
                "orig_match": om, "para_match": pm,
                "same_value": ov_clean == pv_clean,
                "same_cluster": oc == pc,
            })
        _flush(f"{label}: n={n}, same_value={same_value} ({100*same_value/n:.1f}%), "
               f"same_cluster={same_cluster} ({100*same_cluster/n:.1f}%), "
               f"nn_orig={nn_used_orig}, nn_para={nn_used_para}")
        return {
            "n": n,
            "same_value_count": same_value,
            "same_value_pct": 100 * same_value / n,
            "same_cluster_count": same_cluster,
            "same_cluster_pct": 100 * same_cluster / n,
            "nn_used_orig": nn_used_orig,
            "nn_used_para": nn_used_para,
            "details_first_20": details[:20],
        }

    h_stats = cell_stats(h_orig_vals, h_para_vals, "human (orig vs formal)")
    l_stats = cell_stats(l_orig_vals, l_para_vals, "llm (orig vs casual)")

    out = {
        "human_orig_vs_formal": h_stats,
        "llm_orig_vs_casual": l_stats,
        "n_clusters_total": len(cluster_label_by_id),
    }
    path = ANALYSIS / "milestone3_paraphrase_clusters.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")

    # Save full per-pair details too
    pair_path = ANALYSIS / "milestone3_paraphrase_clusters_pairs.jsonl"
    with open(pair_path, "w") as f:
        for ov, pv, oitem in zip(h_orig_vals, h_para_vals, h_items):
            ov_c, _ = to_cluster(ov)
            pv_c, _ = to_cluster(pv)
            f.write(json.dumps({
                "side": "human", "submission_id": oitem["submission_id"],
                "orig_value": ov, "para_value": pv,
                "orig_cluster": ov_c, "para_cluster": pv_c,
                "same_cluster": ov_c == pv_c,
            }) + "\n")
        for ov, pv, oitem in zip(l_orig_vals, l_para_vals, l_items):
            ov_c, _ = to_cluster(ov)
            pv_c, _ = to_cluster(pv)
            f.write(json.dumps({
                "side": "llm", "submission_id": oitem["submission_id"],
                "source": oitem.get("source", "llm"),
                "orig_value": ov, "para_value": pv,
                "orig_cluster": ov_c, "para_cluster": pv_c,
                "same_cluster": ov_c == pv_c,
            }) + "\n")
    _flush(f"saved {pair_path}")
    _flush("ALL DONE")


if __name__ == "__main__":
    main()
