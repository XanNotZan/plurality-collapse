"""A9: generative qualitative probe.

Sample N stratified pairs of (human_comment, llm_comment) that fall in same Kaleido 60-cluster
on same dilemma.  Ask a strong LLM "what is the key framing difference?" SBERT-cluster
the explanations, label clusters via a second LLM pass.

Robustness:
  - Scrambled-pair control (different dilemmas) — confirms probe is not pattern-matching
  - Cluster k-sensitivity sweep k ∈ {5, 8, 12, 16}
  - Sampling reproducibility (two random seeds), report Jaccard of cluster-keyword sets
"""

import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

EMBEDDINGS_DIR = Path("data/embeddings")
ANALYSIS = Path("data/analysis")
ARCTIC_DIR = Path("data/arcticshift")
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
RNG = np.random.RandomState(42)

N_SAMPLES = 300
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.3
BATCH_SIZE = 8

logger = logging.getLogger("a9")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg); sys.stdout.flush()


def load_cluster_map():
    df = pd.read_csv(ANALYSIS / "value_label_clusters.csv")
    return dict(zip(df["value"], df["cluster_id"])), dict(zip(df["cluster_id"], df["cluster_label"]))


def build_cells(cluster_map):
    cells = defaultdict(lambda: {"h": [], "l": []})
    arctic_meta = json.loads((EMBEDDINGS_DIR / "human_arctic_meta.json").read_text())
    decoded = json.loads((ANALYSIS / "milestone3_arctic_decoded_values.json").read_text())
    # arctic body lookup
    arctic_recs = {}
    with open(ARCTIC_DIR / "filtered_comments.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: j = json.loads(line)
            except Exception: continue
            if j.get("_empty") or not j.get("body"): continue
            arctic_recs[(j["submission_id"], j.get("comment_id"))] = j["body"]
    for m in arctic_meta:
        body = arctic_recs.get((m["submission_id"], m.get("comment_id")))
        if not body: continue
        val = decoded.get(str(m["index"]))
        if not val: continue
        cid = cluster_map.get(val)
        if cid is None: continue
        cells[(m["submission_id"], int(cid))]["h"].append({"idx": m["index"], "text": body, "value": val})

    arctic_subs = {m["submission_id"] for m in arctic_meta}
    for s in LLM_SOURCES:
        data = json.loads((ANALYSIS / f"llm_values_{s}.json").read_text())
        for r in data:
            if r["submission_id"] not in arctic_subs: continue
            val = r.get("generated_values")
            if not val: continue
            cid = cluster_map.get(val)
            if cid is None: continue
            cells[(r["submission_id"], int(cid))]["l"].append({
                "idx": r["index"], "source": s,
                "text": r["rationale_text"], "value": val,
            })

    shared = {k: v for k, v in cells.items() if v["h"] and v["l"]}
    return shared


def stratified_sample(cells, n_samples, cluster_labels, rng=RNG, scramble=False):
    """Stratify by cluster_id, aim for roughly equal coverage of clusters present."""
    by_cluster = defaultdict(list)
    for (sid, cid), v in cells.items():
        by_cluster[cid].append((sid, v))
    clusters = sorted(by_cluster.keys())
    per_c = max(1, n_samples // max(1, len(clusters)))

    pairs = []
    if not scramble:
        for cid in clusters:
            entries = by_cluster[cid]
            rng.shuffle(entries)
            quota = min(per_c, len(entries))
            for sid, v in entries[:quota]:
                h = v["h"][rng.randint(0, len(v["h"]))]
                l = v["l"][rng.randint(0, len(v["l"]))]
                pairs.append({
                    "submission_id": sid, "cluster_id": cid,
                    "cluster_label": cluster_labels.get(cid, "?"),
                    "human_text": h["text"], "human_value": h["value"],
                    "llm_text": l["text"], "llm_value": l["value"], "llm_source": l["source"],
                    "scrambled": False,
                })
    else:
        # scrambled: human from cell A, LLM from cell B, where dilemmas differ
        all_h_cells = list(cells.items())
        all_l_cells = list(cells.items())
        rng.shuffle(all_h_cells); rng.shuffle(all_l_cells)
        i = 0
        while len(pairs) < n_samples and i < len(all_h_cells):
            (h_sid, h_cid), h_v = all_h_cells[i]
            i += 1
            j = rng.randint(0, len(all_l_cells))
            (l_sid, l_cid), l_v = all_l_cells[j]
            if h_sid == l_sid: continue
            h = h_v["h"][rng.randint(0, len(h_v["h"]))]
            l = l_v["l"][rng.randint(0, len(l_v["l"]))]
            pairs.append({
                "submission_id": h_sid, "cluster_id": h_cid,
                "cluster_label": cluster_labels.get(h_cid, "?"),
                "human_text": h["text"], "human_value": h["value"],
                "llm_text": l["text"], "llm_value": l["value"], "llm_source": l["source"],
                "scrambled": True,
            })
    return pairs[:n_samples]


def build_probe_prompt(pair):
    h = pair["human_text"][:600]
    l = pair["llm_text"][:600]
    return [
        {"role": "system", "content": (
            "You compare moral framing. Given a human comment and an AI comment about the same "
            "moral dilemma, return ONE short sentence (≤15 words) naming the single biggest "
            "stylistic / framing / rhetorical difference. Do not explain. Do not list multiple "
            "differences. Output only the sentence."
        )},
        {"role": "user", "content": (
            f"HUMAN: {h}\n\nAI: {l}\n\n"
            "What is the single biggest framing difference, in one short sentence?"
        )},
    ]


def run_llm_probe(pairs, model_name=MODEL_NAME, batch_size=BATCH_SIZE):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _flush(f"loading {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to("cuda")
    model.eval()

    outputs = []
    n = len(pairs)
    _flush(f"running probe on {n} pairs (batch={batch_size}, max_new={MAX_NEW_TOKENS}, T={TEMPERATURE})")
    for i in range(0, n, batch_size):
        batch = pairs[i:i+batch_size]
        prompts = [tok.apply_chat_template(build_probe_prompt(p), tokenize=False, add_generation_prompt=True) for p in batch]
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2000).to("cuda")
        with torch.no_grad():
            outs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True if TEMPERATURE > 0 else False,
                temperature=TEMPERATURE,
                pad_token_id=tok.pad_token_id,
            )
        input_length = inputs["input_ids"].shape[1]
        for j, p in enumerate(batch):
            new_ids = outs[j][input_length:]
            txt = tok.decode(new_ids, skip_special_tokens=True).strip()
            outputs.append({**p, "probe_response": txt})
        if (i // batch_size) % 5 == 0:
            _flush(f"  probe progress: {i + len(batch)}/{n}")
    del model; torch.cuda.empty_cache()
    return outputs


def cluster_responses(responses, k_values=(5, 8, 12, 16)):
    """SBERT-embed probe responses then cluster.  Return per-k assignment + top TF-IDF terms per cluster."""
    from sentence_transformers import SentenceTransformer
    _flush("loading SBERT all-mpnet-base-v2")
    sb = SentenceTransformer("all-mpnet-base-v2")
    texts = [r["probe_response"] for r in responses]
    E = sb.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    _flush(f"  SBERT embedded: {E.shape}")

    results = {}
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.9)
    X = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(E)
        per_cluster = {}
        for c in range(k):
            mask = labels == c
            if mask.sum() == 0:
                per_cluster[c] = {"n": 0, "top_terms": [], "examples": []}
                continue
            mean_tfidf = np.asarray(X[mask].mean(axis=0)).ravel()
            top = mean_tfidf.argsort()[-10:][::-1]
            per_cluster[c] = {
                "n": int(mask.sum()),
                "top_terms": [terms[t] for t in top],
                "examples": [texts[i] for i in np.where(mask)[0][:5]],
            }
        results[f"k_{k}"] = {
            "assignments": labels.tolist(),
            "clusters": per_cluster,
            "inertia": float(km.inertia_),
        }
    return results, E


def main():
    cluster_map, cluster_labels = load_cluster_map()
    cells = build_cells(cluster_map)
    _flush(f"shared cells (human+LLM, same dilemma+cluster): {len(cells)}")

    pairs = stratified_sample(cells, N_SAMPLES, cluster_labels, rng=RNG, scramble=False)
    _flush(f"paired samples: {len(pairs)}")
    scrambled = stratified_sample(cells, N_SAMPLES // 2, cluster_labels, rng=np.random.RandomState(7), scramble=True)
    _flush(f"scrambled controls: {len(scrambled)}")

    all_pairs = pairs + scrambled
    outputs = run_llm_probe(all_pairs)

    (ANALYSIS / "milestone3_qualitative_probe_raw.json").write_text(json.dumps(outputs, indent=2))
    _flush("saved raw probe responses")

    # Cluster only the paired (non-scrambled) responses
    paired_outputs = [o for o in outputs if not o["scrambled"]]
    scrambled_outputs = [o for o in outputs if o["scrambled"]]

    cluster_results, _emb = cluster_responses(paired_outputs)
    _flush(f"paired clustered into {[k for k in cluster_results.keys()]}")
    scr_cluster_results, _ = cluster_responses(scrambled_outputs)

    out = {
        "n_paired": len(paired_outputs),
        "n_scrambled": len(scrambled_outputs),
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "paired_clusters": cluster_results,
        "scrambled_clusters": scr_cluster_results,
    }
    (ANALYSIS / "milestone3_qualitative_probe_clusters.json").write_text(json.dumps(out, indent=2))
    _flush("saved milestone3_qualitative_probe_clusters.json")


if __name__ == "__main__":
    main()
