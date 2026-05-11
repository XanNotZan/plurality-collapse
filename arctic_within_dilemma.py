"""Embed ArcticShift-collected human comments via Kaleido, compute within-dilemma diversity vs LLM.

Inputs:
  data/arcticshift/filtered_comments.jsonl

Outputs:
  data/embeddings/human_arctic.npy + meta.json
  data/analysis/milestone3_arctic_within_dilemma.json
"""

import gc
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

EMBEDDINGS_DIR = Path("data/embeddings")
ARCTIC_DIR = Path("data/arcticshift")
ANALYSIS = Path("data/analysis")
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]

logger = logging.getLogger("arctic_within")
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
        torch.cuda.ipc_collect()


def load_kaleido():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    _flush("loading Kaleido")
    tok = AutoTokenizer.from_pretrained("allenai/kaleido-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/kaleido-xl", dtype=torch.float16).to("cuda").eval()
    try:
        template = model.config.task_specific_params["generate"]["template"]
    except Exception:
        template = "[Generate]:\tAction: ACTION"
    return tok, model, template


def embed(tok, model, template, texts, batch=32):
    out = []
    for s in range(0, len(texts), batch):
        b = texts[s:s + batch]
        formatted = [template.replace("ACTION", t if t else "(empty)") for t in b]
        inputs = tok(formatted, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            enc = model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        h = enc.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1)
        out.append(pooled.cpu().float().numpy())
        if (s // batch) % 50 == 0:
            _flush(f"embed {s+len(b)}/{len(texts)}")
    return np.vstack(out)


def main():
    # 1. Load filtered comments
    items = []
    for line in (ARCTIC_DIR / "filtered_comments.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("_empty"):
            continue
        if not j.get("body"):
            continue
        items.append(j)
    _flush(f"loaded {len(items)} ArcticShift comments")

    # Per-dilemma counts
    by_sub = defaultdict(list)
    for it in items:
        by_sub[it["submission_id"]].append(it)
    _flush(f"unique submissions: {len(by_sub)}, "
           f"comments per submission: min={min(len(v) for v in by_sub.values())}, "
           f"median={sorted(len(v) for v in by_sub.values())[len(by_sub)//2]}, "
           f"max={max(len(v) for v in by_sub.values())}")

    # Filter to dilemmas with >= 5 quality comments
    MIN_PER = 5
    qualified = {sid: cs for sid, cs in by_sub.items() if len(cs) >= MIN_PER}
    _flush(f"dilemmas with >= {MIN_PER} comments: {len(qualified)}")

    # Build flat list (preserve order)
    flat = []
    for sid, cs in qualified.items():
        for c in cs:
            flat.append({"submission_id": sid, "comment_id": c["comment_id"], "body": c["body"], "score": c.get("score", 0)})
    texts = [it["body"] for it in flat]
    _flush(f"total comments to embed: {len(texts)}")

    # 2. Embed via Kaleido
    free_gpu()
    tok, model, template = load_kaleido()
    embs = embed(tok, model, template, texts, batch=32)
    _flush(f"embedded shape {embs.shape}")
    np.save(EMBEDDINGS_DIR / "human_arctic.npy", embs)
    meta = [{"index": i, "submission_id": it["submission_id"], "comment_id": it["comment_id"]} for i, it in enumerate(flat)]
    (EMBEDDINGS_DIR / "human_arctic_meta.json").write_text(json.dumps(meta))
    _flush(f"saved human_arctic.npy")

    # 3. Compute within-dilemma cosine for ArcticShift humans
    _flush("computing within-dilemma cosine (ArcticShift humans)")
    arctic_within = []
    arctic_centroids = {}
    sub_to_indices = defaultdict(list)
    for i, it in enumerate(flat):
        sub_to_indices[it["submission_id"]].append(i)
    for sid, idxs in sub_to_indices.items():
        E = embs[idxs]
        # Normalize
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        # Pairwise cosine (1 - dot)
        sim = En @ En.T
        n = sim.shape[0]
        triu = sim[np.triu_indices(n, k=1)]
        if len(triu) > 0:
            arctic_within.append(float(1 - triu.mean()))
        arctic_centroids[sid] = E.mean(axis=0)
    arctic_within = np.array(arctic_within)
    _flush(f"arctic within-human pairwise cos dist: mean={arctic_within.mean():.4f} ± {arctic_within.std():.4f}, n_dilemmas={len(arctic_within)}")

    # 4. Compare to LLM within-dilemma (already computed and re-derive matched)
    # Build per-dilemma LLM embeddings
    _flush("computing within-dilemma cosine (Sachdeva LLMs)")
    LLM_metas = {s: json.loads((EMBEDDINGS_DIR / f"{s}_meta.json").read_text()) for s in LLM_SOURCES}
    LLM_embs = {s: np.load(EMBEDDINGS_DIR / f"{s}.npy") for s in LLM_SOURCES}
    llm_by_sub = defaultdict(list)
    for s in LLM_SOURCES:
        E = LLM_embs[s]
        for m in LLM_metas[s]:
            llm_by_sub[m["submission_id"]].append(E[m["index"]])
    llm_within = []
    matched_human_within = []  # subset of arctic_within where dilemma is in qualified set
    h_to_l_cent = []
    for sid in qualified:
        if sid not in llm_by_sub or len(llm_by_sub[sid]) < 2:
            continue
        L = np.stack(llm_by_sub[sid])
        Ln = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-12)
        sim = Ln @ Ln.T
        n = sim.shape[0]
        triu = sim[np.triu_indices(n, k=1)]
        if len(triu) > 0:
            llm_within.append(float(1 - triu.mean()))
            arctic_idxs = sub_to_indices[sid]
            E = embs[arctic_idxs]
            En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
            sim_h = En @ En.T
            triu_h = sim_h[np.triu_indices(En.shape[0], k=1)] if En.shape[0] > 1 else np.array([])
            if len(triu_h):
                matched_human_within.append(float(1 - triu_h.mean()))
            # Centroid distance human-LLM per dilemma
            hc = E.mean(axis=0); lc = L.mean(axis=0)
            ca = hc / (np.linalg.norm(hc) + 1e-12); cb = lc / (np.linalg.norm(lc) + 1e-12)
            h_to_l_cent.append(float(1 - ca @ cb))
    llm_within = np.array(llm_within)
    matched_human_within = np.array(matched_human_within)
    h_to_l_cent = np.array(h_to_l_cent)
    _flush(f"matched dilemmas: {len(llm_within)}")
    _flush(f"  within-human: {matched_human_within.mean():.4f} ± {matched_human_within.std():.4f}")
    _flush(f"  within-LLM:   {llm_within.mean():.4f} ± {llm_within.std():.4f}")
    _flush(f"  human-LLM centroid: {h_to_l_cent.mean():.4f} ± {h_to_l_cent.std():.4f}")
    frac_h_more_diverse = float(np.mean(matched_human_within > llm_within))
    _flush(f"  fraction of dilemmas where humans MORE diverse than LLMs: {frac_h_more_diverse:.4f}")

    # 5. Corpus-level: PCA on full ArcticShift human pool, compare to original human
    _flush("PCA on ArcticShift human corpus")
    pca = PCA(svd_solver="full")
    pca.fit(embs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    eig = pca.explained_variance_
    arctic_comp90 = int(np.searchsorted(cum, 0.90) + 1)
    arctic_pr = float((eig.sum() ** 2) / (eig ** 2).sum())
    _flush(f"  arctic human (n={embs.shape[0]}): comp90={arctic_comp90}, PR={arctic_pr:.2f}")

    # Match-n: sample arctic to n_orig_human (10826) — but we have ~50K, sample matched n
    n_orig = 10826
    if embs.shape[0] >= n_orig:
        rng = np.random.RandomState(42)
        idx = rng.choice(embs.shape[0], n_orig, replace=False)
        e_match = embs[idx]
        pca2 = PCA(svd_solver="full"); pca2.fit(e_match)
        cum2 = np.cumsum(pca2.explained_variance_ratio_)
        eig2 = pca2.explained_variance_
        arctic_match_comp90 = int(np.searchsorted(cum2, 0.90) + 1)
        arctic_match_pr = float((eig2.sum() ** 2) / (eig2 ** 2).sum())
        _flush(f"  arctic human matched n={n_orig}: comp90={arctic_match_comp90}, PR={arctic_match_pr:.2f}")
    else:
        arctic_match_comp90 = None
        arctic_match_pr = None

    out = {
        "n_arctic_comments": int(embs.shape[0]),
        "n_dilemmas_qualified": int(len(qualified)),
        "min_comments_per_dilemma": MIN_PER,
        "within_human_pairwise_cosine_dist": {
            "mean": float(matched_human_within.mean()),
            "std": float(matched_human_within.std()),
            "n_dilemmas": int(len(matched_human_within)),
        },
        "within_llm_pairwise_cosine_dist": {
            "mean": float(llm_within.mean()),
            "std": float(llm_within.std()),
            "n_dilemmas": int(len(llm_within)),
        },
        "human_to_llm_centroid_dist": {
            "mean": float(h_to_l_cent.mean()),
            "std": float(h_to_l_cent.std()),
        },
        "fraction_dilemmas_human_more_diverse_than_llm": frac_h_more_diverse,
        "arctic_full_corpus": {"n": int(embs.shape[0]), "comp90": arctic_comp90, "pr": arctic_pr},
        "arctic_matched_n_10826": {"comp90": arctic_match_comp90, "pr": arctic_match_pr},
    }
    path = ANALYSIS / "milestone3_arctic_within_dilemma.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")


if __name__ == "__main__":
    main()
