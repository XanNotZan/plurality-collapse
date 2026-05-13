"""Per-dilemma robustness suite, part 2:
  1) SBERT re-embed -> within-dilemma cosine (encoder swap)
  2) Higher-order surface residualization: char n-grams + function-words + sentence stats
  3) Per-LLM-model within-dilemma (within Claude only, within GPT-3.5 only, etc.)
  4) Author-frequency check + author-deduplication
  5) Per-dilemma Kaleido top-K decoded value-cluster diversity (categorical content)
"""

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
from sklearn.feature_extraction.text import TfidfVectorizer

EMBEDDINGS_DIR = Path("data/embeddings")
ANALYSIS = Path("data/analysis")
ARCTIC_DIR = Path("data/arcticshift")
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
RNG = np.random.RandomState(42)

logger = logging.getLogger("arctic_sbert")
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


def cosine_pairwise_mean(E):
    if E.shape[0] < 2:
        return float("nan")
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    sim = En @ En.T
    n = sim.shape[0]
    return float(1 - sim[np.triu_indices(n, k=1)].mean())


# Function-word list (common English function words, sourced from standard lists)
FUNCTION_WORDS = set("""
a about above across after again against all almost alone along already also although always am
among an and another any anybody anyhow anyone anything anywhere are around as at away
back be became because been before being below beside besides between beyond both but by
came can cannot could did do does doing done down during
each either else enough even every everyone everything everywhere
few first for from
get gets got got
had has have having he her here him himself his how however
i if in into is it its itself
just
let like
many may me might more most much must my myself
nearly never no nobody none nor not nothing now
of off often on once one only or other others ought our out over
perhaps possibly
quite
rather really
said same she should so some somebody someone something somewhere still such
that the their them then there these they this those through to too toward toward
under unless until up upon us
very
was we well were what whatever when where which while who whom whose why will with within without would
yet you your yourself
""".split())


def sentence_stats(text):
    """Return [n_sentences, mean_sentence_len, std_sentence_len, n_questions, n_exclamations, n_uppercase_words, n_punct_chars]."""
    if not text:
        return [0.0] * 7
    sents = re.split(r"[.!?]+", text)
    sents = [s.strip() for s in sents if s.strip()]
    n_s = len(sents)
    lens = [len(s.split()) for s in sents]
    mean_len = float(np.mean(lens)) if lens else 0.0
    std_len = float(np.std(lens)) if lens else 0.0
    n_q = text.count("?")
    n_e = text.count("!")
    n_up = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    n_punct = sum(1 for c in text if c in ".,!?;:")
    return [n_s, mean_len, std_len, n_q, n_e, n_up, n_punct]


def function_word_freq_vec(text):
    """Frequency of each function word (sorted alphabetically for consistent vector)."""
    if not text:
        return np.zeros(len(FW_LIST), dtype=np.float32)
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    cnt = Counter(t for t in tokens if t in FUNCTION_WORDS)
    total = sum(cnt.values()) or 1
    return np.array([cnt.get(fw, 0) / total for fw in FW_LIST], dtype=np.float32)


FW_LIST = sorted(FUNCTION_WORDS)


def load_arctic_data():
    H = np.load(EMBEDDINGS_DIR / "human_arctic.npy")
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

    h_by_sub = defaultdict(list)
    idx_to_rec = {}
    for m in arctic_meta:
        sid = m["submission_id"]
        h_by_sub[sid].append(m["index"])
        rec = arctic_recs.get((sid, m.get("comment_id")))
        if rec:
            idx_to_rec[m["index"]] = rec
    return H, arctic_meta, h_by_sub, idx_to_rec, arctic_recs


def load_llm_data():
    _flush("loading HF dataset for LLM text")
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
    return col_data, sub_to_row, LLM_metas, LLM_embs


def main():
    H_kaleido, arctic_meta, h_by_sub, idx_to_rec, arctic_recs = load_arctic_data()
    _flush(f"arctic loaded: {H_kaleido.shape}, n_sub={len(h_by_sub)}")
    col_data, sub_to_row, LLM_metas, LLM_embs = load_llm_data()

    # --- Build aligned LLM data restricted to arctic dilemmas -------------------
    l_by_sub = defaultdict(list)
    for s in LLM_SOURCES:
        for m in LLM_metas[s]:
            sid = m["submission_id"]
            if sid not in h_by_sub:
                continue
            E = LLM_embs[s][m["index"]]
            col = m["column"]
            text = col_data[col][sub_to_row[sid]] if col in col_data else ""
            text = text if isinstance(text, str) else ""
            l_by_sub[sid].append({"source": s, "emb": E, "text": text, "col": col})
    qualified = sorted([s for s in h_by_sub if len(h_by_sub[s]) >= 5 and len(l_by_sub.get(s, [])) >= 2])
    _flush(f"qualified dilemmas: {len(qualified)}")

    out = {"n_dilemmas": len(qualified)}

    # --- 1. SBERT encoder swap -------------------------------------------------
    _flush("=== 1. SBERT re-embed + within-dilemma ===")
    from sentence_transformers import SentenceTransformer
    free_gpu()
    sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

    # Build text lists in same order as embeddings
    h_texts = []
    h_indices = []  # arctic index for each text
    for m in arctic_meta:
        rec = idx_to_rec.get(m["index"])
        h_texts.append(rec["body"] if rec else "")
        h_indices.append(m["index"])
    _flush(f"sbert-encoding {len(h_texts)} human comments")
    t0 = time.time()
    H_sbert = sbert.encode(h_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    _flush(f"sbert humans done in {time.time()-t0:.1f}s, shape {H_sbert.shape}")

    # LLM texts
    l_texts = []
    l_metadata = []
    for sid in qualified:
        for it in l_by_sub[sid]:
            l_texts.append(it["text"])
            l_metadata.append((sid, it["source"], it["col"]))
    _flush(f"sbert-encoding {len(l_texts)} LLM rationales")
    t0 = time.time()
    L_sbert = sbert.encode(l_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    _flush(f"sbert LLM done in {time.time()-t0:.1f}s, shape {L_sbert.shape}")

    del sbert
    free_gpu()

    # Save SBERT embeddings for future use
    np.save(EMBEDDINGS_DIR / "human_arctic_sbert.npy", H_sbert)
    # Build per-dilemma indices for SBERT humans (map arctic index -> sbert position)
    arctic_idx_to_sbert_pos = {m["index"]: i for i, m in enumerate(arctic_meta)}
    # Per-dilemma SBERT cosine
    wh_sbert, wl_sbert = [], []
    sbert_l_offset = 0
    sbert_l_by_sub = {}
    for sid in qualified:
        ll = []
        for it in l_by_sub[sid]:
            ll.append(L_sbert[sbert_l_offset])
            sbert_l_offset += 1
        sbert_l_by_sub[sid] = np.stack(ll)
    for sid in qualified:
        h_pos = [arctic_idx_to_sbert_pos[i] for i in h_by_sub[sid]]
        H_s = H_sbert[h_pos]
        L_s = sbert_l_by_sub[sid]
        wh_sbert.append(cosine_pairwise_mean(H_s))
        wl_sbert.append(cosine_pairwise_mean(L_s))
    wh_sbert, wl_sbert = np.array(wh_sbert), np.array(wl_sbert)
    _flush(f"SBERT per-dilemma: h={wh_sbert.mean():.4f}, l={wl_sbert.mean():.4f}, "
           f"ratio={wh_sbert.mean()/wl_sbert.mean():.3f}, frac_h>l={float(np.mean(wh_sbert>wl_sbert)):.4f}")
    out["sbert_per_dilemma"] = {
        "mean_within_human": float(wh_sbert.mean()),
        "mean_within_llm": float(wl_sbert.mean()),
        "ratio": float(wh_sbert.mean()/wl_sbert.mean()),
        "frac_h>l": float(np.mean(wh_sbert > wl_sbert)),
    }

    # --- 2. Higher-order surface residualization (char n-gram + function-word + sentence-stats) ---
    _flush("=== 2. Higher-order surface residualization (Kaleido) ===")
    all_texts = h_texts + l_texts
    all_embs = np.vstack([H_kaleido[[arctic_idx_to_sbert_pos[m['index']] for m in arctic_meta]], np.vstack([np.stack([it["emb"] for it in l_by_sub[s]]) for s in qualified])])
    # Actually simpler: pool aligned to (h_texts, l_texts)
    H_aligned = H_kaleido[[m["index"] for m in arctic_meta]]
    L_aligned = []
    for sid in qualified:
        for it in l_by_sub[sid]:
            L_aligned.append(it["emb"])
    L_aligned = np.stack(L_aligned)
    all_embs = np.vstack([H_aligned, L_aligned])
    n_h = H_aligned.shape[0]
    _flush(f"pooled: {all_embs.shape}, n_h={n_h}")

    # Char n-gram TF-IDF
    _flush("char n-gram TF-IDF (1K features)")
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=1000, min_df=10, sublinear_tf=True)
    F_char = vec_char.fit_transform(all_texts).toarray().astype(np.float32)
    # Function-word frequencies
    _flush("function-word frequencies")
    F_fw = np.array([function_word_freq_vec(t) for t in all_texts], dtype=np.float32)
    # Sentence stats
    _flush("sentence stats")
    F_ss = np.array([sentence_stats(t) for t in all_texts], dtype=np.float32)
    # Combine + log-length
    log_len = np.log1p(np.array([[len(t.split()), len(t)] for t in all_texts], dtype=np.float32))
    F = np.hstack([F_char, F_fw, F_ss, log_len])
    F_mean = F.mean(axis=0); F = F - F_mean
    _flush(f"design matrix F shape {F.shape}")

    # Ridge
    alpha = 10.0
    p = F.shape[1]
    G = (F.T @ F).astype(np.float64) + alpha * np.eye(p)
    rhs = (F.T @ all_embs).astype(np.float64)
    W = np.linalg.solve(G, rhs).astype(np.float32)
    X_pred = F @ W
    X_resid = (all_embs - X_pred).astype(np.float32)
    frac_var = float(X_resid.var() / all_embs.var())
    _flush(f"frac var preserved (higher-order residual): {frac_var:.3f}")

    H_resid = X_resid[:n_h]
    L_resid = X_resid[n_h:]
    # Reuse offset
    sbert_l_offset = 0  # repurpose
    resid_l_by_sub = {}
    pos = 0
    for sid in qualified:
        cnt = len(l_by_sub[sid])
        resid_l_by_sub[sid] = L_resid[pos:pos + cnt]
        pos += cnt
    wh_hi, wl_hi = [], []
    for sid in qualified:
        h_pos = [arctic_idx_to_sbert_pos[i] for i in h_by_sub[sid]]
        wh_hi.append(cosine_pairwise_mean(H_resid[h_pos]))
        wl_hi.append(cosine_pairwise_mean(resid_l_by_sub[sid]))
    wh_hi, wl_hi = np.array(wh_hi), np.array(wl_hi)
    _flush(f"higher-order residualized: h={wh_hi.mean():.4f}, l={wl_hi.mean():.4f}, ratio={wh_hi.mean()/wl_hi.mean():.3f}, frac_h>l={float(np.mean(wh_hi>wl_hi)):.4f}")
    out["higher_order_residualized"] = {
        "n_features": int(F.shape[1]),
        "frac_var_preserved": frac_var,
        "h": float(wh_hi.mean()), "l": float(wl_hi.mean()),
        "ratio": float(wh_hi.mean()/wl_hi.mean()),
        "frac_h>l": float(np.mean(wh_hi > wl_hi)),
    }

    # --- 3. Per-LLM-model within-dilemma ---------------------------------------
    _flush("=== 3. Per-LLM-model within-dilemma ===")
    per_llm_stats = {}
    for s in LLM_SOURCES:
        whs, wls = [], []
        n_dilemmas = 0
        for sid in qualified:
            same_src = [it["emb"] for it in l_by_sub[sid] if it["source"] == s]
            if len(same_src) >= 2:
                wls.append(cosine_pairwise_mean(np.stack(same_src)))
                # match human
                h_pos = [arctic_idx_to_sbert_pos[i] for i in h_by_sub[sid]]
                whs.append(cosine_pairwise_mean(H_aligned[h_pos]))
                n_dilemmas += 1
        if n_dilemmas == 0:
            continue
        whs, wls = np.array(whs), np.array(wls)
        per_llm_stats[s] = {
            "n_dilemmas": n_dilemmas,
            "within_llm": float(wls.mean()),
            "within_human": float(whs.mean()),
            "ratio": float(whs.mean()/wls.mean()),
            "frac_h>l": float(np.mean(whs > wls)),
        }
        _flush(f"  {s}: n={n_dilemmas}, within-{s}={wls.mean():.4f}, within-human={whs.mean():.4f}, "
               f"ratio={whs.mean()/wls.mean():.3f}, frac_h>l={float(np.mean(whs>wls)):.4f}")
    out["per_llm_within_dilemma"] = per_llm_stats

    # --- 4. Author de-duplication ----------------------------------------------
    _flush("=== 4. Author de-duplication ===")
    # Per-author counts
    author_counts = Counter()
    for rec in idx_to_rec.values():
        a = rec.get("author")
        if a and a != "[deleted]":
            author_counts[a] += 1
    multi_authors = {a for a, c in author_counts.items() if c > 1}
    _flush(f"total unique authors: {len(author_counts)}, with multiple comments: {len(multi_authors)}")
    # Top 10
    for a, c in author_counts.most_common(10):
        _flush(f"  {a}: {c} comments")

    # Per-dilemma: re-run within-dilemma keeping only one comment per author per dilemma (first in time)
    wh_dedup, wl_dedup = [], []
    for sid in qualified:
        h_idxs = h_by_sub[sid]
        # one per author per dilemma
        seen_authors = set()
        keep = []
        for i in h_idxs:
            rec = idx_to_rec.get(i)
            if not rec:
                continue
            a = rec.get("author") or "anon"
            if a not in seen_authors:
                seen_authors.add(a)
                keep.append(i)
        if len(keep) >= 5:
            h_pos = [arctic_idx_to_sbert_pos[i] for i in keep]
            wh_dedup.append(cosine_pairwise_mean(H_aligned[h_pos]))
            # match LLM
            L_e = np.stack([it["emb"] for it in l_by_sub[sid]])
            wl_dedup.append(cosine_pairwise_mean(L_e))
    wh_dedup, wl_dedup = np.array(wh_dedup), np.array(wl_dedup)
    _flush(f"author-deduped: n={len(wh_dedup)}, h={wh_dedup.mean():.4f}, l={wl_dedup.mean():.4f}, "
           f"ratio={wh_dedup.mean()/wl_dedup.mean():.3f}, frac_h>l={float(np.mean(wh_dedup>wl_dedup)):.4f}")
    out["author_deduplicated"] = {
        "n_dilemmas": int(len(wh_dedup)),
        "total_unique_authors": int(len(author_counts)),
        "authors_with_multiple_comments": int(len(multi_authors)),
        "h": float(wh_dedup.mean()), "l": float(wl_dedup.mean()),
        "ratio": float(wh_dedup.mean()/wl_dedup.mean()),
        "frac_h>l": float(np.mean(wh_dedup > wl_dedup)),
    }

    path = ANALYSIS / "milestone3_arctic_part2_robust.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")


if __name__ == "__main__":
    main()
