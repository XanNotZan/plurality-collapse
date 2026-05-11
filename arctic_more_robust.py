"""More per-dilemma robustness tests on ArcticShift data:
  - Sample-size matched (subsample k=5 humans + k=5 LLMs per dilemma)
  - Score-stratified (top-3 humans by score vs rest)
  - Within-verdict (only NTA-cited comments on both sides)
  - Distance-metric robustness (Euclidean)
  - MFD foundation diversity per dilemma
  - Bootstrap CIs
"""

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

EMBEDDINGS_DIR = Path("data/embeddings")
ANALYSIS = Path("data/analysis")
ARCTIC_DIR = Path("data/arcticshift")
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
RNG = np.random.RandomState(42)

logger = logging.getLogger("arctic_more")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg); sys.stdout.flush()


def cosine_pairwise_mean(E):
    if E.shape[0] < 2:
        return float("nan")
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    sim = En @ En.T
    n = sim.shape[0]
    return float(1 - sim[np.triu_indices(n, k=1)].mean())


def euclid_pairwise_mean(E):
    if E.shape[0] < 2:
        return float("nan")
    diffs = E[:, None, :] - E[None, :, :]
    d = np.linalg.norm(diffs, axis=-1)
    n = d.shape[0]
    return float(d[np.triu_indices(n, k=1)].mean())


# MFD
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
FOUNDATIONS = list(MFD.keys())
F_TO_IDX = {f: i for i, f in enumerate(FOUNDATIONS)}


def mfd_count_vec(text):
    counts = np.zeros(len(FOUNDATIONS), dtype=np.int32)
    if not text:
        return counts
    for tok in re.findall(r"\b[a-z]+\b", text.lower()):
        for stem, fnd in MFD_PAIRS:
            if tok.startswith(stem):
                counts[F_TO_IDX[fnd]] += 1
                break
    return counts


VERDICT_PATTERNS = {
    "NTA": re.compile(r"\b(nta|not\s+the\s+asshole|not\s+an?\s+asshole)\b", re.IGNORECASE),
    "YTA": re.compile(r"\b(yta|you'?re\s+the\s+asshole|you\s+are\s+the\s+asshole)\b", re.IGNORECASE),
    "ESH": re.compile(r"\b(esh|everyone\s+sucks)\b", re.IGNORECASE),
    "NAH": re.compile(r"\b(nah|no\s+(?:assholes|aholes)\s+here)\b", re.IGNORECASE),
}


def detect_verdict(text):
    """Return first-mentioned verdict marker, or None."""
    if not text:
        return None
    earliest = (None, len(text) + 1)
    for v, pat in VERDICT_PATTERNS.items():
        m = pat.search(text)
        if m and m.start() < earliest[1]:
            earliest = (v, m.start())
    return earliest[0]


def main():
    H = np.load(EMBEDDINGS_DIR / "human_arctic.npy")
    arctic_meta = json.loads((EMBEDDINGS_DIR / "human_arctic_meta.json").read_text())
    _flush(f"arctic: {H.shape}")

    # Load arctic texts + scores
    arctic_records = {}
    for line in (ARCTIC_DIR / "filtered_comments.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("_empty") or not j.get("body"):
            continue
        arctic_records[(j["submission_id"], j.get("comment_id"))] = j

    # Map index → record
    idx_to_rec = {}
    for m in arctic_meta:
        key = (m["submission_id"], m.get("comment_id"))
        rec = arctic_records.get(key)
        if rec:
            idx_to_rec[m["index"]] = rec

    # Group arctic indices by submission_id
    h_by_sub = defaultdict(list)
    for m in arctic_meta:
        h_by_sub[m["submission_id"]].append(m["index"])

    # Build LLM by_sub
    LLM_metas = {s: json.loads((EMBEDDINGS_DIR / f"{s}_meta.json").read_text()) for s in LLM_SOURCES}
    LLM_embs = {s: np.load(EMBEDDINGS_DIR / f"{s}.npy") for s in LLM_SOURCES}

    # Load HF dataset for LLM texts
    _flush("loading HF dataset for LLM texts")
    ds = load_dataset("ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas", split="test")
    sub_to_row = {ds[i]["submission_id"]: i for i in range(len(ds))}
    needed = set()
    for s in LLM_SOURCES:
        for c in [f"{s}_reason_1", f"{s}_reason_2", f"{s}_reason_3"]:
            if c in ds.column_names:
                needed.add(c)
    col_data = {c: ds[c] for c in needed}

    l_by_sub = defaultdict(list)
    llm_text_by_idx = {}  # (source, idx_in_source) -> text
    for s in LLM_SOURCES:
        for m in LLM_metas[s]:
            sid = m["submission_id"]
            if sid not in h_by_sub:
                continue
            E = LLM_embs[s][m["index"]]
            col = m["column"]
            text = col_data[col][sub_to_row[sid]] if col in col_data else ""
            text = text if isinstance(text, str) else ""
            entry = {"source": s, "embedding": E, "text": text, "col": col}
            l_by_sub[sid].append(entry)

    qualified = [s for s in h_by_sub if len(h_by_sub[s]) >= 5 and len(l_by_sub.get(s, [])) >= 5]
    _flush(f"qualified dilemmas (h>=5, l>=5): {len(qualified)}")

    out = {"n_dilemmas_qualified": len(qualified)}

    # ─── 1. Sample-size matched (k=5 each) ─────────────────────────────────────
    K = 5
    _flush(f"=== Sample-size matched (k={K}) ===")
    wh_arr, wl_arr = [], []
    wh_eu, wl_eu = [], []
    for sid in qualified:
        h_idxs = h_by_sub[sid]
        l_items = l_by_sub[sid]
        h_sub = RNG.choice(h_idxs, K, replace=False)
        l_sub_idx = RNG.choice(len(l_items), K, replace=False)
        Hm = H[h_sub]
        Lm = np.stack([l_items[i]["embedding"] for i in l_sub_idx])
        wh_arr.append(cosine_pairwise_mean(Hm))
        wl_arr.append(cosine_pairwise_mean(Lm))
        wh_eu.append(euclid_pairwise_mean(Hm))
        wl_eu.append(euclid_pairwise_mean(Lm))
    wh_arr, wl_arr = np.array(wh_arr), np.array(wl_arr)
    wh_eu, wl_eu = np.array(wh_eu), np.array(wl_eu)
    _flush(f"k={K} matched cosine: h={wh_arr.mean():.4f}, l={wl_arr.mean():.4f}, ratio={wh_arr.mean()/wl_arr.mean():.3f}, frac_h>l={float(np.mean(wh_arr>wl_arr)):.4f}")
    _flush(f"k={K} matched euclidean: h={wh_eu.mean():.4f}, l={wl_eu.mean():.4f}, ratio={wh_eu.mean()/wl_eu.mean():.3f}, frac_h>l={float(np.mean(wh_eu>wl_eu)):.4f}")
    out["sample_matched_k5"] = {
        "cosine": {"h": float(wh_arr.mean()), "l": float(wl_arr.mean()), "ratio": float(wh_arr.mean()/wl_arr.mean()), "frac_h>l": float(np.mean(wh_arr>wl_arr))},
        "euclidean": {"h": float(wh_eu.mean()), "l": float(wl_eu.mean()), "ratio": float(wh_eu.mean()/wl_eu.mean()), "frac_h>l": float(np.mean(wh_eu>wl_eu))},
    }

    # ─── 2. Score-stratified: top-3 humans by score vs rest ───────────────────
    _flush("=== Score-stratified ===")
    wh_top, wh_rest = [], []
    for sid in qualified:
        h_idxs = h_by_sub[sid]
        scores = [idx_to_rec[i].get("score", 0) for i in h_idxs]
        order = np.argsort(scores)[::-1]
        top = [h_idxs[i] for i in order[:3]]
        rest = [h_idxs[i] for i in order[3:]]
        if len(top) >= 2:
            wh_top.append(cosine_pairwise_mean(H[top]))
        if len(rest) >= 2:
            wh_rest.append(cosine_pairwise_mean(H[rest]))
    wh_top, wh_rest = np.array(wh_top), np.array(wh_rest)
    _flush(f"top-3 humans cosine: {wh_top.mean():.4f} ± {wh_top.std():.4f}")
    _flush(f"rest humans cosine: {wh_rest.mean():.4f} ± {wh_rest.std():.4f}")
    out["score_stratified"] = {
        "top3_humans": {"mean": float(wh_top.mean()), "std": float(wh_top.std()), "n_dilemmas": int(len(wh_top))},
        "rest_humans": {"mean": float(wh_rest.mean()), "std": float(wh_rest.std()), "n_dilemmas": int(len(wh_rest))},
    }

    # ─── 3. Within-verdict (NTA-only) ─────────────────────────────────────────
    _flush("=== Within-NTA only ===")
    wh_nta, wl_nta = [], []
    n_nta_dilemmas = 0
    for sid in qualified:
        h_nta = []
        for i in h_by_sub[sid]:
            rec = idx_to_rec.get(i)
            if rec and detect_verdict(rec.get("body")) == "NTA":
                h_nta.append(i)
        l_nta = [it["embedding"] for it in l_by_sub[sid] if detect_verdict(it["text"]) == "NTA"]
        if len(h_nta) >= 5 and len(l_nta) >= 2:
            n_nta_dilemmas += 1
            wh_nta.append(cosine_pairwise_mean(H[h_nta]))
            wl_nta.append(cosine_pairwise_mean(np.stack(l_nta)))
    wh_nta, wl_nta = np.array(wh_nta), np.array(wl_nta)
    _flush(f"within-NTA: n_dilemmas={n_nta_dilemmas}, h={wh_nta.mean():.4f}, l={wl_nta.mean():.4f}, ratio={wh_nta.mean()/wl_nta.mean():.3f}, frac_h>l={float(np.mean(wh_nta>wl_nta)):.4f}")
    out["within_nta_only"] = {
        "n_dilemmas": n_nta_dilemmas,
        "h": float(wh_nta.mean()), "l": float(wl_nta.mean()),
        "ratio": float(wh_nta.mean()/wl_nta.mean()),
        "frac_h>l": float(np.mean(wh_nta>wl_nta)),
    }

    # ─── 4. Bootstrap CIs on ratio ────────────────────────────────────────────
    _flush("=== Bootstrap CIs ===")
    n_boots = 1000
    # Recompute baseline within-dilemma metrics (cosine)
    wh_full, wl_full = [], []
    for sid in qualified:
        h_idxs = h_by_sub[sid]
        l_items = l_by_sub[sid]
        wh_full.append(cosine_pairwise_mean(H[h_idxs]))
        wl_full.append(cosine_pairwise_mean(np.stack([it["embedding"] for it in l_items])))
    wh_full, wl_full = np.array(wh_full), np.array(wl_full)
    boot_ratios = []
    boot_fracs = []
    n = len(wh_full)
    for b in range(n_boots):
        idx = RNG.randint(0, n, size=n)
        boot_ratios.append(float(wh_full[idx].mean() / wl_full[idx].mean()))
        boot_fracs.append(float(np.mean(wh_full[idx] > wl_full[idx])))
    ratio_q = np.percentile(boot_ratios, [2.5, 50, 97.5])
    frac_q = np.percentile(boot_fracs, [2.5, 50, 97.5])
    _flush(f"bootstrap ratio: median={ratio_q[1]:.4f}, 95% CI=[{ratio_q[0]:.4f}, {ratio_q[2]:.4f}]")
    _flush(f"bootstrap frac_h>l: median={frac_q[1]:.4f}, 95% CI=[{frac_q[0]:.4f}, {frac_q[2]:.4f}]")
    out["bootstrap"] = {
        "n_boots": n_boots,
        "ratio_2.5": float(ratio_q[0]), "ratio_50": float(ratio_q[1]), "ratio_97.5": float(ratio_q[2]),
        "frac_h>l_2.5": float(frac_q[0]), "frac_h>l_50": float(frac_q[1]), "frac_h>l_97.5": float(frac_q[2]),
    }

    # ─── 5. MFD foundation diversity per dilemma ──────────────────────────────
    _flush("=== MFD per-dilemma foundation diversity ===")
    uf_h, uf_l = [], []
    ent_h, ent_l = [], []
    for sid in qualified:
        # Humans
        cnt_h = np.zeros(len(FOUNDATIONS), dtype=np.int32)
        n_h_with = 0
        for i in h_by_sub[sid]:
            rec = idx_to_rec.get(i)
            if not rec:
                continue
            c = mfd_count_vec(rec.get("body", ""))
            if c.sum() > 0:
                n_h_with += 1
            cnt_h += c
        # LLMs
        cnt_l = np.zeros(len(FOUNDATIONS), dtype=np.int32)
        for it in l_by_sub[sid]:
            c = mfd_count_vec(it.get("text", ""))
            cnt_l += c
        # Unique foundations used (with at least 1 lemma)
        uf_h.append(int((cnt_h > 0).sum()))
        uf_l.append(int((cnt_l > 0).sum()))
        # Entropy over foundations
        if cnt_h.sum() > 0:
            p = cnt_h / cnt_h.sum()
            ent_h.append(float(-np.sum(p * np.log(p + 1e-12))))
        else:
            ent_h.append(float("nan"))
        if cnt_l.sum() > 0:
            p = cnt_l / cnt_l.sum()
            ent_l.append(float(-np.sum(p * np.log(p + 1e-12))))
        else:
            ent_l.append(float("nan"))
    uf_h, uf_l = np.array(uf_h), np.array(uf_l)
    ent_h, ent_l = np.array(ent_h), np.array(ent_l)
    val = ~np.isnan(ent_h) & ~np.isnan(ent_l)
    _flush(f"unique MFD foundations per dilemma: h={uf_h.mean():.3f}, l={uf_l.mean():.3f}, "
           f"frac_h>l={float(np.mean(uf_h>uf_l)):.4f}")
    _flush(f"MFD entropy per dilemma: h={ent_h[val].mean():.4f}, l={ent_l[val].mean():.4f}, "
           f"frac_h>l={float(np.mean(ent_h[val]>ent_l[val])):.4f}")
    out["mfd_per_dilemma"] = {
        "unique_foundations": {"h": float(uf_h.mean()), "l": float(uf_l.mean()), "frac_h>l": float(np.mean(uf_h>uf_l))},
        "entropy": {"h": float(ent_h[val].mean()), "l": float(ent_l[val].mean()), "frac_h>l": float(np.mean(ent_h[val]>ent_l[val]))},
    }

    path = ANALYSIS / "milestone3_arctic_more_robust.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")


if __name__ == "__main__":
    main()
