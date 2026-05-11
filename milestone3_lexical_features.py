"""A4+A5+A7: lexical/pragmatic feature extraction for human + LLM comments.

For each comment, compute density-normalised counts of:
  pragmatic markers (hedges, boosters, 1st/2nd-person pronouns, anecdote phrases),
  discourse markers (counterfactual, causal, concessive),
  POS densities (noun, verb, adjective, adverb) via spaCy,
  NER densities (PERSON, ORG, GPE, DATE, CARDINAL) via spaCy,
  concreteness proxies (mean word length, type-token ratio, noun density),
  length features (token_count, sentence_count, mean_sentence_len).

Restrict scope to dilemmas present in ArcticShift human gather (1,991 dilemmas).
Pair humans and LLMs by (submission_id, Kaleido-60-cluster) for within-cluster comparison.

Outputs:
  data/analysis/milestone3_lexical_features.parquet   (per-comment, all sources)
  data/analysis/milestone3_lexical_summary.csv        (per-source means)
  data/analysis/milestone3_lexical_within_cluster.csv (per-cluster source means)
  data/analysis/milestone3_lexical_lengthctrl.csv     (length-residualised diffs)
  data/analysis/milestone3_lexical_bootstrap.json     (bootstrap CIs, perm null)
"""

import csv
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import spacy

EMBEDDINGS_DIR = Path("data/embeddings")
ANALYSIS = Path("data/analysis")
ARCTIC_DIR = Path("data/arcticshift")
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
RNG = np.random.RandomState(42)
N_BOOT = 1000
N_PERM = 1000

logger = logging.getLogger("lexical")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg); sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Pragmatic / discourse marker dictionaries
# ─────────────────────────────────────────────────────────────────────────────

HEDGE_SINGLE = {
    "maybe", "might", "perhaps", "possibly", "probably", "seems", "seem",
    "seemed", "appears", "appear", "appeared", "kinda", "sorta",
    "could", "may", "supposedly", "presumably", "arguably",
}
HEDGE_MULTI = [
    r"\bsort of\b", r"\bkind of\b", r"\bi think\b", r"\bi guess\b",
    r"\bi suppose\b", r"\bi believe\b", r"\bi assume\b", r"\bi feel like\b",
    r"\bcould be\b", r"\bmay be\b", r"\bmight be\b",
    r"\bin my opinion\b", r"\bif you ask me\b", r"\bif anything\b",
]

BOOSTER_SINGLE = {
    "definitely", "clearly", "obviously", "certainly", "undoubtedly",
    "absolutely", "totally", "completely", "exactly", "must", "surely",
    "always", "never", "literally", "always", "indeed", "frankly",
}
BOOSTER_MULTI = [
    r"\bof course\b", r"\bwithout a doubt\b", r"\bno question\b",
    r"\bno way\b", r"\bthere's no\b", r"\bzero chance\b",
]

P1 = {"i", "me", "my", "mine", "we", "us", "our", "ours", "myself", "ourselves",
      "i'm", "i've", "i'd", "i'll", "we've", "we're", "we'll", "we'd"}
P2 = {"you", "your", "yours", "yourself", "yourselves", "you're", "you've",
      "you'll", "you'd", "y'all"}

ANECDOTE_PATTERNS = [
    r"\bwhen i was\b", r"\bback when\b", r"\bi remember\b", r"\bi used to\b",
    r"\bi dated\b", r"\bi married\b", r"\bi grew up\b", r"\bi had a\b",
    r"\bin my experience\b", r"\bi've been\b", r"\bi was\b",
    r"\bmy (friend|sister|brother|mom|mother|dad|father|cousin|aunt|uncle|"
    r"son|daughter|partner|husband|wife|girlfriend|boyfriend|ex|coworker|"
    r"neighbor|parents|grandma|grandpa|grandmother|grandfather|family|kid|child)\b",
    r"\bonce i\b", r"\bthat happened to me\b",
]

# Reused single-word + multi-word for counter/causal/concessive
COUNTERFACTUAL_SINGLE = {"unless", "if", "hypothetically", "supposing", "imagine"}
COUNTERFACTUAL_MULTI = [
    r"\bwould have\b", r"\bcould have\b", r"\bshould have\b",
    r"\bwould've\b", r"\bcould've\b", r"\bshould've\b",
    r"\bhad i\b", r"\bhad you\b", r"\bhad they\b",
    r"\bwere it\b", r"\bif only\b", r"\bin that case\b", r"\bwhat if\b",
]

CAUSAL_SINGLE = {"because", "since", "therefore", "hence", "thus", "consequently",
                 "so", "cuz", "cos"}
CAUSAL_MULTI = [
    r"\bdue to\b", r"\bthanks to\b", r"\bowing to\b", r"\bthat'?s why\b",
    r"\bas a result\b", r"\bso that\b", r"\bin order to\b",
    r"\bbecause of\b", r"\bthe reason\b",
]

CONCESSIVE_SINGLE = {"although", "though", "despite", "however", "nonetheless",
                     "regardless", "yet", "still"}
CONCESSIVE_MULTI = [
    r"\beven though\b", r"\bin spite of\b", r"\bthat said\b",
    r"\bon the other hand\b", r"\bhaving said that\b", r"\beven so\b",
    r"\bat the same time\b",
]


def _compile_patterns(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]


HEDGE_RE = _compile_patterns(HEDGE_MULTI)
BOOSTER_RE = _compile_patterns(BOOSTER_MULTI)
ANECDOTE_RE = _compile_patterns(ANECDOTE_PATTERNS)
CF_RE = _compile_patterns(COUNTERFACTUAL_MULTI)
CAUSAL_RE = _compile_patterns(CAUSAL_MULTI)
CONCESSIVE_RE = _compile_patterns(CONCESSIVE_MULTI)


def count_tokens_in(text_lower, single_set):
    """Count lowercase whole-word matches against a set."""
    if not text_lower:
        return 0
    tokens = re.findall(r"\b[\w']+\b", text_lower)
    if not tokens:
        return 0
    s = single_set
    return sum(1 for t in tokens if t in s)


def count_regex(text_lower, compiled_list):
    return sum(len(p.findall(text_lower)) for p in compiled_list)


def lexical_features(text):
    """Compute regex/lexicon features. Returns dict (counts; densities computed downstream)."""
    if not text:
        return None
    tl = text.lower()
    feats = {
        "hedge_count": count_tokens_in(tl, HEDGE_SINGLE) + count_regex(tl, HEDGE_RE),
        "booster_count": count_tokens_in(tl, BOOSTER_SINGLE) + count_regex(tl, BOOSTER_RE),
        "p1_count": count_tokens_in(tl, P1),
        "p2_count": count_tokens_in(tl, P2),
        "anecdote_count": count_regex(tl, ANECDOTE_RE),
        "cf_count": count_tokens_in(tl, COUNTERFACTUAL_SINGLE) + count_regex(tl, CF_RE),
        "causal_count": count_tokens_in(tl, CAUSAL_SINGLE) + count_regex(tl, CAUSAL_RE),
        "concessive_count": count_tokens_in(tl, CONCESSIVE_SINGLE) + count_regex(tl, CONCESSIVE_RE),
    }
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# spaCy features
# ─────────────────────────────────────────────────────────────────────────────

def spacy_features(doc):
    """POS, NER, length features from a parsed spaCy doc."""
    tokens = [t for t in doc if not t.is_space]
    word_tokens = [t for t in tokens if not t.is_punct]
    n_tok = len(word_tokens)
    n_all = len(tokens)
    if n_tok == 0:
        return None

    pos_counts = defaultdict(int)
    for t in word_tokens:
        pos_counts[t.pos_] += 1

    ner_counts = defaultdict(int)
    for ent in doc.ents:
        ner_counts[ent.label_] += 1

    types = {t.lower_ for t in word_tokens if t.is_alpha}
    ttr = len(types) / n_tok if n_tok else 0.0

    sentence_count = sum(1 for _ in doc.sents)
    mean_sent_len = n_tok / sentence_count if sentence_count else float(n_tok)

    char_lens = [len(t.text) for t in word_tokens if t.is_alpha]
    mean_word_len = float(np.mean(char_lens)) if char_lens else 0.0

    return {
        "token_count": n_tok,
        "type_token_ratio": ttr,
        "sentence_count": sentence_count,
        "mean_sent_len": mean_sent_len,
        "mean_word_len": mean_word_len,
        "noun_count": pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0),
        "verb_count": pos_counts.get("VERB", 0) + pos_counts.get("AUX", 0),
        "adj_count": pos_counts.get("ADJ", 0),
        "adv_count": pos_counts.get("ADV", 0),
        "pron_count": pos_counts.get("PRON", 0),
        "ner_person_count": ner_counts.get("PERSON", 0),
        "ner_org_count": ner_counts.get("ORG", 0),
        "ner_gpe_count": ner_counts.get("GPE", 0),
        "ner_date_count": ner_counts.get("DATE", 0),
        "ner_cardinal_count": ner_counts.get("CARDINAL", 0) + ner_counts.get("ORDINAL", 0),
        "ner_total_count": sum(ner_counts.values()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_arctic_recs():
    _flush("loading ArcticShift human records")
    recs = {}
    with open(ARCTIC_DIR / "filtered_comments.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            if j.get("_empty") or not j.get("body"): continue
            recs[(j["submission_id"], j.get("comment_id"))] = j["body"]
    _flush(f"  {len(recs)} arctic records with body")
    return recs


def load_human_corpus():
    arctic_meta = json.loads((EMBEDDINGS_DIR / "human_arctic_meta.json").read_text())
    arctic_recs = load_arctic_recs()
    decoded = json.loads((ANALYSIS / "milestone3_arctic_decoded_values.json").read_text())
    rows = []
    for m in arctic_meta:
        body = arctic_recs.get((m["submission_id"], m.get("comment_id")))
        if not body: continue
        idx = m["index"]
        val = decoded.get(str(idx))
        rows.append({
            "source": "human",
            "submission_id": m["submission_id"],
            "comment_id": m.get("comment_id"),
            "embed_idx": idx,
            "value_label": val,
            "text": body,
        })
    _flush(f"  human rows with body: {len(rows)} ({sum(1 for r in rows if r['value_label']) } with decoded value)")
    return rows


def load_llm_corpus(arctic_subs):
    rows = []
    for s in LLM_SOURCES:
        data = json.loads((ANALYSIS / f"llm_values_{s}.json").read_text())
        n_kept = 0
        for r in data:
            if r["submission_id"] not in arctic_subs:
                continue
            rows.append({
                "source": s,
                "submission_id": r["submission_id"],
                "comment_id": r.get("column"),
                "embed_idx": r["index"],
                "value_label": r.get("generated_values"),
                "text": r["rationale_text"],
            })
            n_kept += 1
        _flush(f"  {s}: {n_kept} rationales on arctic dilemmas")
    return rows


def load_cluster_mapping():
    df = pd.read_csv(ANALYSIS / "value_label_clusters.csv")
    return dict(zip(df["value"], df["cluster_id"])), dict(zip(df["cluster_id"], df["cluster_label"]))


# ─────────────────────────────────────────────────────────────────────────────
# Feature pipeline
# ─────────────────────────────────────────────────────────────────────────────

def compute_all(rows):
    _flush(f"computing lexical regex features for {len(rows)} comments")
    for r in rows:
        feats = lexical_features(r["text"])
        if feats is None:
            for k in ("hedge_count", "booster_count", "p1_count", "p2_count",
                      "anecdote_count", "cf_count", "causal_count", "concessive_count"):
                r[k] = 0
        else:
            r.update(feats)

    _flush("loading spaCy en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    nlp.max_length = 2_000_000
    _flush(f"  pipeline: {[p[0] for p in nlp.pipeline]}")

    texts = [r["text"][:5000] for r in rows]
    _flush(f"running spaCy pipe (n={len(texts)}, batch_size=128, n_process=4)")
    n_done = 0
    n_total = len(texts)
    try:
        n_proc = max(1, min(4, (os.cpu_count() or 4) - 1))
    except Exception:
        n_proc = 1
    _flush(f"  using n_process={n_proc}")
    spacy_rows = [None] * n_total
    docs_iter = nlp.pipe(texts, batch_size=128, n_process=n_proc)
    for i, doc in enumerate(docs_iter):
        sf = spacy_features(doc)
        spacy_rows[i] = sf
        n_done += 1
        if n_done % 5000 == 0:
            _flush(f"  spaCy progress: {n_done}/{n_total}")
    _flush("spaCy pipe done")

    for r, sf in zip(rows, spacy_rows):
        if sf is None:
            r["token_count"] = 0
        else:
            r.update(sf)
    return rows


DENSITY_FEATS = [
    ("hedge_density", "hedge_count"),
    ("booster_density", "booster_count"),
    ("p1_density", "p1_count"),
    ("p2_density", "p2_count"),
    ("anecdote_density", "anecdote_count"),
    ("cf_density", "cf_count"),
    ("causal_density", "causal_count"),
    ("concessive_density", "concessive_count"),
    ("noun_density", "noun_count"),
    ("verb_density", "verb_count"),
    ("adj_density", "adj_count"),
    ("adv_density", "adv_count"),
    ("pron_density", "pron_count"),
    ("ner_person_density", "ner_person_count"),
    ("ner_org_density", "ner_org_count"),
    ("ner_gpe_density", "ner_gpe_count"),
    ("ner_date_density", "ner_date_count"),
    ("ner_cardinal_density", "ner_cardinal_count"),
    ("ner_total_density", "ner_total_count"),
]
STANDALONE_FEATS = ["mean_word_len", "type_token_ratio", "mean_sent_len"]


def build_dataframe(rows):
    df = pd.DataFrame(rows)
    # densities (per token, multiplied ×100 → "per 100 tokens")
    tc = df["token_count"].astype(float).clip(lower=1)
    for dens_col, count_col in DENSITY_FEATS:
        df[dens_col] = 100.0 * df[count_col].astype(float) / tc
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Statistical comparison
# ─────────────────────────────────────────────────────────────────────────────

def summary_by_source(df, feature_cols):
    out = []
    for src, g in df.groupby("source"):
        rec = {"source": src, "n": len(g)}
        for f in feature_cols:
            rec[f"{f}_mean"] = float(g[f].mean())
            rec[f"{f}_median"] = float(g[f].median())
            rec[f"{f}_std"] = float(g[f].std())
        out.append(rec)
    return pd.DataFrame(out)


def within_cluster_pairs(df_h, df_l, cluster_map, sources_to_keep=None):
    """For each (submission_id, cluster) cell present for both human + at least one LLM,
    compute per-source within-cluster feature means.
    Returns long-format DataFrame: submission_id, cluster_id, source, n, feature means.
    """
    if sources_to_keep is None:
        sources_to_keep = LLM_SOURCES
    def attach_cluster(d):
        d = d.copy()
        d["cluster_id"] = d["value_label"].map(cluster_map)
        return d.dropna(subset=["cluster_id"])
    df_hc = attach_cluster(df_h)
    df_lc = attach_cluster(df_l)
    df_hc["cluster_id"] = df_hc["cluster_id"].astype(int)
    df_lc["cluster_id"] = df_lc["cluster_id"].astype(int)

    out = []
    feats = [f for f, _ in DENSITY_FEATS] + STANDALONE_FEATS + ["token_count"]
    # Iterate cells with both sides present
    cells_h = df_hc.groupby(["submission_id", "cluster_id"])
    h_keys = set(cells_h.groups.keys())
    cells_l = df_lc[df_lc["source"].isin(sources_to_keep)].groupby(["submission_id", "cluster_id"])
    l_keys = set(cells_l.groups.keys())
    shared = h_keys & l_keys
    for sid, cid in shared:
        h_grp = cells_h.get_group((sid, cid))
        l_grp = cells_l.get_group((sid, cid))
        rec_h = {"submission_id": sid, "cluster_id": cid, "source": "human", "n": len(h_grp)}
        rec_l = {"submission_id": sid, "cluster_id": cid, "source": "llm_pooled", "n": len(l_grp)}
        for f in feats:
            rec_h[f] = float(h_grp[f].mean())
            rec_l[f] = float(l_grp[f].mean())
        out.append(rec_h); out.append(rec_l)
    return pd.DataFrame(out)


def paired_diff_stats(df_paired, feature_cols):
    """For each feature, paired (human - llm) difference per (submission_id, cluster_id) cell."""
    h = df_paired[df_paired["source"] == "human"].set_index(["submission_id", "cluster_id"])
    l = df_paired[df_paired["source"] == "llm_pooled"].set_index(["submission_id", "cluster_id"])
    common = h.index.intersection(l.index)
    out = []
    for f in feature_cols:
        diff = (h.loc[common, f] - l.loc[common, f]).values
        mean = float(np.nanmean(diff))
        # Bootstrap CI
        boots = []
        for _ in range(N_BOOT):
            samp = RNG.choice(diff, size=len(diff), replace=True)
            boots.append(float(np.nanmean(samp)))
        boots = np.array(boots)
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        # Permutation null: flip signs (paired)
        perms = []
        for _ in range(N_PERM):
            signs = RNG.choice([-1, 1], size=len(diff))
            perms.append(float(np.nanmean(diff * signs)))
        perms = np.array(perms)
        p = float(np.mean(np.abs(perms) >= abs(mean)))
        # Effect size: standardized
        sd = float(np.nanstd(diff, ddof=1)) if len(diff) > 1 else float("nan")
        cohen_d = mean / sd if sd > 0 else float("nan")
        out.append({
            "feature": f, "n_cells": int(len(common)),
            "h_minus_l_mean": mean,
            "h_minus_l_ci_lo": lo, "h_minus_l_ci_hi": hi,
            "perm_pvalue": p, "cohen_d": cohen_d,
        })
    return pd.DataFrame(out)


def length_residualised_diff(df_h, df_l, feature_cols):
    """Residualize each density feature on log(token_count) globally, then recompute paired diff."""
    df_h = df_h.copy(); df_l = df_l.copy()
    df_h["log_tok"] = np.log1p(df_h["token_count"])
    df_l["log_tok"] = np.log1p(df_l["token_count"])

    out = []
    for f in feature_cols:
        # Fit on combined: simple linear regression of f on log_tok
        x = pd.concat([df_h["log_tok"], df_l["log_tok"]]).values
        y = pd.concat([df_h[f], df_l[f]]).values
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]
        if len(x) < 5:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        rh = df_h[f].values - (slope * df_h["log_tok"].values + intercept)
        rl = df_l[f].values - (slope * df_l["log_tok"].values + intercept)
        raw_diff = float(df_h[f].mean() - df_l[f].mean())
        resid_diff = float(np.nanmean(rh) - np.nanmean(rl))
        out.append({
            "feature": f,
            "raw_diff": raw_diff,
            "resid_diff": resid_diff,
            "diff_attenuation": (resid_diff / raw_diff) if raw_diff != 0 else float("nan"),
            "slope_logtok": slope, "intercept": intercept,
        })
    return pd.DataFrame(out)


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    # Load corpora
    human_rows = load_human_corpus()
    arctic_subs = {r["submission_id"] for r in human_rows}
    llm_rows = load_llm_corpus(arctic_subs)

    cluster_map, cluster_labels = load_cluster_mapping()

    # Compute features
    all_rows = human_rows + llm_rows
    all_rows = compute_all(all_rows)

    # Save raw features
    df = build_dataframe(all_rows)
    feature_cols = [f for f, _ in DENSITY_FEATS] + STANDALONE_FEATS + ["token_count"]
    keep_cols = ["source", "submission_id", "comment_id", "embed_idx", "value_label", "token_count"] + \
                [f for f, _ in DENSITY_FEATS] + STANDALONE_FEATS
    df_out = df[keep_cols]
    csv_path = ANALYSIS / "milestone3_lexical_features.csv"
    df_out.to_csv(csv_path, index=False)
    _flush(f"saved per-comment features: {csv_path}  rows={len(df_out)}")

    # Summary per source
    summary = summary_by_source(df, feature_cols)
    sum_path = ANALYSIS / "milestone3_lexical_summary.csv"
    summary.to_csv(sum_path, index=False)
    _flush(f"saved per-source summary: {sum_path}")

    # Within-cluster paired analysis
    df_h = df[df["source"] == "human"].copy()
    df_l = df[df["source"] != "human"].copy()
    paired = within_cluster_pairs(df_h, df_l, cluster_map)
    paired_path = ANALYSIS / "milestone3_lexical_within_cluster.csv"
    paired.to_csv(paired_path, index=False)
    _flush(f"saved within-cluster paired: {paired_path}  cells={len(paired)//2}")

    # Paired diff statistics
    diff_stats = paired_diff_stats(paired, feature_cols)
    diff_path = ANALYSIS / "milestone3_lexical_pairdiff.csv"
    diff_stats.to_csv(diff_path, index=False)
    _flush(f"saved paired diff stats: {diff_path}")

    # Length-residualised diff
    resid = length_residualised_diff(df_h, df_l, [f for f, _ in DENSITY_FEATS] + STANDALONE_FEATS)
    resid_path = ANALYSIS / "milestone3_lexical_lengthctrl.csv"
    resid.to_csv(resid_path, index=False)
    _flush(f"saved length-controlled diff: {resid_path}")

    # Bootstrap+perm summary as JSON for paper
    boot_out = {
        "n_human_comments": int((df["source"] == "human").sum()),
        "n_llm_comments": int((df["source"] != "human").sum()),
        "n_cells_paired": int(len(paired) // 2),
        "diff_stats": diff_stats.to_dict(orient="records"),
        "length_controlled": resid.to_dict(orient="records"),
    }
    (ANALYSIS / "milestone3_lexical_bootstrap.json").write_text(json.dumps(boot_out, indent=2))
    _flush("saved bootstrap json")


if __name__ == "__main__":
    main()
