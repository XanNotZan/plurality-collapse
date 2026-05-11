"""Time-stratified per-dilemma diversity test.

For each dilemma, sort comments by created_utc.
Compare diversity of early vs late commenters.
Hypothesis: later comments may converge as community consensus forms.
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EMBEDDINGS_DIR = Path("data/embeddings")
ANALYSIS = Path("data/analysis")
ARCTIC_DIR = Path("data/arcticshift")

logger = logging.getLogger("time_strat")
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


def main():
    H_emb = np.load(EMBEDDINGS_DIR / "human_arctic.npy")
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

    # Build h_by_sub with timestamps
    h_by_sub = defaultdict(list)
    for m in arctic_meta:
        rec = arctic_recs.get((m["submission_id"], m.get("comment_id")))
        if rec:
            t = rec.get("created_utc", 0)
            h_by_sub[m["submission_id"]].append({"idx": m["index"], "t": t})

    qualified = sorted([s for s in h_by_sub if len(h_by_sub[s]) >= 10])
    _flush(f"qualified dilemmas (>=10 human comments): {len(qualified)}")

    # Sort by created_utc, split into halves
    early_within = []
    late_within = []
    for sid in qualified:
        items = sorted(h_by_sub[sid], key=lambda x: x["t"])
        n = len(items)
        half = n // 2
        early = items[:half]
        late = items[half:]
        if len(early) >= 5 and len(late) >= 5:
            E_early = H_emb[[e["idx"] for e in early]]
            E_late = H_emb[[e["idx"] for e in late]]
            early_within.append(cosine_pairwise_mean(E_early))
            late_within.append(cosine_pairwise_mean(E_late))
    early_within = np.array(early_within)
    late_within = np.array(late_within)
    _flush(f"early-half within-human cosine: {early_within.mean():.4f} ± {early_within.std():.4f}, n_dilemmas={len(early_within)}")
    _flush(f"late-half within-human cosine: {late_within.mean():.4f} ± {late_within.std():.4f}")
    _flush(f"  ratio (early/late): {early_within.mean() / late_within.mean():.3f}")
    _flush(f"  frac dilemmas early>late: {float(np.mean(early_within > late_within)):.4f}")

    # Per-post age: time since post creation. Need submission's created_utc.
    # ArcticShift doesn't give post time directly here. Approximate: first comment time per dilemma.
    # Then bin comments by relative time within dilemma:
    #   "first 25%", "25-50%", "50-75%", "75-100%" by comment-order index
    # We already did first half vs second half above. Let me do quartiles.

    q_results = {}
    for q_label, lo, hi in [("Q1 (earliest)", 0.0, 0.25),
                            ("Q2", 0.25, 0.5),
                            ("Q3", 0.5, 0.75),
                            ("Q4 (latest)", 0.75, 1.0)]:
        within_q = []
        for sid in qualified:
            items = sorted(h_by_sub[sid], key=lambda x: x["t"])
            n = len(items)
            i_lo = int(n * lo)
            i_hi = int(n * hi)
            sub = items[i_lo:i_hi]
            if len(sub) >= 5:
                E_q = H_emb[[e["idx"] for e in sub]]
                within_q.append(cosine_pairwise_mean(E_q))
        within_q = np.array(within_q)
        q_results[q_label] = {
            "n_dilemmas": int(len(within_q)),
            "mean_within": float(within_q.mean()),
            "std_within": float(within_q.std()),
        }
        _flush(f"{q_label}: n={len(within_q)}, within-human cosine={within_q.mean():.4f} ± {within_q.std():.4f}")

    out = {
        "n_dilemmas_qualified": int(len(qualified)),
        "first_half": {"mean_within": float(early_within.mean()), "std": float(early_within.std())},
        "second_half": {"mean_within": float(late_within.mean()), "std": float(late_within.std())},
        "first_half_more_diverse_frac": float(np.mean(early_within > late_within)),
        "ratio_first_half_over_second": float(early_within.mean() / late_within.mean()),
        "quartiles": q_results,
    }
    path = ANALYSIS / "milestone3_time_stratified.json"
    path.write_text(json.dumps(out, indent=2))
    _flush(f"saved {path}")


if __name__ == "__main__":
    main()
