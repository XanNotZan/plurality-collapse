"""Gather AITA comments via ArcticShift API for all 10,826 Sachdeva submission_ids.

Filter to top-level + non-deleted + non-automod + verdict-bearing + length ≥ 10 words.
Save raw + filtered jsonl. Resumable: skips submission_ids with existing entries.
"""

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

EMBEDDINGS_DIR = Path("data/embeddings")
GEN_DIR = Path("data/generated")
OUT_DIR = Path("data/arcticshift")
OUT_DIR.mkdir(parents=True, exist_ok=True)

API = "https://arctic-shift.photon-reddit.com/api/comments/search"
MAX_LIMIT = 100
MAX_PAGES = 5            # cap fetches per post
MIN_WORDS = 10
VERDICT_RE = re.compile(r"\b(nta|yta|esh|nah|info|y\s*t\s*a|n\s*t\s*a|asshole|not\s+the\s+asshole|you'?re\s+the\s+asshole|everyone\s+sucks|no\s+(?:assholes|aholes)\s+here)\b", re.IGNORECASE)


logger = logging.getLogger("arcticshift")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg)
    sys.stdout.flush()


def is_quality_comment(c, sub_id):
    if c.get("parent_id") != f"t3_{sub_id}":
        return False
    body = c.get("body") or ""
    if body in ("[deleted]", "[removed]", "", None):
        return False
    if c.get("author") in ("AutoModerator", "[deleted]"):
        return False
    if len(body.split()) < MIN_WORDS:
        return False
    if not VERDICT_RE.search(body):
        return False
    return True


def fetch_one(sub_id, session, max_pages=MAX_PAGES):
    """Fetch comments for one submission, paginate via after. Return raw + filtered."""
    raw_kept = []
    filtered = []
    after = None
    for page in range(max_pages):
        params = {"link_id": f"t3_{sub_id}", "limit": MAX_LIMIT}
        if after is not None:
            params["after"] = after
        try:
            r = session.get(API, params=params, timeout=15)
            if r.status_code != 200:
                break
            data = r.json().get("data") or []
        except Exception:
            break
        if not data:
            break
        for c in data:
            if is_quality_comment(c, sub_id):
                # Keep slim representation
                filtered.append({
                    "submission_id": sub_id,
                    "comment_id": c.get("id"),
                    "author": c.get("author"),
                    "score": c.get("score"),
                    "created_utc": c.get("created_utc"),
                    "body": c.get("body"),
                    "permalink": c.get("permalink"),
                })
        # Pagination cursor: max created_utc in this batch
        max_t = max(c.get("created_utc", 0) for c in data)
        if after is not None and max_t <= after:
            break
        after = max_t
        if len(data) < MAX_LIMIT:
            break
    return filtered


def load_submission_ids():
    """Load all unique submission IDs from human meta."""
    meta = json.loads((EMBEDDINGS_DIR / "human_meta.json").read_text())
    seen = []
    seen_set = set()
    for m in meta:
        s = m["submission_id"]
        if s not in seen_set:
            seen.append(s)
            seen_set.add(s)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap submissions for testing (0=all)")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    sub_ids = load_submission_ids()
    if args.limit:
        sub_ids = sub_ids[:args.limit]
    _flush(f"submission ids: {len(sub_ids)}")

    out_path = OUT_DIR / "filtered_comments.jsonl"
    seen_ids = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                seen_ids.add(json.loads(line)["submission_id"])
            except Exception:
                pass
    todo = [s for s in sub_ids if s not in seen_ids]
    _flush(f"already done: {len(seen_ids)}, todo: {len(todo)}")

    if not todo:
        _flush("nothing to do")
        return

    session = requests.Session()
    fout = open(out_path, "a")
    t0 = time.time()
    n_done = 0
    n_with_quality = 0
    total_quality = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_one, s, session): s for s in todo}
        for fut in as_completed(futures):
            sub_id = futures[fut]
            try:
                filtered = fut.result()
            except Exception:
                filtered = []
            # Write a marker line per submission, even if empty (for resumability)
            for f in filtered:
                fout.write(json.dumps(f) + "\n")
            if filtered:
                n_with_quality += 1
                total_quality += len(filtered)
            else:
                # empty marker
                fout.write(json.dumps({"submission_id": sub_id, "_empty": True}) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 50 == 0:
                rate = n_done / (time.time() - t0)
                eta = (len(todo) - n_done) / rate
                _flush(f"{n_done}/{len(todo)} rate={rate:.1f}/s eta={eta:.0f}s "
                       f"with_quality={n_with_quality} total_quality_comments={total_quality}")
    fout.close()
    _flush(f"DONE: {n_done} submissions, {n_with_quality} with ≥1 quality comment, {total_quality} total")


if __name__ == "__main__":
    main()
