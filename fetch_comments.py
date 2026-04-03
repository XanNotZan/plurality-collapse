"""Fetch top-level Reddit comments for r/AITA posts in the plurality-collapse dataset."""

import csv
import json
import logging
import os
import tempfile
import time

import praw
import prawcore.exceptions
from datasets import load_dataset
from dotenv import load_dotenv

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_NAME = "ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas"
DATASET_SPLIT = "test"
OUTPUT_CSV = "data/comments.csv"
CHECKPOINT_FILE = "data/checkpoint.json"
LOG_FILE = "data/fetch.log"
CSV_FIELDS = ["submission_id", "comment_id", "comment_body", "comment_score"]
MAX_RETRIES = 3
RETRY_DELAYS = [5, 30, 120]
SKIP_BODIES = {"[deleted]", "[removed]"}


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("fetch_comments")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_submission_ids() -> list[str]:
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    return ds["submission_id"]


# ── Checkpointing ─────────────────────────────────────────────────────────────
def load_checkpoint() -> set[str]:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        data = json.load(f)
    return set(data.get("completed", []))


def save_checkpoint(completed: set[str]) -> None:
    fd, tmp_path = tempfile.mkstemp(dir="data", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"completed": list(completed)}, f)
        os.replace(tmp_path, CHECKPOINT_FILE)
    except BaseException:
        os.unlink(tmp_path)
        raise


# ── Reddit ─────────────────────────────────────────────────────────────────────
def init_reddit() -> praw.Reddit:
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    placeholders = {"", "your_client_id_here", "your_client_secret_here", "your_user_agent_here"}
    missing = []
    if not client_id or client_id in placeholders:
        missing.append("REDDIT_CLIENT_ID")
    if not client_secret or client_secret in placeholders:
        missing.append("REDDIT_CLIENT_SECRET")
    if not user_agent or user_agent in placeholders:
        missing.append("REDDIT_USER_AGENT")
    if missing:
        raise SystemExit(
            f"Missing or placeholder environment variables: {', '.join(missing)}. "
            "Update .env with your Reddit API credentials from "
            "https://www.reddit.com/prefs/apps"
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        ratelimit_seconds=300,
    )


def fetch_comments_for_submission(
    reddit: praw.Reddit, submission_id: str
) -> list[dict] | None:
    """Fetch top-level comments for a submission. Returns None if inaccessible."""
    submission = reddit.submission(id=submission_id)
    submission.comments.replace_more(limit=0)

    comments = []
    for comment in submission.comments:
        if comment.body in SKIP_BODIES:
            continue
        comments.append(
            {
                "submission_id": submission_id,
                "comment_id": comment.id,
                "comment_body": comment.body,
                "comment_score": comment.score,
            }
        )
    return comments


# ── CSV ────────────────────────────────────────────────────────────────────────
def append_to_csv(rows: list[dict]) -> None:
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    load_dotenv()
    os.makedirs("data", exist_ok=True)
    logger = setup_logging()

    logger.info("Loading submission IDs from HuggingFace dataset...")
    all_ids = load_submission_ids()
    logger.info("Loaded %d submission IDs", len(all_ids))

    completed = load_checkpoint()
    remaining = [sid for sid in all_ids if sid not in completed]
    logger.info(
        "Checkpoint: %d completed, %d remaining", len(completed), len(remaining)
    )

    reddit = init_reddit()
    logger.info("Reddit client initialized. Starting fetch...")

    total_comments = 0
    errors = 0

    try:
        for i, sid in enumerate(remaining, 1):
            comments = None
            should_checkpoint = True

            for attempt in range(MAX_RETRIES):
                try:
                    comments = fetch_comments_for_submission(reddit, sid)
                    break
                except (
                    prawcore.exceptions.NotFound,
                    prawcore.exceptions.Forbidden,
                ) as e:
                    logger.warning("Submission %s inaccessible: %s", sid, e)
                    comments = None
                    break
                except prawcore.exceptions.ResponseException as e:
                    if e.response is not None and e.response.status_code == 401:
                        raise SystemExit(
                            "Reddit API returned 401 Unauthorized. "
                            "Check your credentials in .env."
                        )
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_DELAYS[attempt]
                        logger.warning(
                            "Attempt %d failed for %s: %s. Retrying in %ds...",
                            attempt + 1,
                            sid,
                            e,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts failed for %s: %s. "
                            "Skipping (will retry next run).",
                            MAX_RETRIES,
                            sid,
                            e,
                        )
                        errors += 1
                        should_checkpoint = False
                except (
                    prawcore.exceptions.ServerError,
                    prawcore.exceptions.RequestException,
                ) as e:
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_DELAYS[attempt]
                        logger.warning(
                            "Attempt %d failed for %s: %s. Retrying in %ds...",
                            attempt + 1,
                            sid,
                            e,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts failed for %s: %s. "
                            "Skipping (will retry next run).",
                            MAX_RETRIES,
                            sid,
                            e,
                        )
                        errors += 1
                        should_checkpoint = False

            if should_checkpoint:
                if comments:
                    append_to_csv(comments)
                    total_comments += len(comments)
                completed.add(sid)
                save_checkpoint(completed)

            if i % 100 == 0:
                pct = (len(completed) / len(all_ids)) * 100
                logger.info(
                    "Progress: %d/%d (%.1f%%) | Comments: %d | Errors: %d",
                    len(completed),
                    len(all_ids),
                    pct,
                    total_comments,
                    errors,
                )

    except KeyboardInterrupt:
        logger.info("Interrupted — progress saved through last completed submission.")
        return

    logger.info(
        "Done. Processed %d submissions, collected %d comments, %d errors.",
        len(completed),
        total_comments,
        errors,
    )


if __name__ == "__main__":
    main()
