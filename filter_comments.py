"""Classify r/AITA comments for moral verdicts using a local LLM via OpenAI-compatible API."""

import argparse
import csv
import json
import logging
import os
import tempfile
import time

from openai import OpenAI
from dotenv import load_dotenv

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL = "qwen2.5:14b"  # Change to match your ollama model name
INPUT_CSV = "data/comments.csv"
OUTPUT_CSV = "data/filtered_comments.csv"
CHECKPOINT_FILE = "data/filter_checkpoint.json"
LOG_FILE = "data/filter.log"
OUTPUT_FIELDS = ["submission_id", "comment_id", "comment_body", "comment_score", "verdict"]
MAX_BODY_CHARS = 2000
DEFAULT_DELAY = 0.0
DEFAULT_BATCH_SIZE = 500
MAX_RETRIES = 3
RETRY_DELAYS = [2, 10, 30]

SYSTEM_PROMPT = (
    "You are a classifier for r/AmITheAsshole comments. "
    "Respond with ONLY a JSON object, no other text."
)

USER_PROMPT_TEMPLATE = """\
Classify this r/AmITheAsshole comment. Does it contain a clear moral verdict (NTA, YTA, ESH, or NAH) AND a rationale explaining the judgment?

The verdict may be explicit (using tags like "NTA") or implicit (e.g., "you're clearly in the wrong here" = YTA).

Comment: {comment_body}

Respond with ONLY a JSON object, no other text:
{{"has_verdict": true/false, "verdict": "NTA"/"YTA"/"ESH"/"NAH"/null}}"""


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter AITA comments for verdicts using LLM classification.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between LLM requests (default: 0.0 for local)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Comments per checkpoint save (default: 500)")
    return parser.parse_args()


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("filter_comments")
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


# ── LLM Client ────────────────────────────────────────────────────────────────
def init_llm_client() -> OpenAI:
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")

    if not base_url:
        raise SystemExit(
            "Missing LLM_BASE_URL in .env. "
            "Set to http://localhost:11434/v1 for ollama."
        )
    if not api_key:
        raise SystemExit("Missing LLM_API_KEY in .env.")

    return OpenAI(base_url=base_url, api_key=api_key)


def classify_comment(client: OpenAI, comment_body: str, logger: logging.Logger) -> dict | None:
    """Classify a comment. Returns {"has_verdict": bool, "verdict": str|None} or None on parse failure."""
    truncated = comment_body[:MAX_BODY_CHARS]
    prompt = USER_PROMPT_TEMPLATE.format(comment_body=truncated)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed. Raw response: %s", raw[:500])
        return None

    if "has_verdict" not in result:
        logger.warning("Missing 'has_verdict' key. Raw response: %s", raw[:500])
        return None

    return result


# ── CSV ────────────────────────────────────────────────────────────────────────
def flush_to_csv(rows: list[dict]) -> None:
    if not rows:
        return
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    load_dotenv()
    os.makedirs("data", exist_ok=True)
    logger = setup_logging()

    client = init_llm_client()
    logger.info("LLM client initialized (model: %s)", MODEL)

    completed = load_checkpoint()
    logger.info("Checkpoint: %d comments already processed", len(completed))

    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"Input file not found: {INPUT_CSV}")

    total_processed = len(completed)
    total_with_verdict = 0
    errors = 0
    batch_buffer: list[dict] = []
    batch_completed: set[str] = set()

    try:
        with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                comment_id = row["comment_id"]
                if comment_id in completed:
                    continue

                # Classify with retry
                result = None
                for attempt in range(MAX_RETRIES):
                    try:
                        result = classify_comment(client, row["comment_body"], logger)
                        break
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:
                            delay = RETRY_DELAYS[attempt]
                            logger.warning(
                                "Attempt %d failed for %s: %s. Retrying in %ds...",
                                attempt + 1, comment_id, e, delay,
                            )
                            time.sleep(delay)
                        else:
                            logger.error(
                                "All %d attempts failed for %s: %s. Skipping.",
                                MAX_RETRIES, comment_id, e,
                            )
                            errors += 1

                # Buffer accepted rows
                if result and result.get("has_verdict"):
                    verdict = result.get("verdict")
                    if verdict in ("NTA", "YTA", "ESH", "NAH"):
                        batch_buffer.append({
                            "submission_id": row["submission_id"],
                            "comment_id": comment_id,
                            "comment_body": row["comment_body"],
                            "comment_score": row["comment_score"],
                            "verdict": verdict,
                        })
                        total_with_verdict += 1

                batch_completed.add(comment_id)
                total_processed += 1

                # Flush at batch boundary
                if len(batch_completed) >= args.batch_size:
                    flush_to_csv(batch_buffer)
                    completed.update(batch_completed)
                    save_checkpoint(completed)
                    batch_buffer.clear()
                    batch_completed.clear()

                if total_processed % 1000 == 0:
                    logger.info(
                        "Progress: %d processed | %d with verdict | %d errors",
                        total_processed, total_with_verdict, errors,
                    )

                if args.delay > 0:
                    time.sleep(args.delay)

        # Flush remaining
        flush_to_csv(batch_buffer)
        completed.update(batch_completed)
        save_checkpoint(completed)

    except KeyboardInterrupt:
        logger.info("Interrupted — flushing buffer and saving checkpoint...")
        flush_to_csv(batch_buffer)
        completed.update(batch_completed)
        save_checkpoint(completed)
        logger.info("Progress saved. %d processed, %d with verdict.", total_processed, total_with_verdict)
        return

    logger.info(
        "Done. %d processed, %d with verdict, %d errors.",
        total_processed, total_with_verdict, errors,
    )


if __name__ == "__main__":
    main()
