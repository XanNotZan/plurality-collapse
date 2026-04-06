"""Compare Kaleido-extracted moral value frequencies between human and LLM rationales."""

import csv
import json
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_NAME = "ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas"
DATASET_SPLIT = "test"
EMBEDDINGS_DIR = "data/embeddings"
OUTPUT_DIR = "data/analysis"
MODEL_NAME = "allenai/kaleido-xl"
FALLBACK_TEMPLATE = "[Generate]:\tAction: ACTION"
GEN_BATCH_SIZE = 16
GEN_MAX_NEW_TOKENS = 128
CHECKPOINT_INTERVAL = 1000

HUMAN_VALUES_PATH = "data/analysis/human_values_all.json"
OVERLAP_CSV_PATH = "data/analysis/value_overlap_analysis.csv"

LLM_SOURCE_ORDER = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]

ENRICHMENT_THRESHOLD = 3.0

SOURCE_COLORS = {
    "human":   "#2ca02c",
    "gpt3.5":  "#1f77b4",
    "gpt4":    "#aec7e8",
    "claude":  "#ff7f0e",
    "bison":   "#ffbb78",
    "gemma":   "#d62728",
    "mistral": "#9467bd",
    "llama":   "#8c564b",
    "all_llm": "#e377c2",
}


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("compare_value_frequencies")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger


# ── Data Loading ──────────────────────────────────────────────────────────────
def build_text_lookup(logger: logging.Logger) -> dict[tuple[str, str], str]:
    """Build (submission_id, column) -> text lookup from HF dataset."""
    logger.info("Loading dataset: %s", DATASET_NAME)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    logger.info("Dataset loaded: %d rows", len(ds))

    lookup: dict[tuple[str, str], str] = {}
    for row in ds:
        sid = row["submission_id"]
        for col in row:
            val = row[col]
            if isinstance(val, str) and val.strip():
                lookup[(sid, col)] = val.strip()

    logger.info("Text lookup built: %d entries", len(lookup))
    return lookup


def load_human_values(logger: logging.Logger) -> list[dict]:
    """Load pre-generated human values from disk."""
    if not os.path.exists(HUMAN_VALUES_PATH):
        raise SystemExit(f"{HUMAN_VALUES_PATH} not found. Run value_diversity_gradient.py first.")
    with open(HUMAN_VALUES_PATH, encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded human values: %d records", len(records))
    return records


def load_human_only_values(logger: logging.Logger) -> list[str]:
    """Load the 13 human-only values from value_overlap_analysis.csv."""
    if not os.path.exists(OVERLAP_CSV_PATH):
        logger.warning("%s not found, skipping previously-identified values", OVERLAP_CSV_PATH)
        return []
    with open(OVERLAP_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("human_only_values", "")
            return [v.strip() for v in raw.split(";") if v.strip()]
    return []


# ── Kaleido ──────────────────────────────────────────────────────────────────
def load_kaleido(logger: logging.Logger):
    """Load Kaleido-XL model and tokenizer."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This script requires a GPU.")

    device = torch.device("cuda")
    logger.info("Loading model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.to(device).eval()
    logger.info("Model loaded (fp16) on %s", torch.cuda.get_device_name(0))

    try:
        template = model.config.task_specific_params["generate"]["template"]
        logger.info("Template: %r", template)
    except (AttributeError, KeyError, TypeError):
        template = FALLBACK_TEMPLATE
        logger.warning("Using fallback template: %r", template)

    return model, tokenizer, template, device


def generate_values_batch(
    model, tokenizer, texts: list[str], template: str, device: torch.device,
) -> list[str]:
    """Generate Kaleido value expressions for a batch of rationale texts."""
    formatted = [template.replace("ACTION", t) for t in texts]
    inputs = tokenizer(
        formatted, return_tensors="pt", padding=True, truncation=True, max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=GEN_MAX_NEW_TOKENS,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cleaned = []
    for d in decoded:
        d = d.strip()
        if d.lower().startswith("value:"):
            d = d[len("value:"):].strip()
        cleaned.append(d)
    return cleaned


# ── Per-Source Generation with Checkpointing ─────────────────────────────────
def _checkpoint_path(source: str) -> str:
    return f"data/llm_value_generation_checkpoint_{source}.json"


def _output_path(source: str) -> str:
    return os.path.join(OUTPUT_DIR, f"llm_values_{source}.json")


def _save_checkpoint(results: list[str], source: str, logger: logging.Logger) -> None:
    entries = [{"index": i, "generated_values": v} for i, v in enumerate(results) if v]
    path = _checkpoint_path(source)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)
    logger.info("[%s] Checkpoint saved: %d/%d to %s",
                source, len(entries), len(results), path)


def generate_source_values(
    source: str,
    meta: list[dict],
    text_lookup: dict[tuple[str, str], str],
    model, tokenizer, template: str, device: torch.device,
    logger: logging.Logger,
) -> list[dict]:
    """Generate Kaleido values for all rationales of a single LLM source.

    Returns list of records with generated values.
    """
    # Check if already completed
    out_path = _output_path(source)
    if os.path.exists(out_path):
        logger.info("[%s] Loading existing values from %s", source, out_path)
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)

    n = len(meta)
    texts = [text_lookup.get((m["submission_id"], m["column"]), "") for m in meta]
    results = [""] * n

    # Load checkpoint if exists
    ckpt_path = _checkpoint_path(source)
    completed = set()
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                ckpt_data = json.load(f)
            for entry in ckpt_data:
                idx = entry["index"]
                results[idx] = entry["generated_values"]
                completed.add(idx)
            logger.info("[%s] Resumed from checkpoint: %d/%d completed",
                        source, len(completed), n)
        except (json.JSONDecodeError, KeyError):
            logger.warning("[%s] Checkpoint corrupted, starting fresh", source)
            completed = set()
            results = [""] * n

    remaining = [i for i in range(n) if i not in completed]
    if not remaining:
        logger.info("[%s] All %d rationales already generated", source, n)
    else:
        logger.info("[%s] Generating values for %d remaining rationales...", source, len(remaining))
        n_generated = 0

        try:
            for batch_start in range(0, len(remaining), GEN_BATCH_SIZE):
                batch_indices = remaining[batch_start:batch_start + GEN_BATCH_SIZE]
                batch_texts = [texts[i] for i in batch_indices]
                batch_values = generate_values_batch(model, tokenizer, batch_texts, template, device)

                for idx, val in zip(batch_indices, batch_values):
                    results[idx] = val
                n_generated += len(batch_indices)

                total_done = len(completed) + n_generated
                if total_done % 1000 < GEN_BATCH_SIZE or batch_start + GEN_BATCH_SIZE >= len(remaining):
                    logger.info("[%s] Progress: %d/%d", source, total_done, n)

                if n_generated % CHECKPOINT_INTERVAL < GEN_BATCH_SIZE and n_generated >= CHECKPOINT_INTERVAL:
                    _save_checkpoint(results, source, logger)

        except KeyboardInterrupt:
            logger.warning("[%s] Interrupted! Saving checkpoint...", source)
            _save_checkpoint(results, source, logger)
            raise

    # Build records
    records = []
    for i, (m, t, v) in enumerate(zip(meta, texts, results)):
        records.append({
            "index": i,
            "submission_id": m["submission_id"],
            "column": m["column"],
            "source_model": source,
            "rationale_text": t,
            "generated_values": v,
        })

    # Save and clean up
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    logger.info("[%s] Saved %s (%d records)", source, out_path, len(records))

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return records


# ── Frequency Analysis ───────────────────────────────────────────────────────
def build_freq_dist(records: list[dict]) -> dict[str, float]:
    """Build relative frequency distribution from records."""
    counts: dict[str, int] = {}
    total = 0
    for r in records:
        v = r.get("generated_values", "")
        if v:
            counts[v] = counts.get(v, 0) + 1
            total += 1
    return {v: c / total for v, c in counts.items()} if total > 0 else {}


def build_count_dist(records: list[dict]) -> dict[str, int]:
    """Build absolute count distribution from records."""
    counts: dict[str, int] = {}
    for r in records:
        v = r.get("generated_values", "")
        if v:
            counts[v] = counts.get(v, 0) + 1
    return counts


# ── Outputs ──────────────────────────────────────────────────────────────────
def save_frequency_comparison(
    human_freq: dict[str, float],
    allllm_freq: dict[str, float],
    per_model_freq: dict[str, dict[str, float]],
    output_dir: str, logger: logging.Logger,
) -> None:
    """Save value_frequency_comparison.csv."""
    all_values = sorted(set(human_freq) | set(allllm_freq))

    # Rank values
    human_ranked = sorted(human_freq.items(), key=lambda x: -x[1])
    allllm_ranked = sorted(allllm_freq.items(), key=lambda x: -x[1])
    human_rank = {v: i + 1 for i, (v, _) in enumerate(human_ranked)}
    allllm_rank = {v: i + 1 for i, (v, _) in enumerate(allllm_ranked)}

    path = os.path.join(output_dir, "value_frequency_comparison.csv")
    fieldnames = ["value", "human_freq", "human_rank", "allllm_freq", "allllm_rank", "freq_ratio"]
    for source in LLM_SOURCE_ORDER:
        fieldnames.append(f"{source}_freq")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in all_values:
            hf = human_freq.get(v, 0.0)
            af = allllm_freq.get(v, 0.0)
            ratio = hf / af if af > 0 else (float("inf") if hf > 0 else 0.0)
            row = {
                "value": v,
                "human_freq": f"{hf:.6f}",
                "human_rank": human_rank.get(v, ""),
                "allllm_freq": f"{af:.6f}",
                "allllm_rank": allllm_rank.get(v, ""),
                "freq_ratio": f"{ratio:.4f}" if ratio != float("inf") else "inf",
            }
            for source in LLM_SOURCE_ORDER:
                sf = per_model_freq.get(source, {}).get(v, 0.0)
                row[f"{source}_freq"] = f"{sf:.6f}"
            writer.writerow(row)
    logger.info("Saved %s (%d values)", path, len(all_values))


def save_enriched_csv(
    human_freq: dict[str, float],
    allllm_freq: dict[str, float],
    direction: str,
    output_dir: str, logger: logging.Logger,
) -> None:
    """Save human_enriched_values.csv or llm_enriched_values.csv."""
    rows = []
    for v in set(human_freq) | set(allllm_freq):
        hf = human_freq.get(v, 0.0)
        af = allllm_freq.get(v, 0.0)
        if direction == "human":
            if af > 0 and hf / af >= ENRICHMENT_THRESHOLD:
                rows.append({"value": v, "human_freq": hf, "allllm_freq": af,
                             "ratio": hf / af})
            elif af == 0 and hf > 0:
                rows.append({"value": v, "human_freq": hf, "allllm_freq": 0.0,
                             "ratio": float("inf")})
        else:
            if hf > 0 and af / hf >= ENRICHMENT_THRESHOLD:
                rows.append({"value": v, "human_freq": hf, "allllm_freq": af,
                             "ratio": af / hf})
            elif hf == 0 and af > 0:
                rows.append({"value": v, "human_freq": hf, "allllm_freq": af,
                             "ratio": float("inf")})

    rows.sort(key=lambda r: r["ratio"], reverse=True)

    filename = f"{direction}_enriched_values.csv"
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["value", "human_freq", "allllm_freq", "ratio"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "value": r["value"],
                "human_freq": f"{r['human_freq']:.6f}",
                "allllm_freq": f"{r['allllm_freq']:.6f}",
                "ratio": f"{r['ratio']:.4f}" if r["ratio"] != float("inf") else "inf",
            })
    logger.info("Saved %s (%d values)", path, len(rows))


def save_previously_identified(
    human_only_values: list[str],
    human_counts: dict[str, int],
    human_total: int,
    per_model_counts: dict[str, dict[str, int]],
    per_model_totals: dict[str, int],
    allllm_counts: dict[str, int],
    allllm_total: int,
    output_dir: str, logger: logging.Logger,
) -> None:
    """Save previously_identified_values.csv."""
    path = os.path.join(output_dir, "previously_identified_values.csv")
    fieldnames = ["value", "human_count", "human_freq"]
    for source in LLM_SOURCE_ORDER:
        fieldnames.extend([f"{source}_count", f"{source}_freq"])
    fieldnames.extend(["allllm_count", "allllm_freq"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in human_only_values:
            hc = human_counts.get(v, 0)
            ac = allllm_counts.get(v, 0)
            row = {
                "value": v,
                "human_count": hc,
                "human_freq": f"{hc / human_total:.6f}" if human_total > 0 else "0",
            }
            for source in LLM_SOURCE_ORDER:
                sc = per_model_counts.get(source, {}).get(v, 0)
                st = per_model_totals.get(source, 1)
                row[f"{source}_count"] = sc
                row[f"{source}_freq"] = f"{sc / st:.6f}" if st > 0 else "0"
            row["allllm_count"] = ac
            row["allllm_freq"] = f"{ac / allllm_total:.6f}" if allllm_total > 0 else "0"
            writer.writerow(row)
    logger.info("Saved %s (%d values)", path, len(human_only_values))


def plot_frequency_scatter(
    human_freq: dict[str, float],
    allllm_freq: dict[str, float],
    output_dir: str, logger: logging.Logger,
) -> None:
    """Scatter plot of human vs all_llm value frequencies."""
    all_values = sorted(set(human_freq) | set(allllm_freq))
    hf = np.array([human_freq.get(v, 0.0) for v in all_values])
    af = np.array([allllm_freq.get(v, 0.0) for v in all_values])

    # Spearman on values present in both
    both_mask = (hf > 0) & (af > 0)
    if HAS_SCIPY and both_mask.sum() >= 3:
        rho, p = spearmanr(hf[both_mask], af[both_mask])
    else:
        rho = float(np.corrcoef(hf[both_mask], af[both_mask])[0, 1]) if both_mask.sum() >= 3 else 0.0
        p = float("nan")

    logger.info("Spearman rho=%.3f, p=%s (n=%d values in both)",
                rho, f"{p:.4f}" if not np.isnan(p) else "N/A", both_mask.sum())

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(hf, af, color=SOURCE_COLORS["human"], alpha=0.5, s=30, edgecolors="black",
               linewidth=0.3, zorder=3)

    # Diagonal line
    max_val = max(hf.max(), af.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Label top 10 most divergent values (by absolute freq difference)
    divergence = np.abs(hf - af)
    top_idx = np.argsort(divergence)[-10:][::-1]
    for i in top_idx:
        v = all_values[i]
        ax.annotate(v, (hf[i], af[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.9)

    ax.set_xlabel("Human Frequency")
    ax.set_ylabel("All-LLM Frequency")
    p_str = f", p={p:.4f}" if not np.isnan(p) else ""
    ax.set_title(f"Value Frequency: Human vs All-LLM\n"
                 f"(Spearman \u03c1={rho:.3f}{p_str}, n={both_mask.sum()} shared values)")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    fig.tight_layout()
    path = os.path.join(output_dir, "value_frequency_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    logger = setup_logging()
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load human values
    logger.info("=== Step 1: Load human values ===")
    human_records = load_human_values(logger)
    human_freq = build_freq_dist(human_records)
    human_counts = build_count_dist(human_records)
    human_total = sum(human_counts.values())
    logger.info("Human: %d unique values, %d total", len(human_freq), human_total)

    # Step 2: Load text lookup
    logger.info("=== Step 2: Build text lookup ===")
    text_lookup = build_text_lookup(logger)

    # Step 3: Generate LLM values
    logger.info("=== Step 3: Generate LLM values ===")
    model, tokenizer, template, device = load_kaleido(logger)

    all_llm_records: list[dict] = []
    per_model_records: dict[str, list[dict]] = {}

    for source in LLM_SOURCE_ORDER:
        meta_path = os.path.join(EMBEDDINGS_DIR, f"{source}_meta.json")
        if not os.path.exists(meta_path):
            logger.warning("Missing metadata: %s, skipping", meta_path)
            continue

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        logger.info("[%s] Loaded %d metadata entries", source, len(meta))

        records = generate_source_values(
            source, meta, text_lookup,
            model, tokenizer, template, device, logger)

        per_model_records[source] = records
        all_llm_records.extend(records)

    # Free GPU
    del model, tokenizer
    torch.cuda.empty_cache()
    logger.info("GPU freed. Total LLM records: %d", len(all_llm_records))

    # Step 4: Build frequency distributions
    logger.info("=== Step 4: Frequency analysis ===")
    allllm_freq = build_freq_dist(all_llm_records)
    allllm_counts = build_count_dist(all_llm_records)
    allllm_total = sum(allllm_counts.values())

    per_model_freq: dict[str, dict[str, float]] = {}
    per_model_counts: dict[str, dict[str, int]] = {}
    per_model_totals: dict[str, int] = {}
    for source, records in per_model_records.items():
        per_model_freq[source] = build_freq_dist(records)
        per_model_counts[source] = build_count_dist(records)
        per_model_totals[source] = sum(per_model_counts[source].values())
        logger.info("[%s] %d unique values, %d total",
                    source, len(per_model_freq[source]), per_model_totals[source])

    logger.info("all_llm: %d unique values, %d total", len(allllm_freq), allllm_total)

    # Step 5: Save outputs
    logger.info("=== Step 5: Save outputs ===")
    save_frequency_comparison(human_freq, allllm_freq, per_model_freq, OUTPUT_DIR, logger)
    save_enriched_csv(human_freq, allllm_freq, "human", OUTPUT_DIR, logger)
    save_enriched_csv(human_freq, allllm_freq, "llm", OUTPUT_DIR, logger)

    # Previously identified human-only values
    human_only_values = load_human_only_values(logger)
    if human_only_values:
        save_previously_identified(
            human_only_values, human_counts, human_total,
            per_model_counts, per_model_totals,
            allllm_counts, allllm_total, OUTPUT_DIR, logger)

    plot_frequency_scatter(human_freq, allllm_freq, OUTPUT_DIR, logger)

    elapsed = time.time() - start_time
    logger.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
