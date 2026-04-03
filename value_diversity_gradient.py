"""Plot how moral value diversity changes continuously along the reconstruction error axis."""

import json
import logging
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_NAME = "ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas"
DATASET_SPLIT = "test"
EMBEDDINGS_DIR = "data/embeddings"
OUTPUT_DIR = "data/analysis"
HIDDEN_DIM = 2048
MODEL_NAME = "allenai/kaleido-xl"
FALLBACK_TEMPLATE = "[Generate]:\tAction: ACTION"
GEN_BATCH_SIZE = 16
GEN_MAX_NEW_TOKENS = 128

ALL_SOURCES = ["human", "gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
LLM_SOURCE_ORDER = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]

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

CHECKPOINT_PATH = "data/value_generation_checkpoint.json"
CHECKPOINT_INTERVAL = 500
HUMAN_VALUES_PATH = "data/analysis/human_values_all.json"

WINDOW_SIZE = 200
WINDOW_STEP = 50


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("value_diversity_gradient")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_embeddings(logger: logging.Logger) -> dict[str, np.ndarray]:
    """Load all available embedding matrices from disk."""
    embeddings = {}
    for source in ALL_SOURCES:
        path = os.path.join(EMBEDDINGS_DIR, f"{source}.npy")
        if not os.path.exists(path):
            logger.warning("Missing embedding file: %s", path)
            continue
        matrix = np.load(path)
        if matrix.ndim != 2 or matrix.shape[1] != HIDDEN_DIM:
            logger.error("Unexpected shape for %s: %s, skipping", source, matrix.shape)
            continue
        embeddings[source] = matrix
        logger.info("Loaded %s: shape %s", source, matrix.shape)
    if not embeddings:
        raise SystemExit(f"No embedding files found in {EMBEDDINGS_DIR}/")
    return embeddings


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


# ── Reconstruction Error ─────────────────────────────────────────────────────
def compute_human_reconstruction_errors(
    embeddings: dict[str, np.ndarray], logger: logging.Logger,
) -> tuple[np.ndarray, int]:
    """Fit PCA on all_llm, project human data, return per-rationale errors and k."""
    matrices = [embeddings[s] for s in LLM_SOURCE_ORDER if s in embeddings]
    if not matrices:
        raise SystemExit("No LLM embeddings found.")
    all_llm = np.vstack(matrices)
    logger.info("all_llm matrix: shape %s", all_llm.shape)

    pca = PCA()
    pca.fit(all_llm)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumulative, 0.90) + 1)
    k = min(k, len(cumulative))
    logger.info("all_llm PCA: k=%d for 90%% variance", k)

    human = embeddings["human"]
    components_k = pca.components_[:k]
    centered = human - pca.mean_
    projected = centered @ components_k.T
    reconstructed = projected @ components_k
    residuals = centered - reconstructed
    errors = (residuals ** 2).sum(axis=1)

    logger.info("Human reconstruction: mean=%.4f, median=%.4f, max=%.4f",
                errors.mean(), np.median(errors), errors.max())
    return errors, k


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


# ── Checkpointed Generation ─────────────────────────────────────────────────
def _save_checkpoint(results: list[str], logger: logging.Logger) -> None:
    """Save current generation progress to checkpoint file."""
    entries = [{"index": i, "generated_values": v} for i, v in enumerate(results) if v]
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)
    logger.info("Checkpoint saved: %d/%d rationales to %s",
                len(entries), len(results), CHECKPOINT_PATH)


def generate_all_human_values(
    texts: list[str],
    model, tokenizer, template: str, device: torch.device,
    logger: logging.Logger,
) -> list[str]:
    """Generate Kaleido value expressions for all human rationale texts with checkpointing."""
    n = len(texts)
    results = [""] * n

    # Load checkpoint if exists
    completed = set()
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            for entry in checkpoint_data:
                idx = entry["index"]
                results[idx] = entry["generated_values"]
                completed.add(idx)
            logger.info("Resumed from checkpoint: %d/%d already completed", len(completed), n)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Checkpoint corrupted, starting fresh")
            completed = set()
            results = [""] * n

    remaining = [i for i in range(n) if i not in completed]
    if not remaining:
        logger.info("All %d rationales already generated", n)
        return results

    logger.info("Generating values for %d remaining rationales...", len(remaining))
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
            if total_done % 500 < GEN_BATCH_SIZE or batch_start + GEN_BATCH_SIZE >= len(remaining):
                logger.info("Progress: %d/%d rationales generated", total_done, n)

            if n_generated % CHECKPOINT_INTERVAL < GEN_BATCH_SIZE and n_generated >= CHECKPOINT_INTERVAL:
                _save_checkpoint(results, logger)

    except KeyboardInterrupt:
        logger.warning("Interrupted! Saving checkpoint...")
        _save_checkpoint(results, logger)
        raise

    # Clean up checkpoint on success
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        logger.info("Checkpoint cleaned up")

    return results


# ── Binning and Metrics ──────────────────────────────────────────────────────
def compute_shannon_entropy(values: list[str]) -> float:
    """Compute Shannon entropy H = -sum(p * log2(p)) for value frequency distribution."""
    if not values:
        return 0.0
    freq: dict[str, int] = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    total = len(values)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def _bin_window(records: list[dict], bin_index: int) -> dict:
    """Compute metrics for a single bin/window of records."""
    errors = [r["reconstruction_error"] for r in records]
    values = [r["generated_values"] for r in records]
    return {
        "bin_index": bin_index,
        "n_rationales": len(records),
        "mean_error": float(np.mean(errors)),
        "min_error": float(min(errors)),
        "max_error": float(max(errors)),
        "n_unique_values": len(set(values)),
        "shannon_entropy": compute_shannon_entropy(values),
    }


def compute_nonoverlapping_bins(
    sorted_records: list[dict], window_size: int, logger: logging.Logger,
) -> list[dict]:
    """Bin sorted records into non-overlapping windows."""
    bins = []
    for i in range(0, len(sorted_records), window_size):
        window = sorted_records[i:i + window_size]
        bins.append(_bin_window(window, len(bins)))
    logger.info("Non-overlapping bins: %d (window=%d)", len(bins), window_size)
    return bins


def compute_sliding_bins(
    sorted_records: list[dict], window_size: int, step: int, logger: logging.Logger,
) -> list[dict]:
    """Bin sorted records with a sliding window."""
    bins = []
    for start in range(0, len(sorted_records) - window_size + 1, step):
        window = sorted_records[start:start + window_size]
        bins.append(_bin_window(window, len(bins)))
    logger.info("Sliding bins: %d (window=%d, step=%d)", len(bins), window_size, step)
    return bins


# ── Outputs ──────────────────────────────────────────────────────────────────
def save_gradient_csv(
    bins: list[dict], filename: str, output_dir: str, logger: logging.Logger,
) -> None:
    """Save binned diversity metrics to CSV."""
    import csv
    path = os.path.join(output_dir, filename)
    fieldnames = ["bin_index", "n_rationales", "mean_error", "min_error", "max_error",
                  "n_unique_values", "shannon_entropy"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for b in bins:
            writer.writerow({
                "bin_index": b["bin_index"],
                "n_rationales": b["n_rationales"],
                "mean_error": f"{b['mean_error']:.6f}",
                "min_error": f"{b['min_error']:.6f}",
                "max_error": f"{b['max_error']:.6f}",
                "n_unique_values": b["n_unique_values"],
                "shannon_entropy": f"{b['shannon_entropy']:.4f}",
            })
    logger.info("Saved %s", path)


def plot_diversity_gradient(
    bins: list[dict], filename: str, title_suffix: str,
    output_dir: str, logger: logging.Logger,
) -> None:
    """Dual y-axis plot: unique values + Shannon entropy vs reconstruction error."""
    mean_errors = np.array([b["mean_error"] for b in bins])
    n_unique = np.array([b["n_unique_values"] for b in bins])
    entropy = np.array([b["shannon_entropy"] for b in bins])

    # Correlation
    if HAS_SCIPY:
        r, p = pearsonr(mean_errors, n_unique)
    else:
        r = float(np.corrcoef(mean_errors, n_unique)[0, 1])
        p = float("nan")

    logger.info("%s: Pearson r=%.3f, p=%s", filename,
                r, f"{p:.4f}" if not math.isnan(p) else "N/A")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left axis: unique values
    color1 = SOURCE_COLORS["human"]
    ax1.scatter(mean_errors, n_unique, color=color1, alpha=0.6, s=20, zorder=3)
    ax1.plot(mean_errors, n_unique, color=color1, alpha=0.4, linewidth=1)
    ax1.set_xlabel("Mean Reconstruction Error")
    ax1.set_ylabel("Unique Values per Bin", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Trend line for unique values
    z = np.polyfit(mean_errors, n_unique, 1)
    trend = np.poly1d(z)
    ax1.plot(mean_errors, trend(mean_errors), color=color1, linestyle="--",
             linewidth=1.5, alpha=0.7)

    # Right axis: entropy
    ax2 = ax1.twinx()
    color2 = "#ff7f0e"
    ax2.scatter(mean_errors, entropy, color=color2, alpha=0.6, s=20, zorder=3)
    ax2.plot(mean_errors, entropy, color=color2, alpha=0.4, linewidth=1)
    ax2.set_ylabel("Shannon Entropy (bits)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Trend line for entropy
    z2 = np.polyfit(mean_errors, entropy, 1)
    trend2 = np.poly1d(z2)
    ax2.plot(mean_errors, trend2(mean_errors), color=color2, linestyle="--",
             linewidth=1.5, alpha=0.7)

    p_str = f", p={p:.4f}" if not math.isnan(p) else ""
    ax1.set_title(f"Value Diversity vs Reconstruction Error \u2014 {title_suffix}\n"
                  f"(Pearson r={r:.3f}{p_str})")
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    logger = setup_logging()
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load data
    logger.info("=== Step 1: Load embeddings and text lookup ===")
    embeddings = load_embeddings(logger)
    text_lookup = build_text_lookup(logger)

    if "human" not in embeddings:
        raise SystemExit("Human embeddings not found.")

    # Step 2: Reconstruction errors
    logger.info("=== Step 2: Compute reconstruction errors ===")
    errors, k = compute_human_reconstruction_errors(embeddings, logger)

    # Step 3: Load metadata and resolve texts
    logger.info("=== Step 3: Load human metadata ===")
    meta_path = os.path.join(EMBEDDINGS_DIR, "human_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        human_meta = json.load(f)

    texts = []
    for m in human_meta:
        t = text_lookup.get((m["submission_id"], m["column"]), "")
        texts.append(t)
    logger.info("Resolved %d / %d texts", sum(1 for t in texts if t), len(texts))

    # Step 4: Generate or load values
    logger.info("=== Step 4: Value generation ===")

    if os.path.exists(HUMAN_VALUES_PATH):
        logger.info("Loading existing values from %s", HUMAN_VALUES_PATH)
        with open(HUMAN_VALUES_PATH, encoding="utf-8") as f:
            records = json.load(f)
        logger.info("Loaded %d records", len(records))
    else:
        model, tokenizer, template, device = load_kaleido(logger)
        values = generate_all_human_values(texts, model, tokenizer, template, device, logger)

        del model
        torch.cuda.empty_cache()

        records = []
        for i, (m, t, v, e) in enumerate(zip(human_meta, texts, values, errors)):
            records.append({
                "index": i,
                "submission_id": m["submission_id"],
                "column": m["column"],
                "rationale_text": t,
                "reconstruction_error": float(e),
                "generated_values": v,
            })

        with open(HUMAN_VALUES_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info("Saved %s (%d records)", HUMAN_VALUES_PATH, len(records))

    # Step 5: Sort and filter
    logger.info("=== Step 5: Sort and bin ===")
    valid_records = [r for r in records if r["rationale_text"] and r["generated_values"]]
    logger.info("Valid records: %d / %d", len(valid_records), len(records))
    sorted_records = sorted(valid_records, key=lambda r: r["reconstruction_error"])

    # Step 6: Non-overlapping bins
    nonoverlap_bins = compute_nonoverlapping_bins(sorted_records, WINDOW_SIZE, logger)

    # Step 7: Sliding window bins
    sliding_bins = compute_sliding_bins(sorted_records, WINDOW_SIZE, WINDOW_STEP, logger)

    # Step 8: Save outputs
    logger.info("=== Step 8: Save outputs ===")
    save_gradient_csv(nonoverlap_bins, "value_diversity_gradient.csv", OUTPUT_DIR, logger)

    plot_diversity_gradient(
        nonoverlap_bins, "value_diversity_gradient.png",
        f"Non-Overlapping Bins (n={WINDOW_SIZE})",
        OUTPUT_DIR, logger)

    plot_diversity_gradient(
        sliding_bins, "value_diversity_sliding.png",
        f"Sliding Window (w={WINDOW_SIZE}, step={WINDOW_STEP})",
        OUTPUT_DIR, logger)

    elapsed = time.time() - start_time
    logger.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
