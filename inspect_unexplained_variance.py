"""Inspect what the mutually unexplained variance between human and LLM reasoning represents."""

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
from sklearn.decomposition import PCA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
N_EXEMPLARS = 200
N_EXEMPLARS_JSON = 50

ALL_SOURCES = ["human", "gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
# Order must match extract_embeddings.py's LLM_SOURCES dict iteration order
LLM_SOURCE_ORDER = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("inspect_unexplained_variance")
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


def load_metadata(logger: logging.Logger) -> dict[str, list[dict]]:
    """Load metadata JSON for each source."""
    metadata = {}
    for source in ALL_SOURCES:
        path = os.path.join(EMBEDDINGS_DIR, f"{source}_meta.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            metadata[source] = json.load(f)
        logger.info("Loaded metadata for %s: %d entries", source, len(metadata[source]))
    return metadata


def build_text_lookup(logger: logging.Logger) -> dict[tuple[str, str], str]:
    """Build (submission_id, column) -> text lookup from HF dataset."""
    logger.info("Loading dataset: %s", DATASET_NAME)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    logger.info("Dataset loaded: %d rows", len(ds))

    lookup: dict[tuple[str, str], str] = {}
    # Index by submission_id for fast lookup
    for row in ds:
        sid = row["submission_id"]
        for col in row:
            val = row[col]
            if isinstance(val, str) and val.strip():
                lookup[(sid, col)] = val.strip()

    logger.info("Text lookup built: %d entries", len(lookup))
    return lookup


def build_all_llm(
    embeddings: dict[str, np.ndarray],
    metadata: dict[str, list[dict]],
    logger: logging.Logger,
) -> tuple[np.ndarray, list[dict]]:
    """Concatenate LLM embeddings and metadata in extraction order."""
    matrices = []
    all_meta = []
    for source in LLM_SOURCE_ORDER:
        if source not in embeddings or source not in metadata:
            continue
        matrices.append(embeddings[source])
        for entry in metadata[source]:
            all_meta.append({**entry, "source_model": source})
    combined = np.vstack(matrices)
    logger.info("all_llm: shape %s, %d metadata entries", combined.shape, len(all_meta))
    return combined, all_meta


# ── Reconstruction Error ─────────────────────────────────────────────────────
def compute_reconstruction_errors(
    basis_data: np.ndarray, target_data: np.ndarray, logger: logging.Logger,
) -> tuple[np.ndarray, int]:
    """Compute per-rationale squared reconstruction error.

    Fits PCA on basis_data, projects target_data onto top-k components (k=90%),
    and returns per-rationale squared error and k used.
    """
    pca = PCA()
    pca.fit(basis_data)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumulative, 0.90) + 1)
    k = min(k, len(cumulative))

    components_k = pca.components_[:k]  # [k, d]
    centered = target_data - pca.mean_  # [n, d]
    projected = centered @ components_k.T  # [n, k]
    reconstructed = projected @ components_k  # [n, d]
    residuals = centered - reconstructed  # [n, d]
    errors = (residuals ** 2).sum(axis=1)  # [n]

    logger.info("Reconstruction: k=%d, mean_error=%.2f, median=%.2f, max=%.2f",
                k, errors.mean(), np.median(errors), errors.max())
    return errors, k


# ── Exemplar Selection ───────────────────────────────────────────────────────
def select_exemplars(
    errors: np.ndarray, meta: list[dict], text_lookup: dict,
    n_select: int, direction: str, logger: logging.Logger,
) -> tuple[list[dict], list[dict]]:
    """Select top-n poorly-captured and top-n well-captured rationales.

    Returns (poorly_captured, well_captured) lists of dicts with text attached.
    """
    sorted_idx = np.argsort(errors)
    worst_idx = sorted_idx[-n_select:][::-1]  # highest error first
    best_idx = sorted_idx[:n_select]  # lowest error first

    def build_entries(indices):
        entries = []
        for i in indices:
            m = meta[i]
            sid = m["submission_id"]
            col = m["column"]
            text = text_lookup.get((sid, col), "")
            entry = {
                "index": int(i),
                "submission_id": sid,
                "column": col,
                "reconstruction_error": float(errors[i]),
                "rationale_text": text,
            }
            if "source_model" in m:
                entry["source_model"] = m["source_model"]
            entries.append(entry)
        return entries

    poorly = build_entries(worst_idx)
    well = build_entries(best_idx)
    logger.info("%s: poorly-captured error range [%.2f, %.2f], well-captured [%.2f, %.2f]",
                direction, poorly[-1]["reconstruction_error"], poorly[0]["reconstruction_error"],
                well[0]["reconstruction_error"], well[-1]["reconstruction_error"])
    return poorly, well


# ── Kaleido Value Generation ────────────────────────────────────────────────
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
    return decoded


def generate_values_for_group(
    model, tokenizer, template: str, device: torch.device,
    entries: list[dict], group_name: str, logger: logging.Logger,
) -> list[dict]:
    """Generate value expressions for all entries in a group."""
    texts = [e["rationale_text"] for e in entries]
    all_values = []
    total = len(texts)

    for i in range(0, total, GEN_BATCH_SIZE):
        batch = texts[i:i + GEN_BATCH_SIZE]
        values = generate_values_batch(model, tokenizer, batch, template, device)
        all_values.extend(values)
        processed = min(i + GEN_BATCH_SIZE, total)
        if processed % 100 < GEN_BATCH_SIZE or processed == total:
            logger.info("%s: %d/%d values generated", group_name, processed, total)

    # Attach values to entries
    for entry, val in zip(entries, all_values):
        # Strip "Value: " prefix if present
        clean = val.strip()
        if clean.lower().startswith("value:"):
            clean = clean[len("value:"):].strip()
        entry["generated_values"] = clean

    return entries


# ── Value Distribution Analysis ──────────────────────────────────────────────
def analyze_value_distribution(
    entries: list[dict], group_name: str, logger: logging.Logger,
) -> dict:
    """Analyze the distribution of generated values for a group."""
    values = [e["generated_values"] for e in entries if e["generated_values"]]
    n_rationales = len(entries)

    # Count frequencies
    freq: dict[str, int] = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1

    sorted_values = sorted(freq.items(), key=lambda x: -x[1])
    n_unique = len(freq)
    top_20 = sorted_values[:20]

    logger.info("%s: %d rationales, %d unique values", group_name, n_rationales, n_unique)
    for val, count in top_20[:5]:
        logger.info("  %s: %d (%.1f%%)", val, count, 100 * count / n_rationales)

    return {
        "group_name": group_name,
        "n_rationales": n_rationales,
        "n_unique_values": n_unique,
        "n_with_values": len(values),
        "top_20_values": top_20,
        "frequency": freq,
    }


# ── Outputs ──────────────────────────────────────────────────────────────────
def save_summary_csv(
    analyses: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Write unexplained_variance_summary.csv."""
    import csv
    path = os.path.join(output_dir, "unexplained_variance_summary.csv")
    fieldnames = ["group_name", "n_rationales", "n_unique_values",
                  "avg_values_per_rationale", "top_20_values"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in analyses:
            avg_vpr = a["n_with_values"] / a["n_rationales"] if a["n_rationales"] > 0 else 0
            top20_str = "; ".join(f"{v} ({c})" for v, c in a["top_20_values"])
            writer.writerow({
                "group_name": a["group_name"],
                "n_rationales": a["n_rationales"],
                "n_unique_values": a["n_unique_values"],
                "avg_values_per_rationale": f"{avg_vpr:.2f}",
                "top_20_values": top20_str,
            })
    logger.info("Saved %s", path)


def save_exemplar_json(
    entries: list[dict], filename: str, output_dir: str, logger: logging.Logger,
) -> None:
    """Save exemplar rationales to JSON."""
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s (%d entries)", path, len(entries))


def plot_value_comparison(
    analyses: list[dict], output_dir: str, logger: logging.Logger,
) -> None:
    """Side-by-side bar chart of value distributions: poorly vs well captured."""
    # Expect 4 analyses: human-poorly, human-well, llm-poorly, llm-well
    by_name = {a["group_name"]: a for a in analyses}

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    panels = [
        ("Human Rationales", "human_poorly_captured", "human_well_captured"),
        ("LLM Rationales", "llm_poorly_captured", "llm_well_captured"),
    ]

    for ax, (title, poorly_key, well_key) in zip(axes, panels):
        poorly = by_name.get(poorly_key, {})
        well = by_name.get(well_key, {})

        # Get union of top values from both groups
        poorly_freq = poorly.get("frequency", {})
        well_freq = well.get("frequency", {})

        # Use top 20 from poorly-captured as reference
        top_values = [v for v, _ in poorly.get("top_20_values", [])[:20]]

        if not top_values:
            ax.set_title(f"{title} (no data)")
            continue

        poorly_counts = [poorly_freq.get(v, 0) for v in top_values]
        well_counts = [well_freq.get(v, 0) for v in top_values]

        # Normalize to percentages
        poorly_total = poorly.get("n_rationales", 1)
        well_total = well.get("n_rationales", 1)
        poorly_pct = [100 * c / poorly_total for c in poorly_counts]
        well_pct = [100 * c / well_total for c in well_counts]

        y_pos = np.arange(len(top_values))
        bar_height = 0.35

        ax.barh(y_pos - bar_height / 2, poorly_pct, bar_height,
                label="Poorly captured", color="#d62728", alpha=0.8)
        ax.barh(y_pos + bar_height / 2, well_pct, bar_height,
                label="Well captured", color="#1f77b4", alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency (%)")
        ax.set_title(f"{title}\n(poorly: {poorly.get('n_unique_values', 0)} unique, "
                     f"well: {well.get('n_unique_values', 0)} unique)")
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Value Expressions: Poorly-Captured vs Well-Captured Rationales", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "value_frequency_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    logger = setup_logging()
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    embeddings = load_embeddings(logger)
    metadata = load_metadata(logger)
    text_lookup = build_text_lookup(logger)

    if "human" not in embeddings:
        raise SystemExit("Human embeddings not found.")

    # Build all_llm
    all_llm_emb, all_llm_meta = build_all_llm(embeddings, metadata, logger)

    human_emb = embeddings["human"]
    human_meta = metadata["human"]

    # ── Step 1: Reconstruction errors ────────────────────────────────────────
    logger.info("=== Computing reconstruction errors ===")

    logger.info("Direction: LLM PCs → human data (what LLMs miss about humans)")
    human_errors, llm_k = compute_reconstruction_errors(all_llm_emb, human_emb, logger)

    logger.info("Direction: Human PCs → LLM data (what humans miss about LLMs)")
    llm_errors, human_k = compute_reconstruction_errors(human_emb, all_llm_emb, logger)

    # ── Step 2: Select exemplars ─────────────────────────────────────────────
    logger.info("=== Selecting exemplar rationales ===")

    human_poorly, human_well = select_exemplars(
        human_errors, human_meta, text_lookup, N_EXEMPLARS, "human", logger)
    llm_poorly, llm_well = select_exemplars(
        llm_errors, all_llm_meta, text_lookup, N_EXEMPLARS, "llm", logger)

    # ── Step 3: Generate Kaleido values ──────────────────────────────────────
    logger.info("=== Generating Kaleido value expressions ===")
    model, tokenizer, template, device = load_kaleido(logger)

    groups = [
        ("human_poorly_captured", human_poorly),
        ("human_well_captured", human_well),
        ("llm_poorly_captured", llm_poorly),
        ("llm_well_captured", llm_well),
    ]

    for group_name, entries in groups:
        logger.info("Generating values for %s (%d rationales)...", group_name, len(entries))
        generate_values_for_group(model, tokenizer, template, device,
                                  entries, group_name, logger)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # ── Step 4: Analyze value distributions ──────────────────────────────────
    logger.info("=== Analyzing value distributions ===")

    analyses = []
    for group_name, entries in groups:
        analysis = analyze_value_distribution(entries, group_name, logger)
        analyses.append(analysis)

    # ── Step 5: Save outputs ─────────────────────────────────────────────────
    logger.info("=== Saving outputs ===")

    save_summary_csv(analyses, OUTPUT_DIR, logger)

    save_exemplar_json(human_poorly[:N_EXEMPLARS_JSON],
                       "unexplained_human_exemplars.json", OUTPUT_DIR, logger)
    save_exemplar_json(llm_poorly[:N_EXEMPLARS_JSON],
                       "unexplained_llm_exemplars.json", OUTPUT_DIR, logger)
    save_exemplar_json(human_well[:N_EXEMPLARS_JSON],
                       "wellcaptured_human_exemplars.json", OUTPUT_DIR, logger)
    save_exemplar_json(llm_well[:N_EXEMPLARS_JSON],
                       "wellcaptured_llm_exemplars.json", OUTPUT_DIR, logger)

    plot_value_comparison(analyses, OUTPUT_DIR, logger)

    elapsed = time.time() - start_time
    logger.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
