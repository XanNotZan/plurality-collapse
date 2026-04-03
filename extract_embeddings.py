"""Extract Kaleido-XL encoder embeddings from human and LLM rationales in the Sachdeva dataset."""

import json
import logging
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_NAME = "ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas"
DATASET_SPLIT = "test"
MODEL_NAME = "allenai/kaleido-xl"
OUTPUT_DIR = "data/embeddings"
BATCH_SIZE = 32
HIDDEN_DIM = 2048
MAX_LENGTH = 512

FALLBACK_TEMPLATE = "[Generate]:\tAction: ACTION"

LLM_SOURCES = {
    "gpt3.5":  ["gpt3.5_reason_1", "gpt3.5_reason_2", "gpt3.5_reason_3"],
    "gpt4":    ["gpt4_reason_1", "gpt4_reason_2"],
    "claude":  ["claude_reason_1", "claude_reason_2", "claude_reason_3"],
    "bison":   ["bison_reason_1", "bison_reason_2", "bison_reason_3"],
    "gemma":   ["gemma_reason_1", "gemma_reason_2", "gemma_reason_3"],
    "mistral": ["mistral_reason_1", "mistral_reason_2", "mistral_reason_3"],
    "llama":   ["llama_reason_1", "llama_reason_2", "llama_reason_3"],
}


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("extract_embeddings")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger


# ── Data Collection ────────────────────────────────────────────────────────────
def collect_rationales(ds, logger: logging.Logger) -> dict[str, list[dict]]:
    """Collect rationales per source. Returns {source: [{submission_id, column, text}, ...]}."""
    sources = {}

    # Human rationales
    human = []
    for row in ds:
        text = row.get("top_comment")
        if text and isinstance(text, str) and text.strip():
            human.append({
                "submission_id": row["submission_id"],
                "column": "top_comment",
                "text": text.strip(),
            })
    sources["human"] = human
    logger.info("human: %d rationales", len(human))

    # LLM rationales
    for source_name, columns in LLM_SOURCES.items():
        entries = []
        for row in ds:
            for col in columns:
                text = row.get(col)
                if text and isinstance(text, str) and text.strip():
                    entries.append({
                        "submission_id": row["submission_id"],
                        "column": col,
                        "text": text.strip(),
                    })
        sources[source_name] = entries
        logger.info("%s: %d rationales", source_name, len(entries))

    return sources


# ── Embedding Extraction ──────────────────────────────────────────────────────
def extract_embeddings_batch(
    model, tokenizer, texts: list[str], template: str, device: torch.device
) -> np.ndarray:
    """Encode a batch of texts through Kaleido's encoder and mean-pool."""
    formatted = [template.replace("ACTION", t) for t in texts]
    inputs = tokenizer(
        formatted, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        encoder_outputs = model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    hidden = encoder_outputs.last_hidden_state                    # [batch, seq_len, 2048]
    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype) # [batch, seq_len, 1]
    mean_pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)    # [batch, 2048]
    return mean_pooled.cpu().float().numpy()


def embed_source(
    model, tokenizer, entries: list[dict], template: str, device: torch.device,
    source_name: str, logger: logging.Logger,
) -> tuple[np.ndarray, list[dict]]:
    """Embed all rationales for a source. Returns (embedding_matrix, metadata_list)."""
    texts = [e["text"] for e in entries]
    all_embeddings = []
    total = len(texts)

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        embeddings = extract_embeddings_batch(model, tokenizer, batch, template, device)
        all_embeddings.append(embeddings)

        processed = min(i + BATCH_SIZE, total)
        if processed % 500 < BATCH_SIZE or processed == total:
            logger.info("%s: %d/%d rationales embedded", source_name, processed, total)

    matrix = np.vstack(all_embeddings)  # [n, 2048]

    metadata = [
        {"index": idx, "submission_id": e["submission_id"], "column": e["column"]}
        for idx, e in enumerate(entries)
    ]

    return matrix, metadata


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    load_dotenv()
    logger = setup_logging()
    start_time = time.time()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This script requires a GPU.")

    device = torch.device("cuda")
    logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(0))

    # Load model
    logger.info("Loading model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.to(device).eval()
    logger.info("Model loaded (fp16)")

    # Get template
    try:
        template = model.config.task_specific_params["generate"]["template"]
        logger.info("Template from model config: %r", template)
    except (AttributeError, KeyError, TypeError):
        template = FALLBACK_TEMPLATE
        logger.warning("Template not found in model config, using fallback: %r", template)

    # Load dataset
    logger.info("Loading dataset: %s", DATASET_NAME)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    logger.info("Dataset loaded: %d rows", len(ds))

    # Collect rationales
    sources = collect_rationales(ds, logger)

    # Process each source
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for source_name, entries in sources.items():
        if not entries:
            logger.warning("Skipping %s: no rationales found", source_name)
            continue

        logger.info("Embedding %s (%d rationales)...", source_name, len(entries))
        matrix, metadata = embed_source(
            model, tokenizer, entries, template, device, source_name, logger
        )

        # Save embeddings
        npy_path = os.path.join(OUTPUT_DIR, f"{source_name}.npy")
        np.save(npy_path, matrix)

        # Save metadata
        meta_path = os.path.join(OUTPUT_DIR, f"{source_name}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        logger.info(
            "%s: saved %s (shape %s) and %s",
            source_name, npy_path, matrix.shape, meta_path,
        )

    elapsed = time.time() - start_time
    logger.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
