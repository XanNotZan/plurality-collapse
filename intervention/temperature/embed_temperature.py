"""Embed Qwen-generated rationales for each temperature via Kaleido-XL encoder.

Reads:
  data/generated/intervention/temperature/qwen25_3b_T{T}.jsonl

Writes:
  data/embeddings/intervention/temperature/qwen25_3b_T{T}.npy
  data/embeddings/intervention/temperature/qwen25_3b_T{T}_meta.json
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

GEN_DIR = Path("data/generated/intervention/temperature")
EMB_DIR = Path("data/embeddings/intervention/temperature")
EMB_DIR.mkdir(parents=True, exist_ok=True)

KALEIDO_MODEL = "allenai/kaleido-xl"
TEMPERATURES = [0.3, 0.7, 1.0, 1.3]
BATCH = 32
MAX_LEN = 512
LOG_INTERVAL = 20

logger = logging.getLogger("embed_temp")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg)
    sys.stdout.flush()


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_kaleido():
    _flush(f"loading {KALEIDO_MODEL} (fp16)")
    tok = AutoTokenizer.from_pretrained(KALEIDO_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(KALEIDO_MODEL, dtype=torch.float16)
    model = model.to("cuda").eval()
    try:
        template = model.config.task_specific_params["generate"]["template"]
    except Exception:
        template = "[Generate]:\tAction: ACTION"
    _flush(f"Kaleido template: {template!r}")
    _flush(f"vram: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return tok, model, template


def embed_texts(tok, model, template, texts):
    all_emb = []
    t0 = time.time()
    for s in range(0, len(texts), BATCH):
        batch = texts[s : s + BATCH]
        formatted = [template.replace("ACTION", t if t else "(empty)") for t in batch]
        inputs = tok(formatted, return_tensors="pt", padding=True, truncation=True,
                     max_length=MAX_LEN).to("cuda")
        with torch.no_grad():
            enc = model.encoder(input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"])
        hidden = enc.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_emb.append(pooled.cpu().float().numpy())
        if (s // BATCH) % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            rate = (s + len(batch)) / elapsed if elapsed > 0 else 0
            _flush(f"  embedded {s + len(batch)}/{len(texts)} ({rate:.0f}/s)")
    return np.vstack(all_emb)


def process_temperature(tok, model, template, T):
    jsonl = GEN_DIR / f"qwen25_3b_T{T:.1f}.jsonl"
    if not jsonl.exists():
        _flush(f"missing {jsonl}, skipping T={T}")
        return
    items = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
            items.append(j)
        except Exception:
            continue
    _flush(f"=== T={T} loaded {len(items)} rationales from {jsonl.name}")
    texts = [it["rationale"] for it in items]
    t0 = time.time()
    emb = embed_texts(tok, model, template, texts)
    elapsed = time.time() - t0
    _flush(f"T={T} embedded shape={emb.shape} elapsed={elapsed:.0f}s "
           f"({len(texts) / elapsed:.0f}/s)")
    np.save(EMB_DIR / f"qwen25_3b_T{T:.1f}.npy", emb)
    meta = [
        {"index": i, "submission_id": it["submission_id"],
         "temperature": it["temperature"], "sample_idx": it["sample_idx"]}
        for i, it in enumerate(items)
    ]
    (EMB_DIR / f"qwen25_3b_T{T:.1f}_meta.json").write_text(json.dumps(meta))
    _flush(f"saved {EMB_DIR / f'qwen25_3b_T{T:.1f}.npy'}")


def main():
    free_gpu()
    tok, model, template = load_kaleido()
    for T in TEMPERATURES:
        process_temperature(tok, model, template, T)
    _flush("ALL EMBEDDED")


if __name__ == "__main__":
    main()
