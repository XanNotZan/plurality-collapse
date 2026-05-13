"""Clean Qwen paraphrase output (chat-template artifacts) and re-embed."""

import gc
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

EMBEDDINGS_DIR = Path("data/embeddings")
GEN_DIR = Path("data/generated")

logger = logging.getLogger("clean_paraphrase")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg)
    sys.stdout.flush()


def clean_one(text: str) -> str:
    """Strip chat-template artifacts. Take content after the LAST 'assistant\\n' or 'Formal/Casual rewrite:'."""
    # Common patterns the model emits when chat template was not respected
    markers = [
        "Formal rewrite:\nassistant\n",
        "Casual rewrite:\nassistant\n",
        "\nassistant\n",
        "Formal rewrite:\n",
        "Casual rewrite:\n",
    ]
    cleaned = text
    for m in markers:
        idx = cleaned.rfind(m)
        if idx >= 0:
            cleaned = cleaned[idx + len(m):]
            break
    # Strip leading punctuation/whitespace
    cleaned = cleaned.strip()
    cleaned = re.sub(r"^[\s\.\*\-]+", "", cleaned)
    return cleaned


def clean_jsonl(path_in: Path, path_out: Path):
    items = [json.loads(l) for l in path_in.read_text().splitlines() if l.strip()]
    cleaned_items = []
    n_changed = 0
    n_short_after = 0
    for it in items:
        new_text = clean_one(it["paraphrased"])
        if new_text != it["paraphrased"]:
            n_changed += 1
        if len(new_text.split()) < 5:
            n_short_after += 1
            new_text = it["paraphrased"]  # fallback
        new_it = dict(it)
        new_it["paraphrased_clean"] = new_text
        cleaned_items.append(new_it)
    path_out.write_text("\n".join(json.dumps(x) for x in cleaned_items))
    _flush(f"{path_in.name}: {n_changed}/{len(items)} cleaned, {n_short_after} too short after cleaning (kept original)")
    return cleaned_items


def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_kaleido():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    _flush("loading Kaleido")
    tok = AutoTokenizer.from_pretrained("allenai/kaleido-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/kaleido-xl", dtype=torch.float16).to("cuda").eval()
    try:
        template = model.config.task_specific_params["generate"]["template"]
    except Exception:
        template = "[Generate]:\tAction: ACTION"
    return tok, model, template


def kaleido_embed(tok, model, template, texts, batch=32):
    out = []
    for s in range(0, len(texts), batch):
        b = texts[s:s + batch]
        formatted = [template.replace("ACTION", t if t else "(empty)") for t in b]
        inputs = tok(formatted, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            enc = model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        h = enc.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1)
        out.append(pooled.cpu().float().numpy())
    return np.vstack(out)


def main():
    # Clean
    h_items = clean_jsonl(GEN_DIR / "human_to_formal.jsonl", GEN_DIR / "human_to_formal_clean.jsonl")
    l_items = clean_jsonl(GEN_DIR / "llm_to_casual.jsonl", GEN_DIR / "llm_to_casual_clean.jsonl")

    # Re-embed with cleaned text
    tok, model, template = load_kaleido()
    h_texts = [it["paraphrased_clean"] for it in h_items]
    l_texts = [it["paraphrased_clean"] for it in l_items]

    h_emb = kaleido_embed(tok, model, template, h_texts)
    np.save(EMBEDDINGS_DIR / "human_to_formal_clean.npy", h_emb)
    h_meta = [{"index": i, "submission_id": it.get("submission_id", ""),
               "column": it.get("orig_column", ""),
               "source_origin": it.get("source", "human")}
              for i, it in enumerate(h_items)]
    (EMBEDDINGS_DIR / "human_to_formal_clean_meta.json").write_text(json.dumps(h_meta))
    _flush(f"saved human_to_formal_clean.npy {h_emb.shape}")

    l_emb = kaleido_embed(tok, model, template, l_texts)
    np.save(EMBEDDINGS_DIR / "llm_to_casual_clean.npy", l_emb)
    l_meta = [{"index": i, "submission_id": it.get("submission_id", ""),
               "column": it.get("orig_column", ""),
               "source_origin": it.get("source", "llm")}
              for i, it in enumerate(l_items)]
    (EMBEDDINGS_DIR / "llm_to_casual_clean_meta.json").write_text(json.dumps(l_meta))
    _flush(f"saved llm_to_casual_clean.npy {l_emb.shape}")
    _flush("ALL DONE")


if __name__ == "__main__":
    main()
