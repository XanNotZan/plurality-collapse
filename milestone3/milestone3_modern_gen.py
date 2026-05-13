"""Generate AITA rationales using a modern local LLM (Qwen 2.5), then embed via Kaleido.

Two subtasks:
  --task gen      : produce {N} new rationales on sampled dilemmas (saved as JSONL)
  --task paraphrase : style-transfer 300 human + 300 LLM rationales (saved as JSONL)
  --task embed    : run Kaleido encoder on saved JSONLs, write .npy + meta.json
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

EMBEDDINGS_DIR = Path("data/embeddings")
OUTPUT_DIR = Path("data/analysis")
GEN_DIR = Path("data/generated")
RANDOM_SEED = 42
ALL_SOURCES = ["human", "gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]
LLM_SOURCES = ["gpt3.5", "gpt4", "claude", "bison", "gemma", "mistral", "llama"]

GEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
KALEIDO_MODEL = "allenai/kaleido-xl"
HIDDEN_DIM = 2048

logger = logging.getLogger("modern_gen")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg)
    sys.stdout.flush()


def load_meta(src):
    return json.loads((EMBEDDINGS_DIR / f"{src}_meta.json").read_text())


# --- Generation ----------------------------------------------------------------
RATIONALE_PROMPT = (
    "You are reading a story posted to Reddit's r/AmITheAsshole forum. "
    "The original poster is asking the community to judge their behavior. "
    "Read the post and write a 2--3 sentence judgment of the original poster (OP). "
    "Begin with one of the verdicts (NTA = Not the Asshole, YTA = You're the Asshole, "
    "ESH = Everyone Sucks Here, NAH = No Assholes Here), then give your reasoning.\n\n"
    "Post:\n{post}\n\n"
    "Your judgment:"
)


def load_qwen():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _flush(f"loading {GEN_MODEL}")
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for batch decoder-only generation
    _flush(f"loaded; vram allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return tok, model


def generate_rationales(n_dilemmas=1000, batch_size=4, seed=RANDOM_SEED, max_post_chars=1400):
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GEN_DIR / "qwen25_3b_rationales.jsonl"
    if out_path.exists():
        _flush(f"output exists, will append/skip already-done: {out_path}")

    _flush("loading HF dataset")
    ds = load_dataset("ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas", split="test")

    sub_ids = ds["submission_id"]
    titles = ds["title"]
    selftexts = ds["selftext"]

    # Sample N dilemmas (deterministic)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), n_dilemmas, replace=False)
    indices = sorted(indices.tolist())

    # Skip already done
    done_ids = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done_ids.add(json.loads(line)["submission_id"])
            except Exception:
                pass
    _flush(f"already done: {len(done_ids)}")

    todo = []
    for i in indices:
        sid = sub_ids[i]
        if sid in done_ids:
            continue
        title = titles[i] or ""
        body = selftexts[i] or ""
        post = (title + "\n\n" + body).strip()
        if len(post) > max_post_chars:
            post = post[:max_post_chars] + "..."
        if len(post.split()) < 10:
            continue
        todo.append({"submission_id": sid, "post": post})
    _flush(f"todo: {len(todo)} dilemmas")

    if not todo:
        _flush("nothing to generate")
        return out_path

    tok, model = load_qwen()
    eos = tok.eos_token_id
    pad = tok.pad_token_id or eos
    gen_kwargs = dict(
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=pad,
    )

    fout = open(out_path, "a")
    t0 = time.time()
    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start:batch_start + batch_size]
        msgs_list = [
            [{"role": "user", "content": RATIONALE_PROMPT.format(post=item["post"])}]
            for item in batch
        ]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in msgs_list]
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                     max_length=2048).to("cuda")
        with torch.no_grad():
            outs = model.generate(**inputs, **gen_kwargs)
        # Decode
        input_length = inputs["input_ids"].shape[1]
        for i, item in enumerate(batch):
            new_ids = outs[i][input_length:]
            text = tok.decode(new_ids, skip_special_tokens=True).strip()
            fout.write(json.dumps({
                "submission_id": item["submission_id"],
                "rationale": text,
            }) + "\n")
        fout.flush()
        elapsed = time.time() - t0
        n_done = batch_start + len(batch)
        rate = n_done / elapsed if elapsed > 0 else 0
        if (batch_start // batch_size) % 5 == 0:
            _flush(f"gen {n_done}/{len(todo)} rate={rate:.1f}/s vram={torch.cuda.memory_allocated()/1e9:.1f}GB")
    fout.close()
    _flush(f"saved {out_path}")
    return out_path


# --- Paraphrasing --------------------------------------------------------------
TO_FORMAL_PROMPT = (
    "Rewrite the following Reddit comment as a formal academic moral judgment. "
    "Preserve the meaning exactly. Remove slang, profanity, and Reddit jargon "
    "(e.g. NTA, YTA, OP, lol, lmao, wtf, ngl, tbh). Use complete sentences. "
    "Output only the rewritten text.\n\nOriginal:\n{text}\n\nFormal rewrite:"
)
TO_CASUAL_PROMPT = (
    "Rewrite the following formal moral judgment as a casual Reddit r/AITA comment. "
    "Preserve the meaning exactly. Use Reddit jargon (NTA / YTA / ESH / NAH for verdicts), "
    "informal contractions, and a conversational tone. Output only the rewritten text.\n\n"
    "Original:\n{text}\n\nCasual rewrite:"
)


def paraphrase(n_each=300, batch_size=4, seed=RANDOM_SEED):
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out_h_path = GEN_DIR / "human_to_formal.jsonl"
    out_l_path = GEN_DIR / "llm_to_casual.jsonl"

    # Load source texts
    _flush("loading HF dataset for source rationale text")
    ds = load_dataset("ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas", split="test")
    sub_ids_col = ds["submission_id"]
    needed = ["top_comment"] + [
        c for src in LLM_SOURCES for c in [
            f"{src}_reason_1", f"{src}_reason_2", f"{src}_reason_3",
        ] if c in ds.column_names
    ]
    col_data = {c: ds[c] for c in needed}
    sub_to = {}
    for i, sid in enumerate(sub_ids_col):
        sub_to[sid] = {c: col_data[c][i] for c in col_data}

    # Sample human rationales
    human_meta = load_meta("human")
    rng = np.random.RandomState(seed)
    h_idx = rng.choice(len(human_meta), n_each, replace=False)
    human_items = []
    for i in h_idx:
        m = human_meta[i]
        text = sub_to[m["submission_id"]].get("top_comment", "") or ""
        if 5 < len(text.split()) < 200:
            human_items.append({"submission_id": m["submission_id"], "orig_index": int(i),
                                "source": "human", "orig_column": m["column"], "text": text})
    _flush(f"human items to paraphrase: {len(human_items)}")

    # Sample LLM rationales (mix across LLMs)
    llm_items = []
    n_per_llm = max(1, n_each // len(LLM_SOURCES))
    for src in LLM_SOURCES:
        meta = load_meta(src)
        idx = rng.choice(len(meta), min(n_per_llm + 5, len(meta)), replace=False)
        for i in idx:
            m = meta[i]
            text = sub_to[m["submission_id"]].get(m["column"], "") or ""
            if 5 < len(text.split()) < 200:
                llm_items.append({"submission_id": m["submission_id"], "orig_index": int(i),
                                  "source": src, "orig_column": m["column"], "text": text})
                if len([x for x in llm_items if x["source"] == src]) >= n_per_llm:
                    break
    _flush(f"LLM items to paraphrase: {len(llm_items)}")

    # Skip already-done
    def already_done(path):
        done = set()
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    j = json.loads(line)
                    done.add((j["source"], j["submission_id"], j["orig_column"]))
                except Exception:
                    pass
        return done

    done_h = already_done(out_h_path)
    done_l = already_done(out_l_path)

    human_todo = [x for x in human_items if (x["source"], x["submission_id"], x["orig_column"]) not in done_h]
    llm_todo = [x for x in llm_items if (x["source"], x["submission_id"], x["orig_column"]) not in done_l]
    _flush(f"todo: human={len(human_todo)} llm={len(llm_todo)}")

    if not human_todo and not llm_todo:
        return out_h_path, out_l_path

    tok, model = load_qwen()
    eos = tok.eos_token_id
    pad = tok.pad_token_id or eos
    gen_kwargs = dict(max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=pad)

    def do_batch(items, prompt_tmpl, out_path):
        if not items:
            return
        fout = open(out_path, "a")
        t0 = time.time()
        for s in range(0, len(items), batch_size):
            batch = items[s:s + batch_size]
            msgs_list = [
                [{"role": "user", "content": prompt_tmpl.format(text=item["text"])}]
                for item in batch
            ]
            prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for m in msgs_list]
            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                         max_length=1024).to("cuda")
            with torch.no_grad():
                outs = model.generate(**inputs, **gen_kwargs)
            for i, item in enumerate(batch):
                in_len = inputs["input_ids"][i].ne(pad).sum().item()
                new_ids = outs[i][in_len:]
                text = tok.decode(new_ids, skip_special_tokens=True).strip()
                fout.write(json.dumps({
                    "submission_id": item["submission_id"],
                    "orig_index": item["orig_index"],
                    "source": item["source"],
                    "orig_column": item["orig_column"],
                    "orig_text": item["text"],
                    "paraphrased": text,
                }) + "\n")
            fout.flush()
            elapsed = time.time() - t0
            n_done = s + len(batch)
            if (s // batch_size) % 5 == 0:
                _flush(f"paraphrase {out_path.name} {n_done}/{len(items)} rate={n_done/(elapsed+1e-6):.1f}/s")
        fout.close()
        _flush(f"saved {out_path}")

    do_batch(human_todo, TO_FORMAL_PROMPT, out_h_path)
    do_batch(llm_todo, TO_CASUAL_PROMPT, out_l_path)
    return out_h_path, out_l_path


# --- Embedding via Kaleido -----------------------------------------------------
def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_kaleido():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    _flush(f"loading Kaleido {KALEIDO_MODEL}")
    tok = AutoTokenizer.from_pretrained(KALEIDO_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(KALEIDO_MODEL, torch_dtype=torch.float16)
    model = model.to("cuda").eval()
    try:
        template = model.config.task_specific_params["generate"]["template"]
    except Exception:
        template = "[Generate]:\tAction: ACTION"
    _flush(f"Kaleido template: {template!r}")
    _flush(f"vram after Kaleido: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return tok, model, template


def kaleido_embed_texts(tok, model, template, texts, batch_size=32, max_len=512):
    all_emb = []
    for s in range(0, len(texts), batch_size):
        batch = texts[s:s + batch_size]
        formatted = [template.replace("ACTION", t if t else "(empty)") for t in batch]
        inputs = tok(formatted, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda")
        with torch.no_grad():
            enc = model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        hidden = enc.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_emb.append(pooled.cpu().float().numpy())
        if (s // batch_size) % 10 == 0:
            _flush(f"embed {s + len(batch)}/{len(texts)}")
    return np.vstack(all_emb)


def embed_jsonl(jsonl_path, source_name, text_field):
    if not jsonl_path.exists():
        _flush(f"missing {jsonl_path}, skipping")
        return
    items = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    texts = [it[text_field] for it in items]
    _flush(f"embed {jsonl_path.name}: {len(texts)} items as source={source_name}")
    free_gpu_memory()
    tok, model, template = load_kaleido()
    emb = kaleido_embed_texts(tok, model, template, texts)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_DIR / f"{source_name}.npy", emb)
    meta = []
    for i, it in enumerate(items):
        meta.append({"index": i, "submission_id": it.get("submission_id", ""),
                     "column": it.get("orig_column", "generated"),
                     "source_origin": it.get("source", source_name)})
    (EMBEDDINGS_DIR / f"{source_name}_meta.json").write_text(json.dumps(meta))
    _flush(f"saved {EMBEDDINGS_DIR / f'{source_name}.npy'} shape {emb.shape}")
    # Free
    del model, tok
    free_gpu_memory()


# --- Main ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gen", "paraphrase", "embed_gen", "embed_para", "all"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n_each", type=int, default=300)
    args = ap.parse_args()

    if args.task == "gen":
        generate_rationales(n_dilemmas=args.n, batch_size=args.batch)
    elif args.task == "paraphrase":
        paraphrase(n_each=args.n_each, batch_size=args.batch)
    elif args.task == "embed_gen":
        embed_jsonl(GEN_DIR / "qwen25_3b_rationales.jsonl",
                    source_name="qwen25_3b_modern", text_field="rationale")
    elif args.task == "embed_para":
        embed_jsonl(GEN_DIR / "human_to_formal.jsonl",
                    source_name="human_to_formal", text_field="paraphrased")
        embed_jsonl(GEN_DIR / "llm_to_casual.jsonl",
                    source_name="llm_to_casual", text_field="paraphrased")
    elif args.task == "all":
        generate_rationales(n_dilemmas=args.n, batch_size=args.batch)
        free_gpu_memory()
        paraphrase(n_each=args.n_each, batch_size=args.batch)
        free_gpu_memory()
        embed_jsonl(GEN_DIR / "qwen25_3b_rationales.jsonl",
                    source_name="qwen25_3b_modern", text_field="rationale")
        embed_jsonl(GEN_DIR / "human_to_formal.jsonl",
                    source_name="human_to_formal", text_field="paraphrased")
        embed_jsonl(GEN_DIR / "llm_to_casual.jsonl",
                    source_name="llm_to_casual", text_field="paraphrased")

    _flush("ALL DONE")


if __name__ == "__main__":
    main()
