"""Generate Qwen 2.5 3B Instruct rationales on the 1991 ArcticShift dilemmas
across four temperatures with K samples per dilemma.

Tries vLLM first for PagedAttention + continuous batching. Falls back to HF
transformers with num_return_sequences=K if vLLM is unavailable.

Output:
  data/generated/intervention/temperature/qwen25_3b_T{T}.jsonl
"""

import json
import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

EMBEDDINGS_DIR = Path("data/embeddings")
GEN_DIR = Path("data/generated/intervention/temperature")
GEN_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "Qwen/Qwen2.5-3B-Instruct"
TEMPERATURES = [0.3, 0.7, 1.0, 1.3]
K_PER_TEMP = {0.3: 3, 0.7: 5, 1.0: 8, 1.3: 12}
MAX_NEW_TOKENS = 180
TOP_P = 0.95
REP_PENALTY = 1.05
MAX_POST_CHARS = 1400
LOG_INTERVAL = 20
SEED = 42

RATIONALE_PROMPT = (
    "You are reading a story posted to Reddit's r/AmITheAsshole forum. "
    "The original poster is asking the community to judge their behavior. "
    "Read the post and write a 2-3 sentence judgment of the original poster (OP). "
    "Begin with one of the verdicts (NTA = Not the Asshole, YTA = You're the Asshole, "
    "ESH = Everyone Sucks Here, NAH = No Assholes Here), then give your reasoning.\n\n"
    "Post:\n{post}\n\n"
    "Your judgment:"
)

logger = logging.getLogger("gen_temp")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_h)


def _flush(msg):
    logger.info(msg)
    sys.stdout.flush()


def load_dilemma_ids():
    """Get the 1991 ArcticShift-qualified submission_ids."""
    meta = json.loads((EMBEDDINGS_DIR / "human_arctic_meta.json").read_text())
    ids = sorted({m["submission_id"] for m in meta})
    _flush(f"loaded {len(ids)} dilemma ids from human_arctic_meta.json")
    return ids


def load_posts(dilemma_ids):
    """Pull post text for each submission_id from the Sachdeva HF dataset."""
    from datasets import load_dataset
    _flush("loading Sachdeva HF dataset")
    ds = load_dataset("ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas", split="test")
    by_id = {}
    sids = ds["submission_id"]
    titles = ds["title"]
    selftexts = ds["selftext"]
    want = set(dilemma_ids)
    for i, sid in enumerate(sids):
        if sid not in want:
            continue
        title = titles[i] or ""
        body = selftexts[i] or ""
        post = (title + "\n\n" + body).strip()
        if len(post) > MAX_POST_CHARS:
            post = post[:MAX_POST_CHARS] + "..."
        if len(post.split()) < 10:
            continue
        by_id[sid] = post
    _flush(f"loaded posts for {len(by_id)}/{len(dilemma_ids)} dilemmas")
    return by_id


def out_path(T):
    return GEN_DIR / f"qwen25_3b_T{T:.1f}.jsonl"


def already_done(T):
    """Return set of submission_ids that already have all K samples written."""
    p = out_path(T)
    if not p.exists():
        return set()
    counts = defaultdict(int)
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
            counts[j["submission_id"]] += 1
        except Exception:
            continue
    K = K_PER_TEMP[T]
    return {sid for sid, c in counts.items() if c >= K}


def atomic_append_jsonl(path, lines):
    """Append-only is already atomic on POSIX for short writes; we use it directly."""
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def try_vllm():
    try:
        import vllm  # noqa: F401
        from vllm import LLM, SamplingParams  # noqa: F401
        return True
    except Exception as e:
        _flush(f"vLLM unavailable ({e!r}); will fall back to HF transformers")
        return False


# -----------------------------------------------------------------------------
# vLLM path
# -----------------------------------------------------------------------------
def run_vllm(posts_by_id, dilemma_order):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    _flush(f"loading Qwen tokenizer for chat template")
    tok = AutoTokenizer.from_pretrained(MODEL)

    _flush(f"loading vLLM engine: {MODEL} (bf16)")
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2560,
        seed=SEED,
    )

    # Pre-render chat-templated prompts once
    chat_prompts = {}
    for sid in dilemma_order:
        if sid not in posts_by_id:
            continue
        msgs = [{"role": "user", "content": RATIONALE_PROMPT.format(post=posts_by_id[sid])}]
        chat_prompts[sid] = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    _flush(f"built {len(chat_prompts)} chat prompts")

    for T in TEMPERATURES:
        K = K_PER_TEMP[T]
        path = out_path(T)
        done = already_done(T)
        todo = [sid for sid in dilemma_order if sid in chat_prompts and sid not in done]
        _flush(f"=== T={T} K={K} todo={len(todo)} done={len(done)} path={path.name}")
        if not todo:
            _flush(f"T={T} already complete, skipping")
            continue

        sp = SamplingParams(
            n=K,
            temperature=T,
            top_p=TOP_P,
            max_tokens=MAX_NEW_TOKENS,
            repetition_penalty=REP_PENALTY,
            seed=SEED,
        )

        t0 = time.time()
        n_done_total = 0
        BATCH = 64  # number of dilemmas per generate() call; vLLM handles internal continuous batching
        for s in range(0, len(todo), BATCH):
            chunk = todo[s : s + BATCH]
            prompts = [chat_prompts[sid] for sid in chunk]
            outs = llm.generate(prompts, sp, use_tqdm=False)
            lines = []
            for sid, request_output in zip(chunk, outs):
                for k_idx, completion in enumerate(request_output.outputs):
                    text = completion.text.strip()
                    if len(text.split()) < 6:
                        text = text + " [SHORT]"
                    lines.append(json.dumps({
                        "submission_id": sid,
                        "temperature": T,
                        "sample_idx": k_idx,
                        "rationale": text,
                    }))
            atomic_append_jsonl(path, lines)
            n_done_total += len(chunk)
            elapsed = time.time() - t0
            rate = n_done_total / elapsed if elapsed > 0 else 0
            if (s // BATCH) % LOG_INTERVAL == 0 or n_done_total >= len(todo):
                gen_rate = (n_done_total * K) / elapsed if elapsed > 0 else 0
                _flush(
                    f"T={T} dilemmas {n_done_total}/{len(todo)} "
                    f"({rate:.2f} dil/s, {gen_rate:.1f} gens/s) elapsed={elapsed:.0f}s"
                )
        _flush(f"DONE T={T} elapsed={(time.time()-t0):.0f}s lines_written={n_done_total*K}")

    del llm


# -----------------------------------------------------------------------------
# HF fallback path
# -----------------------------------------------------------------------------
def run_hf(posts_by_id, dilemma_order):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _flush(f"loading HF Qwen: {MODEL} (bf16)")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    pad = tok.pad_token_id or tok.eos_token_id

    PROMPT_BATCH = 8  # number of distinct prompts per forward
    for T in TEMPERATURES:
        K = K_PER_TEMP[T]
        path = out_path(T)
        done = already_done(T)
        todo = [sid for sid in dilemma_order if sid in posts_by_id and sid not in done]
        _flush(f"=== T={T} K={K} todo={len(todo)} done={len(done)} path={path.name}")
        if not todo:
            continue

        t0 = time.time()
        n_done_total = 0
        for s in range(0, len(todo), PROMPT_BATCH):
            chunk = todo[s : s + PROMPT_BATCH]
            msgs_list = [
                [{"role": "user", "content": RATIONALE_PROMPT.format(post=posts_by_id[sid])}]
                for sid in chunk
            ]
            prompts = [
                tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in msgs_list
            ]
            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                         max_length=2048).to("cuda")
            with torch.no_grad():
                outs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=T,
                    top_p=TOP_P,
                    repetition_penalty=REP_PENALTY,
                    num_return_sequences=K,
                    pad_token_id=pad,
                )
            in_len = inputs["input_ids"].shape[1]
            lines = []
            # outs shape: (PROMPT_BATCH * K, seq_len)
            for i, sid in enumerate(chunk):
                for k_idx in range(K):
                    seq = outs[i * K + k_idx][in_len:]
                    text = tok.decode(seq, skip_special_tokens=True).strip()
                    if len(text.split()) < 6:
                        text = text + " [SHORT]"
                    lines.append(json.dumps({
                        "submission_id": sid,
                        "temperature": T,
                        "sample_idx": k_idx,
                        "rationale": text,
                    }))
            atomic_append_jsonl(path, lines)
            n_done_total += len(chunk)
            elapsed = time.time() - t0
            rate = n_done_total / elapsed if elapsed > 0 else 0
            if (s // PROMPT_BATCH) % LOG_INTERVAL == 0 or n_done_total >= len(todo):
                gen_rate = (n_done_total * K) / elapsed if elapsed > 0 else 0
                vram = torch.cuda.memory_allocated() / 1e9
                _flush(
                    f"T={T} dilemmas {n_done_total}/{len(todo)} "
                    f"({rate:.2f} dil/s, {gen_rate:.1f} gens/s) "
                    f"elapsed={elapsed:.0f}s vram={vram:.1f}GB"
                )
        _flush(f"DONE T={T} elapsed={(time.time()-t0):.0f}s lines_written={n_done_total*K}")

    del model


def main():
    dilemma_ids = load_dilemma_ids()
    posts_by_id = load_posts(dilemma_ids)
    dilemma_order = [sid for sid in dilemma_ids if sid in posts_by_id]
    _flush(f"final dilemma count after post filter: {len(dilemma_order)}")

    if try_vllm():
        run_vllm(posts_by_id, dilemma_order)
    else:
        run_hf(posts_by_id, dilemma_order)
    _flush("ALL TEMPERATURES DONE")


if __name__ == "__main__":
    main()
