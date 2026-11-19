"""Stage 1: SFT on verified expert completions (full fp16 LoRA, optional DDP).
    python -m thermo_agent.train.sft DATA_DIR OUT_DIR [--tasks repair design audit ledger] [--model Qwen/Qwen3.5-4B]
        [--dtype bfloat16|float16] [--qlora] [--max-steps N] [--limit-rows N] [--print-sample]
        [--per-device-batch 2] [--grad-accum 8] [--grad-norm 1.0] [--num-workers 2]
Dual T4 (see docs/DUAL_T4.md):
    torchrun --standalone --nproc_per_node=2 --module thermo_agent.train.sft DATA_DIR runs/sft \
        --dtype float16 --per-device-batch 2 --grad-accum 8
DDP replicates the frozen base model; only LoRA gradients cross PCIe. fp16 runs under the Trainer's AMP
GradScaler, so loss overflows are rescaled automatically, and gradients are clipped at --grad-norm before
each optimizer step. Checkpoints are written every --save-steps steps (default 200) and a rerun of the same
command resumes from the newest one, because one epoch does not fit in a single Kaggle session.
Prompts are rendered with the tokenizer exactly as GRPO and inference will render them (see train/render.py).
If the checkpoint loads only through an image-text class, load that class and pass the model object to
SFTTrainer instead of the string."""
from __future__ import annotations
import argparse, json, os, pathlib, random, re

from . import ddp
from .checkpoint import latest
from .config import BASE_MODEL, SftSettings, supported
from . import hf_sync
from .render import CHAT_KW, render_pair, prefix_report  # CHAT_KW is the shared enable_thinking value; render_pair applies it to every row we tokenize


# Tool results are produced by the environment at inference time, so the student must never be
# trained to predict them. A trajectory is stored as one assistant message with these inline
# tags (see teacher.agent_graph.sft_completion); the tags and everything between them are masked.
# Our XML traces use <tool_result>; Qwen3.5's native chat template renders tool results as
# <tool_response>, so both forms are masked. Tool CALLS stay trainable: only the results are
# environment output.
_TOOL_RESULT_TAGS = {
    "tool_result": ("<tool_result>", "</tool_result>"),
    "tool_response": ("<tool_response>", "</tool_response>"),
}
_TOOL_RESULT_OPEN = _TOOL_RESULT_TAGS["tool_result"][0]
_TOOL_RESULT_CLOSE = _TOOL_RESULT_TAGS["tool_result"][1]
_TOOL_RESULT_TAG = re.compile("|".join(
    re.escape(tag) for pair in _TOOL_RESULT_TAGS.values() for tag in pair))


def _tool_result_spans(text: str) -> list[tuple[int, int]]:
    """Character spans ``[start, end)`` of every tool-result block.

    Both our XML ``<tool_result>`` and Qwen's native ``<tool_response>`` forms are recognised.
    The opening and closing tags are included. Nesting is tracked separately per tag name with
    a depth counter; a block that never closes (depth > 0 at the end) masks from its opening tag
    to the end of the text. A stray closing tag with no matching open is ignored.
    """
    spans: list[tuple[int, int]] = []
    opens = {pair[0]: name for name, pair in _TOOL_RESULT_TAGS.items()}
    closes = {pair[1]: name for name, pair in _TOOL_RESULT_TAGS.items()}
    depth = {name: 0 for name in _TOOL_RESULT_TAGS}
    start = 0
    for match in _TOOL_RESULT_TAG.finditer(text):
        tag = match.group(0)
        if tag in opens:
            name = opens[tag]
            if depth[name] == 0:
                start = match.start()
            depth[name] += 1
        else:
            name = closes[tag]
            if depth[name]:
                depth[name] -= 1
                if depth[name] == 0:
                    spans.append((start, match.end()))
    for name, level in depth.items():
        if level:
            spans.append((max(0, start), len(text)))
    return sorted(set(spans))


def mask_tool_result_spans(text, offsets, labels):
    """Mask every token that lies in a ``<tool_result>`` span by setting its label to ``-100``.

    ``text`` is the exact string that was tokenized, ``offsets`` is the fast tokenizer's
    ``offset_mapping`` (one ``(start, end)`` character span per token, aligned 1:1 with
    ``labels``), and ``labels`` is a list or tensor of token ids / ``-100`` entries. A token is
    masked when its half-open character span ``[start, end)`` intersects any tool-result span
    (tags included), so tags split across several tokens are covered too. Multiple spans are
    handled; nested or unclosed blocks mask to end-of-text. Plain text is returned unchanged.
    The input is never mutated (a tensor is cloned, a list is copied).
    """
    spans = _tool_result_spans(text)
    if not spans:
        return labels
    out = labels.clone() if hasattr(labels, "clone") else list(labels)
    for i, (start, end) in enumerate(offsets):
        if start is None or end is None:  # fast tokenizers emit (0, 0) for added special tokens
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            out[i] = -100
    return out


def _common_prefix_length(a, b) -> int:
    """Length of the longest shared prefix of two token-id sequences (TRL's prompt boundary)."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def preprocess_row(tok, row: dict) -> dict:
    """Tokenize one SFT row and build labels with the prompt and tool results masked.

    Produces ``input_ids`` and fully-formed ``labels`` so TRL's ``SFTTrainer`` skips its own
    tokenization and label building. This keeps ``completion_only_loss=True`` (the prompt is
    masked) while additionally masking tool-result spans; the chunked loss already ignores
    ``-100`` labels and normalizes by the unmasked token count over the whole batch.
    """
    pair = render_pair(tok, row)
    prompt, completion = pair["prompt"], pair["completion"]
    eos = getattr(tok, "eos_token", None)
    if eos and not completion.endswith(eos):  # TRL appends EOS to plain-text completions
        completion += eos
    text = prompt + completion
    prompt_ids = tok(prompt)["input_ids"]
    encoded = tok(text, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    prompt_len = _common_prefix_length(prompt_ids, input_ids)
    labels = [-100] * prompt_len + list(input_ids[prompt_len:])
    return {"input_ids": list(input_ids),
            "labels": mask_tool_result_spans(text, encoded["offset_mapping"], labels)}


def preprocess_rows(tok, rows: list[dict]):
    """Tokenize rows into a dataset ready for ``SFTTrainer`` (with tool results masked)."""
    from datasets import Dataset
    return Dataset.from_list([preprocess_row(tok, row) for row in rows])


def read_rows(data_dir, tasks, split, seed=0, limit=None):
    rows = []
    for t in tasks:
        p = pathlib.Path(data_dir) / f"{t}_{split}.jsonl"
        if p.exists(): rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def _report_topology(dtype: str) -> None:
    """Print what this rank sees and refuse a dtype the hardware cannot run."""
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible; SFT needs a GPU")
    capability = torch.cuda.get_device_capability(0)
    ddp.check_dtype_supported(dtype, capability)
    world = ddp.world_size_from_env()
    if torch.cuda.device_count() < world:
        raise SystemExit(f"torchrun started {world} processes but only {torch.cuda.device_count()} GPUs are visible")
    if dtype == "float16":
        # accumulate matmuls in fp32: slower by a few percent, much less overflow-prone on Turing
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    print(f"[rank {ddp.local_rank_from_env()}/{world}] {ddp.topology_line()}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--tasks", nargs="+", default=["repair"], choices=["repair", "design", "audit", "ledger"])
    ap.add_argument("--model", default=BASE_MODEL); ap.add_argument("--dtype", default="float16", choices=list(ddp.DTYPES))
    ap.add_argument("--qlora", action="store_true", help="4-bit base weights; only for cards under 10 GB, not the T4 path")
    ap.add_argument("--max-steps", type=int, default=-1); ap.add_argument("--limit-rows", type=int); ap.add_argument("--print-sample", action="store_true")
    ap.add_argument("--per-device-batch", type=int); ap.add_argument("--grad-accum", type=int); ap.add_argument("--grad-norm", type=float)
    ap.add_argument("--max-length", type=int, help="override the 4096-token window (smoke runs on a small card)")
    ap.add_argument("--save-steps", type=int, help="checkpoint interval; a rerun of the same command resumes automatically")
    ap.add_argument("--loss-token-chunk", type=int, default=int(os.environ.get("LOSS_TOKENS", "512")),
                    help="tokens scored per cross-entropy tile; the full (tokens x 248k vocab) logits never exist")
    ap.add_argument("--loss-vocab-chunk", type=int, default=int(os.environ.get("LOSS_VOCAB", "8192")),
                    help="vocabulary entries per cross-entropy tile (with --loss-token-chunk, bounds peak memory)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--hf-sync", choices=("auto", "on", "off"), default=os.environ.get("THERMO_HF_SYNC", "auto"),
                    help="push full resume state to the private HF repo after each save (auto: only when a token exists)")
    ap.add_argument("--hf-repo", default=None, help="must be <token owner>/thermo-agent-checkpoints")
    a = ap.parse_args(); s = SftSettings()
    if a.per_device_batch: s.per_device_batch = a.per_device_batch
    if a.grad_accum: s.grad_accum = a.grad_accum
    if a.grad_norm is not None: s.max_grad_norm = a.grad_norm
    if a.save_steps: s.save_steps = s.eval_steps = a.save_steps
    if a.max_length: s.max_length = a.max_length
    world = ddp.world_size_from_env()
    if a.qlora and world > 1:
        raise SystemExit("QLoRA + DDP is not supported here: use full fp16 LoRA (drop --qlora), as in docs/DUAL_T4.md")
    if a.print_sample:
        from transformers import AutoTokenizer
        tr = read_rows(a.data, a.tasks, "train", limit=1)
        if not tr: raise SystemExit(f"no training rows for {a.tasks} in {a.data}")
        ex = tr[0]; tok = AutoTokenizer.from_pretrained(a.model); pair = render_pair(tok, ex)
        print("PROMPT TAIL:", repr(pair["prompt"][-200:])); print("COMPLETION HEAD:", repr(pair["completion"][:200]))
        print("PREFIX CHECK:", json.dumps(prefix_report(tok, ex), indent=1)); return
    _report_topology(a.dtype)

    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig
    from .trainers import build_sft_trainer
    tok = AutoTokenizer.from_pretrained(a.model)
    tr, va = read_rows(a.data, a.tasks, "train", limit=a.limit_rows), read_rows(a.data, a.tasks, "val", limit=a.limit_rows)
    if not tr: raise SystemExit(f"no training rows for {a.tasks} in {a.data}")
    # Tokenize here so labels already mask the prompt AND every <tool_result> span. TRL sees
    # `input_ids` + `labels` and therefore skips its own tokenization/label building.
    train = preprocess_rows(tok, tr); val = preprocess_rows(tok, va) if va else None
    init = ddp.model_init_kwargs(a.dtype)
    if a.qlora:
        import torch
        from transformers import BitsAndBytesConfig
        init["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, a.dtype))
    cfg = SFTConfig(**supported(SFTConfig, output_dir=a.out, learning_rate=s.lr, num_train_epochs=s.epochs, max_steps=a.max_steps, max_length=s.max_length,
                    per_device_train_batch_size=s.per_device_batch, gradient_accumulation_steps=s.grad_accum, max_grad_norm=s.max_grad_norm,
                    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, dataloader_num_workers=a.num_workers,
                    logging_steps=5, eval_strategy="steps" if val else "no", eval_steps=s.eval_steps,
                    save_strategy="steps", save_steps=s.save_steps, save_total_limit=2,
                    completion_only_loss=True, model_init_kwargs=init, report_to="none", seed=0,
                    fp16=(a.dtype == "float16"), bf16=(a.dtype == "bfloat16"),
                    gradient_as_bucket_view=True, ddp_find_unused_parameters=False, ddp_bucket_cap_mb=64,
                    optim="adamw_torch"))
    import torch
    from transformers import AutoModelForCausalLM
    # Load explicitly in fp16: TRL's string model loader can silently fall back to fp32, which does not fit a T4.
    base_model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=getattr(torch, a.dtype),
        **{k: v for k, v in init.items() if k not in ("dtype", "torch_dtype")})
    params = sum(p.numel() for p in base_model.parameters())
    if torch.cuda.is_available():
        print(f"[memory] {params / 1e9:.2f}B params, weight dtype {next(base_model.parameters()).dtype}, "
              f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB on {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[attn] requested={init.get('attn_implementation')} "
              f"resolved={getattr(getattr(base_model, 'config', None), '_attn_implementation', None)} "
              f"checkpointing={bool(getattr(base_model, 'is_gradient_checkpointing', False))}", flush=True)
    lora = LoraConfig(r=s.lora.r, lora_alpha=s.lora.alpha, lora_dropout=s.lora.dropout, target_modules=s.lora.target_modules, task_type="CAUSAL_LM")
    repo, callback = hf_sync.prepare(a.hf_repo, "sft", a.out, a.hf_sync)
    resume = latest(a.out)
    if resume:
        print(f"resuming SFT from {resume}", flush=True)
    hooks = [callback] if callback else []
    build_sft_trainer(model=base_model, args=cfg, train_dataset=train, eval_dataset=val, peft_config=lora,
                      processing_class=tok, callbacks=hooks,
                      loss_token_chunk=a.loss_token_chunk, loss_vocab_chunk=a.loss_vocab_chunk
                      ).train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    main()
