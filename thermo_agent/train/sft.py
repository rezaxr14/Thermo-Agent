"""Stage 1: SFT on verified expert completions (LoRA / QLoRA), one adapter for one or several tasks.
    python -m thermo_agent.train.sft DATA_DIR OUT_DIR [--tasks repair design audit ledger] [--model Qwen/Qwen3.5-4B]
        [--dtype bfloat16|float16] [--qlora] [--max-steps N] [--limit-rows N] [--print-sample]
Prompts are rendered with the tokenizer exactly as GRPO and inference will render them (see train/render.py).
Not run in the authoring environment (no GPU). If the checkpoint loads only through an image-text class, load that class and
pass the model object to SFTTrainer instead of the string."""
from __future__ import annotations
import argparse, json, pathlib, random
from .config import BASE_MODEL, SftSettings
from .render import render_pair, prefix_report


def read_rows(data_dir, tasks, split, seed=0, limit=None):
    rows = []
    for t in tasks:
        p = pathlib.Path(data_dir) / f"{t}_{split}.jsonl"
        if p.exists(): rows += [json.loads(l) for l in open(p)]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--tasks", nargs="+", default=["repair"], choices=["repair", "design", "audit", "ledger"])
    ap.add_argument("--model", default=BASE_MODEL); ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1); ap.add_argument("--limit-rows", type=int); ap.add_argument("--print-sample", action="store_true")
    a = ap.parse_args(); s = SftSettings()
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    tok = AutoTokenizer.from_pretrained(a.model)
    tr, va = read_rows(a.data, a.tasks, "train", limit=a.limit_rows), read_rows(a.data, a.tasks, "val", limit=a.limit_rows)
    if not tr: raise SystemExit(f"no training rows for {a.tasks} in {a.data}")
    if a.print_sample:
        ex = tr[0]; pair = render_pair(tok, ex)
        print("PROMPT TAIL:", repr(pair["prompt"][-200:])); print("COMPLETION HEAD:", repr(pair["completion"][:200]))
        print("PREFIX CHECK:", json.dumps(prefix_report(tok, ex), indent=1)); return
    train = Dataset.from_list([render_pair(tok, r) for r in tr]); val = Dataset.from_list([render_pair(tok, r) for r in va]) if va else None
    init = {"dtype": a.dtype}
    if a.qlora:
        import torch
        from transformers import BitsAndBytesConfig
        init["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, a.dtype))
    cfg = SFTConfig(output_dir=a.out, learning_rate=s.lr, num_train_epochs=s.epochs, max_steps=a.max_steps, max_length=s.max_length,
                    per_device_train_batch_size=s.per_device_batch, gradient_accumulation_steps=s.grad_accum, gradient_checkpointing=True,
                    logging_steps=5, eval_strategy="epoch" if val else "no", save_strategy="epoch", completion_only_loss=True,
                    model_init_kwargs=init, report_to="none", fp16=(a.dtype == "float16"), bf16=(a.dtype == "bfloat16"))
    lora = LoraConfig(r=s.lora.r, lora_alpha=s.lora.alpha, lora_dropout=s.lora.dropout, target_modules=s.lora.target_modules, task_type="CAUSAL_LM")
    SFTTrainer(model=a.model, args=cfg, train_dataset=train, eval_dataset=val, peft_config=lora).train()


if __name__ == "__main__":
    main()
