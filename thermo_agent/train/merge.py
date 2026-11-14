"""Merge a LoRA adapter into its base model so GRPO can start from the SFT result.
    python -m thermo_agent.train.merge ADAPTER_DIR MERGED_DIR [--model Qwen/Qwen3.5-4B] [--dtype bfloat16]
Untested in the authoring environment (no GPU)."""
import argparse
from .config import BASE_MODEL


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("adapter"); ap.add_argument("out"); ap.add_argument("--model", default=BASE_MODEL); ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    try: base = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype))
    except TypeError: base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype))
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    merged.save_pretrained(a.out); AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)


if __name__ == "__main__":
    main()
