"""Merge a LoRA adapter into its base model so GRPO can start from the SFT result.
    python -m thermo_agent.train.merge ADAPTER_DIR MERGED_DIR [--model Qwen/Qwen3.5-4B] [--dtype bfloat16]
        [--hf-sync auto|on|off] [--hf-prefix merged/sft]
With a token available the merged weights are pushed to the project's private HF repo, so a fresh Kaggle session
can start GRPO with `--init hf:merged/sft` without repeating SFT and the merge. Untested here (no GPU)."""
import argparse
import os

from . import hf_sync
from .config import BASE_MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter"); ap.add_argument("out")
    ap.add_argument("--model", default=BASE_MODEL); ap.add_argument("--dtype", default="float16")
    ap.add_argument("--hf-sync", choices=("auto", "on", "off"), default=os.environ.get("THERMO_HF_SYNC", "auto"))
    ap.add_argument("--hf-prefix", default="merged/sft")
    ap.add_argument("--hf-repo", default=None)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    try: base = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype))
    except TypeError: base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype))
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    merged.save_pretrained(a.out); AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
    if a.hf_sync == "off":
        return
    try:
        repo = hf_sync.ensure_private_repo(a.hf_repo)
    except SystemExit:
        if a.hf_sync == "auto":
            print("[hf_sync] no HF token; merged model stays local")
            return
        raise
    print(f"[hf_sync] pushing merged model to {repo}/{a.hf_prefix}", flush=True)
    hf_sync.push_merged(a.out, repo, prefix=a.hf_prefix)


if __name__ == "__main__":
    main()
