"""Build the Kaggle smoke kernel: the training notebook with one extra cell that sets small-run env vars.

    python scripts/make_smoke_kernel.py --steps 20 --rows 64 --task ledger --ngpu 1 --batch 1

The generated `kaggle/train_kernel_smoke/train.ipynb` is a copy of the maintained `kaggle/train.ipynb`, so the smoke
run exercises exactly the cells the full run uses; only the environment differs.
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="20")
    ap.add_argument("--rows", default="64")
    ap.add_argument("--task", default="ledger")
    ap.add_argument("--ngpu", default="1")
    ap.add_argument("--batch", default="1")
    ap.add_argument("--accum", default="8")
    ap.add_argument("--maxlen", default="")
    ap.add_argument("--sft-data", default="sft_tools",
                    help="corpus staged as SFT_DATA; sft_tools is the multi-turn tool corpus")
    ap.add_argument("--grpo-data", default="grpo_tools",
                    help="corpus staged as GRPO_DATA; grpo_tools holds the matching tool prompts")
    ap.add_argument("--out", default="kaggle/train_kernel_smoke")
    a = ap.parse_args()

    env = {"STEPS": a.steps, "ROWS": a.rows, "TASK": a.task, "NGPU": a.ngpu,
           "SFT_BATCH": a.batch, "SFT_ACCUM": a.accum,
           "SFT_DATA": a.sft_data, "GRPO_DATA": a.grpo_data}
    if a.maxlen:
        env["MAXLEN"] = a.maxlen
    source = ["import os\n"] + [f'os.environ["{k}"]="{v}"\n' for k, v in env.items()]

    nb = json.loads(pathlib.Path("kaggle/train.ipynb").read_text(encoding="utf-8"))
    nb["cells"].insert(0, {"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": source})
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.ipynb").write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("smoke env:", env, "cells:", len(nb["cells"]))


if __name__ == "__main__":
    main()
