"""Check a combined corpus against the model's real tokenizer and the SFT sequence length.

`combine` already drops rows that do not fit, but the trainer is the one that truncates: this tool re-measures the
written files with the same tokenizer and fails (exit 1) if any row would be cut. It is also the source of the
corpus size statistics quoted in the paper.

    python -m thermo_agent.train.token_audit data/sft --max-length 4096 --out reports/token_audit.json

The tokenizer is `$THERMO_TOKENIZER` or the cached `Qwen/Qwen3.5-4B` `tokenizer.json`; `--reserve` covers the chat
markers that `apply_chat_template` adds and that are not part of the message contents measured here.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from ..synth.combine import TOKEN_RESERVE, tokenizer, tokenizer_path


def _q(xs: list[int], f: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else 0


def audit(data_dir, max_length: int = 4096, reserve: int = TOKEN_RESERVE, tasks=("repair", "ledger", "audit", "design")) -> dict:
    tok = tokenizer()
    if tok is None:
        raise SystemExit("no tokenizer: set THERMO_TOKENIZER or cache the Qwen3.5 tokenizer.json")
    root = pathlib.Path(data_dir)
    report = {"data": str(root), "tokenizer": str(tokenizer_path()), "reserve": reserve,
              "max_length": max_length, "files": {}, "over": []}
    for task in tasks:
        for p in sorted(root.glob(f"{task}_*.jsonl")):
            split = p.stem[len(task) + 1:]
            prompts, comps, totals, harness, memory = [], [], [], 0, 0
            for line in p.open(encoding="utf-8"):
                row = json.loads(line)
                pr = len(tok.encode("".join(m.get("content", "") for m in row["prompt"]), add_special_tokens=False).ids)
                co = len(tok.encode(row["completion"][0]["content"], add_special_tokens=False).ids)
                prompts.append(pr)
                comps.append(co)
                totals.append(pr + co + reserve)
                harness += row.get("harness", "none") != "none"
                memory += bool(row.get("memory_notes"))
                if pr + co + reserve > max_length:
                    report["over"].append({"id": row.get("id"), "task": task, "split": split,
                                           "tokens": pr + co + reserve})
            report["files"][f"{task}_{split}"] = {
                "n": len(totals), "prompt_p50": _q(prompts, .5), "prompt_p99": _q(prompts, .99),
                "prompt_max": max(prompts, default=0), "completion_p50": _q(comps, .5),
                "total_p99": _q(totals, .99), "total_max": max(totals, default=0),
                "over": sum(1 for t in totals if t > max_length), "harness_rows": harness, "memory_rows": memory,
            }
    report["rows"] = sum(f["n"] for f in report["files"].values())
    report["over_total"] = len(report["over"])
    report["fits"] = report["over_total"] == 0
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--reserve", type=int, default=TOKEN_RESERVE)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = audit(a.data, a.max_length, a.reserve)
    if a.out:
        p = pathlib.Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(json.dumps({"data": r["data"], "rows": r["rows"], "max_length": r["max_length"],
                      "tokenizer": r["tokenizer"], "over_total": r["over_total"], "fits": r["fits"],
                      "worst": sorted(r["files"].items(), key=lambda kv: -kv[1]["total_max"])[:3],
                      "over_examples": r["over"][:5]}, indent=1))
    raise SystemExit(0 if r["fits"] else 1)


if __name__ == "__main__":
    main()
