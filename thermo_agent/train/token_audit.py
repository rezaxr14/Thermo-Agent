"""Check a combined corpus against the model's real tokenizer and the SFT sequence length.

`combine` already drops rows that do not fit, but the trainer is the one that truncates: this tool re-measures the
written files with the same tokenizer and fails (exit 1) if any row would be cut. It is also the source of the
corpus size statistics quoted in the paper.

    python -m thermo_agent.train.token_audit data/sft --max-length 8192 --out reports/token_audit.json

The tokenizer is `$THERMO_TOKENIZER` or the cached `Qwen/Qwen3.5-4B` `tokenizer.json`; `--reserve` covers the chat
markers that `apply_chat_template` adds and that are not part of the message contents measured here.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from ..synth.combine import TOKEN_RESERVE, tokenizer, tokenizer_path
from .sft import _tool_result_spans

# Rows longer than this are dropped by `synth.combine` before they reach the trainer; this audit
# counts them so a corpus can be checked without silently truncating a long tool trajectory.
TOOL_TOKEN_GATE = 8000


def tool_result_tokens(tok, text: str) -> int:
    """Number of tokens in `text` whose character offsets fall inside a <tool_result> span.

    These are exactly the tokens the SFT label mask sets to -100 (tags included), measured with
    the same real tokenizer used for the rest of the audit.
    """
    spans = _tool_result_spans(text)
    if not spans:
        return 0
    return sum(1 for s, e in tok.encode(text, add_special_tokens=False).offsets
               if any(s < b and e > a for a, b in spans))


def _q(xs: list[int], f: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else 0


def audit(data_dir, max_length: int = 8192, reserve: int = TOKEN_RESERVE, tasks=("repair", "ledger", "audit", "design")) -> dict:
    tok = tokenizer()
    if tok is None:
        raise SystemExit("no tokenizer: set THERMO_TOKENIZER or cache the Qwen3.5 tokenizer.json")
    root = pathlib.Path(data_dir)
    report = {"data": str(root), "tokenizer": str(tokenizer_path()), "reserve": reserve,
              "max_length": max_length, "tool_token_gate": TOOL_TOKEN_GATE,
              "files": {}, "over": []}
    for task in tasks:
        # Frontier dirs name their files `frontier_<task>_<split>.jsonl`; without this second
        # pattern the audit printed `rows 0 / fits true` for a full frontier corpus.
        for p, kind in [(q, "") for q in sorted(root.glob(f"{task}_*.jsonl"))] + \
                       [(q, "frontier_") for q in sorted(root.glob(f"frontier_{task}_*.jsonl"))]:
            split = p.stem.split(f"{task}_", 1)[1]
            prompts, comps, totals, harness, memory = [], [], [], 0, 0
            tool_masked, tool_rows, dropped = 0, 0, 0
            for line in p.open(encoding="utf-8"):
                row = json.loads(line)
                pr = len(tok.encode("".join(m.get("content", "") for m in row["prompt"]), add_special_tokens=False).ids)
                completion_text = row["completion"][0]["content"]
                co = len(tok.encode(completion_text, add_special_tokens=False).ids)
                masked = tool_result_tokens(tok, completion_text)
                tool_masked += masked
                tool_rows += bool(masked)
                prompts.append(pr)
                comps.append(co)
                total = pr + co + reserve
                totals.append(total)
                harness += row.get("harness", "none") != "none"
                memory += bool(row.get("memory_notes"))
                if total > TOOL_TOKEN_GATE:  # dropped by combine, never truncated by the trainer
                    dropped += 1
                if total > max_length:
                    report["over"].append({"id": row.get("id"), "task": task, "split": split,
                                           "tokens": total})
            report["files"][f"{kind}{task}_{split}"] = {
                "n": len(totals), "prompt_p50": _q(prompts, .5), "prompt_p99": _q(prompts, .99),
                "prompt_max": max(prompts, default=0), "completion_p50": _q(comps, .5),
                "total_p99": _q(totals, .99), "total_max": max(totals, default=0),
                "over": sum(1 for t in totals if t > max_length), "harness_rows": harness, "memory_rows": memory,
                "tool_result_tokens": tool_masked, "tool_result_rows": tool_rows,
                "dropped_8000": dropped,
            }
    report["rows"] = sum(f["n"] for f in report["files"].values())
    report["tool_result_tokens"] = sum(f["tool_result_tokens"] for f in report["files"].values())
    report["tool_result_rows"] = sum(f["tool_result_rows"] for f in report["files"].values())
    report["dropped_8000_total"] = sum(f["dropped_8000"] for f in report["files"].values())
    report["over_total"] = len(report["over"])
    report["fits"] = report["over_total"] == 0
    return report


def summary_table(report: dict) -> str:
    """Printable table of per-file tool-result and 8k-drop counts (plus totals)."""
    cols = (("file", 26), ("rows", 7), ("tool_tok", 9), ("tool_rows", 10), (">8000", 7), ("total_max", 10))
    lines = ["  ".join(f"{name:<{w}}" if i == 0 else f"{name:>{w}}" for i, (name, w) in enumerate(cols))]
    for name in sorted(report["files"]):
        f = report["files"][name]
        vals = (name, f["n"], f["tool_result_tokens"], f["tool_result_rows"], f["dropped_8000"], f["total_max"])
        lines.append("  ".join(f"{v:<{w}}" if i == 0 else f"{v:>{w}}" for i, (v, (_, w)) in enumerate(zip(vals, cols))))
    totals = ("TOTAL", report["rows"], report["tool_result_tokens"], report["tool_result_rows"],
              report["dropped_8000_total"], max((f["total_max"] for f in report["files"].values()), default=0))
    lines.append("  ".join(f"{v:<{w}}" if i == 0 else f"{v:>{w}}" for i, (v, (_, w)) in enumerate(zip(totals, cols))))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--reserve", type=int, default=TOKEN_RESERVE)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = audit(a.data, a.max_length, a.reserve)
    if a.out:
        p = pathlib.Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(summary_table(r))
    print(json.dumps({"data": r["data"], "rows": r["rows"], "max_length": r["max_length"],
                      "tokenizer": r["tokenizer"], "over_total": r["over_total"], "fits": r["fits"],
                      "tool_result_tokens": r["tool_result_tokens"], "tool_result_rows": r["tool_result_rows"],
                      "dropped_8000_total": r["dropped_8000_total"],
                      "worst": sorted(r["files"].items(), key=lambda kv: -kv[1]["total_max"])[:3],
                      "over_examples": r["over"][:5]}, indent=1))
    raise SystemExit(0 if r["fits"] else 1)


if __name__ == "__main__":
    main()
