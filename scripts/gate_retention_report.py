"""Exact 8,192-gate retention per task and tier for a deterministic corpus.

Measures the same sequence training sees: the row's prompt rendered through
``apply_chat_template(tools=...)`` with that row's tool schemas, plus the stored
completion, plus the 96-token chat reserve. Reports attempted/kept per task and
per tier, lists every over-gate id, and reports schema tokens per row.

    .venv/Scripts/python.exe scripts/gate_retention_report.py \
        --corpus data/sft_tools_v3 --out reports/gate_retention_step3.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from thermo_agent.agent import tool_protocol  # noqa: E402
from thermo_agent.synth.combine import TOKEN_RESERVE, tokenizer_path  # noqa: E402

TASKS = ("repair", "design", "audit", "ledger")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="data/sft_tools_v3")
    ap.add_argument("--prompt-variant", default="harness_tools",
                    choices=("plain", "harness_tools"),
                    help="measure the prompt training actually renders")
    ap.add_argument("--gate", type=int, default=8192)
    ap.add_argument("--out", default="reports/gate_retention_step3.json")
    args = ap.parse_args(argv)

    from scripts.build_tool_corpus import final_prompt

    tok = AutoTokenizer.from_pretrained(str(pathlib.Path(tokenizer_path()).parent),
                                        local_files_only=True)

    def n(text: str) -> int:
        return len(tok.encode(str(text), add_special_tokens=False))

    def rendered(messages, tools) -> int:
        return n(tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                         add_generation_prompt=True))

    rows = []
    for path in sorted(pathlib.Path(args.corpus).glob("*_train.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]

    per_task = defaultdict(lambda: {"attempted": 0, "kept": 0, "dropped": []})
    per_tier = defaultdict(lambda: {"attempted": 0, "kept": 0, "dropped": []})
    tokens_by_task = defaultdict(list)
    schema_tokens = []
    for row in rows:
        task = str(row.get("task") or "repair")
        tier = str(row.get("tier") or "unknown")
        prompt, _ = final_prompt(row["prompt"], task, row, args.prompt_variant)
        completion = row.get("completion")
        if isinstance(completion, str):
            completion = [{"role": "assistant", "content": completion}]
        elif not isinstance(completion, list):
            completion = [{"role": "assistant", "content": row.get("sft_completion") or ""}]
        tools = tool_protocol.tool_schemas(task)
        base = rendered(prompt, None)
        with_tools = rendered(prompt, tools)
        # Training text is prompt + completion (the template suffix): one full render.
        full = tok.apply_chat_template(prompt + completion, tools=tools or None,
                                       tokenize=False, enable_thinking=True)
        total = n(full) + TOKEN_RESERVE
        schema_tokens.append(with_tools - base)
        per_task[task]["attempted"] += 1
        per_tier[f"{task}/{tier}"]["attempted"] += 1
        tokens_by_task[task].append(total)
        if total > args.gate:
            entry = {"id": row.get("id"), "tokens": total, "tier": tier}
            per_task[task]["dropped"].append(entry)
            per_tier[f"{task}/{tier}"]["dropped"].append(entry)
        else:
            per_task[task]["kept"] += 1
            per_tier[f"{task}/{tier}"]["kept"] += 1

    def rate(bucket: dict) -> float:
        return round(bucket["kept"] / bucket["attempted"], 4) if bucket["attempted"] else 1.0

    report = {
        "corpus": args.corpus,
        "gate": args.gate,
        "reserve": TOKEN_RESERVE,
        "rows": len(rows),
        "kept": sum(b["kept"] for b in per_task.values()),
        "dropped": sum(len(b["dropped"]) for b in per_task.values()),
        "schema_tokens": {
            "min": min(schema_tokens) if schema_tokens else 0,
            "median": int(statistics.median(schema_tokens)) if schema_tokens else 0,
            "max": max(schema_tokens) if schema_tokens else 0,
        },
        "per_task": {t: {**per_task[t], "retention": rate(per_task[t])} for t in TASKS
                     if per_task[t]["attempted"]},
        "per_tier": {k: {**v, "retention": rate(v)} for k, v in sorted(per_tier.items())},
        "rendered_max_by_task": {t: max(v) for t, v in tokens_by_task.items() if v},
        "dropped_tiers": dict(Counter(d["tier"] for b in per_task.values()
                                      for d in b["dropped"])),
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
