#!/usr/bin/env python3
"""Pick deterministic, interesting traces from every stage for the paper figures.

    python paper/tools/collect_examples.py --datagen data/processed --sft data/sft \
        --frontier data/frontier --harness reports/harness_completions.jsonl \
        --rollouts runs/grpo_*/rollouts.jsonl --evals reports/*.completions.jsonl \
        --out paper/assets/examples.json

Nothing here grades or invents answers: every example is copied verbatim from a file produced by a previous stage,
with the source file recorded so a figure can be traced back to it. Selection is deterministic (sorted by id), so
regenerating the assets reproduces the same examples.
"""
from __future__ import annotations

import argparse
import json
import pathlib

SCHEMA = {
    "examples": [
        {"source": "datagen|sft|frontier|harness|grpo|eval", "id": "row id (or group id)",
         "file": "source file", "task": "repair|design|audit|ledger", "split": "dataset split if known",
         "fault": "fault type if known", "topology": "diagram label", "title": "short caption",
         "fields": "verbatim excerpts: prompt, diagnosis, edits, completion, reasoning, tool_calls, rewards"},
    ],
    "note": "All examples are measured outputs. No figure value is hand-written.",
}


def _read(path):
    p = pathlib.Path(path)
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def _excerpt(text, n=700):
    s = str(text or "")
    return s if len(s) <= n else s[:n] + " ..."


def _graph(row):
    try:
        p = json.loads(row["plant_broken"])
        return {"components": [c["type"] for c in p["components"]],
                "streams": len(p["streams"]), "name": p.get("name")}
    except Exception:
        return None


def from_datagen(dirs, limit=400):
    out = []
    for d in dirs:
        for path in sorted(pathlib.Path(d).glob("repair_test_*.jsonl")):
            split = path.stem[len("repair_"):]
            picked = {}
            for r in _read(path)[:limit]:
                ft = r.get("fault_type", "?")
                if ft not in picked or len(r.get("completion", "")) > len(picked[ft].get("completion", "")):
                    picked[ft] = r
            for ft, r in sorted(picked.items()):
                out.append({"source": "datagen", "id": r["id"], "file": str(path), "task": "repair",
                            "split": split, "fault": ft, "topology": r.get("topology"),
                            "title": f"Ground truth: {ft}", "graph": _graph(r),
                            "fields": {"diagnosis": _excerpt(r.get("completion", "")),
                                       "prompt_tail": _excerpt(r.get("prompt", [{}])[-1].get("content"), 300)}})
    return out


def from_sft(dirs, limit=600):
    out = []
    for d in dirs:
        for task in ("repair", "ledger"):
            path = pathlib.Path(d) / f"{task}_train.jsonl"
            rows = [r for r in _read(path)[:limit]
                    if r.get("harness") in ("agents+skills", "skills") and r.get("memory_notes")]
            for r in sorted(rows, key=lambda x: x["id"])[:2]:
                out.append({"source": "sft", "id": r["id"], "file": str(path), "task": task,
                            "split": "train", "fault": r.get("fault_type"),
                            "topology": r.get("topology"), "title": f"SFT row with harness and memory: {task}",
                            "fields": {"harness": r.get("harness"), "memory_notes": r.get("memory_notes"),
                                       "completion": _excerpt(r.get("completion", [{}])[0].get("content"))}})
    return out


def from_frontier(dirs):
    out = []
    for d in dirs:
        for path in sorted(pathlib.Path(d).glob("frontier_*_train.jsonl")):
            for r in _read(path)[:2]:
                out.append({"source": "frontier", "id": r["id"], "file": str(path),
                            "task": r.get("task"), "split": "train", "topology": r.get("topology"),
                            "title": "Frontier answer accepted by the tool suite",
                            "fields": {"validation": r.get("validation"),
                                       "completion": _excerpt(r.get("completion", [{}])[0].get("content"))}})
    return out


def from_harness(paths, k=3):
    out = []
    for path in paths:
        rows = _read(path)
        ranked = sorted(rows, key=lambda r: (-len(r.get("tool_calls") or []), str(r.get("id"))))[:k]
        for r in ranked:
            out.append({"source": "harness", "id": r.get("id"), "file": str(path), "task": r.get("task"),
                        "topology": r.get("topology"),
                        "title": f"Frontier in the harness ({len(r.get('tool_calls') or [])} tool calls)",
                        "fields": {"completion": _excerpt(r.get("completion")),
                                   "reasoning": _excerpt(r.get("reasoning")),
                                   "tool_calls": r.get("tool_calls"), "usage": r.get("usage")}})
    return out


def from_rollouts(paths, k=2):
    out = []
    for path in paths:
        rows = _read(path)
        groups = {}
        for r in rows:
            groups.setdefault(r.get("group"), []).append(r)
        for gid, group in sorted(groups.items(), key=lambda kv: str(kv[0]))[:k]:
            rewards = [x.get("rewards", []) for x in group]
            spread = max((max(v) - min(v) for v in rewards if v), default=0)
            best = max(group, key=lambda x: max(x.get("rewards") or [0]))
            out.append({"source": "grpo", "id": f"{gid}", "file": str(path), "task": group[0].get("stage"),
                        "title": "GRPO rollout selected by reward",
                        "fields": {"prompt": _excerpt(best.get("prompt")),
                                   "completions": [_excerpt(c) for c in best.get("completions", [])],
                                   "reward_functions": [x.get("reward_fn") for x in group],
                                   "rewards": rewards, "spread": round(spread, 3)}})
    return out


def from_evals(paths, k=2):
    out = []
    for path in paths:
        for r in _read(path)[:k]:
            out.append({"source": "eval", "id": r.get("id"), "file": str(path), "task": r.get("task"),
                        "title": "Model evaluation interaction", "fields": {"completion": _excerpt(r.get("completion"))}})
    return out


def collect(args):
    examples = (from_datagen(args.datagen) + from_sft(args.sft) + from_frontier(args.frontier)
                + from_harness(args.harness) + from_rollouts(args.rollouts) + from_evals(args.evals))
    return {"schema": SCHEMA, "n": len(examples), "examples": examples}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datagen", nargs="*", default=[])
    ap.add_argument("--sft", nargs="*", default=[])
    ap.add_argument("--frontier", nargs="*", default=[])
    ap.add_argument("--harness", nargs="*", default=[])
    ap.add_argument("--rollouts", nargs="*", default=[])
    ap.add_argument("--evals", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    result = collect(a)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out.parent / "examples_schema.json").write_text(json.dumps(SCHEMA, indent=1) + "\n", encoding="utf-8")
    print(f"{result['n']} examples -> {out}")


if __name__ == "__main__":
    main()
