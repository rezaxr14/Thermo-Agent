"""Combine the four data sources into one SFT corpus.

Sources, all kept distinguishable by `source`:

1. ground truth  - the simulator-built rows (`{task}_{split}.jsonl`); for repair, each *train* row is emitted with
                   `--variants` different, equally true paraphrases of its diagnosis (the eval splits keep one
                   row per plant, so a model evaluation never counts the same plant twice);
2. frontier      - teacher answers that passed the tool suite (`frontier_{task}_{split}.jsonl`);
3. harness       - AGENTS.md plus the component skills for this plant, appended to the system prompt for a
                   deterministic fraction of rows so the model learns to use them without every prompt paying
                   for the full text;
4. memory        - hindsight failure notes retrieved for this plant and appended after the harness block.

Every row is measured before it is written and dropped (and counted) when it does not fit the SFT sequence
length, so nothing is ever silently truncated by the trainer. Two gates, both reported:

* characters, `--max-prompt-chars`: a cheap filter that keeps obviously oversized plants out of the tokenizer;
* tokens, `--max-prompt-tokens`: prompt + completion + a small reserve for the chat markers, measured with the
  real `Qwen/Qwen3.5-4B` tokenizer. This is the gate that matters; it is skipped (and reported as such) when
  `tokenizers` or the tokenizer file is unavailable.

Output files are `{task}_{split}.jsonl`, which is exactly what `thermo_agent.train.sft` and
`thermo_agent.train.grpo` read, plus `sft_report.json` for the provenance and the checks.

    python -m thermo_agent.synth.combine data/processed data/frontier data/memory data/sft --variants 4
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib

from ..agent import harness
from ..agent.dsl import plant_to_text
from ..core.schema import Plant
from . import paraphrase
from ..teacher import memory, validate

TASKS = ("repair", "ledger", "audit", "design")
VARIANTS = {"repair": True, "ledger": False, "audit": False, "design": False}
SPLITS = {
    "repair": ("train", "val", "test_id", "test_ood_fault", "test_ood_topology"),
    "ledger": ("train", "val", "test_id", "test_ood_fault", "test_ood_topology"),
    "audit": ("train", "val", "test_id", "test_ood_fault", "test_ood_topology"),
    "design": ("train", "val", "test_id", "test_ood_climate", "test_ood_scale", "test_ood_load"),
}
# Harness text is expensive in tokens: AGENTS.md is 4.7k characters and the component skills add up to 12k. An
# audit plant can carry many components, so audit uses the skill block only (exergy-audit plus the plant's own
# component skills); the other tasks fit AGENTS.md plus their selected skills inside the 4096-token budget.
HARNESS_MODES = {"repair": "agents+skills", "audit": "skills", "ledger": "agents+skills", "design": "agents+skills"}
# Chat markers (`<|im_start|>system\n`, `<|im_end|>\n`, ...) and the generation prompt are not part of the message
# contents that are measured here; 96 tokens covers them for a system+user prompt with room to spare.
TOKEN_RESERVE = 96
TOKENIZER_ENV = "THERMO_TOKENIZER"
DEFAULT_TOKENIZER = pathlib.Path.home() / ".cache" / "thermo" / "qwen3.5-4b" / "tokenizer.json"

_TOKENIZER = None
_TOKENIZER_TRIED = False


def tokenizer_path() -> pathlib.Path | None:
    """`$THERMO_TOKENIZER` if set, else the cached Qwen3.5 tokenizer, else None."""
    for p in (os.environ.get(TOKENIZER_ENV, ""), str(DEFAULT_TOKENIZER)):
        if p and pathlib.Path(p).is_file():
            return pathlib.Path(p)
    return None


def tokenizer():
    """The real tokenizer, or None when the optional dependency or the file is missing."""
    global _TOKENIZER, _TOKENIZER_TRIED
    if not _TOKENIZER_TRIED:
        _TOKENIZER_TRIED = True
        path = tokenizer_path()
        if path is not None:
            try:
                from tokenizers import Tokenizer
            except ImportError:
                pass
            else:
                _TOKENIZER = Tokenizer.from_file(str(path))
    return _TOKENIZER


class Budget:
    """Characters and tokens for one prompt, measured the same way for every row."""

    def __init__(self, max_chars: int = 11500, max_tokens: int = 4000):
        self.max_chars, self.max_tokens = max_chars, max_tokens
        self.tok = tokenizer()
        self.max_kept = 0
        self.max_candidate = 0
        self.dropped_chars = 0
        self.dropped_tokens = 0

    def ntok(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def check(self, messages: list[dict], completion: str) -> tuple[bool, str, int | None]:
        """(fits, reason, tokens). `tokens` is None when no tokenizer is available."""
        if self.max_chars and _chars(messages) > self.max_chars:
            self.dropped_chars += 1
            return False, "chars", None
        if self.tok is None or not self.max_tokens:
            return True, "", None
        n = self.ntok("".join(m.get("content", "") for m in messages)) + self.ntok(completion) + TOKEN_RESERVE
        self.max_candidate = max(self.max_candidate, n)
        if n > self.max_tokens:
            self.dropped_tokens += 1
            return False, "tokens", n
        self.max_kept = max(self.max_kept, n)
        return True, "", n

    def report(self) -> dict:
        return {"max_prompt_chars": self.max_chars, "max_tokens": self.max_tokens,
                "token_reserve": TOKEN_RESERVE, "tokenizer": str(tokenizer_path()) if self.tok else None,
                "max_row_tokens": self.max_kept, "max_candidate_tokens": self.max_candidate,
                "dropped_chars": self.dropped_chars, "dropped_tokens": self.dropped_tokens}


def _plant(task: str, row: dict) -> Plant | None:
    for key in ("plant_broken", "plant"):
        if row.get(key):
            return Plant.model_validate_json(row[key])
    return None


def _query(task: str, row: dict) -> str:
    plant = _plant(task, row)
    head = plant_to_text(plant)[:1200] if plant else ""
    return " ".join([str(row.get("fault_type", "")), " ".join(row.get("fault_types", []) or []), head])


def _mode(task: str, row_id: str, fraction: float) -> str:
    """Deterministic per-row choice, so re-running `combine` puts the harness on exactly the same rows."""
    if fraction <= 0:
        return "none"
    h = int(hashlib.md5(f"harness:{row_id}".encode()).hexdigest(), 16) % 1000
    return HARNESS_MODES[task] if h < fraction * 1000 else "none"


def _augment(task: str, row: dict, mode: str, notes: list[memory.Note], k_memory: int) -> tuple[list[dict], list[str]]:
    plant = _plant(task, row)
    msgs = harness.augment(row["prompt"], mode, task, plant)
    if not notes:
        return msgs, []
    got = memory.retrieve(_query(task, row), notes, k=k_memory, task=task, exclude_record=str(row.get("id", "")))
    if got:
        msgs = memory.augment_with_memory(msgs, task, _query(task, row), notes, k=k_memory,
                                          exclude_record=str(row.get("id", "")))
    return msgs, [n.id for n in got]


def _chars(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages)


def _read(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _emit(task: str, row: dict, variant: int, tag: str, mode: str, notes: list[memory.Note], k_memory: int,
          budget: Budget, stats: collections.Counter) -> dict | None:
    base = paraphrase.to_row(row, variant) if VARIANTS[task] else dict(row)
    msgs, used = _augment(task, base, mode, notes, k_memory)
    fits, reason, n = budget.check(msgs, base["completion"][0]["content"])
    if not fits:
        stats[f"dropped_{reason}:{task}"] += 1
        stats[f"dropped_{reason}_arch:{row.get('topology', '?')}"] += 1
        stats[f"dropped_{reason}_harness:{task}:{mode}"] += 1
        return None
    out = dict(base)
    out["task"] = task
    out["source"] = row.get("source", "ground_truth")
    out["harness"] = mode
    out["memory_notes"] = used
    out["prompt"] = msgs
    if n is not None:
        out["tokens"] = n
    if tag:
        out["id"] = f"{row['id']}#v{tag}"
    stats[f"kept:{task}"] += 1
    stats[f"kept_source:{task}:{out['source']}"] += 1
    stats[f"kept_harness:{task}:{mode}"] += 1
    if used:
        stats[f"kept_memory:{task}"] += 1
    return out


def _check(task: str, rows: list[dict], limit: int, problems: list[str]) -> dict:
    ok = bad = 0
    for row in rows[:limit]:
        v = validate.validate(task, row["completion"][0]["content"], row)
        if v["accepted"]:
            ok += 1
        else:
            bad += 1
            problems.append(f"check:{task}:{row['id']}: {v['reasons']}")
    return {"checked": ok + bad, "accepted": ok, "rejected": bad}


def build(processed, frontier, memory_dir, out_dir, tasks=TASKS, variants: int = 20,
          harness_fraction: float = 0.35, k_memory: int = 3, max_chars: int = 11500,
          splits=None, check_limit: int = 0, extra_processed=(), max_tokens: int = 4000) -> dict:
    proc_dirs = [pathlib.Path(processed)] + [pathlib.Path(p) for p in extra_processed]
    front, mem = pathlib.Path(frontier), pathlib.Path(memory_dir)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    notes: list[memory.Note] = []
    if mem.exists():
        for p in sorted(mem.glob("*.jsonl")):
            notes += memory.load(p)
    notes = memory.dedupe(notes)
    stats = collections.Counter()
    budget = Budget(max_chars, max_tokens)
    report = {"tasks": {}, "memory_notes_available": len(notes), "problems": [], "splits": {}}
    problems = report["problems"]
    seen_ids: dict[str, str] = {}
    for task in tasks:
        report["tasks"][task] = {}
        task_splits = splits or SPLITS[task]
        for split in task_splits:
            rows = []
            for d in proc_dirs:
                rows += _read(d / f"{task}_{split}.jsonl")
            extra = _read(front / f"frontier_{task}_{split}.jsonl")
            emitted = []
            for row in rows:
                # Paraphrases are a training-time device: the eval splits keep one row per ground-truth plant so
                # `model_eval` does not measure the same plant four times.
                n_var = variants if (VARIANTS[task] and split == "train") else 1
                for vno in range(n_var):
                    if n_var > 1:
                        vidx = paraphrase.variant_index(row["id"], paraphrase.variant_count(row), extra=vno)
                    else:
                        vidx = 0
                    mode = _mode(task, row["id"], harness_fraction)
                    r = _emit(task, row, vidx, vno if n_var > 1 else "", mode, notes, k_memory, budget, stats)
                    if r:
                        emitted.append(r)
            for row in extra:
                mode = _mode(task, row["id"], harness_fraction)
                r = _emit(task, row, 0, "", mode, notes, k_memory, budget, stats)
                if r:
                    emitted.append(r)
            for r in emitted:
                if r["id"] in seen_ids and seen_ids[r["id"]] != split:
                    problems.append(f"leak:{r['id']} in {seen_ids[r['id']]} and {split}")
                seen_ids[r["id"]] = split
                text = r["completion"][0]["content"]
                if not text.strip():
                    problems.append(f"empty:{r['id']}")
                if task == "repair" and "<edits>" not in text:
                    problems.append(f"no_edits:{r['id']}")
                if task == "repair" and "<diagnosis>" not in text:
                    problems.append(f"no_diagnosis:{r['id']}")
            path = out / f"{task}_{split}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for r in emitted:
                    fh.write(json.dumps(r) + "\n")
            report["tasks"][task][split] = {
                "n": len(emitted),
                "ground_truth": sum(1 for r in emitted if r["source"] == "ground_truth"),
                "frontier": sum(1 for r in emitted if r["source"] == "frontier"),
                "harness": sum(1 for r in emitted if r["harness"] != "none"),
                "with_memory": sum(1 for r in emitted if r["memory_notes"]),
                "distinct_paraphrases": len({r.get("paraphrase", {}).get("variant") for r in emitted if r.get("paraphrase")}),
                "architectures": dict(collections.Counter(r.get("topology", "?") for r in emitted)),
                "faults": dict(collections.Counter(str(r.get("fault_type", r.get("fault_types", ""))) for r in emitted)),
            }
            if check_limit:
                report["tasks"][task][split]["check"] = _check(task, emitted, check_limit, problems)
    report["stats"] = dict(stats)
    report["budget"] = budget.report()
    all_splits = sorted({s for t in report["tasks"] for s in report["tasks"][t]})
    report["splits"] = {s: sum(report["tasks"][t].get(s, {}).get("n", 0) for t in report["tasks"]) for s in all_splits}
    pathlib.Path(out / "sft_report.json").write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed")
    ap.add_argument("frontier")
    ap.add_argument("memory")
    ap.add_argument("out")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--variants", type=int, default=20)
    ap.add_argument("--harness-fraction", type=float, default=0.35)
    ap.add_argument("--k-memory", type=int, default=3)
    ap.add_argument("--max-prompt-chars", type=int, default=11500)
    ap.add_argument("--max-prompt-tokens", type=int, default=4000,
                    help="prompt + completion + chat-marker reserve; 4000 leaves 96 tokens under SFT max_length 4096")
    ap.add_argument("--check-limit", type=int, default=0)
    ap.add_argument("--splits", nargs="*", default=None, help="override the per-task split lists")
    ap.add_argument("--extra-processed", nargs="*", default=[], help="more processed dirs, e.g. the architecture corpus")
    a = ap.parse_args()
    r = build(a.processed, a.frontier, a.memory, a.out, a.tasks, a.variants, a.harness_fraction,
              a.k_memory, a.max_prompt_chars, a.splits, a.check_limit, a.extra_processed, a.max_prompt_tokens)
    print(json.dumps({"tasks": {t: {s: v["n"] for s, v in d.items()} for t, d in r["tasks"].items()},
                      "problems": len(r["problems"]), "first_problems": r["problems"][:10],
                      "memory_notes": r["memory_notes_available"], "budget": r["budget"]}, indent=1))


if __name__ == "__main__":
    main()
