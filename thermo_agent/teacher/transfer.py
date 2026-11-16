"""Export the teacher prompts, and turn the teacher's answers into validated training rows.

Half of the training corpus is meant to come from a frontier model. That model has to pass the same tool suite as
everything else, so the workflow is deliberately split into two halves that can run on different machines:

    # here: sample prompts from the combined SFT corpus, exactly as the student will see them
    python -m thermo_agent.teacher.export data/sft data/frontier_export --per-task 400

    # on Kaggle: kaggle/teacher_benchmark.py sends each prompt to a frontier model and writes completions.jsonl

    # here: validate every answer with the tool suite and keep only what passes
    python -m thermo_agent.teacher.ingest data/frontier_export completions.jsonl data/frontier \\
        --memory-dir data/memory

`export` keeps one prompt per ground-truth plant (paraphrase variants are a student-side device), and it never
writes the ground-truth completion into the export file, so the notebook cannot leak the answer to the teacher.
`ingest` is resumable: answers already in `data/frontier/` are skipped, so a dropped notebook run costs only the
rows in flight.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib

from . import memory, validate

TASKS = ("repair", "ledger", "audit", "design")


def _read(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _base_id(row_id: str) -> str:
    return str(row_id).split("#", 1)[0]


def _rank(row_id: str, seed: int) -> str:
    return hashlib.md5(f"{seed}:{row_id}".encode()).hexdigest()


def select(rows: list[dict], per_task: int, seed: int = 0) -> list[dict]:
    """One row per ground-truth plant, ground-truth rows only, in a deterministic order."""
    seen: dict[str, dict] = {}
    for row in rows:
        if row.get("source", "ground_truth") != "ground_truth":
            continue
        base = _base_id(row["id"])
        if base not in seen or _rank(row["id"], seed) < _rank(seen[base]["id"], seed):
            seen[base] = row
    picked = sorted(seen.values(), key=lambda r: _rank(_base_id(r["id"]), seed))
    return picked[:per_task] if per_task else picked


def export(sft_dir, out_dir, tasks=TASKS, split: str = "train", per_task: int = 400, seed: int = 0) -> dict:
    src, out = pathlib.Path(sft_dir), pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(src), "split": split, "per_task": per_task, "seed": seed,
                "tasks": {}, "rows": 0}
    for task in tasks:
        rows = select(_read(src / f"{task}_{split}.jsonl"), per_task, seed)
        path = out / f"{task}_{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                clean = {k: v for k, v in row.items() if k != "completion"}
                fh.write(json.dumps(clean) + "\n")
        tokens = [r["tokens"] for r in rows if isinstance(r.get("tokens"), int)]
        manifest["tasks"][task] = {
            "n": len(rows),
            "harness": dict(collections.Counter(r.get("harness", "none") for r in rows)),
            "with_memory": sum(1 for r in rows if r.get("memory_notes")),
            "architectures": len({r.get("topology", "?") for r in rows}),
            "max_prompt_tokens": max(tokens) if tokens else None,
        }
        manifest["rows"] += len(rows)
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


def load_exports(export_dir, tasks=TASKS, split: str = "train") -> dict[str, dict]:
    rows = {}
    for task in tasks:
        for row in _read(pathlib.Path(export_dir) / f"{task}_{split}.jsonl"):
            rows[str(row["id"])] = row
    return rows


def _text(record: dict) -> str:
    for key in ("completion", "text", "answer", "response"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in str(text)).strip("-")[:40] or "teacher"


def ingest(export_dir, completions_path, out_dir, memory_dir=None, tasks=TASKS, split: str = "train",
           min_reward: float = 0.0, model: str = "") -> dict:
    exports = load_exports(export_dir, tasks, split)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing: dict[str, set] = {}
    for task in tasks:
        existing[task] = {str(r.get("record_id") or r.get("id")) for r in _read(out / f"frontier_{task}_{split}.jsonl")}
    accepted: dict[str, list] = collections.defaultdict(list)
    notes: list[memory.Note] = []
    stats = collections.Counter()
    seen_answers: set[str] = set()
    for record in _read(pathlib.Path(completions_path)):
        rid = str(record.get("id", ""))
        text = _text(record)
        row = exports.get(rid)
        stats["attempts"] += 1
        if row is None or not text:
            stats["skipped_unknown_or_empty"] += 1
            continue
        task = row.get("task", "repair")
        if rid in existing[task]:
            stats["skipped_already_accepted"] += 1
            continue
        digest = hashlib.md5(text.encode()).hexdigest()
        if digest in seen_answers:
            stats["skipped_duplicate"] += 1
            continue
        seen_answers.add(digest)
        verdict = validate.validate(task, text, row)
        if verdict["accepted"] and verdict["reward"] >= min_reward:
            accepted[task].append(validate.teacher_row(task, text, row, verdict, tag=_slug(model)))
            stats[f"accepted:{task}"] += 1
        else:
            if verdict["accepted"]:
                verdict = dict(verdict, accepted=False, reasons=verdict["reasons"] + [f"reward {verdict['reward']:.2f} below {min_reward}"])
            notes.append(validate.rejection_note(task, row, verdict, source=f"frontier:{_slug(model)}"))
            stats[f"rejected:{task}"] += 1
            for reason in verdict["reasons"]:
                stats[f"reason:{reason.split(':')[0]}"] += 1
    for task, rows in accepted.items():
        path = out / f"frontier_{task}_{split}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    notes = memory.dedupe_lessons(notes)
    if memory_dir and notes:
        path = pathlib.Path(memory_dir) / "frontier_notes.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = memory.dedupe(memory.load(path) + notes) if path.exists() else notes
        memory.save(merged, path)
        stats["memory_notes"] = len(notes)
    report = {"model": model, "attempts": stats["attempts"], "accepted": sum(v for k, v in stats.items() if k.startswith("accepted:")),
              "rejected": sum(v for k, v in stats.items() if k.startswith("rejected:")), "stats": dict(stats)}
    return report


def main():
    ap = argparse.ArgumentParser(description="export teacher prompts / ingest validated teacher answers")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("export", help="sample prompts from a combined SFT corpus into a teacher export dir")
    ex.add_argument("sft_dir")
    ex.add_argument("out_dir")
    ex.add_argument("--per-task", type=int, default=400)
    ex.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ex.add_argument("--split", default="train")
    ex.add_argument("--seed", type=int, default=0)

    ing = sub.add_parser("ingest", help="validate a completions file and append the accepted rows")
    ing.add_argument("export_dir")
    ing.add_argument("completions")
    ing.add_argument("out_dir")
    ing.add_argument("--memory-dir", default=None)
    ing.add_argument("--model", default="", help="teacher model label, recorded on every row and note")
    ing.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ing.add_argument("--split", default="train")
    ing.add_argument("--min-reward", type=float, default=0.0)

    a = ap.parse_args()
    if a.cmd == "export":
        print(json.dumps(export(a.sft_dir, a.out_dir, a.tasks, a.split, a.per_task, a.seed), indent=1))
    else:
        print(json.dumps(ingest(a.export_dir, a.completions, a.out_dir, a.memory_dir, a.tasks, a.split,
                                a.min_reward, a.model), indent=1))


if __name__ == "__main__":
    main()
