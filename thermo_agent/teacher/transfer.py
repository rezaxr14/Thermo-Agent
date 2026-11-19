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
import re

from . import memory, validate

TASKS = ("repair", "ledger", "audit", "design")

# Provenance helpers.  A frontier row records where it came from on the row itself:
# source="frontier:<model>", model=<slug>, provider=<coarse label>, and run_id when
# a journal id is available.  Only the model *label* is ever stored -- no credentials.
MODEL_SUFFIX_RE = re.compile(r"-t-([A-Za-z0-9][A-Za-z0-9._-]*)$")
PROVIDER_HINTS = (
    ("openai", ("openai", "gpt", "chatgpt", "codex", "o1", "o3", "o4")),
    ("anthropic", ("anthropic", "claude")),
    ("google", ("google", "gemini", "gemma", "vertex")),
)
VALID_PROVIDERS = ("openai", "anthropic", "google", "other")
# A run id is a short label, never a credential.  Refuse anything that looks like one.
_SECRETISH = re.compile(r"(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{16,}|KGAT_|(?i:token|secret|password|api[_-]?key))")


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


def load_exports(export_dir, tasks=TASKS, split: str = "train", extra_dirs=()) -> dict[str, dict]:
    """Export rows keyed by record id, each tagged with the task whose file it came from.

    `data/processed/*_train.jsonl` rows do not carry a `task` column (the file name is the
    only task signal), so without this tag `ingest` fell back to "repair" for every row and
    crashed on the first design/audit row (KeyError: plant_broken).

    `extra_dirs` are searched after the primary dir, without overriding it. A trace shard is
    built from several corpora (`data/processed` plus `data/processed_arch`), and some ids only
    exist in an already-ingested frontier dir. Without the fallback those valid traces are
    counted as `skipped_unknown_or_empty`, i.e. silently lost. Keys prefer `record_id` so a
    teacher row (`<id>-t-<model>`, carrying the same task columns) resolves the same record.
    """
    rows: dict[str, dict] = {}
    for base in (export_dir, *extra_dirs):
        for task in tasks:
            for path in (pathlib.Path(base) / f"{task}_{split}.jsonl",
                         pathlib.Path(base) / f"frontier_{task}_{split}.jsonl"):
                for row in _read(path):
                    row.setdefault("task", task)
                    rows.setdefault(str(row.get("record_id") or row["id"]), row)
    return rows


def _text(record: dict) -> str:
    for key in ("completion", "text", "answer", "response"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in str(text)).strip("-")[:40] or "teacher"


def model_from_id(row_id: str) -> str | None:
    """The producing model slug encoded in a teacher row's ``-t-<model>`` id suffix."""
    match = MODEL_SUFFIX_RE.search(str(row_id or ""))
    return match.group(1) if match else None


def provider_for(model: str) -> str:
    """Coarse provider label derived from a model slug (never from credentials)."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", str(model or "").lower()) if t]
    for provider, hints in PROVIDER_HINTS:
        for token in tokens:
            if any(token.startswith(hint) for hint in hints):
                return provider
    return "other"


def _safe_run_id(value) -> str | None:
    """Keep a short journal/run label; drop empty, oversized or secret-looking values."""
    text = str(value or "").strip()
    if not text or len(text) > 128 or _SECRETISH.search(text):
        return None
    return text


def provenance_fields(model: str = "", run_id=None) -> dict:
    """Explicit provenance stamped on every row produced by :func:`ingest`."""
    slug = _slug(model)
    fields = {"source": f"frontier:{slug}", "model": slug, "provider": provider_for(slug)}
    safe_run = _safe_run_id(run_id)
    if safe_run:
        fields["run_id"] = safe_run
    return fields


def ingest(export_dir, completions_path, out_dir, memory_dir=None, tasks=TASKS, split: str = "train",
           min_reward: float = 0.0, model: str = "", run_id: str | None = None,
           extra_export_dirs=()) -> dict:
    exports = load_exports(export_dir, tasks, split, extra_export_dirs)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing: dict[str, set] = {}
    for task in tasks:
        existing[task] = {str(r.get("record_id") or r.get("id")) for r in _read(out / f"frontier_{task}_{split}.jsonl")}
    accepted: dict[str, list] = collections.defaultdict(list)
    notes: list[memory.Note] = []
    stats = collections.Counter()
    seen_answers: set[str] = set()
    n_success_notes = 0
    base_provenance = provenance_fields(model, run_id)
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
            # A tool-using trace carries its full trajectory (reasoning + tool calls + numeric
            # results + revisions). Persist it as the SFT completion so the student learns to
            # use tools/skills/memory and self-correct, not just to emit the final answer.
            trajectory = None
            traj_raw = record.get("trajectory")
            if traj_raw and record.get("sft_completion"):
                trajectory = dict(traj_raw)
                trajectory["sft_completion"] = record["sft_completion"]
            teacher = validate.teacher_row(task, text, row, verdict, tag=base_provenance["model"],
                                           trajectory=trajectory)
            # The validator tags genuine tool traces "frontier_tools"; keep that signal
            # additively while source itself becomes an explicit per-row provenance field.
            original_source = teacher.get("source")
            provenance = dict(base_provenance)
            if "run_id" not in provenance:
                row_run = _safe_run_id(record.get("run_id") or row.get("run_id"))
                if row_run:
                    provenance["run_id"] = row_run
            teacher.update(provenance)
            if original_source and original_source not in ("frontier", f"frontier:{base_provenance['model']}"):
                teacher["source_detail"] = original_source
            accepted[task].append(teacher)
            stats[f"accepted:{task}"] += 1
            if trajectory is not None:
                stats["accepted_tool_using"] += 1
            # Accepted attempts become reusable "what worked" memory too.  `success_note` reads the fix from
            # the candidate's own completion, so pass it on the row (export rows deliberately carry no answer).
            note_row = dict(row, completion=[{"role": "assistant", "content": text}])
            notes.append(validate.success_note(task, note_row, verdict, source=f"frontier:{_slug(model)}"))
            n_success_notes += 1
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
    # The summary records how often each failure reason appeared; deduped notes teach the class of mistake and
    # frequency notes make the common ones more likely to be retrieved into a prompt.  failure_reasons.json
    # stays failures-only; accepted notes join the retrieval memory alongside them.
    failures = [n for n in notes if not n.accepted]
    summary = memory.summarize(failures)
    frequent = memory.frequency_notes(failures)
    notes = memory.dedupe_lessons(notes)
    if memory_dir:
        path = pathlib.Path(memory_dir) / "frontier_notes.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = memory.dedupe(memory.load(path) + notes + frequent) if path.exists() else memory.dedupe(notes + frequent)
        memory.save(merged, path)
        (pathlib.Path(memory_dir) / "failure_reasons.json").write_text(
            json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        stats["memory_notes"] = len(notes) + len(frequent)
        stats["memory_success_notes"] = n_success_notes
        stats["frequent_lessons"] = len(frequent)
    from .progress import usage_totals
    report = {"model": model, "attempts": stats["attempts"], "accepted": sum(v for k, v in stats.items() if k.startswith("accepted:")),
              "rejected": sum(v for k, v in stats.items() if k.startswith("rejected:")), "stats": dict(stats),
              "failure_reasons": summary, "usage": usage_totals(_read(pathlib.Path(completions_path)))}
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
    ing.add_argument("--run-id", default=None, help="journal/run label recorded on every accepted row (never a secret)")
    ing.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ing.add_argument("--split", default="train")
    ing.add_argument("--min-reward", type=float, default=0.0)
    ing.add_argument("--extra-export-dir", action="append", default=[],
                     help="additional corpus dir searched for task columns when an id is not in export_dir "
                          "(e.g. data/processed_arch for arch ids); earlier dirs win")

    a = ap.parse_args()
    if a.cmd == "export":
        print(json.dumps(export(a.sft_dir, a.out_dir, a.tasks, a.split, a.per_task, a.seed), indent=1))
    else:
        print(json.dumps(ingest(a.export_dir, a.completions, a.out_dir, a.memory_dir, a.tasks, a.split,
                                a.min_reward, a.model, a.run_id, a.extra_export_dir), indent=1))


if __name__ == "__main__":
    main()
