"""Verify a combined corpus on disk against the report that produced it.

`combine` writes `sft_report.json` next to the split files, and `build_corpus.ps1` only swaps a staged corpus into
place after the token audit passes. Neither of those notices that a split file went missing afterwards (a moved
folder, a killed copy, a stray `Rename-Item` leaving one file behind): the report still says 106,420 rows and the
trainer reads whatever is in the directory. This tool re-derives the corpus from the files themselves and compares
it with the report.

    python -m thermo_agent.synth.verify data/sft
    python -m thermo_agent.synth.verify data/grpo --allow-missing-completion

It checks, per task and split:

- every split named in the report has a file, and its row count matches the report;
- no `{task}_*.jsonl` file exists that the report does not know about;
- every row parses, carries `id`/`split`/`task`/`prompt`/`tokens`, and its `split`/`task` match the filename;
- every prompt is a non-empty list of `{role, content}` messages starting with a system or user message, and a
  completion (when present) ends with an assistant message;
- ids are unique across every file and never appear in two splits;
- every row fits `--max-tokens` (the same gate `combine` applies);
- the harness share is within `--harness-tolerance` of the fraction the report recorded.

Exit status is 1 when any check fails, so it can be the last step of a build script.
"""
from __future__ import annotations

import argparse
import json
import pathlib

TASKS = ("repair", "ledger", "audit", "design")


def _rows(path: pathlib.Path):
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if line.strip():
                yield lineno, line.strip()


def verify(data_dir, report_path=None, tasks=TASKS, max_tokens: int = 4000,
           harness_tolerance: float = 0.02, allow_missing_completion: bool = False) -> dict:
    root = pathlib.Path(data_dir)
    report_path = pathlib.Path(report_path) if report_path else root / "sft_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    problems: list[str] = []
    if report is None:
        problems.append(f"report missing: {report_path}")
        expected: dict[str, dict] = {t: {} for t in tasks}
    else:
        expected = {t: report.get("tasks", {}).get(t, {}) for t in tasks}
        if not root.exists():
            problems.append(f"data directory missing: {root}")

    files: dict[str, dict] = {}
    ids: dict[str, str] = {}
    for task in tasks:
        known = set(expected[task])
        present = {p.stem[len(task) + 1:] for p in root.glob(f"{task}_*.jsonl")}
        for split in sorted(known - present):
            problems.append(f"missing file: {task}_{split}.jsonl")
        for split in sorted(present - known):
            problems.append(f"unknown split file: {task}_{split}.jsonl (not in the report)")
        for p in sorted(root.glob(f"{task}_*.jsonl")):
            split = p.stem[len(task) + 1:]
            n = bad = no_completion = 0
            harness = 0
            for lineno, line in _rows(p):
                n += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    bad += 1
                    if bad <= 3:
                        problems.append(f"{p.name}:{lineno}: not JSON ({exc.msg})")
                    continue
                rid = row.get("id")
                if not isinstance(rid, str) or not rid:
                    problems.append(f"{p.name}:{lineno}: missing id")
                    continue
                if row.get("split") != split:
                    problems.append(f"{p.name}:{lineno}: {rid} carries split={row.get('split')!r}")
                if row.get("task") != task:
                    problems.append(f"{p.name}:{lineno}: {rid} carries task={row.get('task')!r}")
                prompt = row.get("prompt")
                if not (isinstance(prompt, list) and prompt and all(
                        isinstance(m, dict) and m.get("role") and isinstance(m.get("content"), str) for m in prompt)):
                    problems.append(f"{p.name}:{lineno}: {rid} has a malformed prompt")
                elif prompt[0]["role"] not in ("system", "user"):
                    problems.append(f"{p.name}:{lineno}: {rid} prompt starts with role={prompt[0]['role']!r}")
                completion = row.get("completion")
                if completion is None:
                    no_completion += 1
                elif not (isinstance(completion, list) and completion
                          and completion[-1].get("role") == "assistant" and str(completion[-1].get("content", "")).strip()):
                    problems.append(f"{p.name}:{lineno}: {rid} has no assistant completion")
                tokens = row.get("tokens")
                if not isinstance(tokens, int) or tokens <= 0:
                    problems.append(f"{p.name}:{lineno}: {rid} has no token count")
                elif tokens > max_tokens:
                    problems.append(f"{p.name}:{lineno}: {rid} is {tokens} tokens (> {max_tokens})")
                harness += row.get("harness", "none") != "none"
                if rid in ids and ids[rid] != split:
                    problems.append(f"leak: {rid} in {ids[rid]} and {split}")
                elif rid in ids:
                    problems.append(f"duplicate id: {rid}")
                ids[rid] = split
            rec = {"n": n, "bad_json": bad, "harness": harness, "without_completion": no_completion,
                   "harness_share": round(harness / n, 4) if n else 0.0}
            if no_completion and not allow_missing_completion:
                problems.append(f"{p.name}: {no_completion} rows have no completion")
            want = expected[task].get(split)
            if want is not None:
                if want.get("n") != n:
                    problems.append(f"{p.name}: {n} rows on disk, report says {want['n']}")
                if want.get("harness") is not None and n:
                    share = harness / n
                    got = want["harness"] / want["n"] if want.get("n") else 0.0
                    if abs(share - got) > harness_tolerance:
                        problems.append(f"{p.name}: harness share {share:.2f} on disk, report says {got:.2f}")
            files[p.stem] = rec

    return {"data": str(root), "report": str(report_path), "files": files,
            "rows": sum(f["n"] for f in files.values()), "problems": problems,
            "ok": not problems}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--report", default=None)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--harness-tolerance", type=float, default=0.02)
    ap.add_argument("--allow-missing-completion", action="store_true",
                    help="GRPO prompts may be stored without a reference completion")
    a = ap.parse_args()
    r = verify(a.data, a.report, a.tasks, a.max_tokens, a.harness_tolerance, a.allow_missing_completion)
    print(json.dumps({"data": r["data"], "rows": r["rows"],
                      "files": len(r["files"]), "problems": r["problems"][:20],
                      "problem_count": len(r["problems"]), "ok": r["ok"]}, indent=1))
    raise SystemExit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
