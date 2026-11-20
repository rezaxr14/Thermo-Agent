"""Local, zero-cost rehearsal of the budget-batch post-processing pipeline.

The hosted Kaggle runner has repeatedly failed *after* a run was scheduled: the
downloaded journal was 0 bytes, rows=0, or contained an ``__error__`` row.  This
script proves the *plumbing* (shard selection -> journal -> token gate -> ingest)
works end-to-end on this machine, with no Kaggle call, no network and no model.

For every allowed model we:

1. load ``harness_config.json`` and the model's shard exactly as the hosted
   runner does (``Config.from_dict``, ``cfg.shard_for``, JSONL read,
   ``cfg.select_rows``);
2. build a fake journal record per selected row in the runner's record shape,
   reusing the export row's ground-truth answer when available (otherwise a
   syntactically valid per-task XML answer);
3. apply the same empty-journal / ``__error__`` guards as
   ``scripts/process_budget_batch.ps1`` (and self-test those guards);
4. run the REAL token gate and the REAL ``thermo_agent.teacher.transfer.ingest``
   into a throwaway output directory.

Everything is written under ``--out`` (default ``data/_rehearsal/<timestamp>``)
and removed unless ``--keep`` is given.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_agent.teacher import transfer  # noqa: E402

DEFAULT_CONFIG = ROOT / "data" / "frontier_tools_cohort_prompts" / "harness_config.json"
PROMPTS_DIR = ROOT / "data" / "frontier_tools_cohort_prompts"
EXPORT_DIR = ROOT / "data" / "processed"
EXTRA_EXPORT_DIRS = (
    ROOT / "data" / "processed_arch",
    ROOT / "data" / "frontier_prov",
    ROOT / "data" / "a_slice",
)
REHEARSAL_ROOT = ROOT / "data" / "_rehearsal"
TOKENGATE = ROOT / "scripts" / "frontier_cohort_tokengate.py"

# Guard text copied verbatim from scripts/process_budget_batch.ps1 so a failure
# here reads exactly like a failure of the real overnight loop.
EMPTY_GUARD_TEXT = (
    "journal is empty: {path} (the runner attempted 0 rows; "
    "check the Kaggle log for [runner][ERROR])"
)
ERROR_GUARD_TEXT = "runner journalled an error instead of traces: {detail}"
_ERROR_ROW = re.compile(r'"id"\s*:\s*"__error__"')


class JournalGuardError(RuntimeError):
    """Raised when a downloaded journal is empty or carries the runner's error row."""


# ---------------------------------------------------------------------------
# The two process_budget_batch.ps1 guards, implemented in testable Python
# ---------------------------------------------------------------------------
def validate_journal(path: str | pathlib.Path) -> int:
    """Mirror the PowerShell guards; return the record count, or raise.

    Mirrors lines 21-28 of ``scripts/process_budget_batch.ps1``: a 0-byte journal
    is never a completed batch, and the runner's explicit ``__error__`` row is
    never a trace.
    """
    p = pathlib.Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise JournalGuardError(EMPTY_GUARD_TEXT.format(path=p))

    detail = None
    count = 0
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            count += 1
            if _ERROR_ROW.search(line):
                detail = line.rstrip("\r\n")
                break
    if detail is not None:
        raise JournalGuardError(ERROR_GUARD_TEXT.format(detail=detail))
    return count


def check_reject_paths(out_dir: pathlib.Path) -> dict:
    """Prove both guards actually fire, using throwaway journals under ``out_dir``."""
    empty = out_dir / "_guard_empty.jsonl"
    empty.write_text("", encoding="utf-8")
    try:
        validate_journal(empty)
    except JournalGuardError as exc:
        assert str(empty) in str(exc), exc
        empty_msg = str(exc)
    else:
        raise AssertionError("empty-journal guard did not fire")

    err = out_dir / "_guard_error.jsonl"
    err.write_text(json.dumps({
        "id": "__error__",
        "error": "rows=0 shard=missing.jsonl source=shard tiers={}",
        "shard": "missing.jsonl", "model": "gpt-6-luna",
    }) + "\n", encoding="utf-8")
    try:
        validate_journal(err)
    except JournalGuardError as exc:
        assert "runner journalled an error instead of traces" in str(exc), exc
        error_msg = str(exc)
    else:
        raise AssertionError("__error__-row guard did not fire")
    return {"empty": empty_msg, "error": error_msg}


# ---------------------------------------------------------------------------
# Hosted-runner parity helpers
# ---------------------------------------------------------------------------
def load_runner_module():
    """Import notebooks/kaggle_model_runner.py the same way its own tests do."""
    path = ROOT / "notebooks" / "kaggle_model_runner.py"
    spec = importlib.util.spec_from_file_location("kaggle_model_runner_rehearsal", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: str | pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in pathlib.Path(path).open(encoding="utf-8")
            if line.strip()]


def find_shard(name: str, config_dir: pathlib.Path) -> pathlib.Path:
    """Resolve a shard name the way the runner's ``_find_shard`` does, locally."""
    for base in (config_dir, PROMPTS_DIR, ROOT, ROOT / "data"):
        candidate = base / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"shard {name!r} not found (config_dir={config_dir})")


def load_export_rows() -> dict[str, dict]:
    """Ground-truth export rows keyed by id, searched like process_budget_batch.ps1."""
    return transfer.load_exports(
        str(EXPORT_DIR),
        extra_dirs=[str(d) for d in EXTRA_EXPORT_DIRS],
    )


def available_tools() -> list[str]:
    """The harness tool list, or a stable fallback if the toolkit cannot import."""
    try:
        from thermo_agent.teacher.toolkit import tool_names
        return sorted(tool_names())
    except Exception:
        return ["solve_plant", "check_exergy", "audit_ranking", "validate_design"]


# ---------------------------------------------------------------------------
# Fake journal construction
# ---------------------------------------------------------------------------
def _text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(m.get("content", "")) for m in value
                         if isinstance(m, dict))
    return str(value or "")


def synthetic_answer(task: str) -> str:
    """Syntactically valid XML for each task (it need not be verifier-accepted)."""
    if task == "repair":
        return "<diagnosis>Candidate repair.</diagnosis>\n<edits>\n</edits>"
    if task == "design":
        return "<rationale>Candidate design.</rationale>\n<design>\n</design>"
    if task == "audit":
        return ("<analysis>Candidate exergy audit.</analysis>\n"
                "<ranking>\n</ranking>\n"
                "<change>SET candidate</change>\n"
                "<prediction>0</prediction>")
    if task == "ledger":
        return "<diagnosis>Candidate ledger fix.</diagnosis>\n<fix>\n</fix>"
    raise ValueError(f"unknown task {task!r}")


def final_answer(row: dict, export_rows: dict[str, dict]) -> tuple[str, bool]:
    """Prefer the export row's ground-truth answer; else synthesize one.

    Returns ``(answer, from_export)``.  Reusing the ground truth makes the
    rehearsal exercise the *kept* paths of both the gate and ingest, not just
    the reject counters.
    """
    task = str(row.get("task") or "repair")
    rid = str(row.get("record_id") or row.get("id") or "")
    export_row = export_rows.get(rid) or export_rows.get(str(row.get("id") or ""))
    if export_row:
        answer = _text(export_row.get("completion")).strip()
        if answer:
            return answer, True
    return synthetic_answer(task), False


def build_record(row: dict, *, model: str, run_id: str, shard: str, cfg,
                 tools: list[str], answer: str) -> dict:
    """One runner-shaped journal record with a 3-turn tool trajectory."""
    rid = str(row.get("id", ""))
    task = str(row.get("task") or "repair")
    prompt = row.get("prompt") if isinstance(row.get("prompt"), list) else []
    system = "\n".join(str(m.get("content", "")) for m in prompt
                       if isinstance(m, dict) and m.get("role") == "system")
    prompt_sha = hashlib.sha256((system or shard).encode("utf-8")).hexdigest()

    specs = [
        ("inspect the plant and the harness skills", "plant_inspect", {"id": rid, "task": task}),
        ("check the candidate against the physics tools", "verify_candidate", {"task": task}),
        ("consolidate the corrected answer", None, None),
    ]
    turns = []
    for index, (focus, tool, arguments) in enumerate(specs, start=1):
        assistant = (
            f"Turn {index}: I {focus}. The plant is '{row.get('topology', 'unknown')}' "
            f"in tier '{row.get('tier', '?')}'; I will use the served skills to check it."
        )
        reasoning = (
            f"Reasoning {index}: for {task} I need to {focus}; the harness prompt "
            f"shares context with the student, so I keep this prose descriptive."
        )
        tool_calls, tool_results = [], []
        if tool is not None:
            tool_calls = [{"name": tool, "arguments": dict(arguments or {})}]
            tool_results = [
                f'<tool_result name="{tool}">deterministic local check for {rid}</tool_result>'
            ]
        turns.append({"assistant": assistant, "reasoning": reasoning,
                      "tool_calls": tool_calls, "tool_results": tool_results})

    blocks = []
    for turn in turns:
        blocks.append(turn["assistant"])
        for call in turn["tool_calls"]:
            blocks.append('<tool_call name="%s">%s</tool_call>'
                          % (call["name"], json.dumps(call["arguments"], sort_keys=True)))
        blocks.extend(turn["tool_results"])
    blocks.append(answer)
    sft_completion = "\n".join(blocks)

    trajectory = {
        "turns": turns,
        "notes_used": [],
        "tools_available": list(tools),
        "final": answer,
        "served_skill": task,
        "system_prompt_sha256": prompt_sha,
        "memory_block": "",
    }
    return {
        "id": rid,
        "record_id": str(row.get("record_id") or rid),
        "task": task,
        "tier": row.get("tier"),
        "model": model,
        "mode": cfg.mode,
        "harness": "agents+skills" if cfg.uses_tools else "none",
        "tools": list(tools),
        "completion": answer,
        "sft_completion": sft_completion,
        "reasoning": "\n\n".join(t["reasoning"] for t in turns),
        "trajectory": trajectory,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                  "total_cost_nanodollars": 0, "seconds": 0.0},
        "error": None,
        "graded": False,
        "provenance": transfer.provenance_fields(model, run_id),
        "run_id": run_id,
        "shard": shard,
        "row_source": "shard",
        "ts": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def build_journal(rows: list[dict], *, model: str, run_id: str, shard: str, cfg,
                  tools: list[str], export_rows: dict[str, dict]) -> tuple[list[dict], int]:
    """Build all records and count how many reused a ground-truth answer."""
    records, from_export = [], 0
    for row in rows:
        answer, used_export = final_answer(row, export_rows)
        from_export += int(used_export)
        records.append(build_record(row, model=model, run_id=run_id, shard=shard,
                                    cfg=cfg, tools=tools, answer=answer))
    return records, from_export


def write_journal(path: pathlib.Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# The real gate and the real ingest
# ---------------------------------------------------------------------------
def run_gate(model: str, journal: pathlib.Path, gated: pathlib.Path,
             manifest: pathlib.Path, gate_tokens: int) -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["HARNESS_GATE_TOKENS"] = str(gate_tokens)
    proc = subprocess.run(
        [sys.executable, str(TOKENGATE), "journal", model,
         str(journal), str(gated), str(manifest)],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"token gate failed for {model} (exit {proc.returncode}):\n"
                           f"{proc.stdout}\n{proc.stderr}")
    if not manifest.is_file() or manifest.stat().st_size == 0:
        raise RuntimeError(f"token gate wrote no manifest for {model}")
    summary = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not summary:
        raise RuntimeError(f"token gate manifest is empty for {model}")
    return summary


def ingest_gated(model: str, run_id: str, gated: pathlib.Path, ingest_out: pathlib.Path,
                 memory_dir: pathlib.Path) -> dict:
    report = transfer.ingest(
        str(EXPORT_DIR), str(gated), str(ingest_out),
        memory_dir=str(memory_dir), model=model, run_id=run_id,
        extra_export_dirs=[str(d) for d in EXTRA_EXPORT_DIRS],
    )
    if not isinstance(report, dict) or not isinstance(report.get("stats"), dict):
        raise RuntimeError(f"ingest did not return a stats dict for {model}")
    return report


def check_provenance(ingest_out: pathlib.Path, model: str, run_id: str) -> list[dict]:
    expected = transfer.provenance_fields(model, run_id)["provenance"]
    accepted = [json.loads(line)
                for path in sorted(ingest_out.glob("frontier_*_train.jsonl"))
                for line in path.open(encoding="utf-8") if line.strip()]
    for row in accepted:
        provenance = row.get("provenance") or {}
        if provenance.get("model") != expected.get("model") or \
                provenance.get("run_id") != expected.get("run_id"):
            raise AssertionError(f"accepted row {row.get('id')} has wrong provenance: "
                                 f"{provenance}")
    return accepted


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_one_model(model: str, *, cfg, rows_limit: int, config_dir: pathlib.Path,
                  out_dir: pathlib.Path, tools: list[str],
                  export_rows: dict[str, dict], gate_tokens: int) -> dict:
    shard_name = cfg.shard_for(model)
    shard_path = find_shard(shard_name, config_dir)
    shard_rows = read_jsonl(shard_path)
    selected = cfg.select_rows(shard_rows)
    if rows_limit > 0:
        selected = selected[:rows_limit]
    if not selected:
        # The exact failure mode this rehearsal exists to catch.
        raise AssertionError(f"{model}: select_rows yielded 0 rows from {shard_path} "
                             f"(tiers={cfg.tiers}, equal_tiers={cfg.equal_tiers})")

    run_id = f"rehearsal-{_dt.datetime.now(_dt.timezone.utc):%Y%m%dT%H%M%S}-{model}"
    model_dir = out_dir / model.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    journal = model_dir / "journal.jsonl"
    gated = model_dir / "gated.jsonl"
    manifest = model_dir / "tokengate_manifest.json"
    ingest_out = model_dir / "ingest"
    memory_dir = model_dir / "memory"

    records, from_export = build_journal(selected, model=model, run_id=run_id,
                                         shard=shard_name, cfg=cfg, tools=tools,
                                         export_rows=export_rows)
    write_journal(journal, records)
    journal_records = validate_journal(journal)
    if journal_records != len(records):
        raise AssertionError(f"{model}: journal has {journal_records} records, "
                             f"expected {len(records)}")

    summary = run_gate(model, journal, gated, manifest, gate_tokens)
    if summary.get("journal_rows") != len(records):
        raise AssertionError(f"{model}: gate manifest accounted for "
                             f"{summary.get('journal_rows')} records, expected {len(records)}")
    gated_records = read_jsonl(gated)

    report = ingest_gated(model, run_id, gated, ingest_out, memory_dir)
    attempts = int(report["stats"].get("attempts", 0))
    if attempts != len(records):
        raise AssertionError(f"{model}: ingest attempts={attempts}, expected {len(records)}")
    accepted = check_provenance(ingest_out, model, run_id)

    tier_counts = {t: sum(1 for r in selected if cfg.tier_of(r) == t) for t in cfg.tiers}
    kept = int(summary.get("survived", 0))
    return {
        "model": model,
        "shard": shard_name,
        "shard_rows": len(shard_rows),
        "selected": len(selected),
        "tier_counts": tier_counts,
        "journal_bytes": journal.stat().st_size,
        "from_export": from_export,
        "gate_kept": kept,
        "gate_dropped": len(records) - kept,
        "gate_summary": summary,
        "ingest_attempts": attempts,
        "ingest_accepted": len(accepted),
        "provenance_examples": [r.get("provenance") for r in accepted[:1]],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="harness_config.json to rehearse (default: cohort prompts config)")
    ap.add_argument("--rows", type=int, default=12,
                    help="max selected rows per model (default: 12)")
    ap.add_argument("--out", default=None,
                    help="scratch directory (default: fresh data/_rehearsal/<timestamp>)")
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    ap.add_argument("--model", dest="models", action="append", default=None,
                    help="model to rehearse; repeatable (default: config allowed_models)")
    args = ap.parse_args(argv)

    config_path = pathlib.Path(args.config).resolve()
    cfg_values = json.loads(config_path.read_text(encoding="utf-8"))
    runner = load_runner_module()
    cfg = runner.Config.from_dict(cfg_values)
    models = tuple(args.models) if args.models else tuple(cfg.allowed_models) or \
        tuple(runner.DEFAULT_ALLOWED)
    gate_tokens = int((cfg.pipeline or {}).get("gate_tokens", 8192))

    auto_out = args.out is None
    if auto_out:
        out = REHEARSAL_ROOT / _dt.datetime.now().strftime("%Y%m%dT%H%M%S%f")
    else:
        out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        guards = check_reject_paths(out)
        print(f"[rehearsal] guard checks OK: empty-journal + __error__ both rejected")
        export_rows = load_export_rows()
        tools = available_tools()
        results = []
        for model in models:
            results.append(run_one_model(
                model, cfg=cfg, rows_limit=args.rows, config_dir=config_path.parent,
                out_dir=out, tools=tools, export_rows=export_rows,
                gate_tokens=gate_tokens,
            ))

        for r in results:
            print(f"[model] {r['model']}")
            print(f"  shard={r['shard']} shard_rows={r['shard_rows']} "
                  f"selected={r['selected']} tiers={r['tier_counts']}")
            print(f"  journal_bytes={r['journal_bytes']} export_answers={r['from_export']}")
            print(f"  gate kept={r['gate_kept']} dropped={r['gate_dropped']}")
            print(f"  ingest attempts={r['ingest_attempts']} accepted={r['ingest_accepted']}")
            print(f"  provenance_examples={r['provenance_examples']}")
        print(f"[rehearsal] out={out} keep={args.keep}")
        print("REHEARSAL OK")
        return 0
    except Exception as exc:
        print(f"REHEARSAL FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.keep:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
