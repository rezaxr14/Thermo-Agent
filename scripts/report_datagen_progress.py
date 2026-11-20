#!/usr/bin/env python3
"""Read-only progress reporter for frontier CoT datagen.

Source of truth: ``data/frontier_tools_cohort/<model>/*.jsonl``. Rows are
accepted traces, deduplicated by ``record_id`` / ``id`` (with any ``#vN``
suffix stripped) within each model -- the same question asked of two different
models is two traces, so dedup is per model rather than global.

Prints a compact table to stdout (the primary human view) and optionally writes
a deterministic JSON payload and a Markdown report. No network, no training,
no credentials, no third-party imports.

    python scripts/report_datagen_progress.py
    python scripts/report_datagen_progress.py --json reports/datagen_progress.json
    python scripts/report_datagen_progress.py --md reports/DATAGEN_PROGRESS.md

With no flags the Markdown report is refreshed and the table is printed; pass
``--json`` (with or without a path) to additionally emit the JSON payload.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
COHORT_DIR = "data/frontier_tools_cohort"
MODELS = ("gpt-6-luna", "claude-haiku-5-5")
TASKS = ("repair", "design", "audit", "ledger")
TIERS = ("easy", "medium", "hard")
UNKNOWN_TIER = "unknown"
FRONTIER_SOURCE = "frontier_tools"

DEFAULT_MD = "reports/DATAGEN_PROGRESS.md"
DEFAULT_JSON = "reports/datagen_progress.json"

_VERSION_SUFFIX = re.compile(r"#v\d+$")


def stripped_id(row: dict) -> str:
    """Return record_id (or id) with a trailing ``#vN`` suffix removed."""
    raw = row.get("record_id")
    if raw is None:
        raw = row.get("id")
    if raw is None:
        return ""
    return _VERSION_SUFFIX.sub("", str(raw)).strip()


def derive_tier(row: dict) -> str:
    """Use ``tier``; if missing/unknown fall back to ``cell`` = ``task/tier``.

    Anything still not one of TIERS becomes ``unknown`` and is reported as a
    data defect rather than being silently dropped.
    """
    tier = str(row.get("tier") or "").strip()
    if tier in TIERS:
        return tier
    cell = str(row.get("cell") or "").strip()
    if "/" in cell:
        candidate = cell.split("/", 1)[1].strip()
        if candidate in TIERS:
            return candidate
    return UNKNOWN_TIER


def derive_task(row: dict) -> str:
    """Use ``task``; fall back to the task half of ``cell`` when absent."""
    task = str(row.get("task") or "").strip()
    if task:
        return task
    cell = str(row.get("cell") or "").strip()
    if "/" in cell:
        return cell.split("/", 1)[0].strip() or UNKNOWN_TIER
    return UNKNOWN_TIER


def load(root: pathlib.Path | str | None = None) -> dict:
    """Read cohort JSONL files.

    Returns ``{"rows": [...], "errors": [...], "malformed": int}``. Rows get
    private ``_model`` / ``_source_file`` / ``_line`` keys. An unreadable file
    is recorded in ``errors`` (caller exits 2); a malformed line is skipped and
    counted (caller exits 0, with a data-defect count).
    """
    out: dict = {"rows": [], "errors": [], "malformed": 0}
    base = pathlib.Path(ROOT if root is None else root) / COHORT_DIR
    for model in MODELS:
        model_dir = base / model
        if not model_dir.is_dir():
            continue
        for path in sorted(model_dir.glob("*.jsonl")):
            try:
                with path.open(encoding="utf-8") as fh:
                    lines = fh.read().splitlines()
            except OSError as exc:
                out["errors"].append(f"{path.as_posix()}: {exc}")
                continue
            for lineno, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    out["malformed"] += 1
                    continue
                if not isinstance(obj, dict):
                    out["malformed"] += 1
                    continue
                obj["_model"] = model
                obj["_source_file"] = path.as_posix()
                obj["_line"] = lineno
                out["rows"].append(obj)
    return out


def resolve_tier(row: dict) -> str:
    """Explicit tier, then `cell`, then the gate's measured tiers (see stamp_tiers)."""
    tier = derive_tier(row)
    if tier != UNKNOWN_TIER:
        return tier
    measured = str(row.get("_measured_tier") or "")
    return measured if measured in TIERS else UNKNOWN_TIER


def stamp_tiers(rows: list[dict]) -> None:
    """Attach `_measured_tier` using the SAME rule as the datagen gate.

    Accepted cohort rows carry no `tier`/`cell`: the gate derives it from the
    measured artefact complexity with `thermo_agent.synth.tiers`. Re-deriving here
    keeps this report consistent with `scripts/check_datagen_complete.py` instead
    of inventing a second tiering scheme.
    """
    try:
        from thermo_agent.synth import tiers as tier_mod
    except Exception:      # standalone use without the package on sys.path
        return
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        task = derive_task(row)
        if task in TASKS:
            groups.setdefault((row["_model"], task), []).append(row)
    for (_, task), group in groups.items():
        scored = [r for r in group if tier_mod.complexity(task, r) is not None]
        if not scored:
            continue
        mapping = tier_mod.assign_tiers(task, scored)
        for row in scored:
            rid = str(row.get("record_id") or row.get("id") or "")
            row["_measured_tier"] = mapping.get(rid)


def build_report(root: pathlib.Path | str | None = None) -> tuple[dict, list[str]]:
    """Return ``(payload, read_errors)`` for the cohort under ``root``."""
    loaded = load(root)
    errors: list[str] = loaded["errors"]
    stamp_tiers(loaded["rows"])

    matrix = {m: {t: {tier: 0 for tier in TIERS} for t in TASKS} for m in MODELS}
    per_model = {m: 0 for m in MODELS}
    per_task_counts: dict[str, int] = {}
    per_tier = {tier: 0 for tier in (*TIERS, UNKNOWN_TIER)}
    defects = {
        "unknown_tier": 0,
        "non_frontier_tools_source": 0,
        "missing_provenance_model": 0,
        "malformed_json": loaded["malformed"],
    }

    seen: set[tuple] = set()
    total = 0
    for row in loaded["rows"]:
        model = row["_model"]
        rid = stripped_id(row)
        key = ((model, rid) if rid
               else (model, row["_source_file"], row["_line"]))
        if key in seen:
            continue
        seen.add(key)

        task = derive_task(row)
        tier = resolve_tier(row)
        total += 1
        per_model[model] = per_model.get(model, 0) + 1
        per_task_counts[task] = per_task_counts.get(task, 0) + 1
        per_tier[tier] = per_tier.get(tier, 0) + 1
        if model in matrix and task in matrix[model] and tier in TIERS:
            matrix[model][task][tier] += 1

        if tier == UNKNOWN_TIER:
            defects["unknown_tier"] += 1
        if not str(row.get("source") or "").startswith("frontier"):
            # Ingest stamps `frontier:<model>`; the training mix canonicalises to
            # `frontier_tools`. Anything else is not a frontier trace at all.
            defects["non_frontier_tools_source"] += 1
        provenance = row.get("provenance")
        if not isinstance(provenance, dict) or not provenance.get("model"):
            defects["missing_provenance_model"] += 1

    zero_cells = [
        f"{model}/{task}/{tier}"
        for model in MODELS
        for task in TASKS
        for tier in TIERS
        if matrix[model][task][tier] == 0
    ]

    balance: dict[str, dict] = {}
    for model in MODELS:
        known = sum(matrix[model][t][tier] for t in TASKS for tier in TIERS)
        entry: dict = {"known_tier_rows": known}
        for tier in TIERS:
            count = sum(matrix[model][t][tier] for t in TASKS)
            entry[tier] = round(100.0 * count / known, 1) if known else 0.0
        balance[model] = entry

    per_task = {t: per_task_counts.get(t, 0) for t in TASKS}
    for task in sorted(k for k in per_task_counts if k not in TASKS):
        per_task[task] = per_task_counts[task]

    payload = {
        "total_accepted_traces": total,
        "per_model": per_model,
        "per_task": per_task,
        "per_tier": per_tier,
        "matrix": matrix,
        "zero_cells": zero_cells,
        "balance": balance,
        "defects": defects,
    }
    return payload, errors


def format_table(payload: dict) -> str:
    """Compact, fixed-width human view (same text goes into the Markdown)."""
    lines = [
        "Frontier CoT datagen progress (accepted traces, deduplicated per model)",
        f"{'model':<18} {'task':<9} {'easy':>6} {'medium':>6} {'hard':>6} {'total':>6}",
    ]
    for model in MODELS:
        for task in TASKS:
            cells = payload["matrix"][model][task]
            total = sum(cells[tier] for tier in TIERS)
            lines.append(
                f"{model:<18} {task:<9} {cells['easy']:>6} {cells['medium']:>6} "
                f"{cells['hard']:>6} {total:>6}"
            )
    grand = payload["total_accepted_traces"]
    tier_totals = [sum(payload["matrix"][m][t][tier] for m in MODELS for t in TASKS)
                   for tier in TIERS]
    lines.append(
        f"{'TOTAL':<18} {'':<9} {tier_totals[0]:>6} {tier_totals[1]:>6} "
        f"{tier_totals[2]:>6} {grand:>6}"
    )
    lines.append("")
    lines.append("per model : " + "  ".join(
        f"{m}={payload['per_model'].get(m, 0)}" for m in MODELS))
    lines.append("per task  : " + "  ".join(
        f"{t}={payload['per_task'].get(t, 0)}" for t in TASKS))
    lines.append("per tier  : " + "  ".join(
        f"{t}={payload['per_tier'].get(t, 0)}"
        for t in (*TIERS, UNKNOWN_TIER)))
    lines.append("balance   : " + "  ".join(
        f"{m} {payload['balance'][m]['easy']:.1f}/"
        f"{payload['balance'][m]['medium']:.1f}/"
        f"{payload['balance'][m]['hard']:.1f}% "
        f"(n={payload['balance'][m]['known_tier_rows']})" for m in MODELS))
    zero = payload["zero_cells"]
    lines.append(f"zero cells: {len(zero)}/24" + (": " + ", ".join(zero) if zero else ""))
    lines.append("defects   : " + "  ".join(
        f"{k}={payload['defects'][k]}" for k in (
            "unknown_tier", "non_frontier_tools_source",
            "missing_provenance_model", "malformed_json")))
    return "\n".join(lines)


def format_markdown(payload: dict, generated: str) -> str:
    """Human-readable Markdown version of the same deterministic payload."""
    out = [
        "# Frontier CoT datagen progress",
        "",
        f"<!-- generated by scripts/report_datagen_progress.py at {generated} -->",
        "",
        f"Accepted traces from `{COHORT_DIR}/<model>/*.jsonl`, deduplicated by "
        "`record_id`/`id` (with `#vN` suffixes stripped) within each model. "
        "The 24-cell matrix is 2 models x 4 tasks x 3 tiers.",
        "",
        "## Compact view",
        "",
        "```text",
        format_table(payload),
        "```",
        "",
        "## 24-cell matrix",
        "",
        "| model | task | easy | medium | hard |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        for task in TASKS:
            c = payload["matrix"][model][task]
            out.append(f"| {model} | {task} | {c['easy']} | {c['medium']} | {c['hard']} |")
    out += [
        "",
        "## Cells still at zero",
        "",
    ]
    if payload["zero_cells"]:
        out += [f"- `{cell}`" for cell in payload["zero_cells"]]
    else:
        out.append("None -- all 24 cells have at least one accepted trace.")
    out += [
        "",
        "## Per-model balance (easy / medium / hard, over known-tier rows)",
        "",
        "| model | known-tier rows | easy % | medium % | hard % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        b = payload["balance"][model]
        out.append(f"| {model} | {b['known_tier_rows']} | {b['easy']:.1f} | "
                   f"{b['medium']:.1f} | {b['hard']:.1f} |")
    out += [
        "",
        "## Data defects",
        "",
        "Rows with an unrecognised/missing tier are counted under `unknown` "
        "(and reported here) rather than silently dropped.",
        "",
        "| defect | rows |",
        "| --- | ---: |",
    ]
    for key, value in payload["defects"].items():
        out.append(f"| `{key}` | {value} |")
    out.append("")
    return "\n".join(out)


def _write(path: str, text: str) -> None:
    target = pathlib.Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--json", nargs="?", const=DEFAULT_JSON, default=None, metavar="PATH",
        help=f"also write the deterministic JSON payload "
             f"(default path when the flag is given: {DEFAULT_JSON})",
    )
    parser.add_argument(
        "--md", default=DEFAULT_MD, metavar="PATH",
        help=f"Markdown report path (default: {DEFAULT_MD})",
    )
    args = parser.parse_args(argv)

    payload, errors = build_report()
    if errors:
        for error in errors:
            print(f"ERROR: cannot read cohort file: {error}", file=sys.stderr)
        return 2

    print(format_table(payload))

    if args.json:
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        try:
            _write(args.json, text)
        except OSError as exc:
            print(f"ERROR: cannot write JSON report {args.json}: {exc}", file=sys.stderr)
            return 2
    if args.md:
        generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
        try:
            _write(args.md, format_markdown(payload, generated))
        except OSError as exc:
            print(f"ERROR: cannot write Markdown report {args.md}: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
