"""Hard gate: refuse training while any datagen item is incomplete.

Reza's directive (DECISIONS 2026-11-19 section 4) says training starts only when
ALL datagen is complete. This script is the machine-checkable form of that rule.
It is read-only and exits non-zero with a list of missing items.

    python scripts/check_datagen_complete.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_dataset_provenance import audit_dirs  # noqa: E402
from scripts.mix_tool_corpora import is_verifier_accepted  # noqa: E402
from thermo_agent.synth import tiers as tier_mod  # noqa: E402

TOOL_ROWS_EXPECTED = 4_800          # DECISIONS E1: 600 plants x 2 variants x 4 tasks
A_ROWS_EXPECTED = 1_600             # DECISIONS E2/J1: round(4,800 x 0.3333)
FRONTIER_CLEAN = "data/frontier_tools_clean"
V2_TOOL_DIRS = ("data/sft_tools_v3", "data/grpo_tools_v3",
                "data/sft_tools_v3_h", "data/grpo_tools_v3_h")
MIX_DIRS = ("data/sft_mix_v3", "data/grpo_mix_v3")

# DECISIONS section 6: the gate stays red after the luna/haiku pilots. Every
# model in the decided cohort must have accepted harness-CoT traces in all four
# tasks and all three tiers. These are coverage floors, never caps (D1..D9).
# Reza (2026-11-20): forget the 72-cell target. Only the 24 BUDGET cells
# (2 models x 4 tasks x 3 tiers) count for the training gate. Premium models are
# not part of this gate and must never be run without an explicit go.
FRONTIER_MODELS: dict[str, tuple[str, ...]] = {
    "gpt-6-luna": ("gpt-6-luna", "openai/gpt-6-luna"),
    "claude-haiku-5-5": ("claude-haiku-5-5", "anthropic/claude-haiku-5-5"),
}
FRONTIER_DIR_GLOBS = ("data/frontier_tools*/**/*.jsonl", "data/frontier_tools_clean*/**/*.jsonl")


def _frontier_rows() -> list[dict]:
    """Accepted tool-bearing rows for the 24 BUDGET cells, deduplicated by diagram.

    Only `gpt-6-luna` and `claude-haiku-5-5` count toward the training gate (Reza,
    2026-11-20, DECISIONS §6). Old gemini/premium traces may still live under
    `data/frontier_tools*`, so they must not inflate the reported cohort count.

    A diagram can appear in more than one store (cohort, accepted, pilot, clean)
    and with a `#vN` retry suffix. Dedupe on the base diagram id so one accepted
    trace is counted once, never once per copy.
    """
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict] = []
    for pattern in FRONTIER_DIR_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                model = _model_name(row)
                if model is None:
                    continue
                task = str(row.get("task") or "")
                blob = json.dumps(row, ensure_ascii=False)
                if "<tool_call" not in blob or "<tool_result" not in blob:
                    continue
                # Same acceptance convention as mix_tool_corpora: `transfer.ingest`
                # records validation={reward, reasons} with no `accepted` key, so a
                # positive reward with no rejection marker must count here too.
                if not is_verifier_accepted(row):
                    continue
                base_id = str(row.get("record_id") or row.get("id") or "").split("#v", 1)[0]
                key = (model, task, base_id)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows


def _model_name(row: dict) -> str | None:
    model = str(row.get("model") or (row.get("provenance") or {}).get("model") or "")
    for canonical, aliases in FRONTIER_MODELS.items():
        if any(alias in model for alias in aliases):
            return canonical
    return None


def _count(directory: str) -> int:
    root = ROOT / directory
    if not root.is_dir():
        return 0
    return sum(1 for path in root.glob("*.jsonl")
               for line in path.open(encoding="utf-8") if line.strip())


def check() -> dict:
    """Return {passed, items: [{name, ok, detail}]} for every datagen item."""
    items: list[dict] = []

    def item(name: str, ok: bool, detail: str) -> None:
        items.append({"name": name, "ok": bool(ok), "detail": detail})

    counts = {d: _count(d) for d in V2_TOOL_DIRS}
    for d in V2_TOOL_DIRS:
        item(f"{d} rows", counts[d] >= TOOL_ROWS_EXPECTED,
             f"{counts[d]} rows; expected >= {TOOL_ROWS_EXPECTED}")
    item("A-slice", _count("data/a_slice") == A_ROWS_EXPECTED,
         f"{_count('data/a_slice')} rows; expected {A_ROWS_EXPECTED}")
    item("frontier cleaned traces", _count(FRONTIER_CLEAN) >= 32,
         f"{_count(FRONTIER_CLEAN)} rows; 32 gemini-3.8-flash traces exist today")
    for d in MIX_DIRS:
        item(f"{d} rows", _count(d) >= TOOL_ROWS_EXPECTED, f"{_count(d)} rows")

    v2_dir = ROOT / "data/heldout_v2"
    manifest = v2_dir / "MANIFEST.json"
    ok = manifest.is_file() and _count("data/heldout_v2") >= 85 * 12
    detail = "frozen MANIFEST.json and >= 85 rows in each of 12 cells" if ok else "not frozen"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            detail += f"; hash={data.get('set_sha256') or data.get('sha256') or '<none>'}"
        except ValueError:
            ok = False
            detail = "unreadable MANIFEST.json"
    item("heldout_v2 frozen", ok, detail)

    # Reza's directive §4.2: equal thirds in the TEST set too, not just training.
    tier_detail: dict[str, dict[str, int]] = {}
    tier_bad: list[str] = []
    for task in tier_mod.TASKS:
        rows = []
        for path in sorted(v2_dir.glob(f"{task}_*.jsonl")):
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
        counts_task = collections.Counter(r.get("tier") for r in rows)
        tier_detail[task] = {t: counts_task[t] for t in tier_mod.TIERS}
        sizes = tier_mod.tier_sizes(len(rows)) if rows else (0, 0, 0)
        if not rows or any(tier_detail[task][t] != sizes[i]
                           for i, t in enumerate(tier_mod.TIERS)):
            tier_bad.append(task)
    item("heldout_v2 tiers equal thirds", not tier_bad,
         f"bad tasks: {tier_bad}; counts={tier_detail}")

    if all(counts.values()):
        prov = audit_dirs(list(MIX_DIRS), root=ROOT)
        item("provenance/tier audit", prov["passed"],
             f"{prov['rows']} rows, {prov['problem_count']} problems")
    else:
        item("provenance/tier audit", False, "v2 tool corpora not built yet")

    # Pilots are NOT enough (DECISIONS section 6). Every decided model must have
    # accepted harness-CoT traces in all four tasks and all three tiers.
    frontier = _frontier_rows()
    by_model: dict[str, list[dict]] = {m: [] for m in FRONTIER_MODELS}
    for row in frontier:
        model = _model_name(row)
        if model:
            by_model[model].append(row)
    missing: list[str] = []
    coverage: dict[str, dict[str, int]] = {}
    for model, model_rows in by_model.items():
        for task in tier_mod.TASKS:
            task_rows = [r for r in model_rows if r.get("task") == task]
            scored = [r for r in task_rows if tier_mod.complexity(task, r) is not None]
            counts = collections.Counter(tier_mod.assign_tiers(task, scored).values())
            coverage[f"{model}/{task}"] = {t: counts[t] for t in tier_mod.TIERS}
            for t in tier_mod.TIERS:
                if counts[t] == 0:
                    missing.append(f"{model}/{task}/{t}")
    item("frontier CoT cohort (models x tasks x tiers)", not missing,
         f"{len(frontier)} accepted tool traces; missing cells: {missing[:12]}"
         + (" ..." if len(missing) > 12 else "") + f"; coverage={coverage}")

    return {"passed": all(i["ok"] for i in items), "items": items}


def main() -> int:
    report = check()
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
