"""Derive the cohort task sources from the proven pilot task.

The payload, bootstrap, compact harness, tool loop and journal format are kept
byte-for-byte; only the task name, the embedded per-model id list and the
journal cell bookkeeping change.  The id lists come from the global assignment
ledger (``scripts/frontier_cohort_assign.py``), so no diagram is sent to two
models unless the first attempt failed verifier grading.  Keeping this
generator makes the hosted task diff reviewable.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle" / "teacher_budget_pilot_task_s00.py"
PLAN_DIR = ROOT / "data" / "frontier_tools_cohort_prompts"
OUT = {
    "proof": ROOT / "kaggle" / "teacher_cohort_proof_task_s00.py",
    "bulk": ROOT / "kaggle" / "teacher_cohort_task_s00.py",
}
TASK_NAME = {
    "proof": "thermo-teacher-cohort-proof-s00",
    "bulk": "thermo-teacher-cohort-s00",
}


def _replace(text: str, old: str, new: str) -> str:
    if old not in text:
        raise AssertionError(f"pilot source missing anchor: {old[:70]!r}")
    return text.replace(old, new, 1)


def _plan(phase: str) -> tuple[dict, dict]:
    model_ids: dict[str, list[str]] = {}
    row_cell: dict[str, str] = {}
    for path in sorted(PLAN_DIR.glob(f"plan_{phase}_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data["model"]
        ids = [r["id"] for r in data["rows"]]
        if ids:
            model_ids[model] = ids
        for r in data["rows"]:
            row_cell[r["id"]] = r["cell"]
    return model_ids, row_cell


def make(text: str, phase: str, model_ids: dict, row_cell: dict) -> str:
    embedded = (
        "# Embedded from scripts/frontier_cohort_assign.py plans (global one-model-per-diagram ledger).\n"
        "MODEL_IDS = " + repr(model_ids) + "\n"
        "ROW_CELL_IDS = " + repr(row_cell) + "\n\n"
    )
    text = _replace(text, '"task_name": "thermo-teacher-budget-pilot-s00"',
                          f'"task_name": "{TASK_NAME[phase]}"')
    text = _replace(text, '"limit": 40,', '"limit": 0,')
    text = _replace(text, 'TASKS = ("repair", "ledger", "audit", "design")\n',
                          'TASKS = ("repair", "ledger", "audit", "design")\n\n' + embedded)
    # Replace the whole _load_rows function (ends just before _memory_dir_candidates).
    start = text.index('def _load_rows(root: pathlib.Path):')
    end = text.index('def _memory_dir_candidates(root: pathlib.Path):')
    new_loader = '''def _load_rows(root: pathlib.Path, model_slug: str):
    """Load ONLY the diagrams the global ledger assigned to this model.

    Rows are the unmodified tool-ready inputs from the pilot's
    ``thermo-teacher-prompts-big`` dataset.  The id list is produced by
    ``scripts/frontier_cohort_assign.py``; no bookkeeping id, tier label or
    manifest text is added to the model-visible prompt.  Per-model journal
    dedupe (below) still handles resume.
    """
    wanted = list(MODEL_IDS.get(model_slug, ()))
    target = set(wanted)
    found: dict[str, dict] = {}
    for base in (pathlib.Path("/kaggle/input"), root, pathlib.Path(".")):
        if not base.exists():
            continue
        for path in base.rglob("prompts_s*.jsonl"):
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                rid = str(row.get("id", ""))
                if rid in target and rid not in found:
                    found[rid] = _adapt_row(row)
    ordered = [found[rid] for rid in wanted if rid in found]
    print("[cohort] model=%s wanted=%d found=%d"
          % (model_slug, len(wanted), len(ordered)), flush=True)
    return ordered


'''
    text = text[:start] + new_loader + text[end:]
    text = _replace(text, 'global MEMORY_NOTES_LOADED, REASONING_LEVEL',
                          'global MEMORY_NOTES_LOADED, REASONING_LEVEL, ROW_CELL')
    text = _replace(text, 'REASONING_LEVEL = None  # "none" when the model rejects tools + reasoning_effort',
                          'REASONING_LEVEL = None  # "none" when the model rejects tools + reasoning_effort\n'
                          'ROW_CELL: dict[str, str] = {}  # id -> task/tier, journal bookkeeping only')
    text = _replace(text, 'rows, ROWS_SOURCE, ROWS_DETAIL = _load_rows(BUNDLE_ROOT)',
                          'rows = _load_rows(BUNDLE_ROOT, MODEL_SLUG)\n'
                          '    ROWS_SOURCE, ROWS_DETAIL = "cohort-ledger", SPEC["task_name"]')
    text = _replace(text, '"shard": SHARD, "row_source": ROWS_SOURCE, "spec_sha": SPEC_SHA,',
                          '"shard": SHARD, "row_source": ROWS_SOURCE, "spec_sha": SPEC_SHA,\n'
                          '                "cell": ROW_CELL.get(rid, ""),')
    fn = "thermo_teacher_cohort_" + phase + "_s00"
    text = _replace(text, 'def thermo_teacher_budget_pilot_s00(llm):',
                          f'def {fn}(llm):')
    text = _replace(text, 'thermo_teacher_budget_pilot_s00.run(kbench.llm)',
                          f'{fn}.run(kbench.llm)')
    return text


def main() -> None:
    src = SRC.read_text(encoding="utf-8")
    for phase in ("proof", "bulk"):
        model_ids, row_cell = _plan(phase)
        OUT[phase].write_text(make(src, phase, model_ids, row_cell), encoding="utf-8")
        print("wrote", OUT[phase], "models:", {m: len(v) for m, v in model_ids.items()})


if __name__ == "__main__":
    main()
