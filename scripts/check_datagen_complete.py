"""Hard gate: refuse training while any datagen item is incomplete.

Reza's directive (DECISIONS 2026-11-19 section 4) says training starts only when
ALL datagen is complete. This script is the machine-checkable form of that rule.
It is read-only and exits non-zero with a list of missing items.

    python scripts/check_datagen_complete.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_dataset_provenance import audit_dirs  # noqa: E402

TOOL_ROWS_EXPECTED = 4_800          # DECISIONS E1: 600 plants x 2 variants x 4 tasks
A_ROWS_EXPECTED = 1_600             # DECISIONS E2/J1: round(4,800 x 0.3333)
FRONTIER_CLEAN = "data/frontier_tools_clean_v2"
V2_TOOL_DIRS = ("data/sft_tools_v2", "data/grpo_tools_v2",
                "data/sft_tools_v2_h", "data/grpo_tools_v2_h")
MIX_DIRS = ("data/sft_mix_v2", "data/grpo_mix_v2")


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

    if all(counts.values()):
        prov = audit_dirs(list(MIX_DIRS), root=ROOT)
        item("provenance/tier audit", prov["passed"],
             f"{prov['rows']} rows, {prov['problem_count']} problems")
    else:
        item("provenance/tier audit", False, "v2 tool corpora not built yet")

    item("budget frontier pilots", (ROOT / "data/frontier_tools_pilot").is_dir()
         and sum(1 for p in (ROOT / "data/frontier_tools_pilot").glob("*.jsonl")
                 for _ in p.open(encoding="utf-8") if _.strip()) > 0,
         "gpt-6-luna / claude-haiku-5-5 payload proof + pilot traces")

    return {"passed": all(i["ok"] for i in items), "items": items}


def main() -> int:
    report = check()
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
