"""heldout_v2 tiers are REAL: equal thirds per task from the documented size measure."""
from __future__ import annotations

import collections
import json
import pathlib

import pytest

from thermo_agent.synth.tiers import assign_tiers, tier_ranges, tier_sizes

ROOT = pathlib.Path(__file__).resolve().parents[1]
V2 = ROOT / "data/heldout_v2"

TASKS = ("repair", "audit", "ledger", "design")
TIERS = ("easy", "medium", "hard")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((V2 / "MANIFEST.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows_by_task() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {t: [] for t in TASKS}
    for path in sorted(V2.glob("*_test_*.jsonl")):
        task = path.name.split("_", 1)[0]
        with path.open(encoding="utf-8") as fh:
            out[task].extend(json.loads(line) for line in fh if line.strip())
    return out


@pytest.mark.parametrize("task", TASKS)
def test_tiers_equal_thirds_and_real(task, rows_by_task, manifest):
    rows = rows_by_task[task]
    assert rows, task
    # Re-derive from the data itself, exactly as the builder does.
    derived = assign_tiers(task, rows)
    assert len(derived) == len(rows), f"{task}: unscored rows cannot be tiered"
    for row in rows:
        rid = str(row.get("record_id") or row["id"])
        assert row.get("tier") in TIERS, (task, rid, row.get("tier"))
        assert row["tier"] == derived[rid], (task, rid, row["tier"], derived[rid])

    counts = collections.Counter(r["tier"] for r in rows)
    expected = dict(zip(TIERS, tier_sizes(len(rows))))
    assert {t: counts[t] for t in TIERS} == expected, (task, counts, expected)

    # Manifest records the same counts and the measured ranges.
    t = manifest["tiers"][task]
    assert t["n"] == len(rows)
    assert t["counts"] == expected
    assert t["ranges"] == tier_ranges(task, rows, derived)


@pytest.mark.parametrize("task", TASKS)
def test_tiers_are_deterministic_not_random(task, rows_by_task):
    rows = rows_by_task[task]
    first = assign_tiers(task, rows)
    second = assign_tiers(task, list(reversed(rows)))
    assert first == second, f"{task}: tier assignment depends on row order"
    assert first == {str(r.get("record_id") or r["id"]): r["tier"] for r in rows}


@pytest.mark.parametrize("task", TASKS)
def test_every_row_has_a_tier_and_a_real_generator_seed(task, rows_by_task):
    for r in rows_by_task[task]:
        assert r["tier"] in TIERS
        assert isinstance(r["gen_seed"], int)
        # Base seed plus a small documented source offset (or +1000*topup attempt).
        assert 20261119 <= r["gen_seed"] <= 20261119 + 11 + 1000 * 8
