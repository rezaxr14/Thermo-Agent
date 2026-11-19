"""Tests for task tiers: equal thirds, determinism, grouped plant variants."""
import collections
import json
import pathlib

import pytest

from thermo_agent.synth.tiers import (
    TIERS, UNKNOWN_TIER, assign_tiers, assign_tiers_grouped, complexity,
    tier_ranges, tier_sizes,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _plant(n: int) -> dict:
    return {"components": [{"id": f"c{i}"} for i in range(n)]}


def _repair_row(rid: str, n: int) -> dict:
    return {"id": rid, "plant_broken": _plant(n)}


def _ledger_row(rid: str, n: int) -> dict:
    topo = {f"c{i}": {} for i in range(n)}
    return {"id": rid, "ledger": json.dumps({"data": {"topo": topo}})}


@pytest.mark.parametrize("n", list(range(1, 31)))
def test_tier_sizes_equal_thirds_remainder_to_hard(n):
    sizes = tier_sizes(n)
    assert sum(sizes) == n
    assert sizes[0] == sizes[1] == n // 3
    assert sizes[2] == n - 2 * (n // 3)
    assert sizes[0] <= sizes[1] <= sizes[2]


@pytest.mark.parametrize("task,row,expected", [
    ("repair", _repair_row("r", 7), 7),
    ("audit", {"id": "a", "plant": json.dumps(_plant(11))}, 11),
    ("ledger", _ledger_row("l", 9), 9),
])
def test_complexity_counts_components(task, row, expected):
    assert complexity(task, row) == expected


def test_complexity_unknown_task():
    with pytest.raises(KeyError):
        complexity("made_up", {})


def test_design_complexity_uses_reference_design():
    row = json.loads((ROOT / "data/processed/design_train.jsonl").open(encoding="utf-8").readline())
    assert complexity("design", row) >= 1  # build_design from site + ref_spec


def test_assign_tiers_equal_thirds_and_deterministic():
    rows = [_repair_row(f"r{i:04d}", 4 + (i % 6)) for i in range(120)]
    a = assign_tiers("repair", rows)
    b = assign_tiers("repair", list(reversed(rows)))
    assert a == b
    counts = collections.Counter(a.values())
    assert counts == {"easy": 40, "medium": 40, "hard": 40}
    assert all(t in TIERS for t in a.values())


def test_assign_tiers_remainder_goes_to_higher_tier():
    rows = [_repair_row(f"r{i:04d}", 4 + i % 4) for i in range(10)]
    counts = collections.Counter(assign_tiers("repair", rows).values())
    assert counts == {"easy": 3, "medium": 3, "hard": 4}


def test_assign_tiers_split_by_measured_size():
    rows = [_repair_row(f"r{i:03d}", i // 10) for i in range(60)]
    tiers = assign_tiers("repair", rows)
    ranges = tier_ranges("repair", rows, tiers)
    assert ranges["easy"]["max"] <= ranges["medium"]["max"] <= ranges["hard"]["max"]
    assert ranges["easy"]["min"] == 0
    assert ranges["hard"]["max"] == 5


def test_unscored_rows_do_not_become_hard():
    rows = [_repair_row(f"r{i:03d}", 4 + i % 3) for i in range(12)]
    rows[0] = {"id": "r000", "record_id": "r000"}  # no plant: complexity is None
    tiers = assign_tiers("repair", rows)
    assert "r000" not in tiers  # never invent a tier for an unmeasurable row


def test_grouped_variants_of_one_plant_share_a_tier():
    plants = [_repair_row(f"p{i:03d}", 4 + i % 5) for i in range(60)]
    rows = []
    for plant in plants:
        for variant in ("direct", "recover"):
            row = dict(plant)
            row["record_id"] = plant["id"]
            row["id"] = f"{plant['id']}-{variant}"
            rows.append(row)
    tiers = assign_tiers_grouped("repair", rows)
    counts = collections.Counter(tiers.values())
    assert counts == {"easy": 20, "medium": 20, "hard": 20}  # 60 plants
    assert len(tiers) == 60  # one entry per plant
    assert all(tier in TIERS for tier in tiers.values())


def test_grouped_rows_with_paraphrase_suffix_in_id_keep_their_tier():
    """Frontier ids are `<plant>#vN-t-<model>`; grouping must still key on `<plant>`.

    Regression: assign_tiers keyed representatives by their full record_id while
    assign_tiers_grouped grouped by the id with `#v...` stripped, so any group whose
    representative carried such a suffix silently lost its tier.
    """
    ids = [f"p{i:03d}" for i in range(60)]
    rows = []
    for i, pid in enumerate(ids):
        plant = _repair_row(pid, 4 + i % 5)
        for suffix in ("#v1-t-google-gemini-3-8-flash", "#v2-t-google-gemini-3-8-flash"):
            row = dict(plant)
            row["record_id"] = f"{pid}{suffix}"
            row["id"] = f"{pid}{suffix}"
            rows.append(row)
    tiers = assign_tiers_grouped("repair", rows)
    assert len(tiers) == 60  # one entry per ground-truth plant, none dropped
    assert all(tier in TIERS for tier in tiers.values())
    assert collections.Counter(tiers.values()) == {"easy": 20, "medium": 20, "hard": 20}


def test_unknown_tier_constant_documented():
    assert UNKNOWN_TIER == "<missing>"
