"""Batch selection must not starve a whole task/tier cell.

History: sorting candidate rows by id alone spent every easy slot on `audit/easy`
because `aud-*` sorts before `led-*`, so `ledger/easy` stayed empty across batches.
"""
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_next_budget_batch", ROOT / "scripts" / "build_next_budget_batch.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def row(rid, cell, priority):
    return (priority, rid, {"id": rid, "cell": cell})


def test_interleave_does_not_starve_a_later_sorting_cell():
    rows = [row("aud-1", "audit/easy", 0), row("aud-2", "audit/easy", 0),
            row("aud-3", "audit/easy", 0), row("led-1", "ledger/easy", 0),
            row("led-2", "ledger/easy", 0)]
    chosen = mod.interleave(rows, 4)
    cells = [entry["cell"] for _, _, entry in chosen]
    assert cells == ["audit/easy", "ledger/easy", "audit/easy", "ledger/easy"]
    assert len(chosen) == 4


def test_interleave_serves_missing_cells_first():
    rows = [row("aud-1", "audit/easy", 1), row("aud-2", "audit/easy", 1),
            row("led-1", "ledger/easy", 0)]
    chosen = mod.interleave(rows, 2)
    assert [entry["cell"] for _, _, entry in chosen] == ["ledger/easy", "audit/easy"]


def test_interleave_respects_quota_and_order_is_deterministic():
    rows = [row("aud-2", "audit/hard", 0), row("aud-1", "audit/hard", 0)]
    first = mod.interleave(rows, 1)
    second = mod.interleave(list(reversed(rows)), 1)
    assert len(first) == 1
    assert first[0][1] == second[0][1] == "aud-1"   # tie broken by id, not input order


def test_equal_third_quotas_are_33_33_34():
    quotas = {"easy": mod.BATCH_SIZE // 3, "medium": mod.BATCH_SIZE // 3,
              "hard": mod.BATCH_SIZE - 2 * (mod.BATCH_SIZE // 3)}
    assert quotas == {"easy": 33, "medium": 33, "hard": 34}

def test_stamp_rows_adds_tier_and_cell():
    all_rows = {"a": {"id": "a", "prompt": []}, "b": {"id": "b", "prompt": []}}
    chosen = [(0, "a", {"cell": "audit/easy"}), (1, "b", {"cell": "ledger/hard"})]
    lines, counts = mod.stamp_rows(chosen, all_rows)
    assert counts == {"easy": 1, "medium": 0, "hard": 1}
    rows = [json.loads(line) for line in lines]
    assert rows[0]["tier"] == "easy" and rows[0]["cell"] == "audit/easy"
    assert rows[1]["tier"] == "hard" and rows[1]["cell"] == "ledger/hard"


def test_stamp_rows_refuses_untiered_rows():
    """A shard without tiers selects 0 rows on the hosted runner (0-byte journal)."""
    import pytest
    all_rows = {"a": {"id": "a"}}
    with pytest.raises(SystemExit):
        mod.stamp_rows([(0, "a", {"cell": ""})], all_rows)
    with pytest.raises(SystemExit):
        mod.stamp_rows([(0, "a", {"cell": "audit"})], all_rows)
