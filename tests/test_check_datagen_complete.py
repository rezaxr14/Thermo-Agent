"""Tests for the no-training-until-datagen-done gate."""
import json

import scripts.check_datagen_complete as gate


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_gate_lists_all_items_and_is_not_passed_on_empty_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    report = gate.check()
    assert report["passed"] is False
    names = {i["name"] for i in report["items"]}
    assert {"data/sft_tools_v2 rows", "A-slice", "heldout_v2 frozen",
            "provenance/tier audit", "budget frontier pilots"} <= names


def test_gate_passes_when_all_artifacts_present(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    good = {"id": "x", "task": "repair", "source": "deterministic_tools",
            "source_model": None, "tier": "easy",
            "provenance": {"generator": "deterministic_tools", "variant": "direct"}}
    for d in gate.V2_TOOL_DIRS + gate.MIX_DIRS:
        _jsonl(tmp_path / d / "repair_train.jsonl", [good] * gate.TOOL_ROWS_EXPECTED)
    _jsonl(tmp_path / "data/a_slice/repair_train.jsonl", [dict(good, source="ground_truth")] * gate.A_ROWS_EXPECTED)
    _jsonl(tmp_path / gate.FRONTIER_CLEAN / "frontier_repair_train.jsonl", [good] * 32)
    _jsonl(tmp_path / "data/frontier_tools_pilot/frontier_repair_train.jsonl", [good])
    heldout = tmp_path / "data/heldout_v2"
    _jsonl(heldout / "repair_test_id.jsonl", [good] * (85 * 12))
    (heldout / "MANIFEST.json").write_text(json.dumps({"set_sha256": "abc"}), encoding="utf-8")
    report = gate.check()
    assert report["passed"] is True, [i for i in report["items"] if not i["ok"]]
