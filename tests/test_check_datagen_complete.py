"""Tests for the no-training-until-datagen-done gate."""
import functools
import json
import pathlib

import scripts.check_datagen_complete as gate

REPO = pathlib.Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=1)
def _design_template() -> dict:
    path = REPO / "data/processed/design_train.jsonl"
    return json.loads(path.open(encoding="utf-8").readline())


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_gate_lists_all_items_and_is_not_passed_on_empty_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    report = gate.check()
    assert report["passed"] is False
    names = {i["name"] for i in report["items"]}
    assert {"data/sft_tools_v2 rows", "A-slice", "heldout_v2 frozen",
            "provenance/tier audit",
            "frontier CoT cohort (models x tasks x tiers)"} <= names


def _frontier_row(model: str, task: str, n: int, rid: str) -> dict:
    if task == "design":
        # design complexity is rebuilt from site + ref_spec; use a real row so the
        # measure is computable (synthetic dicts lack required spec keys).
        template = _design_template()
        site, spec = template["site"], template["ref_spec"]
    else:
        site = json.dumps({"it_load_w": 1e7, "cooling_load_w": 5e6,
                           "ambient": {"T0": 298.15, "P0": 101325.0, "RH": 0.6},
                           "fuel": "CH4", "grid_import_allowed": False})
        spec = json.dumps({"prime_mover": "engine", "cooling": "electric",
                           "rejection": "ext", "bottoming": "none"})
    return {
        "id": rid, "record_id": rid, "task": task, "model": model,
        "plant_broken": json.dumps({"components": [{"id": f"c{i}"} for i in range(n)]}),
        "plant": json.dumps({"components": [{"id": f"c{i}"} for i in range(n)]}),
        "ledger": json.dumps({"data": {"topo": {f"c{i}": {} for i in range(n)}}}),
        "site": site,
        "ref_spec": spec,
        "completion": [{"role": "assistant",
                        "content": "<think>x</think><tool_call name=\"inspect_plant\">{"
                                   "}</tool_call><tool_result>ok</tool_result><answer>x</answer>"}],
        "validation": {"accepted": True},
        "provenance": {"generator": "frontier_tools", "model": model},
    }


def test_cohort_item_is_red_for_pilots_only(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    rows = [_frontier_row("gpt-6-luna", "repair", 4 + i, f"rep-{i}") for i in range(3)]
    _jsonl(tmp_path / "data/frontier_tools_pilot/frontier_repair_train.jsonl", rows)
    report = gate.check()
    cohort = next(i for i in report["items"] if i["name"].startswith("frontier CoT"))
    assert cohort["ok"] is False
    assert "gpt-6-luna/repair" in cohort["detail"] or "claude-haiku-5-5" in cohort["detail"]


def test_gate_passes_when_all_artifacts_present(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    good = {"id": "x", "task": "repair", "source": "deterministic_tools",
            "source_model": None, "tier": "easy",
            "provenance": {"generator": "deterministic_tools", "variant": "direct"}}
    for d in gate.V2_TOOL_DIRS + gate.MIX_DIRS:
        _jsonl(tmp_path / d / "repair_train.jsonl", [good] * gate.TOOL_ROWS_EXPECTED)
    _jsonl(tmp_path / "data/a_slice/repair_train.jsonl",
           [dict(good, source="ground_truth")] * gate.A_ROWS_EXPECTED)
    _jsonl(tmp_path / gate.FRONTIER_CLEAN / "frontier_repair_train.jsonl", [good] * 32)
    for model in gate.FRONTIER_MODELS:
        for task in gate.tier_mod.TASKS:
            rows = [_frontier_row(model, task, 4 + i * 2, f"{model}-{task}-{i}")
                    for i in range(3)]
            _jsonl(tmp_path / f"data/frontier_tools_pilot/frontier_{model}_{task}_train.jsonl", rows)
    heldout = tmp_path / "data/heldout_v2"
    _jsonl(heldout / "repair_test_id.jsonl", [good] * (85 * 12))
    (heldout / "MANIFEST.json").write_text(json.dumps({"set_sha256": "abc"}), encoding="utf-8")
    report = gate.check()
    assert report["passed"] is True, [i for i in report["items"] if not i["ok"]]
