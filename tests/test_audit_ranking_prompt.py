"""Step 4: the audit prompt must advertise exactly the ids the validator accepts.

`audit.truth.ranking` excludes `dc_load` from `ed_json` (the IT load is not part of
the plant), so the harness must not list DC as rankable: a model that follows the
old prompt emitted `DC > ...` and the validator rejected the row as an unknown id.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

from thermo_agent.agent.tool_prompt import build_harness_tool_messages
from thermo_agent.audit.rewards import evaluate
from thermo_agent.core.schema import Plant

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/audit_train.jsonl"
EXCLUDED = ("combustor", "gas_engine", "dc_load")

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def row():
    if not SOURCE.is_file():
        pytest.skip("no real processed audit row")
    return json.loads(SOURCE.read_text(encoding="utf-8").splitlines()[0])


def _rankable_line(system: str) -> str:
    match = re.search(r"Rankable audit ids[^\n]*", system)
    assert match, "the harness must state the rankable audit ids"
    return match.group(0)


def test_audit_harness_states_the_rankable_ids(row):
    plant = Plant.model_validate_json(row["plant"])
    messages = build_harness_tool_messages(list(row["prompt"]), "audit", plant)
    system = messages[0]["content"]
    rankable = [c.id for c in plant.components if c.type not in EXCLUDED]
    assert "Rankable audit ids" in system
    for component_id in rankable:
        assert component_id in _rankable_line(system)
    assert "Never rank any other id." in system


def test_audit_harness_matches_the_validator_ids(row):
    """The advertised ids are exactly the ed_json keys the validator accepts."""
    plant = Plant.model_validate_json(row["plant"])
    system = build_harness_tool_messages(list(row["prompt"]), "audit", plant)[0]["content"]
    advertised = set(re.split(r"[\s,]+", _rankable_line(system).split("): ", 1)[1].split(". Never")[0]))
    valid = set(json.loads(row["ed_json"]))
    excluded = {c.id for c in plant.components if c.type in EXCLUDED}
    assert advertised == valid - excluded


def test_audit_harness_does_not_advertise_dc_load(row):
    plant = Plant.model_validate_json(row["plant"])
    assert any(c.type == "dc_load" for c in plant.components), "fixture must contain the IT load"
    system = build_harness_tool_messages(list(row["prompt"]), "audit", plant)[0]["content"]
    line = _rankable_line(system)
    assert "DC" not in re.split(r"[\s,]+", line)
    assert "DC" not in json.loads(row["ed_json"])


def test_unknown_dc_ranking_is_rejected(row):
    answer = row["completion"][0]["content"]
    corrupted = answer.replace("<ranking>", "<ranking>DC > ", 1)
    result = evaluate(corrupted, row["plant"], row.get("site", ""), row["ed_json"],
                      float(row["fuel_w"]), float(row["net_w"]))
    assert result["parsed"] is False
    assert "DC" not in json.loads(row["ed_json"])
