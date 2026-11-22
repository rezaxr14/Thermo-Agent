"""Step 3: the audit prompt must name rankable ids, and DC must be rejected."""
from __future__ import annotations

import json
import pathlib

import pytest

from thermo_agent.agent.tool_prompt import build_harness_tool_messages
from thermo_agent.audit.rewards import evaluate
from thermo_agent.core.schema import Plant

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/audit_train.jsonl"

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def row():
    if not SOURCE.is_file():
        pytest.skip("no real processed audit row")
    return json.loads(SOURCE.read_text(encoding="utf-8").splitlines()[0])


def test_audit_harness_states_the_rankable_ids(row):
    plant = Plant.model_validate_json(row["plant"])
    messages = build_harness_tool_messages(list(row["prompt"]), "audit", plant)
    system = messages[0]["content"]
    rankable = [c.id for c in plant.components if c.type not in ("combustor", "gas_engine")]
    assert "Rankable audit ids" in system
    for component_id in rankable:
        assert component_id in system
    assert "Never rank any other id." in system


def test_unknown_dc_ranking_is_rejected(row):
    answer = row["completion"][0]["content"]
    corrupted = answer.replace("<ranking>", "<ranking>DC > ", 1)
    result = evaluate(corrupted, row["plant"], row.get("site", ""), row["ed_json"],
                      float(row["fuel_w"]), float(row["net_w"]))
    assert result["parsed"] is False
    assert "DC" not in json.loads(row["ed_json"])
