"""Gate 2: the single shared multi-turn ThermoEnv (no network; one real repair row).

The row comes from ``data/processed/repair_train.jsonl`` and the whole module is
marked ``corpus`` so it skips cleanly on a fresh clone where that untracked file
is absent (see tests/conftest.py).
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import get_type_hints

import pytest

from thermo_agent.agent.env import ThermoEnv
from thermo_agent.teacher import toolkit
from thermo_agent.teacher.toolkit import SKILL_TOOL_NAMES
from thermo_agent.teacher.thermo_tools import DYNAMIC_TOOL_NAMES
from thermo_agent.teacher.validate import validate

pytestmark = pytest.mark.corpus

ROOT = Path(__file__).resolve().parents[1]
ROW_PATH = ROOT / "data" / "processed" / "repair_train.jsonl"
TOOL_NAMES = ["inspect_plant", "first_law_check", "second_law_check",
              "validity_check", "solve_plant"]
ALL_TOOL_NAMES = TOOL_NAMES + list(DYNAMIC_TOOL_NAMES) + list(SKILL_TOOL_NAMES)


@pytest.fixture(scope="module")
def row() -> dict:
    with ROW_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return json.loads(line)
    pytest.skip(f"no real repair row in {ROW_PATH}")


def _truth(row: dict) -> str:
    return row["completion"][0]["content"]


def _prompt(row: dict) -> str:
    return "\n\n".join(m["content"] for m in row["prompt"]
                       if m.get("role") != "system" and m.get("content"))


# ---------------------------------------------------------------------------
# tool list
# ---------------------------------------------------------------------------
def test_tools_names_equal_toolkit(row: dict):
    env = ThermoEnv(row, "repair")
    assert [fn.__name__ for fn in env.tools] == toolkit.tool_names()
    assert toolkit.tool_names() == TOOL_NAMES
    # Opt-in hosted mode exposes all 16 tools (14 physics + 2 skill) via the parser path.
    assert toolkit.tool_names(include_dynamic=True) == ALL_TOOL_NAMES
    dyn = ThermoEnv(row, "repair", include_dynamic=True)
    assert [fn.__name__ for fn in dyn.tools] == toolkit.tool_names(include_dynamic=True)
    assert [fn.__name__ for fn in dyn.tools] == ALL_TOOL_NAMES
    assert "check_answer" not in ALL_TOOL_NAMES


def test_no_oracle_tool(row: dict):
    names = [fn.__name__ for fn in ThermoEnv(row, "repair").tools]
    assert "check_answer" not in names
    assert not any("answer" in n or "oracle" in n for n in names)


def test_dynamic_tools_are_callable_through_the_shared_parser(row: dict):
    """The hosted opt-in binds the 9 JSON-object tools behind the same fn(candidate) path."""
    env = ThermoEnv(row, "repair", include_dynamic=True)
    fluid = json.loads(env._call("fluid_state", json.dumps(
        {"fluid": "Water", "T_K": 300.0, "P_kPa": 101.325})))
    assert fluid["ok"] is True and fluid["fluid"] == "Water"
    assert isinstance(fluid["T_K"], float)
    # Invalid JSON is a uniform ok:false payload, never an exception.
    bad = json.loads(env._call("turbomachine", "not json"))
    assert bad["ok"] is False and bad["error"]
    # Default env still has only the five plant tools.
    assert len(ThermoEnv(row, "repair").tools) == 5


def test_observation_block_names_dynamic_tools_only_when_enabled(row: dict):
    default = ThermoEnv(row, "repair").observation_block()
    dynamic = ThermoEnv(row, "repair", include_dynamic=True).observation_block()
    assert all(n not in default for n in DYNAMIC_TOOL_NAMES)
    assert all(n in dynamic for n in DYNAMIC_TOOL_NAMES)
    assert "JSON object string" in dynamic


# ---------------------------------------------------------------------------
# tool signatures, docstrings and real JSON output
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", TOOL_NAMES)
def test_tool_method_contract(row: dict, name: str):
    env = ThermoEnv(row, "repair")
    method = getattr(env, name)

    doc = inspect.getdoc(method)
    assert doc and doc.strip(), name
    assert "Args:" in doc and "Returns:" in doc, name  # Google-style (TRL schema)

    sig = inspect.signature(method)
    hints = get_type_hints(method)
    assert sig.return_annotation is not inspect.Signature.empty
    for param in sig.parameters.values():
        if param.name == "self":
            continue
        assert param.annotation is not inspect.Parameter.empty, (name, param.name)
        assert param.name in hints, (name, param.name)

    candidate = "" if name == "inspect_plant" else _truth(row)
    out = method(candidate)
    assert isinstance(out, str), name
    payload = json.loads(out)  # valid JSON, returned unchanged from the toolkit
    assert isinstance(payload, dict), name
    assert ("ok" in payload) or ("error" in payload), (name, payload)


# ---------------------------------------------------------------------------
# reset / tool-call cap / step
# ---------------------------------------------------------------------------
def test_reset_returns_prompt_and_zeroes_state(row: dict):
    env = ThermoEnv(row, "repair", max_tool_calls=1)
    assert env.reset() == _prompt(row)

    env.first_law_check(_truth(row))
    env.step(_truth(row))
    assert env.done

    assert env.reset() == _prompt(row)
    assert env.turn == 0
    assert env.tool_calls == 0
    assert env.history == []
    assert env.done is False
    assert env.final_answer is None
    assert env.reward is None


def test_reset_rebinds_row_and_task(row: dict):
    env = ThermoEnv(row, "repair")
    assert env.reset(row=row, task="ledger") == _prompt(row)
    assert env.task == "ledger"


def test_exceeding_max_tool_calls_raises(row: dict):
    env = ThermoEnv(row, "repair", max_tool_calls=2)
    env.first_law_check(_truth(row))
    env.second_law_check(_truth(row))
    with pytest.raises(RuntimeError, match="tool-call limit"):
        env.validity_check(_truth(row))
    assert env.tool_calls == 2  # the rejected call is not counted


def test_step_returns_verifier_reward_and_done(row: dict):
    env = ThermoEnv(row, "repair")
    answer = _truth(row)
    expected = validate("repair", answer, row)["reward"]
    got = env.step(answer)
    assert got == pytest.approx(expected)
    assert env.done is True
    assert env.final_answer == answer
    assert env.reward == pytest.approx(expected)
    assert env.history[-1] == {"final": answer}


# ---------------------------------------------------------------------------
# observation block: compact, D11-clean, per-task final XML contract
# ---------------------------------------------------------------------------
def test_observation_block_no_bookkeeping(row: dict):
    block = ThermoEnv(row, "repair").observation_block()

    # required content
    assert "MULTI-TURN TOOL ENVIRONMENT" in block
    assert "6 assistant turns" in block and "8 tool calls" in block
    assert "inventory" in block.lower()
    for name in TOOL_NAMES:
        assert name in block

    # no model names, ISO timestamps, or cost / token figures (D11)
    assert not re.search(r"\b(?:gemini|gpt[- ]?\d|claude|haiku|sonnet|luna|qwen|llama)\b", block, re.I)
    assert not re.search(r"\b20\d{2}-\d{2}-\d{2}(?:[T ][0-9:.+-]+)?", block)
    assert not re.search(r"\b(?:cost|tokens?|usd|latency|price|dollars?|model[_-]?id)\b", block, re.I)
    for bad in ("$", "€", "time=", "timestamp"):
        assert bad not in block


@pytest.mark.parametrize("task, tags", [
    ("repair", ("<diagnosis>", "<edits>")),
    ("design", ("<rationale>", "<design>")),
    ("audit", ("<analysis>", "<ranking>", "<change>", "<prediction>")),
    ("ledger", ("<diagnosis>", "<fix>")),
])
def test_observation_block_final_xml_per_task(row: dict, task: str, tags: tuple):
    block = ThermoEnv(row, task).observation_block()
    for tag in tags:
        assert tag in block, (task, tag)
