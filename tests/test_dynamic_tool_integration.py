"""End-to-end wiring of the nine continuous-input tools into the live harness.

Covers the explicit opt-in (OFF by default), the shared ``fn(candidate)`` parser
path, the compact token contract, the repair skill map, and the hosted
``pipeline.dynamic_tools`` flag that the Kaggle runner reads.

No model calls, no network, no Kaggle: everything here is local CPU only.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from thermo_agent.agent import harness as H
from thermo_agent.agent.tool_prompt import build_compact_harness_messages
from thermo_agent.teacher import agent_graph
from thermo_agent.teacher.toolkit import SKILL_TOOL_NAMES, make_tools, tool_names
from thermo_agent.teacher.thermo_tools import DYNAMIC_TOOL_NAMES, dynamic_tools

ROOT = Path(__file__).resolve().parents[1]
PLANT_TOOLS = ["inspect_plant", "first_law_check", "second_law_check",
               "validity_check", "solve_plant"]
ALL_TOOLS = PLANT_TOOLS + list(DYNAMIC_TOOL_NAMES) + list(SKILL_TOOL_NAMES)


# --------------------------------------------------------------------------- #
# 1. opt-in registry: default 5, hosted 16 (14 physics + 2 skill), no oracle
# --------------------------------------------------------------------------- #
def test_registry_default_off_and_opt_in_on():
    assert tool_names() == PLANT_TOOLS
    assert tool_names(include_dynamic=True) == ALL_TOOLS
    assert len(ALL_TOOLS) == 16
    assert len(set(ALL_TOOLS)) == 16
    assert "check_answer" not in tool_names()
    assert "check_answer" not in tool_names(include_dynamic=True)

    row = {"prompt": [{"role": "system", "content": "SYS"},
                      {"role": "user", "content": "U"}]}
    assert [f.__name__ for f in make_tools("repair", row)] == PLANT_TOOLS
    assert [f.__name__ for f in make_tools("repair", row, include_dynamic=True)] == ALL_TOOLS


def test_dynamic_callables_keep_the_single_candidate_slot():
    import inspect

    fns = dynamic_tools()
    assert [fn.__name__ for fn in fns] == list(DYNAMIC_TOOL_NAMES)
    for fn in fns:
        sig = inspect.signature(fn)
        assert list(sig.parameters) == ["candidate"]
        assert sig.parameters["candidate"].default == ""
        assert json.loads(fn(""))["ok"] is False          # never raises
        assert isinstance(fn(json.dumps({})), str)


def test_dynamic_tools_return_real_numbers():
    dyn = {fn.__name__: fn for fn in dynamic_tools()}
    state = json.loads(dyn["fluid_state"](json.dumps(
        {"fluid": "Water", "T_K": 300.0, "P_kPa": 101.325})))
    assert state["ok"] is True
    assert state["fluid"] == "Water"
    assert isinstance(state["T_K"], float) and state["T_K"] == pytest.approx(300.0)
    sat = json.loads(dyn["saturation_state"](json.dumps({"fluid": "Water", "T_K": 373.15})))
    assert sat["ok"] is True and sat["T_sat_K"] == pytest.approx(373.15, abs=0.01)


# --------------------------------------------------------------------------- #
# 2. the shared parser path executes fn(candidate) for dynamic tools too
# --------------------------------------------------------------------------- #
def test_agent_graph_call_tools_executes_dynamic_candidate_json(corpus_fixture_rows):
    row = corpus_fixture_rows["repair"][0]
    ctx = agent_graph.build_context({"row": row, "task": "repair", "notes": [],
                                     "mode": "agents+skills", "include_dynamic": True})
    assert [t.__name__ for t in ctx["tools"]] == ALL_TOOLS

    candidate = json.dumps({"fluid": "Water", "T_K": 300.0, "P_kPa": 101.325})
    state = {"last_calls": [{"name": "fluid_state", "arguments": {"candidate": candidate}}],
             "last_text": "", "last_reasoning": "", "tool_map": ctx["tool_map"], "turns": []}
    out = agent_graph.call_tools(state)
    result = json.loads(out["turns"][0]["tool_results"][0])
    assert result["ok"] is True and result["fluid"] == "Water"
    # The exact JSON-object string reached the tool (not a wrapped answer).
    assert out["turns"][0]["tool_calls"][0]["arguments"]["candidate"] == candidate


def test_compact_prompt_names_dynamic_tools_only_when_included(corpus_fixture_rows):
    row = corpus_fixture_rows["design"][0]
    base = [{"role": "system", "content": "base"}, {"role": "user", "content": "u"}]
    without = build_compact_harness_messages(list(base), "design", None)[0]["content"]
    with_dyn = build_compact_harness_messages(
        list(base), "design", None, tools=make_tools("design", row, include_dynamic=True)
    )[0]["content"]
    assert all(n not in without for n in DYNAMIC_TOOL_NAMES)
    assert all(n in with_dyn for n in DYNAMIC_TOOL_NAMES)
    assert "JSON object string" not in without
    assert "JSON object string" in with_dyn
    assert "check_answer" not in with_dyn


# --------------------------------------------------------------------------- #
# 3. skills: repair has a skill; component skills stay out of the compact path
# --------------------------------------------------------------------------- #
def test_repair_skill_and_component_skills_excluded_from_compact(corpus_fixture_rows):
    assert H.TASK_SKILLS["repair"] == ["repair-diagnosis"]
    # Component skills are selected by frontmatter for the verbose/augment path.
    row = corpus_fixture_rows["repair"][0]
    from thermo_agent.core.schema import Plant
    plant = None
    for key in ("plant_broken", "plant", "plant_original"):
        if row.get(key):
            try:
                plant = Plant.model_validate_json(row[key])
                break
            except Exception:
                pass
    if plant is not None:
        assert any(n not in H.TASK_SKILLS["repair"] for n in H.skills_for(plant, "repair"))

    # The compact builder deliberately serves only the task's first skill.
    compact = build_compact_harness_messages(
        [{"role": "system", "content": "base"}], "repair", plant)[0]["content"]
    assert "SKILL repair-diagnosis" in compact
    for name in H.skill_names():
        if name != "repair-diagnosis":
            assert f"SKILL {name}" not in compact, f"component skill {name} leaked into compact"


# --------------------------------------------------------------------------- #
# 4. the hosted frontier harness reads the flag and forwards it
# --------------------------------------------------------------------------- #
def test_hosted_config_enables_dynamic_tools():
    cfg = json.loads((ROOT / "data" / "frontier_tools_cohort_prompts" /
                      "harness_config.json").read_text(encoding="utf-8"))
    assert cfg["pipeline"]["dynamic_tools"] is True
    # The batch contract is otherwise untouched.
    assert cfg["limit"] == 100 and cfg["rounds"] == 6 and cfg["k_memory"] == 3
    assert cfg["pipeline"]["gate_tokens"] == 8192
    assert set(cfg["allowed_models"]) == {"gpt-6-luna", "claude-haiku-5-5"}


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "runner_dyn_under_test", ROOT / "notebooks" / "kaggle_model_runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_forwards_dynamic_flag_to_tools_context_and_direct():
    runner = _load_runner()
    assert runner.DEFAULTS["pipeline"]["dynamic_tools"] is False  # default stays off
    assert runner.Config.from_dict({}).dynamic_tools is False
    assert runner.Config.from_dict(
        {"pipeline": {"dynamic_tools": True}}).dynamic_tools is True
    assert runner.Config.from_dict(
        {"pipeline": {"dynamic_tools": "true"}}).dynamic_tools is True

    code = (ROOT / "notebooks" / "kaggle_model_runner.py").read_text(encoding="utf-8")
    assert "tool_names(include_dynamic=CFG.dynamic_tools)" in code
    assert '"include_dynamic": CFG.dynamic_tools' in code
    assert "include_dynamic=CFG.dynamic_tools)" in code  # run_direct call


def test_runner_config_shard_limit_and_models_unchanged():
    cfg = json.loads((ROOT / "data" / "frontier_tools_cohort_prompts" /
                      "harness_config.json").read_text(encoding="utf-8"))
    runner = _load_runner()
    parsed = runner.Config.from_dict(cfg)
    assert parsed.limit == 100 and parsed.rounds == 6 and parsed.k_memory == 3
    assert parsed.uses_tools is True and parsed.dynamic_tools is True
    assert set(parsed.allowed_models) == {"gpt-6-luna", "claude-haiku-5-5"}
