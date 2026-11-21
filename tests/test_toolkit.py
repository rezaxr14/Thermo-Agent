"""Gate 0 (P0-4 .. P0-8): physics tools return real numbers with units, no answer oracle.

Uses the FIRST REAL ROW of each processed train file -- no synthetic fixtures.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from thermo_agent.teacher.toolkit import SKILL_TOOL_NAMES, make_tools, tool_names
from thermo_agent.teacher.thermo_tools import DYNAMIC_TOOL_NAMES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

PHYSICS_TOOLS = ["inspect_plant", "first_law_check", "second_law_check", "validity_check", "solve_plant"]
ALL_TOOLS = PHYSICS_TOOLS + list(DYNAMIC_TOOL_NAMES) + list(SKILL_TOOL_NAMES)
NUM = (int, float)


@lru_cache(maxsize=None)
def _row(task: str) -> dict:
    with (DATA / f"{task}_train.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return json.loads(line)
    raise AssertionError(f"no row in {DATA / (task + '_train.jsonl')}")


def _truth(task: str) -> str:
    return _row(task)["completion"][0]["content"]


def _tools(task: str) -> dict:
    return {fn.__name__: fn for fn in make_tools(task, _row(task))}


def _j(text: str) -> dict:
    return json.loads(text)


# --------------------------------------------------------------------------------------
# P0-8: the oracle is gone from teacher AND student tool lists.
# --------------------------------------------------------------------------------------
def test_check_answer_absent_from_tool_lists():
    # 2026-11-20: all 16 tools are the default; the legacy 5-tool set is the opt-out.
    assert "check_answer" not in tool_names()
    assert tool_names() == ALL_TOOLS
    assert tool_names(include_dynamic=False) == PHYSICS_TOOLS
    assert len(ALL_TOOLS) == 16
    assert "check_answer" not in tool_names(include_dynamic=True)
    for task in ("repair", "audit", "ledger", "design"):
        tools = _tools(task)
        assert "check_answer" not in tools, task
        assert set(tools) == set(ALL_TOOLS), task
        legacy = {fn.__name__: fn for fn in make_tools(task, _row(task), include_dynamic=False)}
        assert "check_answer" not in legacy, task
        assert list(legacy) == PHYSICS_TOOLS, task
        dynamic = {fn.__name__: fn for fn in make_tools(task, _row(task), include_dynamic=True)}
        assert "check_answer" not in dynamic, task
        assert list(dynamic) == ALL_TOOLS, task
        # The nine extras are the unchanged module-level callables: fn(candidate).
        for name in DYNAMIC_TOOL_NAMES:
            assert json.loads(dynamic[name](""))["ok"] is False, (task, name)


# --------------------------------------------------------------------------------------
# P0-6: a plant that solves still carries first-class signal; units are real.
# --------------------------------------------------------------------------------------
def test_first_law_reports_units_and_first_class_signals():
    out = _j(_tools("repair")["first_law_check"](_truth("repair")))
    assert out["ok"] is True, out
    for key in ("max_mass_residual_kg_s", "max_energy_residual_W", "W_net_W", "P_IT_W",
                "Q_fuel_W", "Q_heat_W", "Q_cooling_W"):
        assert isinstance(out[key], NUM), (key, out.get(key))
    assert out["max_mass_residual_kg_s"] >= 0.0
    assert out["per_component"] and "mass_residual_kg_s" in out["per_component"][0]
    # first-class convergence / tear / unit-range / failed-check fields
    assert out["converged"] is True
    assert isinstance(out["tear_stream_residual"], NUM)
    assert isinstance(out["unit_range_flags"], list)
    assert isinstance(out["failed_checks"], list)


def test_second_law_and_validity_and_solve_have_real_units():
    repair = _tools("repair")
    gt = _truth("repair")
    second = _j(repair["second_law_check"](gt))
    assert second["ok"] is True
    assert isinstance(second["total_exergy_destruction_MW"], NUM)
    assert second["per_component_exergy_destruction_MW"]
    assert "exergy_destruction_MW" in second["per_component_exergy_destruction_MW"][0]
    assert isinstance(second["failed_checks"], list)

    validity = _j(repair["validity_check"](gt))
    assert validity["ok"] is True
    assert isinstance(validity.get("valid"), bool)
    assert isinstance(validity["score"], NUM)
    assert isinstance(validity["unit_range_flags"], list)

    solved = _j(repair["solve_plant"](gt))
    assert solved["ok"] is True and solved["n_streams"] > 0
    s0 = solved["streams"][0]
    for key in ("T_K", "P_kPa", "h_kJ_kg", "s_kJ_kgK", "m_dot_kg_s", "e_ph_kJ_kg"):
        assert isinstance(s0[key], NUM), (key, s0.get(key))


def test_adversarial_candidates_have_nonempty_failure_list():
    """wrong_scale / echo_broken must expose a failure, not an empty first-law result."""
    from thermo_agent.teacher.validate import adversarial_candidates

    row = _row("repair")
    tools = _tools("repair")
    by_name = dict(adversarial_candidates(row))
    for name in ("wrong_scale", "echo_broken"):
        assert name in by_name, (name, list(by_name))
        out = _j(tools["first_law_check"](by_name[name]))
        assert out["ok"] is False, (name, out)
        assert out["failed_checks"], (name, out)


# --------------------------------------------------------------------------------------
# P0-7: inspect the broken plant / ledger as given, with no candidate.
# --------------------------------------------------------------------------------------
def test_inspect_plant_on_broken_repair_plant():
    tools = _tools("repair")
    out = _j(tools["inspect_plant"](""))
    assert out["ok"] is True
    assert out["task"] == "repair"
    assert out["solved"] is False
    assert out["solver_status"] != "OK"
    assert out["failed_checks"], out                      # solver failure is first-class
    assert isinstance(out["max_energy_residual_W"], NUM)
    # candidate is ignored: inspecting first never needs an <edits> block
    assert _j(tools["inspect_plant"]("<edits>\nSET NOPE.x = 1\n</edits>")) == out


def test_inspect_plant_on_ledger_reports_real_per_line_residuals():
    out = _j(_tools("ledger")["inspect_plant"](""))
    assert out["ok"] is True and out["task"] == "ledger"
    assert isinstance(out["ledger_closure_residual_W"], NUM)
    lines = out["per_line_residuals"]
    assert lines
    assert any(x["line"].startswith("E[") for x in lines)
    assert any(x["line"].startswith("ED[") for x in lines)
    assert all("residual_W" in x for x in lines)
    # F4 (2026-11-20): the tool now states only recomputed values and residuals —
    # it must NOT reveal the faulty line via failed_lines / ok flags / line names.
    assert "failed_lines" not in out
    assert all("ok" not in x for x in lines)
    assert all("recomputed_MW" in x for x in lines)
    # residuals are real per-line numbers (MW-scale), not closure divided by line count
    assert any(abs(x["residual_W"]) > 1e6 for x in lines), lines


# --------------------------------------------------------------------------------------
# P0-4: real first-law residuals for ledger; malformed candidate is an error, not a crash.
# --------------------------------------------------------------------------------------
def test_ledger_first_law_uses_per_line_residuals():
    out = _j(_tools("ledger")["first_law_check"](_truth("ledger")))
    assert out["ok"] is True
    assert isinstance(out["max_energy_residual_W"], NUM)
    lines = out["per_line_residuals"]
    assert lines and all("residual_W" in x for x in lines)
    assert any(x["line"].startswith("E[") for x in lines)
    assert any(x["line"].startswith("ED[") for x in lines)
    assert isinstance(out["ledger_closure_residual_W"], NUM)


def test_malformed_ledger_candidate_returns_error():
    tools = _tools("ledger")
    candidate_tools = [n for n in PHYSICS_TOOLS if n != "inspect_plant"]
    for bad in ("", "garbage with no fix block", "<fix>\nNOT A FIX LINE\n</fix>"):
        for name in candidate_tools:
            out = _j(tools[name](bad))
            assert out["ok"] is False, (name, bad, out)
            assert isinstance(out["error"], str) and out["error"], (name, out)
            assert out["failed_checks"], (name, out)


# --------------------------------------------------------------------------------------
# P0-5: the audit candidate's SET change is applied and re-solved.
# --------------------------------------------------------------------------------------
def test_two_audit_candidates_give_different_output():
    tools = _tools("audit")

    def candidate(change: str) -> str:
        return ("<analysis>fuel 194 MW</analysis><ranking>CC > GT > AC</ranking>"
                f"<change>{change}</change><prediction>fuel_change_mw = 0</prediction>")

    lower = candidate("SET CC.T_out = 1559.19")   # saves fuel (ground-truth direction)
    higher = candidate("SET CC.T_out = 1275.7")   # costs fuel

    for name in ("first_law_check", "second_law_check", "validity_check", "solve_plant"):
        a, b = _j(tools[name](lower)), _j(tools[name](higher))
        assert a["candidate_effect"]["change"] != b["candidate_effect"]["change"], name
        assert a["candidate_effect"]["applied"] is True, (name, a)
        assert b["candidate_effect"]["applied"] is True, (name, b)
        assert a["candidate_effect"]["delta_fuel_W"] != b["candidate_effect"]["delta_fuel_W"], name
        assert a != b, name

    first_a = _j(tools["first_law_check"](lower))
    first_b = _j(tools["first_law_check"](higher))
    assert first_a["Q_fuel_W"] != first_b["Q_fuel_W"]
    assert first_a["candidate_effect"]["new_fuel_W"] != first_b["candidate_effect"]["new_fuel_W"]
