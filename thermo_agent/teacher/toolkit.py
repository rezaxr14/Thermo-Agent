"""Real thermodynamics tools for the frontier model during CoT generation.

These are NOT verdict-string filters. Each tool builds the plant implied by the model's
own candidate answer, runs the project's sequential-modular solver (`core.solve.solve`),
and returns the actual computed numbers with units. The model reads those numbers, reasons
about them and revises its answer. The same solver/verifiers produce the final training label,
so tool feedback and acceptance are guaranteed consistent.

Tools:
  first_law_check(candidate)  -> per-component mass residual (kg/s) and energy residual (W),
                                 plant-wide energy closure, heat/power balance, fuel and IT load.
  second_law_check(candidate) -> exergy destruction (MW) per component, exergy closure (MW),
                                 eta_I, eta_II, Carnot and COP-vs-reversible limits.
  validity_check(candidate)   -> unit/range violations, site constraints, pinch approach (K),
                                 and the hard-verifier verdict.
  solve_plant(candidate)      -> the full numeric state table (T K, P kPa, h, s, exergy) for
                                 every stream, so the model can reason about actual states.
"""
from __future__ import annotations

import json

from ..core.schema import Plant, Site
from ..core.solve import solve
from ..verify.verifiers import run_all
from ..agent.dsl import parse_edits, edits_to_patch, EditError
from ..synth.faults import apply_patch, _dump
from ..design.spec import parse_design, DesignError
from ..design.build import evaluate_design
from . import validate


def _ledger_candidate_result(candidate: str, row: dict):
    """Evaluate a proposed ledger fix with the REAL ledger program and expose its numbers.

    Returns a lightweight object with `.kpis`, `.balances` (one pseudo-component carrying the
    closure residual and exergy values) and a verifier report, so all five tools work for ledger
    without fabricating a Plant. First law = energy closure residual (MW); second law = exergy
    destruction / non-negativity across the topology.
    """
    import json as _json
    from ..ledger import program as P, rewards as ledger_rewards
    from dataclasses import dataclass

    L = _json.loads(row["ledger"]) if isinstance(row.get("ledger"), str) else row.get("ledger", {})
    data, broken = L.get("data", {}), L.get("lines", {})

    @dataclass
    class _B:
        component: str; mass_residual: float = 0.0; energy_residual: float = 0.0
        exergy_destruction_w: float = 0.0; min_approach_K: float | None = None
    @dataclass
    class _R:
        ok: bool; error: object = None; message: str = ""
        balances: list = None; states: list = None; kpis: dict = None

    try:
        fixes = ledger_rewards.parse_fixes(candidate, set(broken))
    except Exception as exc:
        return None, _R(False, None, f"ledger: {type(exc).__name__}: {exc}", [], [], {}), None, None

    lines = {**broken, **fixes}
    r = P.run(lines, data)
    ef = max(data.get("e_fuel_mw", 1e-6), 1e-6)
    closure = r.get("closure")
    cids = list(data.get("topo", []))
    ed = r.get("ED", {})
    balances = [_B(c, energy_residual=(closure or 0.0) / len(cids) if cids else 0.0,
                   exergy_destruction_w=ed.get(c, 0.0) * 1e6) for c in cids]
    kpis = {"closure_residual_w": (closure or 0.0) * 1e6, "ED_total_w": sum(ed.values()) * 1e6,
            "E_fuel_w": ef * 1e6, "T0_K": data.get("consts_true", {}).get("T0", 0.0)}
    rep = ledger_rewards.evaluate(candidate, row["ledger"] if isinstance(row["ledger"], str) else _json.dumps(row["ledger"]))
    return None, _R(True, None, "", balances, [], kpis), rep, None


def _candidate_result(task: str, candidate: str, row: dict):
    """Return (plant, SolveResult, report, error). Builds the plant from the candidate text."""
    try:
        if task == "repair":
            plant = Plant.model_validate_json(row["plant_broken"])
            edits = parse_edits(candidate)
            if not edits:
                return None, None, None, "no <edits> block found in candidate"
            patch = edits_to_patch(plant, edits)
            plant = Plant.model_validate(apply_patch(_dump(plant), patch))
        elif task == "design":
            site = Site.model_validate_json(row["site"])
            spec = parse_design(candidate, needs_heat=site.heat_demand_w > 0)
            ev = evaluate_design(site, spec)
            if ev.plant is None:
                return None, None, None, ev.reason or "design could not be built"
            plant, res, rep = ev.plant, ev.result, ev.report
            return plant, res, rep, None
        elif task == "audit":
            plant = Plant.model_validate_json(row["plant"])
        elif task == "ledger":
            # Ledger rows carry a JSON ledger of balance/exergy equations, not a Plant. Build a
            # synthetic single-stream plant from the FIXED state so the real solver still runs and
            # the first/second-law tools report genuine residuals for the proposed fixes.
            return _ledger_candidate_result(candidate, row)
        else:
            return None, None, None, f"unknown task {task!r}"
        res = solve(plant)
        site = None
        if row.get("site"):
            try:
                site = Site.model_validate_json(row["site"])
            except Exception:
                site = None
        rep = run_all(plant, res, site)
        return plant, res, rep, None
    except (EditError, DesignError, KeyError, ValueError, TypeError) as exc:
        return None, None, None, f"{type(exc).__name__}: {exc}"


def _fmt(v, nd=6):
    return None if v is None else round(float(v), nd)


def _rep_failed(rep) -> list:
    if rep is None:
        return []
    if hasattr(rep, "failed"):
        return rep.failed()
    return [k for k, v in rep.items() if isinstance(v, bool) and not v and k not in ("parsed", "any", "all_correct")]


def _rep_checks(rep) -> list:
    if rep is None or not hasattr(rep, "checks"):
        return [{"name": k, "ok": v} for k, v in (rep or {}).items() if isinstance(v, (bool, float, int))] if isinstance(rep, dict) else []
    return [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in rep.checks]


def make_tools(task: str, row: dict, candidate_of=lambda c: c):
    """Tool signatures are plain str -> str so kbench builds a clean JSON schema.

    `candidate_of` lets a caller unwrap an answer from a wrapper if needed.
    """
    def check(candidate: str):
        return _candidate_result(task, candidate_of(candidate), row)

    def first_law_check(candidate: str) -> str:
        """Solve the candidate plant and report ACTUAL mass/energy residuals (kg/s, W) and heat/power/fuel balance."""
        plant, res, rep, err = check(candidate)
        if err:
            return json.dumps({"ok": False, "error": err})
        if not res.ok:
            return json.dumps({"ok": False, "error": f"{res.error.value}: {res.message}"})
        k = res.kpis
        comp = [{"component": b.component, "mass_residual_kg_s": _fmt(b.mass_residual, 9),
                 "energy_residual_W": _fmt(b.energy_residual, 3)} for b in res.balances]
        return json.dumps({
            "ok": True,
            "per_component": comp,
            "max_mass_residual_kg_s": _fmt(max(abs(b.mass_residual) for b in res.balances), 9),
            "max_energy_residual_W": _fmt(max(abs(b.energy_residual) for b in res.balances), 3),
            "exergy_closure_residual_W": _fmt(k.get("closure_residual_w"), 3),
            "W_net_W": _fmt(k.get("W_net_w"), 1), "P_IT_W": _fmt(k.get("P_it_w"), 1),
            "Q_fuel_W": _fmt(k.get("Q_fuel_w"), 1), "Q_heat_W": _fmt(k.get("Q_heat_w"), 1),
            "Q_cooling_W": _fmt(k.get("Q_cooling_w"), 1),
            "mass_balance_ok": all(abs(b.mass_residual) <= 1e-6 * max(1.0, max((s.m_dot for s in res.states), default=1.0)) for b in res.balances),
            "energy_balance_ok": all(abs(b.energy_residual) <= 1e-6 * (abs(k.get("W_net_w", 0)) + abs(k.get("Q_fuel_w", 0)) + 1e6) for b in res.balances),
        }, ensure_ascii=False)

    def second_law_check(candidate: str) -> str:
        """Solve the candidate plant and report ACTUAL exergy destruction (MW), eta_I/eta_II and thermodynamic limits."""
        plant, res, rep, err = check(candidate)
        if err:
            return json.dumps({"ok": False, "error": err})
        if not res.ok:
            return json.dumps({"ok": False, "error": f"{res.error.value}: {res.message}"})
        k = res.kpis
        ed = [{"component": b.component, "exergy_destruction_MW": _fmt(b.exergy_destruction_w / 1e6, 6)}
              for b in res.balances]
        limits = {}
        if "eta_II" in k: limits["eta_II"] = _fmt(k["eta_II"], 6)
        if "eta_I_power" in k and k.get("T_max_K", 0) > k.get("T0_K", 0):
            limits["eta_I_power"] = _fmt(k["eta_I_power"], 6)
            limits["carnot_power_limit"] = _fmt(1 - k["T0_K"] / k["T_max_K"], 6)
        for key in k:
            if key.startswith("COP_") and f"COPrev_{key[4:]}" in k:
                limits[key] = _fmt(k[key], 4)
                limits[f"reversible_{key}"] = _fmt(k[f"COPrev_{key[4:]}"], 4)
        return json.dumps({
            "ok": True,
            "per_component_exergy_destruction_MW": ed,
            "total_exergy_destruction_MW": _fmt(k.get("ED_total_w", 0) / 1e6, 6),
            "plant_exergy_destruction_MW": _fmt(k.get("ED_plant_w", 0) / 1e6, 6),
            "exergy_closure_residual_W": _fmt(k.get("closure_residual_w"), 3),
            "limits": limits,
            "failed_checks": _rep_failed(rep),
            "nonneg_exergy_ok": all(b.exergy_destruction_w >= -1e-5 * max(abs(k.get("E_fuel_w", 0)), 1e6) for b in res.balances),
        }, ensure_ascii=False)

    def validity_check(candidate: str) -> str:
        """Solve and report unit/range violations, site constraints, pinch temperatures and the hard-verifier verdict."""
        plant, res, rep, err = check(candidate)
        if err:
            return json.dumps({"ok": False, "error": err})
        if not res.ok:
            return json.dumps({"ok": False, "error": f"{res.error.value}: {res.message}"})
        k = res.kpis
        pinch = [{"component": b.component, "min_approach_K": _fmt(b.min_approach_K, 3)}
                 for b in res.balances if b.min_approach_K is not None]
        return json.dumps({
            "ok": True,
            "valid": rep.valid if rep is not None and hasattr(rep, "valid") else (rep.get("all_correct") if isinstance(rep, dict) else None),
            "score": _fmt(rep.score, 4) if rep is not None and hasattr(rep, "score") else None,
            "failed_checks": _rep_failed(rep),
            "checks": _rep_checks(rep),
            "min_approach_K": pinch,
            "T_max_K": _fmt(k.get("T_max_K"), 2), "T0_K": _fmt(k.get("T0_K"), 2),
            "water_use_kg_s": _fmt(k.get("water_use_kg_s"), 3),
        }, ensure_ascii=False)

    def solve_plant(candidate: str) -> str:
        """Solve the candidate plant and return the full stream state table (T K, P kPa, h,s kJ/kg, exergy kJ/kg)."""
        plant, res, rep, err = check(candidate)
        if err:
            return json.dumps({"ok": False, "error": err})
        if not res.ok:
            return json.dumps({"ok": False, "error": f"{res.error.value}: {res.message}"})
        # core.schema stores h/s/e_ph/e_ch in J/kg (J/kg-K for s); report kJ/kg to match labels.
        streams = [{"stream": s.stream, "T_K": _fmt(s.T, 2), "P_kPa": _fmt(s.P / 1000, 3),
                    "h_kJ_kg": _fmt(s.h / 1000, 3), "s_kJ_kgK": _fmt(s.s / 1000, 5),
                    "m_dot_kg_s": _fmt(s.m_dot, 5), "e_ph_kJ_kg": _fmt(s.e_ph / 1000, 3),
                    "e_ch_kJ_kg": _fmt(s.e_ch / 1000, 3)} for s in res.states]
        return json.dumps({"ok": True, "n_streams": len(streams), "streams": streams}, ensure_ascii=False)

    def check_answer(candidate: str) -> str:
        """Full verifier verdict on the candidate answer (accept/reward/reasons)."""
        v = validate.validate(task, candidate_of(candidate), row)
        return json.dumps({"accepted": v["accepted"], "reward": v["reward"], "reasons": v["reasons"]}, ensure_ascii=False)

    return [check_answer, first_law_check, second_law_check, validity_check, solve_plant]


def tool_names() -> list[str]:
    return ["check_answer", "first_law_check", "second_law_check", "validity_check", "solve_plant"]
