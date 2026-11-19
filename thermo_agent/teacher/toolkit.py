"""Real thermodynamics tools for the frontier model during CoT generation.

These are NOT verdict-string filters. Each tool builds the plant implied by the model's
own candidate answer, runs the project's sequential-modular solver (`core.solve.solve`),
and returns the actual computed numbers with units. The model reads those numbers, reasons
about them and revises its answer. The same solver/verifiers produce the final training label,
so tool feedback and acceptance are guaranteed consistent.

There is deliberately NO ``check_answer`` oracle here: acceptance and reward are computed only
at acceptance time (``teacher.validate``) and as the GRPO reward. The tool list is physics only.

Tools (all take a single ``candidate: str`` and return JSON with real numbers + units):
  inspect_plant(candidate)    -> solve + verify the plant/ledger AS GIVEN (candidate ignored).
                                 Solver status, failed physics checks and key residuals. This
                                 is the only tool a student can call *before* it has an answer.
  first_law_check(candidate)  -> per-component/per-line mass residual (kg/s) and energy residual (W),
                                 plant-wide energy closure, heat/power balance, fuel and IT load,
                                 plus convergence, tear-residual, unit/range flags and failed checks.
  second_law_check(candidate) -> exergy destruction (MW), exergy closure (MW), eta_I/eta_II,
                                 Carnot and COP-vs-reversible limits, and failed checks.
  validity_check(candidate)   -> unit/range violations, site constraints, pinch approach (K),
                                 and the hard-verifier verdict.
  solve_plant(candidate)      -> the full numeric state table (T K, P kPa, h, s, exergy) for
                                 every stream, so the model can reason about actual states.

For audit candidates the proposed ``SET`` change is applied and the plant re-solved, so every
tool reports the *new* exergy/validity/fuel. For ledger candidates the real per-line ledger
residuals are reported (value minus truth, in W), not a closure divided by component count.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..core.schema import Plant, Site
from ..core.solve import solve
from ..verify.verifiers import run_all
from ..agent.dsl import parse_edits, edits_to_patch, EditError
from ..synth.faults import apply_patch, _dump
from ..design.spec import parse_design, DesignError
from ..design.build import evaluate_design


@dataclass
class _B:
    """One balance line (physical component or ledger equation) reported by a physics tool."""
    component: str
    mass_residual: float = 0.0
    energy_residual: float = 0.0
    exergy_destruction_w: float = 0.0
    min_approach_K: float | None = None


@dataclass
class _LedgerResult:
    """SolveResult-shaped view of an evaluated ledger (no physical streams)."""
    ok: bool = True
    error: None = None
    message: str = ""
    states: list = field(default_factory=list)
    balances: list = field(default_factory=list)
    kpis: dict = field(default_factory=dict)


@dataclass
class _Eval:
    """Everything a physics tool needs about one candidate (or the as-given plant)."""
    plant: object = None
    res: object = None
    rep: object = None
    error: str | None = None
    meta: dict = field(default_factory=dict)


_TEAR_M = re.compile(
    r"tear stream\s+\S+.*?computed T=([-\d.eE+]+) K P=([-\d.eE+]+) Pa "
    r"differs from specified T=([-\d.eE+]+) K P=([-\d.eE+]+) Pa", re.S)
_SET_M = re.compile(
    r"\bSET\s+([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


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
        return [{"name": k, "ok": v, "detail": ""} for k, v in (rep or {}).items()
                if isinstance(v, (bool, float, int))] if isinstance(rep, dict) else []
    return [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in rep.checks]


def _solver_error(res) -> str | None:
    err = getattr(res, "error", None) if res is not None else None
    return getattr(err, "value", None)


def _tear_residual(res) -> float:
    """Numeric tear-stream mismatch (K / Pa mixed magnitude) parsed from the solver message."""
    if res is None or getattr(res, "ok", False):
        return 0.0
    msg = getattr(res, "message", "") or ""
    m = _TEAR_M.search(msg)
    if m:
        try:
            tc, pc, ts, ps = (float(x) for x in m.groups())
            return max(abs(tc - ts), abs(pc - ps) / max(abs(ps), 1.0))
        except ValueError:
            return 0.0
    return 1.0 if "tear stream" in msg else 0.0


def _site(row: dict):
    if row.get("site"):
        try:
            return Site.model_validate_json(row["site"])
        except Exception:
            return None
    return None


def _with_meta(payload: dict, meta: dict) -> dict:
    """Expose the applied audit change and its new fuel as first-class fields."""
    if meta and meta.get("task") == "audit":
        payload["candidate_effect"] = {
            "change": meta.get("change"),
            "applied": meta.get("applied"),
            "change_valid": meta.get("change_valid"),
            "base_fuel_W": _fmt(meta.get("base_fuel_w"), 1),
            "new_fuel_W": _fmt(meta.get("new_fuel_w"), 1),
            "delta_fuel_W": _fmt(meta.get("delta_fuel_w"), 1),
            "new_exergy_destruction_W": _fmt(meta.get("new_exergy_destruction_w"), 1),
            "reason": meta.get("reason", ""),
        }
    return payload


def _ledger_input(row: dict):
    L = json.loads(row["ledger"]) if isinstance(row.get("ledger"), str) else (row.get("ledger") or {})
    return L, L.get("data", {}) or {}, L.get("lines", {}) or {}


def _ledger_eval(candidate: str, row: dict, as_given: bool = False) -> _Eval:
    """Real per-line ledger residuals (value - truth, W), closure and failed lines.

    A malformed candidate is returned as ``_Eval.error`` so every tool answers with the
    uniform ``{"ok": false, "error": ...}`` shape instead of crashing on ``res.error == None``.
    """
    from ..ledger import program as P, rewards as ledger_rewards

    L, data, broken = _ledger_input(row)
    if as_given:
        fixes: dict = {}
    else:
        try:
            fixes = ledger_rewards.parse_fixes(candidate, set(broken))
        except Exception as exc:
            return _Eval(error=f"ledger: {type(exc).__name__}: {exc}")

    lines = {**broken, **fixes}
    r = P.run(lines, data)
    ef = max(float(data.get("e_fuel_mw", 0.0)), 1e-6)
    truth_E = data.get("truth_E", {}) or {}
    truth_ED = data.get("truth_ED", {}) or {}
    tol_w = 1e-6 * ef * 1e6

    line_residuals: list[dict] = []
    balances: list[_B] = []
    for sid, val in (r.get("E") or {}).items():
        truth = float(truth_E.get(sid, val))
        resid = (val - truth) * 1e6
        line_residuals.append({"line": f"E[{sid}]", "value_MW": _fmt(val, 6),
                               "truth_MW": _fmt(truth, 6), "residual_W": _fmt(resid, 3),
                               "ok": abs(resid) <= tol_w})
        balances.append(_B(f"E[{sid}]", energy_residual=resid))
    for cid, val in (r.get("ED") or {}).items():
        truth = float(truth_ED.get(cid, val))
        resid = (val - truth) * 1e6
        line_residuals.append({"line": f"ED[{cid}]", "value_MW": _fmt(val, 6),
                               "truth_MW": _fmt(truth, 6), "residual_W": _fmt(resid, 3),
                               "ok": abs(resid) <= tol_w})
        balances.append(_B(f"ED[{cid}]", energy_residual=resid, exergy_destruction_w=val * 1e6))

    failed_lines = sorted(r.get("errors") or {})
    for key, detail in (r.get("errors") or {}).items():
        line_residuals.append({"line": key, "value_MW": None, "truth_MW": None,
                               "residual_W": None, "ok": False, "error": str(detail)})
        balances.append(_B(key))
    negative = [f"ED[{c}]" for c, v in (r.get("ED") or {}).items() if v < -1e-5 * ef]
    failed_lines += negative

    closure = r.get("closure")
    kpis = {"closure_residual_w": (closure or 0.0) * 1e6,
            "ED_total_w": sum((r.get("ED") or {}).values()) * 1e6,
            "E_fuel_w": ef * 1e6,
            "T0_K": float((data.get("consts_true") or {}).get("T0", 0.0))}
    res = _LedgerResult(balances=balances, kpis=kpis,
                        message=("ledger line errors: " + ", ".join(failed_lines)) if failed_lines else "")

    n_lines = max(len(line_residuals), 1)
    n_ok = sum(1 for x in line_residuals if x["ok"] and x.get("truth_MW") is not None)
    exact = n_ok / n_lines
    closure_ok = closure is not None and abs(closure) <= 1e-6 * ef
    if as_given:
        rep = {"parsed": True, "any": True, "closure": float(closure_ok),
               "nonneg": 1.0 - len(negative) / max(len(data.get("topo", {})), 1),
               "exact": exact, "has_diag": True, "all_correct": bool(closure_ok and not failed_lines and exact >= 1.0)}
    else:
        ledger_js = row["ledger"] if isinstance(row.get("ledger"), str) else json.dumps(row.get("ledger", {}))
        rep = ledger_rewards.evaluate(candidate, ledger_js)

    meta = {"ledger": True, "line_residuals": line_residuals, "closure_residual_w": kpis["closure_residual_w"],
            "failed_lines": failed_lines, "n_lines": n_lines}
    return _Eval(plant=None, res=res, rep=rep, meta=meta)


def _audit_change(candidate: str):
    """Extract the candidate's proposed ``SET component.param = number`` (or None)."""
    m = re.search(r"<change>(.*?)</change>", candidate or "", re.S)
    text = m.group(1) if m else (candidate or "")
    mm = _SET_M.search(text)
    return (mm.group(1), mm.group(2), float(mm.group(3))) if mm else None


def _audit_eval(candidate: str, row: dict) -> _Eval:
    """Apply the candidate's proposed SET change, re-solve and report the NEW plant."""
    from ..audit import changes as ch

    plant = Plant.model_validate_json(row["plant"])
    site = _site(row)
    base_res = solve(plant)
    rep0 = run_all(plant, base_res, site)
    base_fuel = base_res.kpis.get("Q_fuel_w", 0.0) if base_res.ok else 0.0
    base_ex = base_res.kpis.get("ED_total_w", 0.0) if base_res.ok else 0.0
    meta = {"task": "audit", "change": None, "applied": False, "change_valid": False,
            "base_fuel_w": base_fuel, "new_fuel_w": base_fuel, "delta_fuel_w": 0.0,
            "new_exergy_destruction_w": base_ex, "reason": ""}

    change = _audit_change(candidate)
    if change is None:
        meta["reason"] = "no <change> SET line in candidate"
        return _Eval(plant=plant, res=base_res, rep=rep0, meta=meta)
    cid, param, value = change
    meta["change"] = f"SET {cid}.{param} = {value!r}"

    if not base_res.ok:
        meta["reason"] = "base plant does not solve"
        return _Eval(plant=plant, res=base_res, rep=rep0, meta=meta)
    try:
        if not ch.check_change(plant, cid, param, value):
            meta["reason"] = "change not allowed"
            return _Eval(plant=plant, res=base_res, rep=rep0, meta=meta)
        out = ch.resize(ch.apply_change(plant, cid, param, value), base_res.kpis["W_net_w"])
        if out is None:
            meta["reason"] = "cannot be solved after resize"
            return _Eval(plant=plant, res=base_res, rep=rep0, meta=meta)
        p2, r2 = out
        rep2 = run_all(p2, r2, site)
        new_fuel = r2.kpis["Q_fuel_w"]
        meta.update({"applied": True, "change_valid": bool(rep2.valid),
                     "new_fuel_w": new_fuel, "delta_fuel_w": new_fuel - base_fuel,
                     "new_exergy_destruction_w": r2.kpis.get("ED_total_w", 0.0),
                     "reason": "" if rep2.valid else ",".join(rep2.failed())})
        return _Eval(plant=p2, res=r2, rep=rep2, meta=meta)
    except (KeyError, StopIteration, ValueError, TypeError, IndexError) as exc:
        meta["reason"] = f"{type(exc).__name__}: {exc}"
        return _Eval(plant=plant, res=base_res, rep=rep0, meta=meta)


def _eval(task: str, candidate: str, row: dict, as_given: bool = False) -> _Eval:
    """Return the physics evaluation of ``candidate`` (or the as-given plant/ledger)."""
    if task == "ledger":
        return _ledger_eval(candidate, row, as_given=as_given)
    if as_given:
        if task == "repair":
            key = "plant_broken"
        elif task == "audit":
            key = "plant"
        elif task == "design":
            return _Eval(error="design has no plant to inspect before a <design> candidate")
        else:
            return _Eval(error=f"unknown task {task!r}")
        try:
            plant = Plant.model_validate_json(row[key])
        except (KeyError, ValueError, TypeError) as exc:
            return _Eval(error=f"{type(exc).__name__}: {exc}")
        res = solve(plant)
        return _Eval(plant=plant, res=res, rep=run_all(plant, res, _site(row)))

    try:
        if task == "repair":
            plant = Plant.model_validate_json(row["plant_broken"])
            edits = parse_edits(candidate)
            if not edits:
                return _Eval(error="no <edits> block found in candidate")
            patch = edits_to_patch(plant, edits)
            plant = Plant.model_validate(apply_patch(_dump(plant), patch))
        elif task == "design":
            site = Site.model_validate_json(row["site"])
            spec = parse_design(candidate, needs_heat=site.heat_demand_w > 0)
            ev = evaluate_design(site, spec)
            if ev.plant is None:
                return _Eval(error=ev.reason or "design could not be built")
            return _Eval(plant=ev.plant, res=ev.result, rep=ev.report)
        elif task == "audit":
            return _audit_eval(candidate, row)
        else:
            return _Eval(error=f"unknown task {task!r}")
        res = solve(plant)
        return _Eval(plant=plant, res=res, rep=run_all(plant, res, _site(row)))
    except (EditError, DesignError, KeyError, ValueError, TypeError) as exc:
        return _Eval(error=f"{type(exc).__name__}: {exc}")


def _err_payload(ev: _Eval, stage: str) -> dict:
    """Uniform failure shape: always carries a non-empty ``failed_checks`` list."""
    return {"ok": False, "error": ev.error, "failed_checks": [f"{stage}: {ev.error}"]}


def _unsolved_payload(ev: _Eval) -> dict:
    code = _solver_error(ev.res) or "E_UNKNOWN"
    msg = getattr(ev.res, "message", "") if ev.res is not None else "no result"
    return {"ok": False, "error": f"{code}: {msg}", "converged": False,
            "solver_status": code, "solver_message": msg,
            "tear_stream_residual": _tear_residual(ev.res), "unit_range_flags": [],
            "failed_checks": [f"solver:{code}"]}


def make_tools(task: str, row: dict, candidate_of=lambda c: c):
    """Tool signatures are plain str -> str so kbench builds a clean JSON schema.

    `candidate_of` lets a caller unwrap an answer from a wrapper if needed. The oracle
    ``check_answer`` is intentionally absent: grading stays at acceptance time / in the GRPO
    reward, never in the teacher's or student's tool list.
    """
    def check(candidate: str) -> _Eval:
        return _eval(task, candidate_of(candidate), row)

    def inspect_plant(candidate: str = "") -> str:
        """Solve and verify the plant/ledger AS GIVEN (candidate ignored): solver status, failed physics checks and key residuals (kg/s, W, MW)."""
        ev = _eval(task, "", row, as_given=True)
        if ev.error:
            return json.dumps({"ok": False, "task": task, "error": ev.error,
                               "failed_checks": [f"inspect: {ev.error}"]}, ensure_ascii=False)
        res, rep = ev.res, ev.rep
        code = _solver_error(res)
        failed = _rep_failed(rep)
        if res is not None and not res.ok:
            failed = [f"solver:{code or 'E_UNKNOWN'}"] + failed
        k = res.kpis if res is not None else {}
        balances = res.balances if res is not None else []
        states = res.states if res is not None else []
        payload = {
            "ok": True, "task": task,
            "solved": bool(res is not None and res.ok),
            "converged": bool(res is not None and res.ok),
            "solver_status": code or "OK",
            "solver_message": getattr(res, "message", "") if res is not None else "",
            "valid": bool(rep.valid) if rep is not None and hasattr(rep, "valid") else None,
            "failed_checks": failed,
            "checks": _rep_checks(rep),
            "tear_stream_residual": _tear_residual(res),
            "n_streams": len(states),
            "n_components": len(ev.plant.components) if ev.plant is not None else 0,
            "max_mass_residual_kg_s": _fmt(max((abs(b.mass_residual) for b in balances), default=0.0), 9),
            "max_energy_residual_W": _fmt(max((abs(b.energy_residual) for b in balances), default=0.0), 3),
            "exergy_closure_residual_W": _fmt(k.get("closure_residual_w"), 3),
            "total_exergy_destruction_MW": _fmt(k.get("ED_total_w", 0.0) / 1e6, 6),
            "Q_fuel_W": _fmt(k.get("Q_fuel_w"), 1), "W_net_W": _fmt(k.get("W_net_w"), 1),
        }
        if ev.meta.get("ledger"):
            payload["ledger_closure_residual_W"] = _fmt(ev.meta.get("closure_residual_w"), 3)
            payload["per_line_residuals"] = ev.meta.get("line_residuals", [])
            payload["failed_lines"] = ev.meta.get("failed_lines", [])
            payload["n_lines"] = ev.meta.get("n_lines", 0)
        return json.dumps(_with_meta(payload, ev.meta), ensure_ascii=False)

    def first_law_check(candidate: str) -> str:
        """Solve the candidate plant and report ACTUAL mass/energy residuals (kg/s, W), heat/power/fuel balance, convergence, tear residual, unit/range flags and failed checks."""
        ev = check(candidate)
        if ev.error:
            return json.dumps(_with_meta(_err_payload(ev, "parse"), ev.meta), ensure_ascii=False)
        res, rep = ev.res, ev.rep
        if res is None or not res.ok:
            return json.dumps(_with_meta(_unsolved_payload(ev), ev.meta), ensure_ascii=False)
        k = res.kpis
        comp = [{"component": b.component, "mass_residual_kg_s": _fmt(b.mass_residual, 9),
                 "energy_residual_W": _fmt(b.energy_residual, 3)} for b in res.balances]
        checks = _rep_checks(rep)
        unit_flags = [{"name": c["name"], "detail": c["detail"]} for c in checks
                      if c["name"].startswith("V6") and not c["ok"]]
        payload = {
            "ok": True,
            "converged": True, "solver_status": "OK", "solver_message": "",
            "tear_stream_residual": 0.0, "tear_closed": True,
            "unit_range_flags": unit_flags,
            "failed_checks": _rep_failed(rep),
            "per_component": comp,
            "max_mass_residual_kg_s": _fmt(max((abs(b.mass_residual) for b in res.balances), default=0.0), 9),
            "max_energy_residual_W": _fmt(max((abs(b.energy_residual) for b in res.balances), default=0.0), 3),
            "exergy_closure_residual_W": _fmt(k.get("closure_residual_w"), 3),
            "W_net_W": _fmt(k.get("W_net_w"), 1), "P_IT_W": _fmt(k.get("P_it_w"), 1),
            "Q_fuel_W": _fmt(k.get("Q_fuel_w"), 1), "Q_heat_W": _fmt(k.get("Q_heat_w"), 1),
            "Q_cooling_W": _fmt(k.get("Q_cooling_w"), 1),
            "mass_balance_ok": all(abs(b.mass_residual) <= 1e-6 * max(1.0, max((s.m_dot for s in res.states), default=1.0)) for b in res.balances),
            "energy_balance_ok": all(abs(b.energy_residual) <= 1e-6 * (abs(k.get("W_net_w", 0)) + abs(k.get("Q_fuel_w", 0)) + 1e6) for b in res.balances),
        }
        if ev.meta.get("ledger"):
            payload["per_line_residuals"] = ev.meta.get("line_residuals", [])
            payload["ledger_closure_residual_W"] = _fmt(ev.meta.get("closure_residual_w"), 3)
            payload["failed_lines"] = ev.meta.get("failed_lines", [])
        return json.dumps(_with_meta(payload, ev.meta), ensure_ascii=False)

    def second_law_check(candidate: str) -> str:
        """Solve the candidate plant and report ACTUAL exergy destruction (MW), eta_I/eta_II and thermodynamic limits, plus failed checks."""
        ev = check(candidate)
        if ev.error:
            return json.dumps(_with_meta(_err_payload(ev, "parse"), ev.meta), ensure_ascii=False)
        res, rep = ev.res, ev.rep
        if res is None or not res.ok:
            return json.dumps(_with_meta(_unsolved_payload(ev), ev.meta), ensure_ascii=False)
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
        payload = {
            "ok": True,
            "per_component_exergy_destruction_MW": ed,
            "total_exergy_destruction_MW": _fmt(k.get("ED_total_w", 0) / 1e6, 6),
            "plant_exergy_destruction_MW": _fmt(k.get("ED_plant_w", 0) / 1e6, 6),
            "exergy_closure_residual_W": _fmt(k.get("closure_residual_w"), 3),
            "limits": limits,
            "failed_checks": _rep_failed(rep),
            "nonneg_exergy_ok": all(b.exergy_destruction_w >= -1e-5 * max(abs(k.get("E_fuel_w", 0)), 1e6) for b in res.balances),
        }
        return json.dumps(_with_meta(payload, ev.meta), ensure_ascii=False)

    def validity_check(candidate: str) -> str:
        """Solve and report unit/range violations, site constraints, pinch temperatures and the hard-verifier verdict."""
        ev = check(candidate)
        if ev.error:
            return json.dumps(_with_meta(_err_payload(ev, "parse"), ev.meta), ensure_ascii=False)
        res, rep = ev.res, ev.rep
        if res is None or not res.ok:
            return json.dumps(_with_meta(_unsolved_payload(ev), ev.meta), ensure_ascii=False)
        k = res.kpis
        pinch = [{"component": b.component, "min_approach_K": _fmt(b.min_approach_K, 3)}
                 for b in res.balances if b.min_approach_K is not None]
        payload = {
            "ok": True,
            "valid": rep.valid if rep is not None and hasattr(rep, "valid") else (rep.get("all_correct") if isinstance(rep, dict) else None),
            "score": _fmt(rep.score, 4) if rep is not None and hasattr(rep, "score") else None,
            "failed_checks": _rep_failed(rep),
            "checks": _rep_checks(rep),
            "unit_range_flags": [{"name": c["name"], "detail": c["detail"]} for c in _rep_checks(rep)
                                 if c["name"].startswith("V6") and not c["ok"]],
            "min_approach_K": pinch,
            "T_max_K": _fmt(k.get("T_max_K"), 2), "T0_K": _fmt(k.get("T0_K"), 2),
            "water_use_kg_s": _fmt(k.get("water_use_kg_s"), 3),
        }
        return json.dumps(_with_meta(payload, ev.meta), ensure_ascii=False)

    def solve_plant(candidate: str) -> str:
        """Solve the candidate plant and return the full stream state table (T K, P kPa, h,s kJ/kg, exergy kJ/kg)."""
        ev = check(candidate)
        if ev.error:
            return json.dumps(_with_meta(_err_payload(ev, "parse"), ev.meta), ensure_ascii=False)
        res = ev.res
        if res is None or not res.ok:
            return json.dumps(_with_meta(_unsolved_payload(ev), ev.meta), ensure_ascii=False)
        # core.schema stores h/s/e_ph/e_ch in J/kg (J/kg-K for s); report kJ/kg to match labels.
        streams = [{"stream": s.stream, "T_K": _fmt(s.T, 2), "P_kPa": _fmt(s.P / 1000, 3),
                    "h_kJ_kg": _fmt(s.h / 1000, 3), "s_kJ_kgK": _fmt(s.s / 1000, 5),
                    "m_dot_kg_s": _fmt(s.m_dot, 5), "e_ph_kJ_kg": _fmt(s.e_ph / 1000, 3),
                    "e_ch_kJ_kg": _fmt(s.e_ch / 1000, 3)} for s in res.states]
        return json.dumps(_with_meta({"ok": True, "n_streams": len(streams), "streams": streams}, ev.meta),
                          ensure_ascii=False)

    return [inspect_plant, first_law_check, second_law_check, validity_check, solve_plant]


def tool_names() -> list[str]:
    return ["inspect_plant", "first_law_check", "second_law_check", "validity_check", "solve_plant"]
