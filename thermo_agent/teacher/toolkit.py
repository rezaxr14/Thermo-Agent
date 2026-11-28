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
from ..agent import harness as H
from ..agent.build_harness import frontmatter
from ..agent.dsl import parse_edits, edits_to_patch, EditError
from ..synth.faults import apply_patch, _dump
from ..design.spec import parse_design, DesignError
from ..design.build import evaluate_design
from .thermo_tools import DYNAMIC_TOOL_NAMES, dynamic_tools

# The eleven extra tools (nine continuous-input component tools plus the two
# on-demand skill tools below) are ON by default since the 2026-11-20 hard fix:
# ``make_tools``/``tool_names``/``ThermoEnv`` all default to include_dynamic=True.
# ``include_dynamic=False`` exists only for unit tests that need the legacy
# five-plant-tool contract; the old 5-tool traces remain valid and are not migrated.

# The two on-demand skill tools are kept SEPARATE from ``DYNAMIC_TOOL_NAMES``
# (thermo_tools.py, nine CoolProp tools): that registry and its tests are frozen,
# and the skill tools read the SKILL.md library instead of calling CoolProp.
SKILL_TOOL_NAMES = ("list_skills", "load_skill")
_SKILL_HINT = "call load_skill(name) for the full text"


# --------------------------------------------------------------------------- #
# on-demand skill library (metadata catalog + full-text loader)
# --------------------------------------------------------------------------- #
def _meta_list(value) -> list[str]:
    """Normalise a frontmatter value (list or ``[a, b]`` string) to a clean list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value).strip().strip("[]").split(",") if x.strip()]


def skill_metadata(name: str) -> dict | None:
    """Frontmatter-only record for one skill: no skill body ever leaves here.

    Returns ``None`` for a name that is not in the library, so callers can validate
    untrusted names without touching the filesystem (no path traversal).
    """
    if name not in H.skill_names():
        return None
    path = H.ROOT / "skills" / name / "SKILL.md"
    fm = frontmatter(path.read_text(encoding="utf-8"))
    return {
        "name": name,
        "description": " ".join(str(fm.get("description", "")).split()),
        "components": _meta_list(fm.get("components")),
        "variants": _meta_list(fm.get("variants")),
        "task": (str(fm["task"]).strip() or None) if fm.get("task") else None,
    }


def all_skill_metadata() -> list[dict]:
    """Compact catalog records for the whole skill library, in deterministic order."""
    return [m for n in H.skill_names() if (m := skill_metadata(n)) is not None]


def skill_body(name: str) -> str | None:
    """The full SKILL.md text with frontmatter stripped — exactly what
    ``H.read_skill(name)`` returns minus its ``SKILL <name>: <description>``
    header line. ``None`` for an unknown name. Never called with an
    unvalidated path: membership in the library is checked first.
    """
    if name not in H.skill_names():
        return None
    text = (H.ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
    return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)


# A served skill is a *tool result*, so it shares the 8,192-token training
# budget with the prompt and the rest of the trace. Keep the formulas, limits
# and failure modes; drop the duplicated `## Environment:` block (the compact
# contract already states it) and any section that does not fit the budget.
_SKILL_BUDGET_CHARS = 1000
_SKILL_PRIORITY = ("physics", "equation", "correct", "parameter", "constraint",
                   "failure", "fix", "finding", "rule")


def compact_skill_body(text: str, budget: int = _SKILL_BUDGET_CHARS) -> str:
    """A budgeted essentials view of one SKILL.md body.

    Priority sections (formulas, parameters, failure modes, fixes) are kept
    first, then the remaining sections while they fit. The full file stays on
    disk; this only bounds what one tool result costs in a training trace.
    """
    text = text.split("\n## Environment:", 1)[0].rstrip()
    parts = re.split(r"(?m)^## ", text)
    head, sections = parts[0].rstrip(), []
    for part in parts[1:]:
        title = part.splitlines()[0].strip().lower() if part.strip() else ""
        rank = 0 if any(key in title for key in _SKILL_PRIORITY) else 1
        sections.append((rank, part.rstrip()))
    sections.sort(key=lambda pair: pair[0])       # stable: library order inside a rank
    out = head
    for _rank, part in sections:
        candidate = f"{out}\n## {part}"
        if len(candidate) <= budget:
            out = candidate
    return out.strip()


def _skill_object(candidate: str) -> dict:
    """Decode the single ``candidate`` slot into a JSON object (or raise)."""
    if not isinstance(candidate, str):
        raise ValueError("candidate must be a JSON object string")
    obj = json.loads(candidate) if candidate.strip() else {}   # blank == empty filter object
    if not isinstance(obj, dict):
        raise ValueError("candidate must decode to a JSON object")
    return obj


def _skill_matches(meta: dict, task: str, components: list[str], query: str) -> bool:
    haystack = " ".join([meta["name"], meta["description"],
                         *meta["components"], *meta["variants"],
                         meta["task"] or ""]).lower()
    if task and task not in haystack:
        return False
    if components and not (set(meta["components"]) & set(components)):
        return False
    if query and query not in haystack:
        return False
    return True


def list_skills(candidate: str = "") -> str:
    """List skills matching optional task/components/query filters.

    Args:
        candidate (str): JSON object with optional ``task``, ``components``
            (list) and ``query`` (string) filters. An empty object lists the
            whole library. All filters are lowercase substring/exact-component
            matches; multiple filters are ANDed.

    Returns:
        str: JSON with ``ok``, ``hint``, ``n`` and a compact ``skills`` catalog
        (name, description, components, variants, task). NEVER a skill body.
        Malformed input returns ``{"ok": false, ...}``; the tool never raises.
    """
    try:
        obj = _skill_object(candidate)
        task = str(obj["task"]).strip().lower() if obj.get("task") else ""
        components = _meta_list(obj.get("components"))
        query = str(obj["query"]).strip().lower() if obj.get("query") else ""
        catalog = [m for m in all_skill_metadata()
                   if not (task or components or query) or _skill_matches(m, task, components, query)]
        return json.dumps({"ok": True, "hint": _SKILL_HINT, "n": len(catalog),
                           "skills": catalog}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                           "hint": _SKILL_HINT, "available": list(H.skill_names())},
                          ensure_ascii=False)


def load_skill(candidate: str = "") -> str:
    """Return a budgeted essentials view of one named skill's SKILL.md body.

    Args:
        candidate (str): JSON object with a single ``name`` key, e.g.
            ``{"name": "compressor"}``. The name is validated against the
            skill library exactly, so paths/traversal are rejected.

    Returns:
        str: JSON with ``ok``, ``name`` and the frontmatter-stripped, budgeted
        ``text`` (formulas, limits and failure modes first), or
        ``{"ok": false, "error": "unknown skill", "available": [...]}``.
        The text is a tool result only — the harness never writes it back into
        the system prompt. The tool never raises.
    """
    try:
        obj = _skill_object(candidate)
        name = obj.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("missing non-empty 'name'")
        text = skill_body(name.strip())
        if text is None:
            raise KeyError("unknown skill")
        return json.dumps({"ok": True, "name": name.strip(),
                           "text": compact_skill_body(text)},
                          ensure_ascii=False)
    except Exception:
        return json.dumps({"ok": False, "error": "unknown skill",
                           "available": list(H.skill_names())}, ensure_ascii=False)


def skill_tools() -> list:
    """The two on-demand skill tools as ``(candidate: str) -> str`` callables."""
    return [list_skills, load_skill]


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


# Tool results are training tokens too. A list of dicts repeats every column name
# on every row; the same numbers as one header + rows is several hundred tokens
# smaller and still ordinary JSON that the model can read.
STREAM_COLUMNS = ("stream", "T_K", "P_kPa", "h_kJ_kg", "s_kJ_kgK",
                  "m_dot_kg_s", "e_ph_kJ_kg", "e_ch_kJ_kg")
LINE_COLUMNS = ("line", "value_MW", "recomputed_MW", "residual_W")


def _table(rows: list[dict], columns: tuple[str, ...]) -> dict:
    """Compact column/row view of homogeneous records."""
    return {"columns": list(columns),
            "rows": [[row.get(col) for col in columns] for row in rows]}


def compact_result(name: str, text: str) -> str:
    """Rewrite an older stored result into today's compact table shape.

    Values are preserved byte-for-byte; only the repeated JSON keys are removed.
    Used when a corpus that predates the compact tool output is re-served, so a
    stored trace and a fresh tool call describe the same result.
    """
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return text
    if not isinstance(payload, dict):
        return text
    changed = False
    streams = payload.get("streams")
    if isinstance(streams, list) and streams and isinstance(streams[0], dict):
        payload["columns"] = list(STREAM_COLUMNS)
        payload["rows"] = [[row.get(col) for col in STREAM_COLUMNS] for row in streams]
        del payload["streams"]
        changed = True
    lines = payload.get("per_line_residuals")
    if isinstance(lines, list):
        errors = {str(row.get("line")): row.get("error") for row in lines
                  if isinstance(row, dict) and row.get("error")}
        payload["line_table"] = _table([row for row in lines if isinstance(row, dict)],
                                       LINE_COLUMNS)
        if errors:
            payload["line_errors"] = errors
        del payload["per_line_residuals"]
        changed = True
    # The solver's one-line hint is part of the served result too. Older stored
    # rows point at a key that no longer exists. Do not reuse `text` as the loop
    # variable: the unchanged path below returns the caller's original string.
    for key in ("solver_message", "message"):
        hint = payload.get(key)
        if isinstance(hint, str) and "per_line_residuals" in hint:
            payload[key] = hint.replace("per_line_residuals", "line_table")
            changed = True
    return json.dumps(payload, ensure_ascii=False) if changed else text


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
                               "recomputed_MW": _fmt(truth, 6), "residual_W": _fmt(resid, 3)})
        balances.append(_B(f"E[{sid}]", energy_residual=resid))
    for cid, val in (r.get("ED") or {}).items():
        truth = float(truth_ED.get(cid, val))
        resid = (val - truth) * 1e6
        line_residuals.append({"line": f"ED[{cid}]", "value_MW": _fmt(val, 6),
                               "recomputed_MW": _fmt(truth, 6), "residual_W": _fmt(resid, 3)})
        balances.append(_B(f"ED[{cid}]", energy_residual=resid, exergy_destruction_w=val * 1e6))

    failed_lines = sorted(r.get("errors") or {})
    for key, detail in (r.get("errors") or {}).items():
        line_residuals.append({"line": key, "value_MW": None, "recomputed_MW": None,
                               "residual_W": None, "error": str(detail)})
        balances.append(_B(key))
    negative = [f"ED[{c}]" for c, v in (r.get("ED") or {}).items() if v < -1e-5 * ef]
    failed_lines += negative

    closure = r.get("closure")
    kpis = {"closure_residual_w": (closure or 0.0) * 1e6,
            "ED_total_w": sum((r.get("ED") or {}).values()) * 1e6,
            "E_fuel_w": ef * 1e6,
            "T0_K": float((data.get("consts_true") or {}).get("T0", 0.0))}
    res = _LedgerResult(balances=balances, kpis=kpis,
                        message=(f"{len(failed_lines)} ledger line(s) fail the checks; read line_table") if failed_lines else "")

    n_lines = max(len(line_residuals), 1)
    n_ok = sum(1 for x in line_residuals
               if x.get("recomputed_MW") is not None and abs(x["residual_W"]) <= tol_w and "error" not in x)
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


def make_tools(task: str, row: dict, candidate_of=lambda c: c, include_dynamic: bool = True):
    """Tool signatures are plain str -> str so kbench builds a clean JSON schema.

    `candidate_of` lets a caller unwrap an answer from a wrapper if needed. The oracle
    ``check_answer`` is intentionally absent: grading stays at acceptance time / in the GRPO
    reward, never in the teacher's or student's tool list.

    With ``include_dynamic=True`` the nine continuous-input component tools from
    :mod:`thermo_agent.teacher.thermo_tools` are appended, followed by the two
    on-demand skill tools (:func:`list_skills`, :func:`load_skill`). All eleven are
    ``(candidate: str) -> str`` callables (``candidate`` is a JSON object string), so the
    shared tool-call parser executes them exactly like the five plant tools. The default
    is ``True`` (2026-11-20): every harness, trace, training and eval path uses the full on-demand toolset; pass ``False`` only for the five-tool unit tests.
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
            lines = ev.meta.get("line_residuals", [])
            payload["line_table"] = _table([x for x in lines if isinstance(x, dict)],
                                           LINE_COLUMNS)
            errors = {str(x.get("line")): x.get("error") for x in lines
                      if isinstance(x, dict) and x.get("error")}
            if errors:
                payload["line_errors"] = errors
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
            lines = ev.meta.get("line_residuals", [])
            payload["line_table"] = _table([x for x in lines if isinstance(x, dict)],
                                           LINE_COLUMNS)
            errors = {str(x.get("line")): x.get("error") for x in lines
                      if isinstance(x, dict) and x.get("error")}
            if errors:
                payload["line_errors"] = errors
            payload["ledger_closure_residual_W"] = _fmt(ev.meta.get("closure_residual_w"), 3)
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
        table = _table(streams, STREAM_COLUMNS)
        return json.dumps(_with_meta({"ok": True, "n_streams": len(streams), **table}, ev.meta),
                          ensure_ascii=False)

    tools = [inspect_plant, first_law_check, second_law_check, validity_check, solve_plant]
    if include_dynamic:
        tools.extend(dynamic_tools())
        tools.extend(skill_tools())
    return tools


def tool_names(include_dynamic: bool = True) -> list[str]:
    names = ["inspect_plant", "first_law_check", "second_law_check", "validity_check", "solve_plant"]
    if include_dynamic:
        names += list(DYNAMIC_TOOL_NAMES) + list(SKILL_TOOL_NAMES)
    return names
