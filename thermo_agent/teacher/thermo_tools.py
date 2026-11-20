"""Continuous, component-level thermodynamics tools (CoolProp-backed).

The five plant-level tools in :mod:`thermo_agent.teacher.toolkit` all answer a single
``candidate`` string about a *whole plant*.  They cannot answer a component question
with arbitrary inputs ("what is the outlet of an R245fa turbine at 1.2 MPa, 400 K
with eta_s = 0.85?").  This module adds that capability: nine tools with continuous
inputs, real property evaluation and no hard-coded per-task answers.

Compatibility contract
----------------------
The shared tool-call parser executes a tool as ``fn(payload.get("candidate", ""))``.
Every public tool therefore has the exact signature
``def <name>(candidate: str = "") -> str`` where ``candidate`` is a JSON *object*
string.  Each tool parses it, computes real numbers and returns a JSON string.
No tool ever raises: all exceptions are converted to ``{"ok": false, "error": ...}``
and non-finite results are rejected by ``json.dumps(..., allow_nan=False)``.

Units at the boundary are SI with the conventional engineering prefixes used in the
rest of the project: K, kPa, kJ/kg, kJ/kg/K, kg/s, W, W/K.  Internally CoolProp is
called in Pa / J/kg / J/kg/K.  Molar ideal-gas mixtures (``CGAM_AIR``, ``CH4``,
``FLUE``) use the NASA-7 polynomials in :mod:`thermo_agent.core.gasprops`; the
``model`` field on each result says which property model produced the numbers.
"""
from __future__ import annotations

import json
import math

try:  # CoolProp is present in the project venv; keep import failure graceful.
    import CoolProp.CoolProp as CP
except ModuleNotFoundError:  # pragma: no cover - depends on the interpreter
    CP = None

from ..core import gasprops as gp
from ..core import fluids as fl
from ..core.props import GAS_NAMES, GAS_PRESETS


# --------------------------------------------------------------------------- #
# parsing / validation / serialisation helpers
# --------------------------------------------------------------------------- #
_MISSING = object()


class ToolError(Exception):
    """A precise, user-facing input error (never leaves a tool as an exception)."""


def _parse(candidate: str) -> dict:
    """Decode the single ``candidate`` argument into a JSON object."""
    if candidate is None:
        raise ToolError("candidate is empty; expected a JSON object string")
    if not isinstance(candidate, str):
        raise ToolError(f"candidate must be a JSON string, got {type(candidate).__name__}")
    text = candidate.strip()
    if not text:
        raise ToolError("candidate is empty; expected a JSON object string")
    try:
        obj = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ToolError(f"invalid JSON in candidate: {exc}") from exc
    if not isinstance(obj, dict):
        raise ToolError(f"candidate must decode to a JSON object, got {type(obj).__name__}")
    return obj


def _str(p: dict, key: str, default=_MISSING) -> str:
    if key not in p:
        if default is _MISSING:
            raise ToolError(f"missing required key '{key}'")
        return default
    value = p[key]
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"'{key}' must be a non-empty string, got {value!r}")
    return value.strip()


def _num(p: dict, key: str, default=_MISSING, *, positive: bool = False) -> float:
    """Return a required finite float; ``positive`` additionally requires > 0."""
    value = p[key] if key in p else default
    if value is _MISSING:
        raise ToolError(f"missing required key '{key}'")
    if value is None or isinstance(value, bool):
        raise ToolError(f"'{key}' must be a number, got {value!r}")
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"'{key}' must be a number, got {value!r}") from exc
    if not math.isfinite(x):
        raise ToolError(f"'{key}' must be finite, got {value!r}")
    if positive and x <= 0.0:
        raise ToolError(f"'{key}' must be > 0, got {x!r}")
    return x


def _optnum(p: dict, key: str, *, positive: bool = False) -> float | None:
    """Like :func:`_num` but returns ``None`` when the key is absent or null."""
    if key not in p or p[key] is None:
        return None
    return _num(p, key, positive=positive)


def _r(value, nd: int = 6):
    return None if value is None else round(float(value), nd)


def _dump(payload: dict) -> str:
    # allow_nan=False makes a stray NaN/Inf a hard error, converted to ok:false below.
    return json.dumps(payload, allow_nan=False, ensure_ascii=False)


def _fail(exc: Exception) -> str:
    try:
        message = str(exc)
    except Exception:  # pragma: no cover - defensive
        message = repr(exc)
    return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {message}"},
                      allow_nan=False, ensure_ascii=False)


def _cp():
    if CP is None:
        raise ToolError("CoolProp is not installed; real-fluid properties are unavailable")
    return CP


# --------------------------------------------------------------------------- #
# fluid property engine: real fluids (CoolProp) + ideal-gas mixtures (NASA-7)
# --------------------------------------------------------------------------- #
_FLUE_COMP = {"N2": 0.75, "O2": 0.06, "CO2": 0.12, "H2O": 0.07}


def _gas_comp(fluid: str) -> dict | None:
    if fluid in GAS_PRESETS:
        return GAS_PRESETS[fluid]
    if fluid == "FLUE":
        return dict(_FLUE_COMP)
    return None


def _is_gas(fluid: str) -> bool:
    return fluid in GAS_NAMES


def _gas_m(comp: dict) -> float:
    """Molar mass [kg/mol] of a composition."""
    return gp.mw(comp) / 1000.0


def _gas_state(fluid: str, T=None, P=None, h=None, s=None, q=None) -> dict:
    """Ideal-gas mixture state. Internal units: K, Pa, J/kg, J/kg/K."""
    comp = _gas_comp(fluid)
    if comp is None:
        raise ToolError(f"'{fluid}' is not a supported ideal-gas mixture")
    if q is not None:
        raise ToolError(f"phase quality is not defined for ideal gas '{fluid}'")
    M = _gas_m(comp)
    Rs = gp.R / M  # J/kg/K

    def h_of(T_):
        return gp.h_mix(comp, T_) / M

    def s_of(T_, P_):
        return gp.s_mix(comp, T_, P_, mixing=True) / M

    if T is not None and P is not None:
        T_, P_ = T, P
        h_, s_ = h_of(T_), s_of(T_, P_)
    elif h is not None and P is not None:
        P_ = P
        T_ = gp.T_from_h(comp, h * M)
        h_, s_ = h, s_of(T_, P_)
    elif s is not None and P is not None:
        P_ = P
        T_ = gp.T_from_s_mixing(comp, s * M, P)
        h_, s_ = h_of(T_), s
    else:
        raise ToolError(
            f"fluid '{fluid}': need a consistent input pair (T_K,P_kPa), (P,h) or (P,s)")

    cp = (gp.h_mix(comp, T_ + 1.0) - gp.h_mix(comp, T_ - 1.0)) / (2.0 * M)
    return {"T": T_, "P": P_, "h": h_, "s": s_, "rho": P_ / (Rs * T_),
            "cp": cp, "phase": "gas", "quality": None, "model": "ideal-gas-NASA7"}


def _real_state(fluid: str, T=None, P=None, h=None, s=None, q=None) -> dict:
    """Real-fluid state via CoolProp. Internal units: K, Pa, J/kg, J/kg/K."""
    cp = _cp()
    try:
        cp.PropsSI("M", fluid)  # forces a precise error for an unknown fluid name
    except Exception as exc:
        raise ToolError(f"unknown fluid '{fluid}': {exc}") from exc

    if q is not None:
        if T is None and P is None:
            raise ToolError("quality requires T_K or P_kPa")
        if T is not None and P is None:
            P = cp.PropsSI("P", "T", T, "Q", q, fluid)
        elif P is not None and T is None:
            T = cp.PropsSI("T", "P", P, "Q", q, fluid)
        else:
            # Fully specified: verify the saturation inputs agree before trusting them.
            try:
                ps = cp.PropsSI("P", "T", T, "Q", q, fluid)
            except Exception:
                ps = P
            if not math.isfinite(ps) or abs(P - ps) > 0.02 * max(abs(P), 1.0):
                raise ToolError(f"inconsistent T_K / P_kPa / q for '{fluid}'")
        T_, P_ = T, P
        h_ = cp.PropsSI("H", "T", T_, "Q", q, fluid)
        s_ = cp.PropsSI("S", "T", T_, "Q", q, fluid)
        quality = q
    else:
        if T is not None and P is not None:
            h_ = cp.PropsSI("H", "T", T, "P", P, fluid)
            s_ = cp.PropsSI("S", "T", T, "P", P, fluid)
            T_, P_ = T, P
        elif h is not None and P is not None:
            h_ = h
            T_ = cp.PropsSI("T", "H", h, "P", P, fluid)
            s_ = cp.PropsSI("S", "H", h, "P", P, fluid)
            P_ = P
        elif s is not None and P is not None:
            s_ = s
            h_ = cp.PropsSI("H", "S", s, "P", P, fluid)
            T_ = cp.PropsSI("T", "H", h_, "P", P, fluid)
            P_ = P
        else:
            raise ToolError(
                f"fluid '{fluid}': need a consistent pair (T_K,P_kPa), (P_kPa,h_kJ_kg), "
                "(P_kPa,s_kJ_kgK), (T_K,q) or (P_kPa,q)")
        try:
            qq = cp.PropsSI("Q", "H", h_, "P", P_, fluid)
            quality = qq if 0.0 <= qq <= 1.0 else None
        except Exception:
            quality = None
        T_ = float(T_)
        P_ = float(P_)

    if quality is not None:
        phase = "two-phase"
    else:
        try:
            ts = cp.PropsSI("T", "P", P_, "Q", 0, fluid)
            tol = 1e-2 * max(abs(ts), 1.0)
            if T_ < ts - tol:
                phase = "liquid"
            elif T_ > ts + tol:
                phase = "gas"
            else:
                phase = "saturated"
        except Exception:
            phase = "supercritical"

    try:
        rho = cp.PropsSI("D", "T", T_, "P", P_, fluid)
    except Exception:
        rho = None
    try:
        cpv = cp.PropsSI("C", "P", P_, "T", T_, fluid)
        if not math.isfinite(cpv) or cpv > 1e6:  # formally infinite in the dome
            cpv = None
    except Exception:
        cpv = None
    return {"T": T_, "P": P_, "h": h_, "s": s_, "rho": rho, "cp": cpv,
            "phase": phase, "quality": quality, "model": "CoolProp"}


def _state(fluid: str, T=None, P=None, h=None, s=None, q=None) -> dict:
    """Resolve a state from any consistent pair (SI internal units)."""
    if _is_gas(fluid):
        return _gas_state(fluid, T=T, P=P, h=h, s=s, q=q)
    return _real_state(fluid, T=T, P=P, h=h, s=s, q=q)


def _state_fields(st: dict) -> dict:
    """Boundary representation of a state (kPa, kJ/kg, kJ/kg/K)."""
    return {"T_K": _r(st["T"], 4), "P_kPa": _r(st["P"] / 1000.0, 5),
            "h_kJ_kg": _r(st["h"] / 1000.0, 5), "s_kJ_kgK": _r(st["s"] / 1000.0, 6),
            "phase": st["phase"], "quality": st["quality"]}


# --------------------------------------------------------------------------- #
# 1. fluid_state
# --------------------------------------------------------------------------- #
def fluid_state(candidate: str = "") -> str:
    """Resolve a fluid state from any consistent pair of properties.

    Args:
        candidate (str): JSON object with ``fluid`` and two of ``T_K``, ``P_kPa``,
            ``h_kJ_kg``, ``s_kJ_kgK`` (or a quality ``q`` in [0, 1] with ``T_K`` or
            ``P_kPa``).  Optional ``m_dot_kg_s``.  An absent quality is ``-1``.

    Returns:
        str: JSON with T, P, h, s, density, cp, phase, quality and mass flow.
    """
    try:
        p = _parse(candidate)
        fluid = _str(p, "fluid")
        T = _optnum(p, "T_K", positive=True)
        P = _optnum(p, "P_kPa", positive=True)
        h = _optnum(p, "h_kJ_kg")
        s = _optnum(p, "s_kJ_kgK")
        q = _optnum(p, "q")
        if q is not None:
            if q < 0:
                q = None  # the parser's "-1 if absent" convention
            elif not 0.0 <= q <= 1.0:
                raise ToolError("q must be in [0, 1] (or negative for absent)")
        m_dot = _optnum(p, "m_dot_kg_s", positive=True)

        st = _state(fluid, T=T, P=None if P is None else P * 1000.0,
                    h=None if h is None else h * 1000.0,
                    s=None if s is None else s * 1000.0, q=q)
        out = {"ok": True, "fluid": fluid, "model": st["model"], **_state_fields(st),
               "rho_kg_m3": _r(st["rho"], 5),
               "cp_kJ_kgK": None if st["cp"] is None else _r(st["cp"] / 1000.0, 6),
               "m_dot_kg_s": m_dot}
        return _dump(out)
    except Exception as exc:
        return _fail(exc)


# --------------------------------------------------------------------------- #
# 2. saturation_state
# --------------------------------------------------------------------------- #
def saturation_state(candidate: str = "") -> str:
    """Saturated-liquid/vapour properties of a real fluid at T or P.

    Args:
        candidate (str): JSON object with ``fluid``, exactly one of ``T_K`` or
            ``P_kPa``, and optional ``quality`` (default 0).

    Returns:
        str: JSON with T_sat, P_sat, h_f/g/fg, s_f/g, v_f/g and the state at
        the requested quality.
    """
    try:
        p = _parse(candidate)
        fluid = _str(p, "fluid")
        if _is_gas(fluid):
            raise ToolError(f"fluid '{fluid}' is an ideal gas and has no saturation state")
        cpc = _cp()
        T = _optnum(p, "T_K", positive=True)
        P = _optnum(p, "P_kPa", positive=True)
        q = _optnum(p, "quality") or 0.0
        if not 0.0 <= q <= 1.0:
            raise ToolError("quality must be in [0, 1]")
        if (T is None) == (P is None):
            raise ToolError("saturation_state needs exactly one of T_K or P_kPa")
        if T is not None:
            Ts = T
            Ps = cpc.PropsSI("P", "T", Ts, "Q", 0, fluid)
        else:
            Ps = P * 1000.0
            Ts = cpc.PropsSI("T", "P", Ps, "Q", 0, fluid)

        hf = cpc.PropsSI("H", "T", Ts, "Q", 0, fluid)
        hg = cpc.PropsSI("H", "T", Ts, "Q", 1, fluid)
        sf = cpc.PropsSI("S", "T", Ts, "Q", 0, fluid)
        sg = cpc.PropsSI("S", "T", Ts, "Q", 1, fluid)
        vf = 1.0 / cpc.PropsSI("D", "T", Ts, "Q", 0, fluid)
        vg = 1.0 / cpc.PropsSI("D", "T", Ts, "Q", 1, fluid)
        h = hf + q * (hg - hf)
        s = sf + q * (sg - sf)
        out = {
            "ok": True, "fluid": fluid, "model": "CoolProp", "quality": q,
            "T_sat_K": _r(Ts, 5), "P_sat_kPa": _r(Ps / 1000.0, 6),
            "h_f_kJ_kg": _r(hf / 1000.0, 5), "h_g_kJ_kg": _r(hg / 1000.0, 5),
            "h_fg_kJ_kg": _r((hg - hf) / 1000.0, 5),
            "s_f_kJ_kgK": _r(sf / 1000.0, 7), "s_g_kJ_kgK": _r(sg / 1000.0, 7),
            "v_f_m3_kg": _r(vf, 9), "v_g_m3_kg": _r(vg, 7),
            "h_kJ_kg": _r(h / 1000.0, 5), "s_kJ_kgK": _r(s / 1000.0, 7),
            # short aliases matching the task's field names
            "h_f": _r(hf / 1000.0, 5), "h_g": _r(hg / 1000.0, 5),
            "h_fg": _r((hg - hf) / 1000.0, 5), "s_f": _r(sf / 1000.0, 7),
            "s_g": _r(sg / 1000.0, 7), "v_f": _r(vf, 9), "v_g": _r(vg, 7),
        }
        return _dump(out)
    except Exception as exc:
        return _fail(exc)


# --------------------------------------------------------------------------- #
# 3. turbomachine
# --------------------------------------------------------------------------- #
def turbomachine(candidate: str = "") -> str:
    """Compressor / turbine / pump / fan from inlet state and isentropic efficiency.

    Args:
        candidate (str): JSON with ``kind``, ``fluid``, ``T_in_K``, ``P_in_kPa``,
            ``P_out_kPa`` or ``pressure_ratio``, ``eta_s``, optional
            ``m_dot_kg_s`` (default 1) and ``T0_K`` (default 298.15).

    Returns:
        str: JSON with the actual outlet state, specific work, power, efficiency,
        exergy destruction and warnings.
    """
    try:
        p = _parse(candidate)
        kind = _str(p, "kind").lower()
        if kind not in ("compressor", "turbine", "pump", "fan"):
            raise ToolError("kind must be one of compressor, turbine, pump, fan")
        fluid = _str(p, "fluid")
        T_in = _num(p, "T_in_K", positive=True)
        P_in = _num(p, "P_in_kPa", positive=True) * 1000.0
        P_out = _optnum(p, "P_out_kPa", positive=True)
        PR = _optnum(p, "pressure_ratio", positive=True)
        eta = _num(p, "eta_s", positive=True)
        if eta > 1.0:
            raise ToolError(f"eta_s must be in (0, 1], got {eta}")
        m_dot = _optnum(p, "m_dot_kg_s", positive=True) or 1.0
        T0 = _optnum(p, "T0_K", positive=True) or 298.15

        if P_out is None:
            if PR is None:
                raise ToolError("turbomachine needs P_out_kPa or pressure_ratio")
            P_out = P_in * PR if kind in ("compressor", "fan", "pump") else P_in / PR
        else:
            P_out *= 1000.0
        if P_out <= 0:
            raise ToolError("outlet pressure must be positive")

        si = _state(fluid, T=T_in, P=P_in)
        ss = _state(fluid, s=si["s"], P=P_out)
        dh_s = ss["h"] - si["h"]
        warnings: list[str] = []
        if kind == "turbine":
            dh = eta * dh_s                       # negative (real enthalpy drop)
            h_out = si["h"] + dh
            w_spec = -dh / 1000.0                 # > 0, output
            power = m_dot * w_spec * 1000.0       # > 0, produced
            if P_out >= P_in:
                warnings.append("no pressure drop across turbine")
        else:
            dh = dh_s / eta                       # positive (real enthalpy rise)
            h_out = si["h"] + dh
            w_spec = dh / 1000.0                  # > 0, input
            power = m_dot * w_spec * 1000.0       # > 0, consumed
            if P_out <= P_in:
                warnings.append("no pressure rise across machine")
        so = _state(fluid, h=h_out, P=P_out)
        if so["quality"] is not None and 0.0 < so["quality"] < 1.0:
            warnings.append(f"wet outlet (quality={so['quality']:.3f})")
        if eta == 1.0:
            warnings.append("ideal isentropic machine (eta_s = 1)")

        sgen = m_dot * (so["s"] - si["s"])        # W/K
        ex_dest = T0 * sgen
        out = {
            "ok": True, "kind": kind, "fluid": fluid, "model": si["model"],
            "inlet": _state_fields(si), "outlet": _state_fields(so),
            "specific_work_kJ_kg": _r(w_spec, 6),
            "power_W": _r(power, 3),
            "isentropic_efficiency": eta,
            "entropy_generation_W_per_K": _r(sgen, 6),
            "exergy_destruction_W": _r(ex_dest, 3),
            "warnings": warnings,
        }
        return _dump(out)
    except Exception as exc:
        return _fail(exc)


# --------------------------------------------------------------------------- #
# 4. heat_exchanger
# --------------------------------------------------------------------------- #
def _hx_side(p: dict, name: str) -> dict:
    d = p.get(name)
    if not isinstance(d, dict):
        raise ToolError(f"'{name}' must be an object with fluid, T_K, P_kPa, m_dot_kg_s")
    fluid = _str(d, "fluid")
    T = _num(d, "T_K", positive=True)
    P = _num(d, "P_kPa", positive=True) * 1000.0
    m = _num(d, "m_dot_kg_s", positive=True)
    st = _state(fluid, T=T, P=P)
    return {"fluid": fluid, "T": T, "P": P, "m": m, "st": st,
            "C": m * st["cp"] if st["cp"] is not None else None}


def heat_exchanger(candidate: str = "") -> str:
    """Two-stream heat exchanger with a specified outlet, duty or effectiveness.

    Args:
        candidate (str): JSON with ``hot`` and ``cold`` side objects and exactly
            one of ``hot_T_out_K``, ``cold_T_out_K``, ``duty_W``, ``effectiveness``.
            Optional ``flow_arrangement`` (counter|parallel) and ``T0_K``.

    Returns:
        str: JSON with the duty, both outlet states, endpoint pinch, LMTD, UA,
        energy residual and warnings.
    """
    try:
        p = _parse(candidate)
        hot = _hx_side(p, "hot")
        cold = _hx_side(p, "cold")
        specs = [k for k in ("hot_T_out_K", "cold_T_out_K", "duty_W", "effectiveness")
                 if k in p and p[k] is not None]
        if len(specs) != 1:
            raise ToolError("heat_exchanger needs exactly one of hot_T_out_K, "
                            "cold_T_out_K, duty_W, effectiveness")
        arrangement = _str(p, "flow_arrangement", default="counter").lower()
        if arrangement not in ("counter", "parallel"):
            raise ToolError("flow_arrangement must be 'counter' or 'parallel'")
        T0 = _optnum(p, "T0_K", positive=True) or 298.15
        spec = specs[0]

        hi, ci = hot["st"], cold["st"]
        Thi, Tci = hi["T"], ci["T"]
        warnings: list[str] = []
        if Thi <= Tci:
            raise ToolError("hot inlet temperature must exceed cold inlet temperature")

        if spec == "hot_T_out_K":
            Tho = _num(p, spec, positive=True)
            sho = _state(hot["fluid"], T=Tho, P=hot["P"])
            Q = hot["m"] * (hi["h"] - sho["h"])
        elif spec == "cold_T_out_K":
            Tco = _num(p, spec, positive=True)
            sco = _state(cold["fluid"], T=Tco, P=cold["P"])
            Q = cold["m"] * (sco["h"] - ci["h"])
        elif spec == "duty_W":
            Q = _num(p, spec)
        else:  # effectiveness
            eps = _num(p, spec, positive=True)
            if eps > 1.0:
                raise ToolError("effectiveness must be in (0, 1]")
            if hot["C"] is None or cold["C"] is None:
                raise ToolError("effectiveness needs single-phase sides with finite cp")
            Cmin = min(hot["C"], cold["C"])
            Q = eps * Cmin * (Thi - Tci)
            if eps == 1.0:
                warnings.append("ideal exchanger (effectiveness = 1)")

        if Q < 0:
            raise ToolError(f"computed duty {Q:.3g} W is negative; check inlet temperatures")
        sho = _state(hot["fluid"], h=hi["h"] - Q / hot["m"], P=hot["P"])
        sco = _state(cold["fluid"], h=ci["h"] + Q / cold["m"], P=cold["P"])
        Tho, Tco = sho["T"], sco["T"]

        Q_hot = hot["m"] * (hi["h"] - sho["h"])
        Q_cold = cold["m"] * (sco["h"] - ci["h"])
        resid = Q_cold - Q_hot
        if arrangement == "counter":
            dT1, dT2 = Thi - Tco, Tho - Tci
        else:
            dT1, dT2 = Thi - Tci, Tho - Tco
        pinch = min(dT1, dT2)
        if pinch < 0:
            warnings.append(f"temperature cross: min approach {pinch:.2f} K")
        if dT1 <= 0 or dT2 <= 0:
            lmtd = None
        elif abs(dT1 - dT2) < 1e-8:
            lmtd = dT1
        else:
            lmtd = (dT1 - dT2) / math.log(dT1 / dT2)
        ua = Q / lmtd if lmtd and lmtd > 0 else None
        for label, st in (("hot", sho), ("cold", sco)):
            if st["quality"] is not None:
                warnings.append(f"{label} side two-phase outlet (quality={st['quality']:.3f})")
        out = {
            "ok": True, "flow_arrangement": arrangement, "model": hi["model"],
            "duty_W": _r(Q, 3),
            "hot_inlet": _state_fields(hi), "hot_outlet": _state_fields(sho),
            "cold_inlet": _state_fields(ci), "cold_outlet": _state_fields(sco),
            "min_approach_K": _r(pinch, 5),
            "LMTD_K": _r(lmtd, 5), "UA_W_per_K": _r(ua, 3),
            "energy_residual_W": _r(resid, 6),
            "relative_energy_residual": _r(abs(resid) / max(abs(Q), 1.0), 9),
            "warnings": warnings,
        }
        return _dump(out)
    except Exception as exc:
        return _fail(exc)


# --------------------------------------------------------------------------- #
# 5. combustor
# --------------------------------------------------------------------------- #
_FUELS = {
    "CH4": {"afr": 17.16, "lhv": 50.0e6, "co2": 2.75, "name": "methane"},
    "H2": {"afr": 34.05, "lhv": 120.0e6, "co2": 0.0, "name": "hydrogen"},
}


def _fuel(spec: str) -> dict:
    key = spec.strip().upper()
    if key in ("NG", "NATURAL_GAS", "NATURALGAS", "METHANE"):
        key = "CH4"
    if key in ("HYDROGEN",):
        key = "H2"
    if key not in _FUELS:
        raise ToolError(f"unknown fuel '{spec}' (supported: CH4, NG, H2)")
    return _FUELS[key]


def combustor(candidate: str = "") -> str:
    """Stoichiometric combustor mass/energy balance and adiabatic flame estimate.

    Args:
        candidate (str): JSON with ``fuel``, ``air_excess``, ``T_air_K``,
            ``eta_comb``, optional ``fuel_LHV_MJ_kg``, ``P_kPa`` and either
            ``fuel_flow_kg_s`` or ``T_out_K``.

    Returns:
        str: JSON with fuel/air/flue flows, heat release, useful heat, AFR,
        adiabatic flame temperature, CO2 flow and warnings.
    """
    try:
        p = _parse(candidate)
        fuel = _fuel(_str(p, "fuel"))
        lam = _optnum(p, "air_excess", positive=True) or 1.0
        if lam < 1.0:
            raise ToolError("air_excess must be >= 1")
        T_air = _optnum(p, "T_air_K", positive=True) or 298.15
        eta = _optnum(p, "eta_comb", positive=True) or 0.98
        if eta > 1.0:
            raise ToolError("eta_comb must be in (0, 1]")
        LHV = (_optnum(p, "fuel_LHV_MJ_kg", positive=True) or fuel["lhv"] / 1.0e6) * 1.0e6
        P_kPa = _optnum(p, "P_kPa", positive=True) or 101.325
        mf = _optnum(p, "fuel_flow_kg_s", positive=True)
        T_out = _optnum(p, "T_out_K", positive=True)
        if mf is None and T_out is None:
            raise ToolError("combustor needs fuel_flow_kg_s or T_out_K")

        afr = lam * fuel["afr"]
        cp_flue = 1150.0  # J/kg/K, representative flue-gas value (ideal-gas cp model)
        warnings = ["flue gas modelled with constant ideal-gas cp = 1.15 kJ/kg/K"]
        if mf is None:
            # Absolute scale is not determined by T_out alone; size on 1 kg/s air.
            m_air = 1.0
            mf = m_air / afr
            warnings.append("fuel flow not given: sized on a 1 kg/s air basis")
        else:
            m_air = afr * mf
        m_flue = m_air + mf
        heat_release = mf * LHV
        useful = eta * heat_release
        T_ad = T_air + useful / (m_flue * cp_flue)
        if T_out is not None:
            if T_out > T_ad + 1.0:
                warnings.append(f"T_out {T_out:.1f} K exceeds adiabatic flame {T_ad:.1f} K")
        out = {
            "ok": True, "fuel": fuel["name"], "model": "stoichiometric ideal-gas cp",
            "fuel_flow_kg_s": _r(mf, 8), "air_flow_kg_s": _r(m_air, 6),
            "flue_flow_kg_s": _r(m_flue, 6),
            "heat_release_W": _r(heat_release, 3), "useful_heat_W": _r(useful, 3),
            "air_fuel_ratio": _r(afr, 5), "excess_air_ratio": lam,
            "T_air_K": T_air, "T_out_K": T_out, "P_kPa": P_kPa,
            "adiabatic_flame_K": _r(T_ad, 2),
            "CO2_kg_s": _r(mf * fuel["co2"], 8), "warnings": warnings,
        }
        return _dump(out)
    except Exception as exc:
        return _fail(exc)


# --------------------------------------------------------------------------- #
# 6. cycle_performance
# --------------------------------------------------------------------------- #
def _rankine(fluid: str, Th: float, Tc: float, eta_t: float, eta_p: float) -> tuple[float, float]:
    """Simple saturated Rankine/ORC cycle -> (thermal efficiency, net specific work kJ/kg)."""
    cpc = _cp()
    Pl = cpc.PropsSI("P", "T", Tc, "Q", 0, fluid)
    Ph = cpc.PropsSI("P", "T", Th, "Q", 0, fluid)
    h1 = cpc.PropsSI("H", "T", Tc, "Q", 0, fluid)
    v1 = 1.0 / cpc.PropsSI("D", "T", Tc, "Q", 0, fluid)
    wp = v1 * (Ph - Pl) / eta_p
    h2 = h1 + wp
    h3 = cpc.PropsSI("H", "T", Th, "Q", 1, fluid)
    s3 = cpc.PropsSI("S", "T", Th, "Q", 1, fluid)
    h4s = cpc.PropsSI("H", "P", Pl, "S", s3, fluid)
    wt = eta_t * (h3 - h4s)
    qin = h3 - h2
    if qin <= 0:
        raise ToolError("no heat input in Rankine cycle")
    wnet = wt - wp
    return wnet / qin, wnet / 1000.0


def cycle_performance(candidate: str = "") -> str:
    """Carnot/Lorenz limits and a real efficiency or COP for common cycles.

    Args:
        candidate (str): JSON with ``kind``, ``T_hot_K``, ``T_cold_K``, optional
            ``eta_second_law``, ``eta_turbine``, ``eta_pump``, ``fluid`` and
            log-mean input temperatures for a Lorenz COP.

    Returns:
        str: JSON with Carnot (efficiency or COP), Lorenz COP for HP/chillers,
        real efficiency/COP, specific work and warnings.
    """
    try:
        p = _parse(candidate)
        kind = _str(p, "kind").lower()
        if kind not in ("rankine", "orc", "heat_pump", "chiller", "absorption"):
            raise ToolError("kind must be rankine, orc, heat_pump, chiller or absorption")
        Th = _num(p, "T_hot_K", positive=True)
        Tc = _num(p, "T_cold_K", positive=True)
        if Th <= Tc:
            raise ToolError(f"T_hot_K ({Th}) must exceed T_cold_K ({Tc})")
        eta2 = _optnum(p, "eta_second_law", positive=True) or 0.5
        if eta2 > 1.0:
            raise ToolError("eta_second_law must be in (0, 1]")
        warnings: list[str] = []
        out: dict = {"ok": True, "kind": kind, "T_hot_K": Th, "T_cold_K": Tc,
                     "eta_second_law": eta2, "warnings": warnings}
        lift = Th - Tc
        if lift <= 0:
            raise ToolError("temperature lift must be positive")

        if kind in ("rankine", "orc"):
            carnot = 1.0 - Tc / Th
            fluid = _str(p, "fluid")
            eta_t = _optnum(p, "eta_turbine", positive=True) or 0.85
            eta_p = _optnum(p, "eta_pump", positive=True) or 0.75
            w_spec = None
            if _is_gas(fluid):
                real = eta2 * carnot
                warnings.append(f"'{fluid}' is an ideal gas; Rankine not solvable, used eta_second_law * Carnot")
            else:
                try:
                    real, w_spec = _rankine(fluid, Th, Tc, eta_t, eta_p)
                    out["model"] = "CoolProp saturated Rankine cycle"
                except Exception as exc:
                    real = eta2 * carnot
                    warnings.append(f"Rankine evaluation failed ({type(exc).__name__}: {exc}); "
                                    "used eta_second_law * Carnot")
            if real > carnot:
                warnings.append("real efficiency exceeded Carnot; clamped")
                real = carnot
            out.update({"model": out.get("model", "Carnot + second-law factor"),
                        "carnot_efficiency": _r(carnot, 6),
                        "real_efficiency": _r(real, 6),
                        "specific_work_kJ_kg": _r(w_spec, 6)})
            return _dump(out)

        # COP cycles ------------------------------------------------------- #
        carnot_cop = Th / lift if kind in ("heat_pump",) else Tc / lift
        out["carnot_cop"] = _r(carnot_cop, 6)
        Thi = _optnum(p, "T_hot_in_K", positive=True)
        Tho = _optnum(p, "T_hot_out_K", positive=True)
        Tci = _optnum(p, "T_cold_in_K", positive=True)
        Tco = _optnum(p, "T_cold_out_K", positive=True)
        if all(v is not None for v in (Thi, Tho, Tci, Tco)):
            def lm(a, b):
                return a if abs(a - b) < 1e-9 else (a - b) / math.log(a / b)
            Tlm_h, Tlm_c = lm(Thi, Tho), lm(Tci, Tco)
            if Tlm_h <= Tlm_c:
                raise ToolError("hot log-mean temperature must exceed cold log-mean")
            lorenz = Tlm_h / (Tlm_h - Tlm_c) if kind == "heat_pump" else Tlm_c / (Tlm_h - Tlm_c)
            out["model"] = "Lorenz (log-mean stream temperatures) + second-law factor"
        else:
            lorenz = carnot_cop
            out["model"] = "Carnot (two-temperature) + second-law factor"
            warnings.append("no hot/cold inlet-outlet temperatures: Lorenz = Carnot")
        out["lorenz_cop"] = _r(lorenz, 6)
        real = eta2 * lorenz
        if real > carnot_cop:
            warnings.append("real COP exceeded Carnot; clamped")
            real = carnot_cop
        out["real_cop"] = _r(real, 6)
        # Work per unit of reference heat transfer (kJ per kJ of heat).
        out["specific_work_kJ_kg"] = _r(1.0 / carnot_cop, 6)
        if kind == "absorption":
            warnings.append("absorption cycle approximated by a second-law fraction of Carnot COP")
        return _dump(out)
    except Exception as exc:
        return _fail(exc)


# --------------------------------------------------------------------------- #
# 7. flow_network
# --------------------------------------------------------------------------- #
def _stream(p: dict) -> dict:
    if not isinstance(p, dict):
        raise ToolError("each stream must be an object with fluid, T_K, P_kPa, m_dot_kg_s")
    fluid = _str(p, "fluid")
    T = _num(p, "T_K", positive=True)
    P = _num(p, "P_kPa", positive=True) * 1000.0
    m = _num(p, "m_dot_kg_s", positive=True)
    return {"fluid": fluid, "T": T, "P": P, "m": m, "st": _state(fluid, T=T, P=P)}


def flow_network(candidate: str = "") -> str:
    """Mixer, splitter, valve or pipe junction with mass/energy/entropy balances.

    Args:
        candidate (str): JSON with ``kind``, ``streams`` list, plus ``fraction``
            (splitter), ``dp_kPa`` or ``dp_fraction`` (valve/pipe) and ``T0_K``.

    Returns:
        str: JSON with outlet stream(s), mass and energy residuals, entropy
        generation and exergy destruction.
    """
    try:
        p = _parse(candidate)
        kind = _str(p, "kind").lower()
        if kind not in ("mixer", "splitter", "valve", "pipe"):
            raise ToolError("kind must be mixer, splitter, valve or pipe")
        raw = p.get("streams")
        if not isinstance(raw, list) or not raw:
            raise ToolError("'streams' must be a non-empty list")
        streams = [_stream(s) for s in raw]
        T0 = _optnum(p, "T0_K", positive=True) or 298.15
        warnings: list[str] = []
        m_in = sum(s["m"] for s in streams)
        h_in = sum(s["m"] * s["st"]["h"] for s in streams)
        s_in = sum(s["m"] * s["st"]["s"] for s in streams)
        outs: list[dict] = []
        raw: list[tuple[float, float, float]] = []  # (m_dot, h[J/kg], s[J/kg/K]) unrounded

        if kind == "mixer":
            if len(streams) < 2:
                raise ToolError("mixer needs at least two inlet streams")
            fluids = {s["fluid"] for s in streams}
            if len(fluids) > 1:
                warnings.append("mixing different fluids with a single-fluid outlet model")
            fluid = streams[0]["fluid"]
            P_out = min(s["P"] for s in streams)
            h_out = h_in / m_in
            st = _state(fluid, h=h_out, P=P_out)
            outs.append({"fluid": fluid, "m_dot_kg_s": _r(m_in, 8),
                         **_state_fields(st)})
            raw.append((m_in, st["h"], st["s"]))
        elif kind == "splitter":
            if len(streams) != 1:
                raise ToolError("splitter needs exactly one inlet stream")
            f = _num(p, "fraction", positive=True)
            if f > 1.0:
                raise ToolError("fraction must be in (0, 1)")
            s = streams[0]
            if f == 1.0:
                warnings.append("fraction = 1: second outlet has zero flow")
            for m in (f * s["m"], (1.0 - f) * s["m"]):
                outs.append({"fluid": s["fluid"], "m_dot_kg_s": _r(m, 8),
                             **_state_fields(s["st"])})
                raw.append((m, s["st"]["h"], s["st"]["s"]))
        else:  # valve / pipe: isenthalpic pressure drop
            if len(streams) != 1:
                raise ToolError(f"{kind} needs exactly one inlet stream")
            s = streams[0]
            dP = _optnum(p, "dp_kPa", positive=True)
            dpf = _optnum(p, "dp_fraction", positive=True)
            if (dP is None) == (dpf is None):
                raise ToolError(f"{kind} needs exactly one of dp_kPa or dp_fraction")
            if dpf is not None and dpf >= 1.0:
                raise ToolError("dp_fraction must be in (0, 1)")
            P_out = s["P"] - (dP * 1000.0 if dP is not None else s["P"] * dpf)
            if P_out <= 0:
                raise ToolError("outlet pressure would be non-positive")
            st = _state(s["fluid"], h=s["st"]["h"], P=P_out)
            outs.append({"fluid": s["fluid"], "m_dot_kg_s": _r(s["m"], 8),
                         **_state_fields(st)})
            if kind == "valve" and st["quality"] is not None:
                warnings.append("flashing valve: two-phase outlet")
            raw.append((s["m"], st["h"], st["s"]))

        m_out = sum(m for m, _, _ in raw)
        h_out_tot = sum(m * h for m, h, _ in raw)
        s_out_tot = sum(m * s for m, _, s in raw)
        sgen = s_out_tot - s_in
        out = {
            "ok": True, "kind": kind, "model": streams[0]["st"]["model"],
            "outlets": outs,
            "mass_residual_kg_s": _r(m_out - m_in, 10),
            "energy_residual_W": _r(h_out_tot - h_in, 6),
            "entropy_generation_W_per_K": _r(sgen, 6),
            "exergy_destruction_W": _r(T0 * sgen, 3),
            "warnings": warnings,
        }
        return _dump(out)
    except Exception as exc:
        return _fail(exc)


# --------------------------------------------------------------------------- #
# 8. psychrometrics
# --------------------------------------------------------------------------- #
def psychrometrics(candidate: str = "") -> str:
    """Humid-air state ( CoolProp ``HAPropsSI`` ) and cooling-tower approach.

    Args:
        candidate (str): JSON with ``T_dry_K``, ``T_wet_K`` or ``RH``, optional
            ``P_kPa`` (default 101.325) and ``T_water_K``.

    Returns:
        str: JSON with humidity ratio, enthalpy, dew point, wet bulb, density,
        approach and warnings.
    """
    try:
        cpc = _cp()
        p = _parse(candidate)
        Td = _num(p, "T_dry_K", positive=True)
        P = (_optnum(p, "P_kPa", positive=True) or 101.325) * 1000.0
        Twet = _optnum(p, "T_wet_K", positive=True)
        rh = _optnum(p, "RH")
        Twater = _optnum(p, "T_water_K", positive=True)
        warnings: list[str] = []
        if (Twet is None) == (rh is None):
            raise ToolError("psychrometrics needs exactly one of T_wet_K or RH")
        if rh is not None:
            if rh > 1.0:
                rh /= 100.0
                warnings.append("RH > 1 interpreted as a percentage")
            if not 0.0 <= rh <= 1.0:
                raise ToolError("RH must be in [0, 1] (or [0, 100] as a percentage)")
            RH = rh
            Twb = cpc.HAPropsSI("Twb", "T", Td, "P", P, "R", RH)
        else:
            RH = cpc.HAPropsSI("R", "T", Td, "P", P, "Twb", Twet)
            Twb = Twet
            if RH > 1.0 + 1e-6:
                warnings.append(f"computed RH {RH:.3f} > 1: wet bulb above dry bulb")
        W = cpc.HAPropsSI("W", "T", Td, "P", P, "R", RH)
        h = cpc.HAPropsSI("H", "T", Td, "P", P, "R", RH)
        Tdp = cpc.HAPropsSI("Tdp", "T", Td, "P", P, "R", RH)
        # In HAPropsSI "D" is the dew point, not density: rho = (1+W)/v with v per kg dry air.
        v_da = cpc.HAPropsSI("V", "T", Td, "P", P, "R", RH)
        rho = (1.0 + W) / v_da
        if Twb > Td + 1e-3:
            warnings.append("wet-bulb temperature exceeds dry-bulb temperature")
        approach = None if Twater is None else Twater - Twb
        if approach is not None and approach < 0:
            warnings.append("negative cooling-tower approach (water colder than wet bulb)")
        out = {
            "ok": True, "model": "CoolProp humid-air (HAPropsSI)",
            "T_dry_K": _r(Td, 4), "P_kPa": _r(P / 1000.0, 5),
            "relative_humidity": _r(RH, 6),
            "humidity_ratio_kg_per_kg": _r(W, 8),
            "enthalpy_kJ_kg_da": _r(h / 1000.0, 5),
            "dew_point_K": _r(Tdp, 4), "wet_bulb_K": _r(Twb, 4),
            "density_kg_m3": _r(rho, 5),
            "T_water_K": Twater, "approach_K": _r(approach, 4),
            "warnings": warnings,
        }
        return _dump(out)
    except Exception as exc:
        return _fail(exc)


# --------------------------------------------------------------------------- #
# 9. exergy_stream
# --------------------------------------------------------------------------- #
def exergy_stream(candidate: str = "") -> str:
    """Physical + chemical flow exergy of one stream relative to a dead state.

    Args:
        candidate (str): JSON with ``fluid``, ``T_K``, ``P_KPa`` (``P_kPa`` also
            accepted), ``m_dot_kg_s``, optional ``T0_K``, ``P0_kPa`` and a molar
            ``composition`` for gas-mixture chemical exergy.

    Returns:
        str: JSON with specific physical/chemical exergy, flow exergy and h/s.
    """
    try:
        p = _parse(candidate)
        fluid = _str(p, "fluid")
        pkey = "P_KPa" if "P_KPa" in p else "P_kPa"
        P = _num(p, pkey, positive=True) * 1000.0
        T = _num(p, "T_K", positive=True)
        m_dot = _num(p, "m_dot_kg_s", positive=True)
        T0 = _optnum(p, "T0_K", positive=True) or 298.15
        P0 = (_optnum(p, "P0_kPa", positive=True) or 101.325) * 1000.0
        comp = p.get("composition")

        st = _state(fluid, T=T, P=P)
        dead = _state(fluid, T=T0, P=P0)
        e_ph = (st["h"] - dead["h"]) - T0 * (st["s"] - dead["s"])  # J/kg
        e_ch = 0.0
        if isinstance(comp, dict) and comp:
            from ..core import exergy as ex
            env = ex.Env(T0=T0, P0=P0)
            Mcomp = _gas_m(comp)
            e_ch = ex.e_ch_gas(comp, env) / Mcomp
        elif fluid == "CH4":
            from ..core import exergy as ex
            env = ex.Env(T0=T0, P0=P0)
            eps = ex._eps(env)
            e_ch = eps["CH4"] / _gas_m(GAS_PRESETS["CH4"])
        e_ph_kj = e_ph / 1000.0
        e_ch_kj = e_ch / 1000.0
        out = {
            "ok": True, "fluid": fluid, "model": st["model"],
            "specific_physical_exergy_kJ_kg": _r(e_ph_kj, 6),
            "specific_chemical_exergy_kJ_kg": _r(e_ch_kj, 6),
            "flow_exergy_W": _r(m_dot * (e_ph + e_ch), 3),
            "h_kJ_kg": _r(st["h"] / 1000.0, 6), "h": _r(st["h"] / 1000.0, 6),
            "s_kJ_kgK": _r(st["s"] / 1000.0, 8), "s": _r(st["s"] / 1000.0, 8),
            "m_dot_kg_s": m_dot, "T0_K": T0, "P0_kPa": _r(P0 / 1000.0, 5),
        }
        return _dump(out)
    except Exception as exc:
        return _fail(exc)


# --------------------------------------------------------------------------- #
# public tool registry
# --------------------------------------------------------------------------- #
DYNAMIC_TOOL_NAMES = (
    "fluid_state",
    "saturation_state",
    "turbomachine",
    "heat_exchanger",
    "combustor",
    "cycle_performance",
    "flow_network",
    "psychrometrics",
    "exergy_stream",
)


def dynamic_tools() -> list:
    """Return the nine continuous-input tools as plain module-level functions.

    The order matches :data:`DYNAMIC_TOOL_NAMES` exactly; each callable keeps its
    ``(candidate: str = "") -> str`` signature for the shared tool-call parser.
    """
    return [fluid_state, saturation_state, turbomachine, heat_exchanger,
            combustor, cycle_performance, flow_network, psychrometrics,
            exergy_stream]


__all__ = ["DYNAMIC_TOOL_NAMES", "dynamic_tools", *DYNAMIC_TOOL_NAMES]
