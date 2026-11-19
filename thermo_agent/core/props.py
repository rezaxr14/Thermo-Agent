"""Unified state objects over ideal-gas mixtures (NASA-7) and real fluids (CoolProp). Molar SI basis."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
try:  # hosted benchmarks venv may lack CoolProp; ideal-gas states do not need it
    import CoolProp.CoolProp as CP
except ModuleNotFoundError:  # pragma: no cover - depends on the interpreter
    CP = None
from . import gasprops as gp, fluids as fl

GAS_PRESETS: dict[str, dict] = {"CGAM_AIR": gp.AIR_CGAM, "CH4": {"CH4": 1.0}}
GAS_NAMES = set(GAS_PRESETS) | {"FLUE"}


class SolveError(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(msg); self.code = code; self.msg = msg


def is_gas(fluid: str) -> bool:
    return fluid in GAS_NAMES


def check_fluid(fluid: str):
    if is_gas(fluid):
        return
    try:
        fl.M(fluid)
    except Exception:
        raise SolveError("E_UNKNOWN_FLUID", f"unknown fluid '{fluid}'")


@dataclass
class State:
    stream: str
    fluid: str
    comp: Optional[dict]
    T: float; P: float; h: float; s: float          # K, Pa, J/mol, J/mol/K
    n: Optional[float]                               # mol/s
    x: Optional[float] = None

    @property
    def M(self) -> float:                            # kg/mol
        return gp.mw(self.comp) / 1000 if self.comp is not None else fl.M(self.fluid)

    @property
    def m(self) -> Optional[float]:
        return None if self.n is None else self.n * self.M


def comp_of(fluid: str, comp: Optional[dict]) -> Optional[dict]:
    if comp is not None: return comp
    return GAS_PRESETS.get(fluid)


def M_of(fluid, comp):
    c = comp_of(fluid, comp)
    return gp.mw(c) / 1000 if c is not None else fl.M(fluid)


def _quality(fluid, h, P):
    try:
        q = fl._cp().PropsSI("Q", "H", h / fl.M(fluid), "P", P, fluid)
        return q if 0 <= q <= 1 else None
    except Exception:
        return None


def _positive(**kw):
    for k, v in kw.items():
        if not (v > 0 and v < float("inf")):
            raise SolveError("E_PROP_RANGE", f"{k}={v} is not a positive finite number")


def state_TP(stream, fluid, comp, T, P, n) -> State:
    _positive(T=T, P=P)
    c = comp_of(fluid, comp)
    try:
        if c is not None:
            return State(stream, fluid, c, T, P, gp.h_mix(c, T), gp.s_mix(c, T, P, mixing=True), n)
        h = fl.h_TP(fluid, T, P)
        return State(stream, fluid, None, T, P, h, fl.s_TP(fluid, T, P), n, _quality(fluid, h, P))
    except SolveError: raise
    except Exception as e:
        raise SolveError("E_PROP_RANGE", f"{fluid} at T={T:.1f} K P={P:.3g} Pa: {e}")


def state_hP(stream, fluid, comp, h, P, n) -> State:
    _positive(P=P)
    c = comp_of(fluid, comp)
    try:
        if c is not None:
            try: T = gp.T_from_h(c, h)
            except ValueError:
                raise SolveError("E_PROP_RANGE", f"{fluid}: enthalpy {h:.4g} J/mol is outside the property data (T would be below 200 K or above 3500 K)")
            return State(stream, fluid, c, T, P, h, gp.s_mix(c, T, P, mixing=True), n)
        return State(stream, fluid, None, fl.T_hP(fluid, h, P), P, h, fl.s_hP(fluid, h, P), n, _quality(fluid, h, P))
    except SolveError: raise
    except Exception as e:
        raise SolveError("E_PROP_RANGE", f"{fluid} at h={h:.4g} J/mol P={P:.3g} Pa: {e}")


def state_sP(stream, fluid, comp, s, P, n) -> State:
    _positive(P=P)
    c = comp_of(fluid, comp)
    try:
        if c is not None:
            T = gp.T_from_s_mixing(c, s, P) if hasattr(gp, "T_from_s_mixing") else None
            h = gp.h_mix(c, T)
            return State(stream, fluid, c, T, P, h, s, n)
        h = fl.h_sP(fluid, s, P)
        return State(stream, fluid, None, fl.T_hP(fluid, h, P), P, h, s, n, _quality(fluid, h, P))
    except SolveError: raise
    except Exception as e:
        raise SolveError("E_PROP_RANGE", f"{fluid} isentropic state s={s:.4g} P={P:.3g} Pa: {e}")


def state_sat(stream, fluid, P, q, n) -> State:
    _positive(P=P)
    try:
        h = fl.h_Pq(fluid, P, q)
        return State(stream, fluid, None, fl.Tsat_P(fluid, P), P, h, fl.s_Pq(fluid, P, q), n, q)
    except Exception as e:
        raise SolveError("E_PROP_RANGE", f"{fluid} saturation at P={P:.3g} Pa: {e}")
