"""Thin molar-basis wrappers around CoolProp for real fluids (water, R245fa, ...).

All functions return molar quantities: h [J/mol], s [J/mol/K], v [m3/mol].
"""
from __future__ import annotations

try:  # the hosted benchmarks venv may lack CoolProp; gas cycles still work without it
    import CoolProp.CoolProp as CP
except ModuleNotFoundError:  # pragma: no cover - depends on the interpreter
    CP = None

_M: dict[str, float] = {}


def _cp():
    if CP is None:
        raise RuntimeError("CoolProp is not installed; real-fluid states are unavailable in this environment")
    return CP



def M(fluid: str) -> float:
    """Molar mass [kg/mol]."""
    if fluid not in _M:
        _M[fluid] = _cp().PropsSI("M", fluid)
    return _M[fluid]


def _p(out, a, x, b, y, fluid):
    return _cp().PropsSI(out, a, x, b, y, fluid)


def h_TP(f, T, P): return _p("H", "T", T, "P", P, f) * M(f)
def s_TP(f, T, P): return _p("S", "T", T, "P", P, f) * M(f)
def h_Tq(f, T, q): return _p("H", "T", T, "Q", q, f) * M(f)
def h_Pq(f, P, q): return _p("H", "P", P, "Q", q, f) * M(f)
def s_Pq(f, P, q): return _p("S", "P", P, "Q", q, f) * M(f)
def Tsat_P(f, P): return _p("T", "P", P, "Q", 0, f)
def Psat_T(f, T): return _p("P", "T", T, "Q", 0, f)
def T_hP(f, h, P): return _p("T", "H", h / M(f), "P", P, f)
def s_hP(f, h, P): return _p("S", "H", h / M(f), "P", P, f) * M(f)
def h_sP(f, s, P): return _p("H", "S", s / M(f), "P", P, f) * M(f)
def Tmax(f): return _cp().PropsSI("Tmax", f)
def v_Pq(f, P, q): return M(f) / _p("D", "P", P, "Q", q, f)
