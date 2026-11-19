"""Ideal-gas properties for N2, O2, CO2, H2O(g), CH4 from NASA-7 polynomials (GRI-Mech 3.0).

Units: SI molar. h [J/mol] (== kJ/kmol numerically), s [J/mol/K], T [K], P [Pa].
h includes the formation enthalpy (needed for combustion energy balances).

Two entropy conventions are offered for mixtures:
  mixing=False : s = sum y_i s_i(T, P_total)           (no mixing term)
  mixing=True  : s = sum y_i s_i(T, y_i P) (partial pressures; thermodynamically correct)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from ._roots import brentq

R = 8.314462618
_DATA = json.loads((Path(__file__).parent / "data" / "nasa7_gri30.json").read_text())["species"]
MW = {k: v["MW"] for k, v in _DATA.items()}  # kg/kmol
Composition = dict  # {species: mole fraction}


def _a(sp: str, T: float):
    d = _DATA[sp]
    return d["low"] if T < d["Tmid"] else d["high"]


def h_sp(sp: str, T: float) -> float:
    a = _a(sp, T)
    return R * T * (a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T)


def s0_sp(sp: str, T: float) -> float:
    a = _a(sp, T)
    return R * (a[0]*np.log(T) + a[1]*T + a[2]*T**2/2 + a[3]*T**3/3 + a[4]*T**4/4 + a[6])


def s_sp(sp: str, T: float, P: float) -> float:
    return s0_sp(sp, T) - R*np.log(P/_DATA[sp]["Pref"])


def mw(comp: Composition) -> float:
    return sum(y*MW[k] for k, y in comp.items())


def h_mix(comp: Composition, T: float) -> float:
    return sum(y*h_sp(k, T) for k, y in comp.items())


def s_mix(comp: Composition, T: float, P: float, mixing: bool = False) -> float:
    if mixing:
        return sum(y*s_sp(k, T, y*P) for k, y in comp.items() if y > 0)
    return sum(y*s_sp(k, T, P) for k, y in comp.items())


def T_from_h(comp: Composition, h: float, lo: float = 200.0, hi: float = 3500.0) -> float:
    return brentq(lambda T: h_mix(comp, T) - h, lo, hi, xtol=1e-10)


def T_from_s(comp: Composition, s: float, P: float, lo: float = 200.0, hi: float = 3500.0) -> float:
    return brentq(lambda T: s_mix(comp, T, P) - s, lo, hi, xtol=1e-10)


def T_from_s_mixing(comp: Composition, s: float, P: float, lo: float = 200.0, hi: float = 3500.0) -> float:
    return brentq(lambda T: s_mix(comp, T, P, mixing=True) - s, lo, hi, xtol=1e-10)


def lhv_molar(T: float = 298.15) -> float:
    """CH4 + 2 O2 -> CO2 + 2 H2O(g), J per mol CH4 (from NASA-7 formation data)."""
    return h_sp("CH4", T) + 2*h_sp("O2", T) - h_sp("CO2", T) - 2*h_sp("H2O", T)


# CGAM air composition (Bejan et al.)
AIR_CGAM: Composition = {"N2": 0.7748, "O2": 0.2059, "CO2": 0.0003, "H2O": 0.019}
