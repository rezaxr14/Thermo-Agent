"""Exergy of streams relative to an explicit dead state (T0, P0, environment gas composition).

Gas chemical exergy (per mol): sum_i y_i eps_i + R T0 sum_i y_i ln y_i
  eps_i for species in the environment: -R T0 ln(y_env_i)   -> ambient air has exactly zero exergy
  eps_CH4: from the Gibbs energy of CH4 + 2 O2 -> CO2 + 2 H2O(g) at (T0, 1 atm)
Liquids / refrigerants: chemical exergy set to 0 (they never react; constant terms cancel in every balance).
Physical exergy: (h - h0) - T0 (s - s0) with h0,s0 at (T0,P0) for the SAME composition/fluid.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
import math
from . import gasprops as gp, fluids as fl
from .props import State, is_gas

R = gp.R


@dataclass(frozen=True)
class Env:
    T0: float = 298.15
    P0: float = 101325.0
    y_env: tuple = tuple(sorted(gp.AIR_CGAM.items()))   # hashable


@lru_cache(maxsize=None)
def _eps(env: Env) -> dict:
    y = dict(env.y_env); T0 = env.T0
    eps = {sp: -R * T0 * math.log(yy) for sp, yy in y.items() if yy > 0}
    g = lambda sp: gp.h_sp(sp, T0) - T0 * gp.s0_sp(sp, T0)
    dG = g("CO2") + 2 * g("H2O") - g("CH4") - 2 * g("O2")
    eps["CH4"] = -dG + eps["CO2"] + 2 * eps["H2O"] - 2 * eps["O2"]
    return eps


def e_ch_gas(comp: dict, env: Env) -> float:
    eps = _eps(env)
    return sum(y * eps[k] for k, y in comp.items()) + R * env.T0 * sum(y * math.log(y) for y in comp.values() if y > 0)


def e_molar(st: State, env: Env) -> tuple[float, float]:
    """(e_ph, e_ch) in J/mol."""
    if st.comp is not None:
        h0 = gp.h_mix(st.comp, env.T0); s0 = gp.s_mix(st.comp, env.T0, env.P0, mixing=True)
        return (st.h - h0) - env.T0 * (st.s - s0), e_ch_gas(st.comp, env)
    h0 = fl.h_TP(st.fluid, env.T0, env.P0); s0 = fl.s_TP(st.fluid, env.T0, env.P0)
    return (st.h - h0) - env.T0 * (st.s - s0), 0.0
