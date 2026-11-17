"""V7: every number with a unit in a reasoning trace must match a value the solver or the plant spec actually contains."""
from __future__ import annotations
import re
from ..core.schema import Plant, SolveResult

_UNITS = {"MW": ("W", 1e6), "kW": ("W", 1e3), "W": ("W", 1.0), "K": ("K", 1.0), "°C": ("K", None), "degC": ("K", None),
          "bar": ("Pa", 1e5), "kPa": ("Pa", 1e3), "Pa": ("Pa", 1.0), "kg/s": ("kg/s", 1.0), "%": ("frac", 0.01)}
_RX = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(MW|kW|W|K|°C|degC|bar|kPa|Pa|kg/s|%)(?![\w/])")


def _candidates(plant: Plant, res: SolveResult) -> dict[str, list[float]]:
    c: dict[str, list[float]] = {"W": [], "K": [], "Pa": [], "kg/s": [], "frac": []}
    for s in res.states:
        c["K"].append(s.T); c["Pa"].append(s.P); c["kg/s"].append(s.m_dot)
    for b in res.balances:
        c["W"] += [abs(b.Q_w), abs(b.W_w), abs(b.exergy_destruction_w)]
    for k, v in res.kpis.items():
        if k.endswith("_w"): c["W"].append(abs(v))
        elif k.endswith("_K"): c["K"].append(v)
        elif k.startswith("eta"): c["frac"].append(v)
    for s in plant.streams:
        if s.m_dot: c["kg/s"].append(s.m_dot)
        if s.fixed:
            if s.fixed.T: c["K"].append(s.fixed.T)
            if s.fixed.P: c["Pa"].append(s.fixed.P)
    for comp in plant.components:
        for k, v in comp.params.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)): continue
            if k == "P_it": c["W"].append(v)
            elif k.startswith("T_"): c["K"].append(v)
            elif k.startswith("P_"): c["Pa"].append(v)
            elif k.startswith(("eta", "dp", "q_loss")): c["frac"].append(v)
            elif k.startswith(("pinch", "dT")): c["K"].append(v)       # temperature differences
    return c


def check_numbers(text: str, plant: Plant, res: SolveResult, rel: float = 0.005) -> dict:
    cand = _candidates(plant, res)
    grounded, ungrounded = [], []
    for val, unit in _RX.findall(text):
        cls, scale = _UNITS[unit]; v = float(val)
        si = v + 273.15 if scale is None else v * scale
        pool = cand[cls]
        ok = any(abs(si - x) <= rel * max(abs(x), 1e-9) + (0.05 if cls == "K" else 0) for x in pool)
        if cls == "K" and not ok:                       # allow temperature differences quoted in K
            ok = any(abs(v - x) <= rel * max(abs(x), 1e-9) + 0.05 for x in pool)
        (grounded if ok else ungrounded).append(f"{val} {unit}")
    n = len(grounded) + len(ungrounded)
    # No numbers to check: a substantive diagnosis is still grounded (e.g. a sign flip described in words).
    # An empty or trivial tag is not: without this guard, "<diagnosis></diagnosis>" earned the full reward.
    words = len(re.findall(r"[A-Za-z]{3,}", text))
    return {"n_numbers": n, "grounded": grounded, "ungrounded": ungrounded,
            "fraction": (len(grounded) / n) if n else (1.0 if words >= 5 else 0.0)}
