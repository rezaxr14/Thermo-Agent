"""Optimizer-style baselines for Design. Cost unit: design evaluations (build + solve + verify, about 4 solver runs)."""
from __future__ import annotations
import itertools, random
import numpy as np
from ..core.schema import Site
from .spec import DISCRETE, CONTINUOUS, HEAT_SOURCES, default_spec, required_keys
from .build import evaluate_design

INVALID = 1e3


def enumerate_designs(site: Site) -> list[tuple[dict, object]]:
    """Every discrete combination with default continuous parameters."""
    heat = HEAT_SOURCES if site.heat_demand_w > 0 else (None,)
    return [(s, evaluate_design(site, s)) for s in (default_spec(*c, heat_source=h) for c in itertools.product(*DISCRETE.values()) for h in heat)]


def polish(site: Site, spec: dict, max_evals: int = 60, trace: list | None = None) -> tuple[dict, float | None, int]:
    """Nelder-Mead on the continuous parameters of a fixed discrete configuration (minimise fuel, invalid = penalty)."""
    keys = [k for k in required_keys(spec) if k in CONTINUOUS]; n = {"evals": 0}
    lo = np.array([CONTINUOUS[k][0] for k in keys]); hi = np.array([CONTINUOUS[k][1] for k in keys])
    to_spec = lambda x: {**spec, **{k: float(lo[i] + np.clip(x[i], 0, 1) * (hi[i] - lo[i])) for i, k in enumerate(keys)}}
    best = {"fuel": None, "spec": spec}

    def f(x):
        n["evals"] += 1; s = to_spec(x); e = evaluate_design(site, s)
        if e.fuel_w is not None and (best["fuel"] is None or e.fuel_w < best["fuel"]): best.update(fuel=e.fuel_w, spec=s)
        if trace is not None: trace.append((n["evals"], best["fuel"]))                  # best fuel after each evaluation
        return INVALID if e.fuel_w is None else e.fuel_w / 1e6
    x0 = (np.array([spec[k] for k in keys]) - lo) / (hi - lo)
    if keys:
        from scipy.optimize import minimize  # lazy: hosted task venv has no scipy and never designs via polish
        minimize(f, x0, method="Nelder-Mead", options={"maxfev": max_evals, "xatol": 1e-3, "fatol": 1e-4, "initial_simplex": _simplex(x0)})
    else: f(x0)
    return best["spec"], best["fuel"], n["evals"]


def _simplex(x0, step=0.15):
    pts = [x0] + [np.clip(x0 + step * np.eye(len(x0))[i] * (1 if x0[i] < 0.5 else -1), 0, 1) for i in range(len(x0))]
    return np.array(pts)


def reference_design(site: Site, top: int = 2, max_evals: int = 60) -> dict | None:
    """Best known design: enumerate, then polish the best `top` valid configurations."""
    table = enumerate_designs(site); evals = len(table)
    valid = sorted([(e.fuel_w, s) for s, e in table if e.fuel_w is not None], key=lambda t: t[0])
    if not valid: return None
    best_fuel, best_spec = valid[0]
    for _, s in valid[:top]:
        sp, fuel, n = polish(site, s, max_evals); evals += n
        if fuel is not None and fuel < best_fuel: best_fuel, best_spec = fuel, sp
    return {"spec": best_spec, "fuel_w": best_fuel, "evals": evals, "table": [(s, e.fuel_w, e.reason) for s, e in table]}


def anytime_enumeration(site: Site) -> list[float | None]:
    """Best fuel after each of the default configurations, in catalog order."""
    best, out = None, []
    for _, e in enumerate_designs(site):
        if e.fuel_w is not None and (best is None or e.fuel_w < best): best = e.fuel_w
        out.append(best)
    return out


def anytime_reference(site: Site, top: int = 2, max_evals: int = 60) -> list[float | None]:
    """Best fuel after each evaluation of enumerate-then-polish (the reference procedure)."""
    curve = anytime_enumeration(site); table = enumerate_designs(site)       # (enumeration is repeated: cheap and keeps this function standalone)
    valid = sorted([(e.fuel_w, s) for s, e in table if e.fuel_w is not None], key=lambda t: t[0])
    best = curve[-1]
    for _, s in valid[:top]:
        tr: list = []; polish(site, s, max_evals, tr)
        for _, f in tr:
            if f is not None and (best is None or f < best): best = f
            curve.append(best)
    return curve


def random_search(site: Site, rng: random.Random, budget: int = 20) -> list[float | None]:
    """Best fuel found after each evaluation (None until a valid design appears)."""
    best, out = None, []
    for _ in range(budget):
        s = {k: rng.choice(v) for k, v in DISCRETE.items()}
        if site.heat_demand_w > 0: s["heat_source"] = rng.choice(HEAT_SOURCES)
        s.update({k: rng.uniform(*CONTINUOUS[k][:2]) for k in required_keys(s) if k in CONTINUOUS})
        e = evaluate_design(site, s)
        if e.fuel_w is not None and (best is None or e.fuel_w < best): best = e.fuel_w
        out.append(best)
    return out
