"""Frontier supercomputer energy dataset (Sun et al., Scientific Data 2024): turns a measured trace into Site specs and
dc_chp cooling-loop settings.

The real file's column names have not been inspected here, so nothing is guessed: call describe(path) to list the
columns, then pass an explicit `columns` mapping to load_frontier. Required keys: it_power, t_supply, t_return.
Optional: waste_heat. Units are declared, not inferred."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..core.schema import Site

_P = {"W": 1.0, "kW": 1e3, "MW": 1e6}


def _read(path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)


def describe(path) -> pd.DataFrame:
    df = _read(path)
    return pd.DataFrame({"dtype": df.dtypes.astype(str), "non_null": df.notna().sum(), "min": df.min(numeric_only=True),
                         "median": df.median(numeric_only=True), "max": df.max(numeric_only=True)})


def load_frontier(path, columns: dict[str, str], power_unit: str = "MW", temp_unit: str = "C") -> pd.DataFrame:
    """-> DataFrame with it_power_w, t_supply_k, t_return_k (+ waste_heat_w). Fails loudly on missing columns or bad units."""
    for k in ("it_power", "t_supply", "t_return"):
        if k not in columns: raise KeyError(f"columns must map '{k}' to a column name")
    if power_unit not in _P: raise ValueError(f"power_unit must be one of {list(_P)}")
    if temp_unit not in ("C", "K"): raise ValueError("temp_unit must be 'C' or 'K'")
    df = _read(path); missing = [v for v in columns.values() if v not in df.columns]
    if missing: raise KeyError(f"columns not in file: {missing}; available: {list(df.columns)}")
    off = 273.15 if temp_unit == "C" else 0.0
    out = pd.DataFrame({"it_power_w": df[columns["it_power"]] * _P[power_unit],
                        "t_supply_k": df[columns["t_supply"]] + off, "t_return_k": df[columns["t_return"]] + off})
    if "waste_heat" in columns: out["waste_heat_w"] = df[columns["waste_heat"]] * _P[power_unit]
    out = out.dropna()
    bad = out[(out.t_return_k <= out.t_supply_k) | (out.it_power_w <= 0)]
    if len(bad) > 0.2 * len(out): raise ValueError(f"{len(bad)} of {len(out)} rows have return <= supply or non-positive power; check units")
    return out.drop(bad.index)


def site_from_trace(df: pd.DataFrame, quantile: float = 0.95, grid_import_allowed: bool = False) -> tuple[Site, dict]:
    """Design-point Site from a load quantile, plus the median loop settings and heat-capture ratio for dc_chp."""
    P = float(df.it_power_w.quantile(quantile))
    ratio = float((df.waste_heat_w / df.it_power_w).median()) if "waste_heat_w" in df else None
    loop = {"T_supply": float(df.t_supply_k.median()), "T_return": float(df.t_return_k.median())}
    if ratio is not None: loop["chilled_fraction"] = min(max(ratio, 0.05), 1.0)
    site = Site(it_load_w=P, cooling_load_w=(P * ratio if ratio else None), grid_import_allowed=grid_import_allowed)
    return site, {"P_it": P, **loop}
