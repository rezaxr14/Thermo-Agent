"""Design catalog and the text form the model writes.

    prime_mover = turbine | engine
    cooling     = absorption | electric
    rejection   = dry | wet
    bottoming   = none | orc_r245fa | orc_cyclopentane | orc_toluene
    heat_source = exhaust | heat_pump          (only when the site has a heat demand)
    dT_rej      = cooling-water temperature rise on the chiller / ORC rejection side (K)
    PR, T4      = turbine pressure ratio, turbine inlet temperature (K)                [turbine only]
    T_hw, dT_hw = hot-water supply temperature (K) and its drop across the generator (K)   [absorption only]
    P_orc_bar, orc_frac = ORC evaporation pressure (bar) and flow position between its minimum and maximum   [ORC only]
"""
from __future__ import annotations
import re

DISCRETE = {"prime_mover": ("turbine", "engine"), "cooling": ("absorption", "electric"), "rejection": ("dry", "wet"),
            "bottoming": ("none", "orc_r245fa", "orc_cyclopentane", "orc_toluene")}
HEAT_SOURCES = ("exhaust", "heat_pump")
# name: (low, high, default)
CONTINUOUS = {"dT_rej": (5.0, 12.0, 8.0), "PR": (10.0, 30.0, 18.0), "T4": (1300.0, 1600.0, 1500.0),
              "T_hw": (353.15, 383.15, 368.15), "dT_hw": (5.0, 20.0, 10.0),
              "P_orc_bar": (10.0, 25.0, 12.0), "orc_frac": (0.1, 1.0, 0.5)}
ORC_FLUID = {"orc_r245fa": "R245fa", "orc_cyclopentane": "Cyclopentane", "orc_toluene": "Toluene"}


class DesignError(ValueError):
    pass


def required_keys(spec: dict, needs_heat: bool | None = None) -> list[str]:
    if needs_heat is None: needs_heat = "heat_source" in spec
    keys = list(DISCRETE) + ["dT_rej"]
    if spec.get("prime_mover") == "turbine": keys += ["PR", "T4"]
    if spec.get("cooling") == "absorption": keys += ["T_hw", "dT_hw"]
    if spec.get("bottoming", "none") != "none": keys += ["P_orc_bar", "orc_frac"]
    if needs_heat: keys += ["heat_source"]
    return keys


def default_spec(prime_mover: str, cooling: str, rejection: str, bottoming: str = "none", heat_source: str | None = None) -> dict:
    s = {"prime_mover": prime_mover, "cooling": cooling, "rejection": rejection, "bottoming": bottoming}
    if heat_source: s["heat_source"] = heat_source
    s.update({k: CONTINUOUS[k][2] for k in required_keys(s) if k in CONTINUOUS})
    return s


def validate(spec: dict, needs_heat: bool = False) -> dict:
    for k, opts in DISCRETE.items():
        if spec.get(k) not in opts: raise DesignError(f"{k} must be one of {opts}")
    if needs_heat and spec.get("heat_source") not in HEAT_SOURCES: raise DesignError(f"heat_source must be one of {HEAT_SOURCES}")
    need = required_keys(spec, needs_heat)
    if set(spec) != set(need): raise DesignError(f"keys must be exactly {sorted(need)}; got {sorted(spec)}")
    for k in need:
        if k in CONTINUOUS:
            lo, hi, _ = CONTINUOUS[k]
            if not (lo <= spec[k] <= hi): raise DesignError(f"{k}={spec[k]} outside [{lo}, {hi}]")
    return spec


def parse_design(text: str, needs_heat: bool = False) -> dict:
    m = re.search(r"<design>(.*?)</design>", text, flags=re.S)
    if not m: raise DesignError("no <design> block")
    spec = {}
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line: continue
        mm = re.match(r"^([A-Za-z0-9_]+)\s*=\s*(\S+)$", line)
        if not mm: raise DesignError(f"bad line: {line[:50]!r}")
        k, v = mm.groups()
        if k in spec: raise DesignError(f"duplicate key {k}")
        if k in DISCRETE or k == "heat_source": spec[k] = v
        elif k in CONTINUOUS:
            try: spec[k] = float(v)
            except ValueError: raise DesignError(f"{k} is not a number")
        else: raise DesignError(f"unknown key {k}")
    return validate(spec, needs_heat)


def format_design(spec: dict) -> str:
    lines = [f"{k} = {spec[k]}" if (k in DISCRETE or k == "heat_source") else f"{k} = {spec[k]!r}" for k in required_keys(spec)]
    return "<design>\n" + "\n".join(lines) + "\n</design>"


def catalog_text() -> str:
    parts = [f"{k} in {{{', '.join(v)}}}" for k, v in DISCRETE.items()]
    parts += [f"{k} in [{lo:g}, {hi:g}]" for k, (lo, hi, _) in CONTINUOUS.items()]
    return "; ".join(parts)
