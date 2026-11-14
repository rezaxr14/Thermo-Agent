"""GRPO rewards for Design (TRL contract). Dataset columns: site (JSON string), ref_fuel_w (float)."""
from __future__ import annotations
from functools import lru_cache
from ..core.schema import Site
from ..verify.grounding import check_numbers
from .spec import parse_design, DesignError
from .build import evaluate_design
import re


def completion_text(c) -> str:
    return c if isinstance(c, str) else c[-1]["content"]


@lru_cache(maxsize=8192)
def evaluate(text: str, site_js: str) -> dict:
    out = {"parsed": False, "built": False, "valid": False, "fuel_w": None, "grounded": 0.0, "has_rationale": False, "reason": ""}
    site = Site.model_validate_json(site_js)
    m = re.search(r"<rationale>(.*?)</rationale>", text, flags=re.S); out["has_rationale"] = m is not None
    try: spec = parse_design(text, needs_heat=site.heat_demand_w > 0); out["parsed"] = True
    except DesignError as e: out["reason"] = str(e); return out
    ev = evaluate_design(site, spec)
    out.update(built=ev.built, valid=ev.valid, fuel_w=ev.fuel_w, reason=ev.reason)
    if m and ev.result is not None and ev.result.ok: out["grounded"] = check_numbers(m.group(1), ev.plant, ev.result, rel=0.03)["fraction"]
    return out


def _each(completions, kw, fn):
    return [fn(evaluate(completion_text(c), kw["site"][i]), i, kw) for i, c in enumerate(completions)]


def d_format(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: 1.0 if (e["parsed"] and e["has_rationale"]) else (0.3 if e["parsed"] else 0.0))

def d_valid(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: 1.0 if e["valid"] else (0.2 if e["built"] else 0.0))

def d_quality(prompts, completions, **kw):
    """1.0 at the reference fuel, falling linearly to 0 at 10 % more fuel; up to 1.05 for beating the reference."""
    def f(e, i, k):
        if not e["valid"]: return 0.0
        ref = float(k["ref_fuel_w"][i]); return max(0.0, min(1.05, 1.0 - (e["fuel_w"] - ref) / (0.1 * ref)))
    return _each(completions, kw, f)

def d_grounded(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: e["grounded"] if e["valid"] else 0.0)


REWARD_FUNCS = [d_format, d_valid, d_quality, d_grounded]
REWARD_WEIGHTS = [0.5, 2.0, 2.0, 0.5]
