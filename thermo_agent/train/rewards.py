"""Verifiable reward functions for GRPO (TRL contract: f(prompts, completions, **dataset_columns) -> list[float | None]).
Dataset columns used: plant_broken, plant_original, site (JSON strings; '' when absent), deducible, min_edits."""
from __future__ import annotations
import json, re
from functools import lru_cache
from pydantic import ValidationError
from ..core.schema import Plant, Site
from ..core.solve import solve
from ..verify.verifiers import run_all
from ..verify.grounding import check_numbers
from ..agent.dsl import parse_edits, edits_to_patch, EditError
from ..synth.faults import apply_patch, _dump
from ..eval.repair import _close


def completion_text(c) -> str:
    return c if isinstance(c, str) else c[-1]["content"]


def _diagnosis(text: str):
    m = re.search(r"<diagnosis>(.*?)</diagnosis>", text, flags=re.S)
    return m.group(1).strip() if m else None


@lru_cache(maxsize=8192)
def evaluate(text: str, broken_js: str, original_js: str, site_js: str, deducible: bool) -> dict:
    out = {"parsed": False, "applied": False, "solved": False, "physics_ok": False, "site_ok": None, "exact": False,
           "n_edits": 0, "score": 0.0, "grounded": 0.0, "has_diag": False}
    broken = Plant.model_validate_json(broken_js); original = Plant.model_validate_json(original_js)
    site = Site.model_validate_json(site_js) if site_js else None
    diag = _diagnosis(text); out["has_diag"] = diag is not None
    try:
        edits = parse_edits(text); out["parsed"] = True; out["n_edits"] = len(edits)
        patch = edits_to_patch(broken, edits)
        fixed = Plant.model_validate(apply_patch(_dump(broken), patch)); out["applied"] = True
    except (EditError, ValidationError, KeyError, IndexError, ValueError, TypeError):
        fixed = None
    bres = solve(broken)
    if diag is not None:
        out["grounded"] = check_numbers(diag, broken, bres)["fraction"]
    if fixed is None: return out
    res = solve(fixed); out["solved"] = res.ok
    if res.ok:
        rep = run_all(fixed, res, site)
        phys = [c for c in rep.checks if not c.name.startswith("V8")]
        out["physics_ok"] = all(c.ok for c in phys if c.hard)
        out["score"] = sum(c.ok for c in phys) / max(len(phys), 1)
        if site is not None: out["site_ok"] = all(c.ok for c in rep.checks if c.name.startswith("V8"))
    out["exact"] = _close(_dump(fixed), _dump(original))
    return out


def _each(completions, kw, fn):
    n = len(completions)
    site = kw.get("site") or [""] * n
    return [fn(evaluate(completion_text(c), kw["plant_broken"][i], kw["plant_original"][i], site[i], bool(kw["deducible"][i])), i, kw)
            for i, c in enumerate(completions)]


def r_format(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: 1.0 if (e["applied"] and e["has_diag"]) else (0.3 if e["parsed"] else 0.0))

def r_valid(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: (1.0 if e["physics_ok"] else 0.5 * e["score"]) if e["solved"] else 0.0)

def r_site(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: None if e["site_ok"] is None else float(bool(e["site_ok"] and e["physics_ok"])))

def r_exact(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: float(e["exact"]) if kw["deducible"][i] else None)

def r_minimal(prompts, completions, **kw):
    def f(e, i, k):
        ok = e["physics_ok"] and e["site_ok"] is not False
        return 1.0 / (1 + max(0, e["n_edits"] - int(k["min_edits"][i]))) if ok else 0.0
    return _each(completions, kw, f)

def r_grounded(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: e["grounded"] if e["has_diag"] else 0.0)


REWARD_FUNCS = [r_format, r_valid, r_site, r_exact, r_minimal, r_grounded]
REWARD_WEIGHTS = [0.5, 2.0, 1.0, 1.0, 0.5, 0.5]
