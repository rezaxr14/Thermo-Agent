"""GRPO rewards for Audit. The ranking reward scores only the components other than the combustor / gas engine. Dataset columns: plant, site (JSON; '' if none), ed_json (component -> destruction W, plant only),
fuel_w, net_w, best_gain_w."""
from __future__ import annotations
import json, math
from functools import lru_cache
from ..core.schema import Plant, Site
from ..core.solve import solve
from ..verify.grounding import check_numbers
from .spec import parse_audit, AuditError
from .truth import change_effect


def completion_text(c) -> str:
    return c if isinstance(c, str) else c[-1]["content"]


def ndcg(pred: list[str], ed: dict[str, float], k: int = 3) -> float:
    rel = lambda i: max(ed.get(i, 0.0), 0.0)
    dcg = sum(rel(c) / math.log2(i + 2) for i, c in enumerate(pred[:k]))
    ideal = sum(v / math.log2(i + 2) for i, v in enumerate(sorted((max(v, 0.0) for v in ed.values()), reverse=True)[:k]))
    return dcg / ideal if ideal > 0 else 0.0


@lru_cache(maxsize=8192)
def evaluate(text: str, plant_js: str, site_js: str, ed_js: str, fuel_w: float, net_w: float) -> dict:
    out = {"parsed": False, "any": any(t in text for t in ("<ranking>", "<change>", "<prediction>")), "ndcg": 0.0,
           "change_ok": False, "applied": False, "delta_w": None, "pred_err": None, "grounded": 0.0, "reason": ""}
    plant = Plant.model_validate_json(plant_js); site = Site.model_validate_json(site_js) if site_js else None
    ed = json.loads(ed_js)
    try: a = parse_audit(text, set(ed)); out["parsed"] = True
    except AuditError as e: out["reason"] = str(e); return out
    pm = {c.id for c in plant.components if c.type in ("combustor", "gas_engine")}     # always the largest destroyer: not scored
    out["ndcg"] = ndcg([x for x in a["ranking"] if x not in pm], {k: v for k, v in ed.items() if k not in pm})
    if a["analysis"]:
        res = solve(plant); out["grounded"] = check_numbers(a["analysis"], plant, res)["fraction"]
    e = change_effect(plant, site, *a["change"], net_w, fuel_w)
    out["applied"] = e["reason"] != "change not allowed"; out["change_ok"] = e["valid"]; out["reason"] = e["reason"]
    if e["valid"]:
        out["delta_w"] = e["delta_w"]; out["pred_err"] = abs(a["prediction_mw"] * 1e6 - e["delta_w"]) / max(abs(e["delta_w"]), 0.002 * fuel_w)     # relative to the effect, floored at 0.2 % of fuel
    return out


def _each(completions, kw, fn):
    n = len(completions); site = kw.get("site") or [""] * n
    return [fn(evaluate(completion_text(c), kw["plant"][i], site[i], kw["ed_json"][i], float(kw["fuel_w"][i]), float(kw["net_w"][i])), i, kw)
            for i, c in enumerate(completions)]


def a_format(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: 1.0 if e["parsed"] else (0.3 if e["any"] else 0.0))

def a_rank(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: e["ndcg"])

def a_valid(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: 1.0 if e["change_ok"] else (0.2 if e["applied"] else 0.0))

def a_gain(prompts, completions, **kw):
    """1.0 when the change saves as much fuel as the best allowed change; 0 when it saves nothing or costs fuel."""
    def f(e, i, k):
        if not e["change_ok"]: return 0.0
        return max(0.0, min(1.05, -e["delta_w"] / float(k["best_gain_w"][i])))
    return _each(completions, kw, f)

def a_pred(prompts, completions, **kw):
    """1 - |predicted - actual| / max(|actual|, 0.2 % of fuel): predicting no effect scores 0 on any real effect."""
    return _each(completions, kw, lambda e, i, k: max(0.0, 1.0 - e["pred_err"]) if e["pred_err"] is not None else 0.0)

def a_grounded(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: e["grounded"] if e["parsed"] else 0.0)


REWARD_FUNCS = [a_format, a_rank, a_valid, a_gain, a_pred, a_grounded]
REWARD_WEIGHTS = [0.5, 1.5, 1.0, 2.0, 1.0, 0.5]
