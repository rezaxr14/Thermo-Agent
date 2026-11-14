"""GRPO rewards for ledger repair. Dataset column: `ledger` (JSON string: data, broken lines, faulty keys, numbers shown)."""
from __future__ import annotations
import json, re
from functools import lru_cache
from . import program as P
from .expr import parse, ExprError

_FIX = re.compile(r"^FIX\s+(T0|E\[[A-Za-z0-9_]+\]|ED\[[A-Za-z0-9_]+\])\s*=\s*(.+?)\s*$")
_MW = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*MW(?![\w/])")
MAX_FIXES = 4


def completion_text(c) -> str:
    return c if isinstance(c, str) else c[-1]["content"]


def parse_fixes(text: str, known: set[str]) -> dict[str, str]:
    m = re.search(r"<fix>(.*?)</fix>", text, flags=re.S)
    if not m: raise ValueError("no <fix> block")
    out = {}
    for line in [l.strip() for l in m.group(1).strip().splitlines() if l.strip()]:
        mm = _FIX.match(line)
        if not mm: raise ValueError(f"bad line {line[:40]!r}")
        key, expr = mm.groups()
        if key not in known: raise ValueError(f"no such line {key}")
        if key in out: raise ValueError(f"duplicate fix for {key}")
        parse(expr)
        out[key] = expr
    if not out or len(out) > MAX_FIXES: raise ValueError("need 1 to 4 fixes")
    return out


@lru_cache(maxsize=8192)
def evaluate(text: str, ledger_js: str) -> dict:
    L = json.loads(ledger_js); data, broken, faulty = L["data"], L["lines"], L["faulty"]
    out = {"parsed": False, "any": "<fix>" in text, "closure": 0.0, "nonneg": 0.0, "exact": 0.0, "n_fixes": 0, "grounded": 0.0, "has_diag": False}
    diag = re.search(r"<diagnosis>(.*?)</diagnosis>", text, flags=re.S); out["has_diag"] = diag is not None
    if diag:
        nums = L["numbers"]; found = [float(x) for x in _MW.findall(diag.group(1))]
        ok = [any(abs(f - n) <= 0.005 * abs(n) + 1e-9 for n in nums) for f in found]
        out["grounded"] = 1.0 if not found else sum(ok) / len(found)
    try: fixes = parse_fixes(text, set(broken)); out["parsed"] = True; out["n_fixes"] = len(fixes)
    except (ValueError, ExprError): return out
    lines = {**broken, **fixes}; r = P.run(lines, data); ef = max(data["e_fuel_mw"], 1e-6)
    if r["closure"] is not None: out["closure"] = 1.0 if abs(r["closure"]) <= 1e-6 * ef else max(0.0, 1.0 - abs(r["closure"]) / (0.01 * ef))
    cids = list(data["topo"]); bad = [c for c in cids if f"ED[{c}]" in r["errors"] or r["ED"].get(c, 0.0) < -1e-5 * ef]
    out["nonneg"] = 1.0 - len(bad) / len(cids)
    fixed = sum(P.matches_truth(r, data, k) for k in faulty)
    broken_now = sum(1 for k in broken if k not in faulty and not P.matches_truth(r, data, k))
    out["exact"] = max(0.0, min(1.0, (fixed - broken_now) / len(faulty)))
    out["all_correct"] = fixed == len(faulty) and broken_now == 0
    return out


def _each(completions, kw, fn):
    return [fn(evaluate(completion_text(c), kw["ledger"][i]), i, kw) for i, c in enumerate(completions)]

def l_format(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: 1.0 if (e["parsed"] and e["has_diag"]) else (0.3 if e["any"] else 0.0))

def l_closure(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: e["closure"])

def l_nonneg(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: e["nonneg"] if e["parsed"] else 0.0)

def l_exact(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: e["exact"])

def l_minimal(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: 1.0 / (1 + max(0, e["n_fixes"] - len(json.loads(k["ledger"][i])["faulty"]))) if e.get("all_correct") else 0.0)

def l_grounded(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: e["grounded"] if e["has_diag"] else 0.0)


REWARD_FUNCS = [l_format, l_closure, l_nonneg, l_exact, l_minimal, l_grounded]
REWARD_WEIGHTS = [0.5, 1.0, 0.5, 2.5, 0.5, 0.5]
