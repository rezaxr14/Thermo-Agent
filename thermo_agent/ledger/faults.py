"""Equation-level faults in a ledger. Each returns the broken line and a record {type, key, correct, buggy}."""
from __future__ import annotations
import random
from . import program as P

STREAM_FAULTS = ("missing_m", "missing_ech", "unit_slip", "sign_T0", "wrong_ref", "t0_literal")
COMPONENT_FAULTS = ("sign_flip", "dropped_term", "dropped_work", "unit_slip", "wrong_factor", "wrong_stream")
CONST_FAULTS = ("t0_celsius", "t0_offset")
HELD_OUT = {"wrong_stream", "dropped_work", "wrong_ref"}          # fault types never seen in training data


def _stream_fault(rng, key, data, ftype):
    x = key[2:-1]; ids = data["stream_order"]
    if ftype == "missing_m": return "(h[{x}] - T0*s[{x}] - g0[{x}] + ech[{x}])/1e6".format(x=x), {"x": x}
    if ftype == "missing_ech":
        if data["tables"]["ech"][x] == 0: return None
        return "m[{x}]*(h[{x}] - T0*s[{x}] - g0[{x}])/1e6".format(x=x), {"x": x}
    if ftype == "unit_slip": return "m[{x}]*(h[{x}] - T0*s[{x}] - g0[{x}] + ech[{x}])/1e3".format(x=x), {"x": x}
    if ftype == "sign_T0": return "m[{x}]*(h[{x}] + T0*s[{x}] - g0[{x}] + ech[{x}])/1e6".format(x=x), {"x": x}
    if ftype == "t0_literal": return "m[{x}]*(h[{x}] - 25*s[{x}] - g0[{x}] + ech[{x}])/1e6".format(x=x), {"x": x}
    if ftype == "wrong_ref":
        y = rng.choice([i for i in ids if i != x]); return "m[{x}]*(h[{x}] - T0*s[{x}] - g0[{y}] + ech[{x}])/1e6".format(x=x, y=y), {"x": x, "y": y}


def _component_fault(rng, key, data, ftype):
    cid = key[3:-1]; terms = P.component_terms(data["topo"][cid], cid in data["tables"]["W"], cid)
    e_idx = [i for i, t in enumerate(terms) if t[2] == "E"]; w_idx = [i for i, t in enumerate(terms) if t[2] == "W"]
    t = list(terms); detail = {}
    if ftype == "sign_flip":
        i = rng.randrange(len(t)); detail["ident"] = t[i][3]; t[i] = (-t[i][0],) + t[i][1:]
    elif ftype == "dropped_term":
        if len(e_idx) < 2: return None
        i = rng.choice(e_idx); detail["ident"] = t[i][3]; del t[i]
    elif ftype == "dropped_work":
        if not w_idx: return None
        del t[w_idx[0]]
    elif ftype in ("unit_slip", "wrong_factor"):
        i = rng.randrange(len(t)); fac = rng.choice([1000.0, 0.001] if ftype == "unit_slip" else [0.1, 10.0, 2.0, 0.5])
        detail.update(ident=t[i][3], factor=fac); t[i] = (t[i][0], fac) + t[i][2:]
    elif ftype == "wrong_stream":
        i = rng.choice(e_idx); others = [s for s in data["stream_order"] if s not in {x[3] for x in terms}]
        if not others: return None
        detail["right"] = t[i][3]; detail["wrong"] = rng.choice(others); t[i] = t[i][:3] + (detail["wrong"],)
    return P._render(t), detail


def _const_fault(rng, key, data, ftype):
    T0 = data["consts_true"]["T0"]
    return (repr(T0 - 273.15) if ftype == "t0_celsius" else "273.15"), {}


def inject(rng: random.Random, data: dict, n_faults: int = 1, allowed: set | None = None, tries: int = 60):
    """Break `n_faults` distinct lines so that each broken line really changes a value. Returns (lines, records) or None."""
    lines = dict(data["lines"]); recs = []; used = set()
    while len(recs) < n_faults and tries > 0:
        tries -= 1
        layer = rng.choices(["stream", "component", "const"], [0.4, 0.5, 0.1])[0]
        if layer == "stream": key = f"E[{rng.choice(data['stream_order'])}]"; ftype = rng.choice(STREAM_FAULTS); fn = _stream_fault
        elif layer == "component": key = f"ED[{rng.choice(list(data['topo']))}]"; ftype = rng.choice(COMPONENT_FAULTS); fn = _component_fault
        else: key = "T0"; ftype = rng.choice(CONST_FAULTS); fn = _const_fault
        if key in used or (allowed is not None and ftype not in allowed): continue
        made = fn(rng, key, data, ftype)
        if made is None or made[0] == data["lines"][key]: continue
        buggy, detail = made
        trial = {**lines, key: buggy}; r = P.run(trial, data)
        if P.matches_truth(r, data, key): continue                                               # changed nothing observable
        if r["errors"]: continue                                                                 # must still evaluate
        lines[key] = buggy; used.add(key); recs.append({"type": ftype, "key": key, "correct": data["lines"][key], "buggy": buggy, "detail": detail})
    return (lines, recs) if len(recs) == n_faults else None
