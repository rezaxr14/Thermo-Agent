"""Repair baselines that do not use a language model. Cost is counted in solver calls."""
from __future__ import annotations
import re
from ..core.schema import Plant, Site
from ..core.solve import solve
from ..verify.verifiers import run_all
from ..synth.faults import apply_patch, _dump

TRANSFORMS = [lambda v: v * 1e5, lambda v: v * 1e-5, lambda v: v * 1e3, lambda v: v * 1e-3, lambda v: v * 1e6, lambda v: v * 1e-6,
              lambda v: v + 273.15, lambda v: -v, lambda v: v / 100.0]
EFF_DEFAULTS = (0.85, 0.75, 0.5)


def _numeric_slots(d: dict):
    for i, c in enumerate(d["components"]):
        for k, v in c["params"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool): yield f"/components/{i}/params/{k}", c["id"], k, v
    for i, s in enumerate(d["streams"]):
        if s.get("m_dot") is not None: yield f"/streams/{i}/m_dot", s["id"], "m_dot", s["m_dot"]
        for k, v in (s.get("fixed") or {}).items():
            if v is not None: yield f"/streams/{i}/fixed/{k}", s["id"], k, v


def candidate_patches(plant: Plant, hint: str = ""):
    """Single-edit candidates, slots mentioned in the verifier output first."""
    d = plant.model_dump(by_alias=True, mode="json")
    slots = list(_numeric_slots(d))
    slots.sort(key=lambda t: 0 if (t[1] in hint and t[2] in hint) else (1 if t[1] in hint else 2))
    for path, cid, key, v in slots:
        seen = {v}
        vals = [t(v) for t in TRANSFORMS] + (list(EFF_DEFAULTS) if key.startswith(("eta", "eps")) else [])
        for nv in vals:
            if nv not in seen and nv == nv:
                seen.add(nv); yield [{"op": "replace", "path": path, "value": float(nv)}]


def structural_candidates(plant: Plant):
    """Single structural edits: exchange the destinations of two same-fluid streams entering one component, or relabel a stream's fluid."""
    d = plant.model_dump(by_alias=True, mode="json"); st = d["streams"]
    for i, a in enumerate(st):
        for j, b in enumerate(st):
            if i < j and not a["to"].startswith("ext:") and not b["to"].startswith("ext:") and a["to"].split(".")[0] == b["to"].split(".")[0] \
                    and a["to"] != b["to"] and a["fluid"] == b["fluid"]:
                yield [{"op": "replace", "path": f"/streams/{i}/to", "value": b["to"]}, {"op": "replace", "path": f"/streams/{j}/to", "value": a["to"]}]
    for i, a in enumerate(st):
        for f in ("Water", "FLUE", "CGAM_AIR", "CH4", "R245fa"):
            if f != a["fluid"]: yield [{"op": "replace", "path": f"/streams/{i}/fluid", "value": f}]


def search_repair(broken: Plant, site: Site | None, hint: str = "", max_solves: int = 400) -> tuple[list[dict] | None, int]:
    base = _dump(broken); n = 0
    import itertools
    for patch in itertools.chain(candidate_patches(broken, hint), structural_candidates(broken)):
        if n >= max_solves: break
        try: cand = Plant.model_validate(apply_patch(base, patch))
        except Exception: continue
        n += 1
        res = solve(cand)
        if res.ok and run_all(cand, res, site).valid: return patch, n
    return None, n
