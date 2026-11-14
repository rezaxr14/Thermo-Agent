"""Fault injectors. Each returns the broken plant (as a validated Plant) and a Fault with the exact inverse patch."""
from __future__ import annotations
import copy, random
from ..core.schema import Plant, Fault

Num = (int, float)


class NotApplicable(Exception):
    """The plant has no site where this fault type can be injected."""


def _dump(p: Plant) -> dict:
    return p.model_dump(by_alias=True, mode="json")


def _get(d, ptr):
    for k in ptr.strip("/").split("/"):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def _set(d, ptr, v):
    keys = ptr.strip("/").split("/"); parent = _get(d, "/" + "/".join(keys[:-1])) if len(keys) > 1 else d
    k = keys[-1]
    if isinstance(parent, list): parent[int(k)] = v
    else: parent[k] = v


def apply_patch(plant_dict: dict, patch: list[dict]) -> dict:
    d = copy.deepcopy(plant_dict)
    for op in patch:
        if op.get("op") != "replace": raise ValueError("only 'replace' operations are supported")
        _set(d, op["path"], op["value"])
    return d


def _targets(d: dict, keys: tuple[str, ...], prefix: bool = False):
    out = []
    for i, c in enumerate(d["components"]):
        for k, v in c["params"].items():
            if isinstance(v, Num) and not isinstance(v, bool) and (k.startswith(keys) if prefix else k in keys):
                out.append((f"/components/{i}/params/{k}", c["id"], v))
    return out


def _stream_targets(d, field_name):
    return [(f"/streams/{i}/fixed/{field_name}", s["id"], s["fixed"][field_name]) for i, s in enumerate(d["streams"])
            if s.get("fixed") and s["fixed"].get(field_name) is not None and s["from"].startswith("ext:")]


def _finish(plant, d, path, comp, before, after, ftype, deducible=True):
    d2 = copy.deepcopy(d); _set(d2, path, after)
    return Plant.model_validate(d2), Fault(type=ftype, component=comp, path=path, before=before, after=after,
                                          inverse_patch=[{"op": "replace", "path": path, "value": before}], deducible=deducible)


def unit_slip(plant: Plant, rng: random.Random):
    d = _dump(plant)
    pool = (_targets(d, ("P_out", "P_steam", "P_it", "P_el")) + _stream_targets(d, "P")
            + _targets(d, ("T_out", "T_cold_out", "T_chw_out", "T_gen_out", "T_return", "T_hw")))
    path, comp, v = rng.choice(pool)
    key = path.rsplit("/", 1)[-1]
    if key in ("P_it", "P_el"):
        after = v * rng.choice([1e-6, 1e-3, 1e3])        # W entered as MW / kW
    elif key.startswith("P") or path.endswith("/fixed/P"):
        after = v * rng.choice([1e-5, 1e5, 1e-3])        # bar<->Pa, kPa<->Pa slips
    else:
        after = v - 273.15                                # degC entered where K expected
    return _finish(plant, d, path, comp, v, after, "unit_slip")


def sign_flip(plant: Plant, rng: random.Random):
    d = _dump(plant)
    pool = [t for t in _targets(d, ("dp", "dT", "pinch", "approach"), prefix=True) if t[2] != 0]
    path, comp, v = rng.choice(pool)
    return _finish(plant, d, path, comp, v, -v, "sign_flip")


def efficiency_gt_1(plant: Plant, rng: random.Random):
    d = _dump(plant)
    path, comp, v = rng.choice(_targets(d, ("eta", "eps"), prefix=True))
    percent = rng.random() < 0.5
    after = round(v * 100, 6) if percent else round(1 + rng.uniform(0.02, 0.3), 4)   # percent entered as fraction, or >1
    return _finish(plant, d, path, comp, v, after, "efficiency_gt_1", deducible=percent)


def swapped_ports(plant: Plant, rng: random.Random):
    """Two streams of the same fluid entering the same component are connected to each other's ports."""
    d = _dump(plant); pairs = []
    for i, a in enumerate(d["streams"]):
        for j, b in enumerate(d["streams"]):
            if i < j and not a["to"].startswith("ext:") and not b["to"].startswith("ext:") \
                    and a["to"].split(".")[0] == b["to"].split(".")[0] and a["to"] != b["to"] and a["fluid"] == b["fluid"]:
                pairs.append((i, j))
    if not pairs: raise NotApplicable("no component has two same-fluid inlets")
    i, j = rng.choice(pairs); ta, tb = d["streams"][i]["to"], d["streams"][j]["to"]
    d2 = copy.deepcopy(d); d2["streams"][i]["to"], d2["streams"][j]["to"] = tb, ta
    inv = [{"op": "replace", "path": f"/streams/{i}/to", "value": ta}, {"op": "replace", "path": f"/streams/{j}/to", "value": tb}]
    return Plant.model_validate(d2), Fault(type="swapped_ports", component=ta.split(".")[0], path=f"/streams/{i}/to", before=ta, after=tb,
                                           inverse_patch=inv, deducible=True)


def wrong_fluid(plant: Plant, rng: random.Random):
    """A stream is relabelled with a fluid of the other class (gas <-> liquid)."""
    d = _dump(plant); gas = {"CGAM_AIR", "FLUE", "CH4"}
    i = rng.choice([k for k, s in enumerate(d["streams"]) if s["fluid"] in gas | {"Water"}])
    before = d["streams"][i]["fluid"]; after = "Water" if before in gas else rng.choice(["FLUE", "CGAM_AIR"])
    return _finish(plant, d, f"/streams/{i}/fluid", d["streams"][i]["id"], before, after, "wrong_fluid")


INJECTORS = {"unit_slip": unit_slip, "sign_flip": sign_flip, "efficiency_gt_1": efficiency_gt_1,
             "swapped_ports": swapped_ports, "wrong_fluid": wrong_fluid}
