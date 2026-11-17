"""Expert completions (SFT targets) and dataset rows for Repair. The expert writes number-grounded diagnoses from the
known fault; values to restore come from the fault record."""
from __future__ import annotations
import hashlib
import json
from ..core.schema import RepairRecord
from ..agent.dsl import patch_to_edits_text
from ..train.prompts import repair_messages
from ..verify.verifiers import run_all
from ..core.solve import solve
from .splits import is_ood_topology, topology_label


def _target(path: str, plant) -> tuple[str, str]:
    parts = path.strip("/").split("/")
    if parts[0] == "components":
        c = plant.components[int(parts[1])]; return f"{c.id}.{parts[3]}", parts[3]
    s = plant.streams[int(parts[1])]; return f"@{s.id}.{'.'.join(parts[2:])}", parts[-1]


def _dst(plant, sid):
    return next(s.dst for s in plant.streams if s.id == sid)


def expert_completion(rec: RepairRecord) -> str:
    f = rec.faults[0]
    if f.type == "wrong_fluid":
        sid = f.component; port = _dst(rec.plant_broken, sid)
        why = (f"Stream {sid} is labelled {f.after}, but it feeds {port}, which expects a stream of the other fluid class; "
               f"its neighbours in the same loop carry {f.before}.")
        return f"<diagnosis>{why}</diagnosis>\n{patch_to_edits_text(rec.plant_broken, f.inverse_patch)}"
    if f.type == "swapped_ports":
        a, b = (rec.plant_broken.streams[int(op['path'].split('/')[2])].id for op in f.inverse_patch)
        why = (f"Streams {a} and {b} enter {f.component} through each other's ports; the solver output shows the inlet "
               f"conditions at {f.component} do not match the roles of those ports, so the two connections are exchanged.")
        return f"<diagnosis>{why}</diagnosis>\n{patch_to_edits_text(rec.plant_broken, f.inverse_patch)}"
    name, key = _target(f.path, rec.plant_broken); a = f.after
    if f.type == "unit_slip":
        if key in ("P_it", "P_el"): why = f"{name} = {a:g} W is not a credible electrical power; it is off by a power-of-ten factor, a W/kW/MW unit slip."
        elif key.startswith("P") or key == "P": why = f"{name} = {a:g} Pa is far from the loop's other pressures, a bar/Pa or kPa/Pa unit slip."
        else: why = f"{name} = {a:g} K is far below the plant's other temperatures; it looks like a degC value entered where K is required."
    elif f.type == "sign_flip":
        why = f"{name} = {a:g} is negative; this quantity must be non-negative, so the sign was flipped."
    else:
        why = (f"{name} = {a:g} is a percentage entered where a fraction is required." if f.deducible
               else f"{name} = {a:g} exceeds 1, but efficiencies and second-law fractions must lie in (0, 1].")
    return f"<diagnosis>{why}</diagnosis>\n{patch_to_edits_text(rec.plant_broken, f.inverse_patch)}"


def topology_of(rec: RepairRecord) -> str:
    return topology_label(rec.plant_original)


def assign_split(rec: RepairRecord) -> str:
    """Splits are by structure, never random: unseen diagram combination, unseen fault type, then in-distribution."""
    topo, ft = topology_of(rec), rec.faults[0].type
    if is_ood_topology(topo): return "test_ood_topology"      # steam/HRSG export plant, engine with ideal rejection
    if ft in ("sign_flip", "swapped_ports"): return "test_ood_fault"
    h = int(hashlib.md5(rec.id.encode()).hexdigest(), 16) % 10
    return "val" if h == 0 else ("test_id" if h == 1 else "train")


def to_row(rec: RepairRecord) -> dict:
    rep = run_all(rec.plant_broken, rec.solve_broken, rec.site)
    return {"id": rec.id, "split": assign_split(rec), "topology": topology_of(rec), "fault_type": rec.faults[0].type,
            "prompt": repair_messages(rec.plant_broken, rec.site, rec.solve_broken, rep),
            "completion": [{"role": "assistant", "content": expert_completion(rec)}],
            "plant_broken": rec.plant_broken.model_dump_json(by_alias=True), "plant_original": rec.plant_original.model_dump_json(by_alias=True),
            "site": rec.site.model_dump_json() if rec.site else "", "deducible": bool(rec.faults[0].deducible),
            "min_edits": sum(len(f.inverse_patch) for f in rec.faults), "detected": bool(rec.verifier_report_broken["detected"])}
