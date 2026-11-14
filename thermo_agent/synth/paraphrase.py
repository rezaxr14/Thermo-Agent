"""Deterministic paraphrase variants of the expert diagnosis.

Every variant is rendered from the same ground-truth facts (the injected fault and its inverse patch), so the
wording varies but the content cannot drift. A repair row picks a variant by a hash of its id, which keeps the
dataset reproducible while removing the one-template-per-fault-type memorisation shortcut: the model sees 20 to
80 phrasings of each fault class across the corpus, all of them true by construction.

    from thermo_agent.synth import paraphrase
    paraphrase.variant_count(row)          # how many renderings this fault class has
    paraphrase.to_row(row, variant=7)      # copy of the row with a different, equally true completion
"""
from __future__ import annotations

import hashlib
import itertools
import re

from ..agent.dsl import parse_edits
from ..core.schema import Plant

_EDITS = re.compile(r"<edits>.*?</edits>", re.S)

FLUIDS = ("Air", "FLUE", "CH4", "Water", "R245fa", "Cyclopentane", "Toluene")
GAS = frozenset({"Air", "FLUE", "CH4"})


def _edits_block(text: str) -> str:
    m = _EDITS.search(text)
    if not m:
        raise ValueError("row completion has no <edits> block")
    return m.group(0)


def _broken_value(plant: Plant, e: dict):
    if e["kind"] == "fluid":
        return next(s.fluid for s in plant.streams if s.id == e["id"])
    if e["kind"] == "rewire":
        s = next(s for s in plant.streams if s.id == e["id"])
        return getattr(s, e["end"])
    if e["stream"]:
        s = next(s for s in plant.streams if s.id == e["id"])
        if e["field"] == "m_dot":
            return s.m_dot
        return getattr(s.fixed, e["field"].split(".", 1)[1])
    return next(c for c in plant.components if c.id == e["id"]).params[e["field"]]


def facts(row: dict) -> dict:
    """Ground-truth slots for the paraphrase, read from the broken plant and the expert edit block."""
    plant = Plant.model_validate_json(row["plant_broken"])
    edits = parse_edits(row["completion"][0]["content"])
    ft = row["fault_type"]
    if ft == "wrong_fluid":
        e = next(x for x in edits if x["kind"] == "fluid")
        stream = next(s for s in plant.streams if s.id == e["id"])
        return {"sid": e["id"], "after": stream.fluid, "before": e["value"], "port": stream.dst}
    if ft == "swapped_ports":
        ids = [x["id"] for x in edits if x["kind"] == "rewire"]
        return {"a": ids[0], "b": ids[1], "comp": edits[0]["value"].split(".", 1)[0]}
    e = edits[0]
    name = f"@{e['id']}.{e['field']}" if e["stream"] else f"{e['id']}.{e['field']}"
    return {"name": name, "key": e["field"], "after": _broken_value(plant, e), "before": e["value"]}


def _slots(fault_type: str, f: dict, deducible: bool = True) -> list[list[str]]:
    if fault_type == "wrong_fluid":
        return [
            ["Stream {sid} is labelled {after}", "Stream {sid} carries {after}",
             "The stream {sid} is declared as {after}", "{sid} is tagged {after}",
             "The fluid label on stream {sid} is {after}"],
            [", but it feeds {port}, which expects a stream of the other fluid class",
             ", yet it enters {port}, which requires the other fluid class",
             ", while its destination {port} only accepts the other fluid class",
             "; the component it supplies, {port}, expects the other fluid class"],
            ["; its neighbours in the same loop carry {before}.",
             ". Every other stream in that loop carries {before}.",
             ", and the rest of the loop carries {before}.",
             ". The loop's other streams are {before}."],
        ]
    if fault_type == "swapped_ports":
        return [
            ["Streams {a} and {b} enter {comp} through each other's ports",
             "{a} and {b} are connected to the wrong ports of {comp}",
             "The two inlets of {comp} are exchanged, so {a} takes the port of {b} and {b} takes that of {a}",
             "{comp} receives {a} and {b} on swapped ports"],
            ["; the inlet conditions at {comp} do not match the roles of those ports, so the two connections are exchanged.",
             "; the solver shows inlet states at {comp} that belong to the other port, so the two connections must be swapped back.",
             "; the stream entering each port has the state the other port expects, which identifies the exchange.",
             "; matching the solver states to the port roles shows the two streams are interchanged.",
             ". Reconnecting each stream to its own port restores the design states."],
        ]
    if fault_type == "unit_slip":
        key = f["key"]
        if key in ("P_it", "P_el"):
            subjects = ["{name} = {after:g} W is not a credible electrical power",
                        "{name} reads {after:g} W, far from the plant's other power values",
                        "The value {name} = {after:g} W is off by orders of magnitude",
                        "{name} is given as {after:g} W, which no unit in this plant produces",
                        "At {name} the value {after:g} W is written where a plant-scale power belongs"]
            tails = ["; it is off by a power-of-ten factor, a W/kW/MW unit slip.",
                     ", so the unit is a slip between W, kW and MW.",
                     "; the digits are right but the power of ten is not, a W/kW/MW unit slip.",
                     ", consistent with a value written in kW or MW but labelled W."]
        elif key.startswith("P"):
            subjects = ["{name} = {after:g} Pa is far from the loop's other pressures",
                        "{name} reads {after:g} Pa, which is not a plant-scale pressure",
                        "The value {name} = {after:g} Pa is a bar/Pa or kPa/Pa unit slip",
                        "{name} is given as {after:g} Pa, orders of magnitude off the loop",
                        "At {name} the pressure {after:g} Pa belongs to a different unit system"]
            tails = ["; the loop's pressures are of order 1e5 Pa, so this is a unit slip.",
                     ", so a bar or kPa value was written in Pa.",
                     "; the digits are right but the scale is not, a pressure unit slip.",
                     ", consistent with a value written in bar or kPa but labelled Pa."]
        else:
            subjects = ["{name} = {after:g} K is far below the plant's other temperatures",
                        "{name} reads {after:g} K, colder than any stream in the loop",
                        "The value {name} = {after:g} K looks like a degC value entered where K is required",
                        "{name} is given as {after:g} K, a Celsius number in a kelvin field",
                        "At {name} the temperature {after:g} K is off by roughly 273 K"]
            tails = ["; adding 273.15 gives the temperature the loop expects.",
                     ", so the value is in degC but the field is in K.",
                     "; the offset of about 273 K is the giveaway.",
                     ", consistent with a Celsius value entered in kelvin."]
        return [subjects, tails]
    if fault_type == "sign_flip":
        return [
            ["{name} = {after:g} is negative", "The value {name} = {after:g} carries a minus sign",
             "{name} is stored as {after:g}, below zero", "A negative number appears at {name} ({after:g})",
             "{name} = {after:g} has the wrong sign"],
            ["; this quantity must be non-negative, so the sign was flipped.",
             "; no physical value of this quantity is negative, so the sign is inverted.",
             "; the magnitude is plausible but the sign is not, so it must be flipped.",
             "; this quantity is a magnitude, and a magnitude cannot be negative.",
             "; the minus sign is the error, not the number."],
        ]
    if fault_type == "efficiency_gt_1":
        if deducible:
            subjects = ["{name} = {after:g} is a percentage entered where a fraction is required",
                        "{name} reads {after:g}, so a 0-1 fraction was written in percent",
                        "The value {name} = {after:g} is a percentage in a fraction field",
                        "{name} is given as {after:g}, which is the percentage form of a fraction",
                        "At {name} the number {after:g} looks like percent rather than a fraction"]
            tails = ["; the same efficiency written as a fraction is below 1.",
                     ", and efficiencies must lie in (0, 1].",
                     "; dividing by 100 gives the value the field expects.",
                     ", so the field is a fraction, not a percentage.",
                     "; the percentage sign was dropped but the scale was not."]
        else:
            subjects = ["{name} = {after:g} exceeds 1, but efficiencies and second-law fractions must lie in (0, 1]",
                        "{name} reads {after:g}, above the physical bound of 1",
                        "The value {name} = {after:g} is impossible for an efficiency or second-law fraction",
                        "{name} is given as {after:g}, which cannot be an efficiency",
                        "At {name} the number {after:g} breaks the (0, 1] bound"]
            tails = ["; no efficiency can exceed unity.",
                     ", so the value is physically impossible as written.",
                     "; the bound is 1, and the value is above it.",
                     ", which would create energy from nothing.",
                     "; a second-law fraction above 1 contradicts the entropy balance."]
        return [subjects, tails]
    raise KeyError(f"no paraphrase templates for fault type {fault_type!r}")


def variants(row: dict) -> list[str]:
    """All diagnosis sentences (without tags) for this row, in a fixed order."""
    f = facts(row)
    return ["".join(parts).format(**f) for parts in itertools.product(*_slots(row["fault_type"], f, row.get("deducible", True)))]


def variant_count(row: dict) -> int:
    return len(variants(row))


def variant_index(row_id: str, n: int, extra: int = 0) -> int:
    """Deterministic index into the variant list; different rows get different phrasings."""
    if n < 1:
        raise ValueError("n must be >= 1")
    h = int(hashlib.md5(f"{row_id}:{extra}".encode()).hexdigest(), 16)
    return h % n


def completion(row: dict, variant: int = 0) -> str:
    vs = variants(row)
    why = vs[variant % len(vs)]
    return f"<diagnosis>{why}</diagnosis>\n{_edits_block(row['completion'][0]['content'])}"


def to_row(row: dict, variant: int = 0) -> dict:
    """Copy of the row with the diagnosis paraphrased; the edit block and every label column are unchanged."""
    n = variant_count(row)
    out = dict(row)
    out["completion"] = [{"role": "assistant", "content": completion(row, variant)}]
    out["paraphrase"] = {"variant": variant % n, "n": n}
    return out
