"""Token-efficient text forms: the plant the model reads, and the edit language it writes.

Plant text (one line per item, JSON only for the numeric payload):
    PLANT <name> T0=<K> P0=<Pa> RH=<->
    C <id> <type> {"PR":18.0,"eta_s":0.86}
    S <id> <from> -> <to> <fluid> {"m_dot":..., "fixed":{"T":...,"P":...}}
Edit language (what the model answers with, inside <edits>...</edits>):
    SET <component>.<param> = <number>
    SET @<stream>.<field> = <number>          field in m_dot, fixed.T, fixed.P, fixed.x, fixed.h
    REWIRE @<stream>.<to|from> = <component>.<port>     (reconnect one end of an existing stream)
    FLUID @<stream> = <fluid>                           (change the fluid carried by an existing stream)
Edits can only change numbers, endpoints and fluids that already exist; they cannot add or remove components or streams.
"""
from __future__ import annotations
import json, re
from ..core.schema import Plant, Component, Stream, FixedState, Ambient

MAX_EDITS = 6
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_SET = re.compile(rf"^SET\s+(@?)([A-Za-z0-9_]+)\.([A-Za-z0-9_.]+)\s*=\s*({_NUM})\s*$")
_REWIRE = re.compile(r"^REWIRE\s+@([A-Za-z0-9_]+)\.(to|from)\s*=\s*([A-Za-z0-9_]+\.[A-Za-z0-9_]+|ext:[A-Za-z0-9_]+)\s*$")
_FLUID = re.compile(r"^FLUID\s+@([A-Za-z0-9_]+)\s*=\s*([A-Za-z0-9_\-]+)\s*$")


def _j(x) -> str:
    return json.dumps(x, separators=(",", ":"))


def plant_to_text(p: Plant) -> str:
    a = p.ambient
    out = [f"PLANT {p.name} T0={a.T0!r} P0={a.P0!r} RH={a.RH!r}"]
    for c in p.components:
        out.append(f"C {c.id} {c.type}" + (f" {_j(c.params)}" if c.params else ""))
    for s in p.streams:
        extra = {}
        if s.m_dot is not None: extra["m_dot"] = s.m_dot
        if s.fixed is not None: extra["fixed"] = {k: v for k, v in s.fixed.model_dump().items() if v is not None}
        if s.composition: extra["composition"] = s.composition
        out.append(f"S {s.id} {s.src} -> {s.dst} {s.fluid}" + (f" {_j(extra)}" if extra else ""))
    return "\n".join(out)


def text_to_plant(t: str) -> Plant:
    comps, streams, name, amb = [], [], "plant", Ambient()
    for line in t.strip().splitlines():
        head, _, rest = line.partition(" ")
        if head == "PLANT":
            toks = rest.split(); name = toks[0]
            kv = dict(x.split("=") for x in toks[1:])
            amb = Ambient(T0=float(kv["T0"]), P0=float(kv["P0"]), RH=float(kv["RH"]))
        elif head == "C":
            body, _, js = rest.partition("{")
            cid, ctype = body.split()
            comps.append(Component(id=cid, type=ctype, params=json.loads("{" + js) if js else {}))
        elif head == "S":
            body, _, js = rest.partition("{")
            sid, src, arrow, dst, fluid = body.split()
            assert arrow == "->"
            extra = json.loads("{" + js) if js else {}
            streams.append(Stream(id=sid, **{"from": src, "to": dst}, fluid=fluid, m_dot=extra.get("m_dot"),
                                  fixed=FixedState(**extra["fixed"]) if "fixed" in extra else None,
                                  composition=extra.get("composition")))
    return Plant(name=name, ambient=amb, components=comps, streams=streams)


class EditError(ValueError):
    pass


def parse_edits(text: str) -> list[dict]:
    """-> list of {"kind": "set"|"rewire"|"fluid", ...}. Raises EditError on any malformed or excessive edit."""
    m = re.search(r"<edits>(.*?)</edits>", text, flags=re.S)
    if not m: raise EditError("no <edits> block")
    lines = [l.strip() for l in m.group(1).strip().splitlines() if l.strip()]
    if len(lines) > MAX_EDITS: raise EditError(f"more than {MAX_EDITS} edits")
    out = []
    for l in lines:
        if mm := _SET.match(l):
            out.append({"kind": "set", "stream": mm.group(1) == "@", "id": mm.group(2), "field": mm.group(3), "value": float(mm.group(4))})
        elif mm := _REWIRE.match(l):
            out.append({"kind": "rewire", "id": mm.group(1), "end": mm.group(2), "value": mm.group(3)})
        elif mm := _FLUID.match(l):
            out.append({"kind": "fluid", "id": mm.group(1), "value": mm.group(2)})
        else:
            raise EditError(f"bad edit line: {l[:60]!r}")
    return out


def edits_to_patch(plant: Plant, edits: list[dict]) -> list[dict]:
    """Resolve edits against the plant into replace-only JSON-patch ops; only existing fields may change."""
    from ..core.components import REQUIRED_PORTS
    from ..core.props import check_fluid, SolveError
    d = plant.model_dump(by_alias=True, mode="json")
    sidx = lambda ident: next((i for i, s in enumerate(d["streams"]) if s["id"] == ident), None)
    patch = []
    for e in edits:
        if e["kind"] == "rewire":
            i = sidx(e["id"])
            if i is None: raise EditError(f"unknown stream {e['id']}")
            tgt = e["value"]
            if not tgt.startswith("ext:"):
                cid, port = tgt.split(".", 1)
                comp = next((c for c in d["components"] if c["id"] == cid), None)
                if comp is None or port not in REQUIRED_PORTS.get(comp["type"], []): raise EditError(f"no such port {tgt}")
            patch.append({"op": "replace", "path": f"/streams/{i}/{e['end']}", "value": tgt})
        elif e["kind"] == "fluid":
            i = sidx(e["id"])
            if i is None: raise EditError(f"unknown stream {e['id']}")
            try: check_fluid(e["value"])
            except SolveError: raise EditError(f"unknown fluid {e['value']}")
            patch.append({"op": "replace", "path": f"/streams/{i}/fluid", "value": e["value"]})
        elif e["stream"]:
            i = sidx(e["id"]); field, value = e["field"], e["value"]
            if i is None: raise EditError(f"unknown stream {e['id']}")
            s = d["streams"][i]
            if field == "m_dot":
                if s.get("m_dot") is None: raise EditError(f"{e['id']}.m_dot is not set")
                patch.append({"op": "replace", "path": f"/streams/{i}/m_dot", "value": value})
            elif field.startswith("fixed.") and s.get("fixed") and s["fixed"].get(field[6:]) is not None:
                patch.append({"op": "replace", "path": f"/streams/{i}/fixed/{field[6:]}", "value": value})
            else:
                raise EditError(f"cannot edit @{e['id']}.{field}")
        else:
            idx = next((i for i, c in enumerate(d["components"]) if c["id"] == e["id"]), None)
            if idx is None: raise EditError(f"unknown component {e['id']}")
            pr = d["components"][idx]["params"].get(e["field"])
            if not isinstance(pr, (int, float)) or isinstance(pr, bool): raise EditError(f"{e['id']}.{e['field']} is not a numeric parameter")
            patch.append({"op": "replace", "path": f"/components/{idx}/params/{e['field']}", "value": e["value"]})
    return patch


def patch_to_edits_text(plant: Plant, patch: list[dict]) -> str:
    """Inverse of edits_to_patch for replace ops (used to write expert completions)."""
    lines = []
    for op in patch:
        parts = op["path"].strip("/").split("/"); v = op["value"]
        if parts[0] == "components":
            lines.append(f"SET {plant.components[int(parts[1])].id}.{parts[3]} = {v!r}")
        else:
            sid = plant.streams[int(parts[1])].id
            if parts[2] in ("to", "from"): lines.append(f"REWIRE @{sid}.{parts[2]} = {v}")
            elif parts[2] == "fluid": lines.append(f"FLUID @{sid} = {v}")
            else: lines.append(f"SET @{sid}.{'.'.join(parts[2:])} = {v!r}")
    return "<edits>\n" + "\n".join(lines) + "\n</edits>"
