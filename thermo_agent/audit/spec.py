"""Audit answer format.
    <analysis>...cites only numbers (MW, K) of the plant...</analysis>
    <ranking>ID1 > ID2 > ID3</ranking>          components by exergy destruction, largest first (1 to 5 ids)
    <change>SET <component>.<param> = <number></change>
    <prediction>fuel_change_mw = <number></prediction>   change in fuel input after resizing generation to the same net electricity
"""
from __future__ import annotations
import re

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


class AuditError(ValueError):
    pass


def parse_audit(text: str, valid_ids: set[str]) -> dict:
    g = lambda tag: (re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S) or [None, None])[1]
    out = {"analysis": g("analysis")}
    r, c, p = g("ranking"), g("change"), g("prediction")
    if r is None or c is None or p is None: raise AuditError("need <ranking>, <change> and <prediction> blocks")
    ids = [x for x in re.split(r"[>,\s]+", r.strip()) if x]
    if not (1 <= len(ids) <= 5) or len(set(ids)) != len(ids): raise AuditError("ranking needs 1 to 5 distinct ids")
    bad = [i for i in ids if i not in valid_ids]
    if bad: raise AuditError(f"unknown component ids {bad}")
    out["ranking"] = ids
    m = re.match(rf"^\s*SET\s+([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\s*=\s*({_NUM})\s*$", c)
    if not m: raise AuditError("change must be one SET line")
    out["change"] = (m.group(1), m.group(2), float(m.group(3)))
    m = re.match(rf"^\s*fuel_change_mw\s*=\s*({_NUM})\s*$", p)
    if not m: raise AuditError("prediction must be fuel_change_mw = <number>")
    out["prediction_mw"] = float(m.group(1))
    return out
