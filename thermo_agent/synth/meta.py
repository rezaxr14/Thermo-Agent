"""Reproducibility metadata written next to each generated dataset: meta_<task>.json."""
from __future__ import annotations
import importlib.metadata as md, json, pathlib, platform, datetime
from ..core.schema import SCHEMA_VERSION


def write_meta(out_dir, task: str, **params) -> dict:
    meta = {"task": task, "params": params, "schema_version": SCHEMA_VERSION, "python": platform.python_version(),
            "coolprop": md.version("CoolProp"), "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    pathlib.Path(out_dir, f"meta_{task}.json").write_text(json.dumps(meta, indent=1))
    return meta
