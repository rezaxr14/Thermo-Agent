"""Build tier-targeted prompt shards for the frontier cohort campaign.

Per task x tier (easy/medium/hard) rows are selected from the deterministic
ground-truth export with ``thermo_agent.synth.tiers.assign_tiers`` (equal thirds,
sha256 tie-break on the row id).  Exactly one paraphrase per ground-truth plant
is present -- the export was built with ``transfer.select`` -- and no heldout id
is ever written.

Only INPUT columns are projected onto a shard row (the same allowlist as
``scripts/make_hosted_prompts.py``).  The compact harness text is NOT stored: the
hosted task regenerates it at run time with the SAME builder used by training and
eval (``thermo_agent.agent.tool_prompt.build_harness_tool_messages`` via
``agent_graph.build_context``), so no bookkeeping id, manifest or ground-truth
answer reaches the model.

Usage::

    .venv/Scripts/python.exe scripts/frontier_cohort_shards.py
    .venv/Scripts/python.exe scripts/frontier_cohort_shards.py --per-cell 6
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib

from thermo_agent.synth import tiers as tier_mod

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPORT = ROOT / "data" / "frontier_export"
OUT = ROOT / "data" / "frontier_tools_cohort_prompts"
DATASET_ID = "rezanadimi/thermo-teacher-cohort-prompts"

# Identical to make_hosted_prompts.FIELD_ALLOWLIST: the tools read these input
# columns, and every answer-derived column is absent by construction.
FIELD_ALLOWLIST = {
    "repair": ["id", "split", "topology", "prompt", "plant_broken", "site", "task"],
    "audit": ["id", "split", "topology", "prompt", "plant", "site", "task"],
    "ledger": ["id", "split", "topology", "prompt", "ledger", "task"],
    "design": ["id", "split", "prompt", "site", "task"],
}
FORBIDDEN_KEYS = (
    "plant_original", "min_edits", "deducible", "detected", "ed_json", "fuel_w",
    "net_w", "best_gain_w", "n_candidates", "ref_spec", "ref_fuel_w", "n_valid",
    "ref_evals", "fault_type", "fault_types", "completion", "validation",
    "trajectory", "answer", "tier", "memory_notes", "tokens", "harness", "paraphrase",
)


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_id(rid: str) -> str:
    return hashlib.sha256(rid.encode()).hexdigest()


def _base_id(rid: str) -> str:
    return str(rid).split("#v", 1)[0]


def project(row: dict, task: str) -> dict:
    keys = FIELD_ALLOWLIST[task]
    out = {}
    for k in keys:
        if k not in row:
            raise KeyError(f"{task} row {row.get('id', '?')} missing input key {k!r}")
        out[k] = row[k]
    bad = sorted(set(out) & set(FORBIDDEN_KEYS))
    if bad:
        raise AssertionError(f"forbidden keys projected for {task}: {bad}")
    return out


def _heldout_ids() -> set[str]:
    ids: set[str] = set()
    for name in ("heldout", "heldout_v2"):
        base = ROOT / "data" / name
        for path in base.glob("*.jsonl"):
            for row in read_jsonl(path):
                ids.add(str(row.get("record_id") or row.get("id") or ""))
                ids.add(_base_id(str(row.get("id") or "")))
    return ids


def build(per_cell: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    heldout = _heldout_ids()
    manifest = {
        "generator": "scripts/frontier_cohort_shards.py",
        "dataset": DATASET_ID,
        "source": "data/frontier_export (deterministic ground truth, one paraphrase per plant)",
        "tier_rule": "thermo_agent.synth.tiers.assign_tiers (equal thirds, sha256(id) tie-break)",
        "per_cell": per_cell,
        "harness": "regenerated at run time by agent_graph.build_context (compact builder); "
                   "no bookkeeping ids/manifests or ground truth in model-visible text",
        "tasks": {},
        "cells": {},
        "source_hashes": {},
        "proof_ids": [],
        "heldout_overlap": [],
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    all_rows: list[dict] = []
    diagram_index: dict[str, dict] = {}
    for task in tier_mod.TASKS:
        path = EXPORT / f"{task}_train.jsonl"
        rows = read_jsonl(path)
        manifest["source_hashes"][path.name] = {"sha256": sha256(path), "rows": len(rows)}
        tier_map = tier_mod.assign_tiers(task, rows, id_key="id")
        if len(tier_map) != len(rows):
            raise AssertionError(f"{task}: {len(rows) - len(tier_map)} unscored rows")
        bases = [_base_id(r["id"]) for r in rows]
        if len(set(bases)) != len(rows):
            raise AssertionError(f"{task}: more than one paraphrase per plant")
        ranges = tier_mod.tier_ranges(task, rows, tier_map)
        manifest["tasks"][task] = {
            "n": len(rows),
            "tiers": {t: ranges[t] for t in tier_mod.TIERS},
        }
        for tier in tier_mod.TIERS:
            group = [r for r in rows if tier_map[str(r["id"])] == tier]
            group.sort(key=lambda r: _sha_id(str(r["id"])))  # deterministic, sha256 tie-break
            picked = group[:per_cell]
            if len(picked) < per_cell:
                raise AssertionError(f"{task}/{tier}: only {len(picked)} rows")
            overlap = [str(r["id"]) for r in picked if _base_id(str(r["id"])) in heldout]
            if overlap:
                raise AssertionError(f"{task}/{tier}: heldout overlap {overlap}")
            manifest["heldout_overlap"].extend(overlap)
            name = f"prompts_{task}_{tier}.jsonl"
            (OUT / name).write_text(
                "".join(json.dumps(project(r, task), ensure_ascii=False) + "\n" for r in picked),
                encoding="utf-8",
            )
            manifest["cells"][f"{task}/{tier}"] = {
                "shard": name,
                "ids": [str(r["id"]) for r in picked],
                "score_min": ranges[tier]["min"],
                "score_max": ranges[tier]["max"],
            }
            for r in picked:
                all_rows.append((task, tier, project(r, task)))
                base = _base_id(str(r["id"]))
                if base in diagram_index:
                    raise AssertionError(f"duplicate diagram {base} across cells")
                diagram_index[base] = {
                    "id": str(r["id"]), "task": task, "tier": tier,
                    "cell": f"{task}/{tier}",
                }

    # Combined shard: deterministic task/tier order, stable sha256 order inside a cell.
    (OUT / "prompts_all.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for _, _, r in all_rows),
        encoding="utf-8",
    )
    # Proof shard: one row from each of the four tasks, plus one extra repair row.
    chosen = []
    seen_tasks: set[str] = set()
    for task in tier_mod.TASKS:
        cand = [row for t, _, row in all_rows if t == task and task not in seen_tasks]
        if cand:
            chosen.append(cand[0])
            seen_tasks.add(task)
    extra = [row for t, _, row in all_rows if t == "repair" and row not in chosen]
    if extra:
        chosen.append(extra[0])
    (OUT / "prompts_proof.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen),
        encoding="utf-8",
    )
    manifest["proof_ids"] = [str(r["id"]) for r in chosen]
    manifest["total_rows"] = len(all_rows)
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # Global identity index: the campaign ledger keys on the #vN-stripped plant id.
    (OUT / "diagram_index.json").write_text(
        json.dumps(diagram_index, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")

    metadata = {
        "title": "thermo-agent frontier cohort tier shards",
        "id": DATASET_ID,
        "description": "Tier-targeted tool-ready task inputs only (no answers). "
                       "Compact harness regenerated at run time by the shared builder.",
        "licenses": [{"name": "MIT"}],
        "files": [
            {"name": f"prompts_{t}_{ti}.jsonl", "description": f"{t}/{ti} inputs only"}
            for t in tier_mod.TASKS for ti in tier_mod.TIERS
        ] + [
            {"name": "prompts_all.jsonl", "description": "all tier-targeted rows"},
            {"name": "prompts_proof.jsonl", "description": "5-row payload-proof shard"},
        ],
    }
    (OUT / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "out": str(OUT),
        "per_cell": per_cell,
        "total_rows": len(all_rows),
        "cells": {k: len(v["ids"]) for k, v in manifest["cells"].items()},
        "proof_ids": manifest["proof_ids"],
        "heldout_overlap": manifest["heldout_overlap"],
    }, indent=1, ensure_ascii=False))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-cell", type=int, default=24,
                    help="rows per task x tier cell (default: 24; enough distinct diagrams "
                         "to assign to all six models without cross-model duplicates)")
    args = ap.parse_args()
    build(args.per_cell)


if __name__ == "__main__":
    main()
