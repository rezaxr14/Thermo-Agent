"""Build the hosted teacher prompt dataset: INPUTS ONLY, never answers.

The big-model harness runs the real physics tools via
``thermo_agent.teacher.toolkit.make_tools``. Those tools read *input* columns
(``plant_broken`` for repair, ``plant``/``site`` for audit, ``ledger`` for
ledger, ``site`` for design). The previous compact shards carried only the
system/user prompt text, so every hosted row was ungradable and made zero tool
calls. This builder re-projects the frontier export into shards that carry the
tool-ready inputs while deliberately stripping every answer-derived column.

Grading is deferred to local ingest against ``data/frontier_export`` by id.

Usage::

    .venv/Scripts/python.exe scripts/make_hosted_prompts.py
    .venv/Scripts/python.exe scripts/make_hosted_prompts.py --out /tmp/check
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
import random
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPORT = ROOT / "data" / "frontier_export"
OUT = ROOT / "data" / "kaggle_datasets" / "teacher_big_prompts"

# Fixed order so the concatenated + seeded-shuffled stream is reproducible.
TASK_ORDER = ("repair", "audit", "ledger", "design")
N_SHARDS = 5
SEED = 20261119

# The ONLY keys copied from each export row. Answer-derived columns are absent
# by construction (see FORBIDDEN_KEYS / ANSWER_KEYS below).
FIELD_ALLOWLIST = {
    "repair": ["id", "split", "topology", "prompt", "plant_broken", "site", "task"],
    "audit": ["id", "split", "topology", "prompt", "plant", "site", "task"],
    "ledger": ["id", "split", "topology", "prompt", "ledger", "task"],
    "design": ["id", "split", "prompt", "site", "task"],
}

# Keys that must never appear on an output row (answers, verdicts, trajectories,
# and any other leakage channel listed by the task contract).
FORBIDDEN_KEYS = (
    "plant_original", "min_edits", "deducible", "detected", "ed_json", "fuel_w",
    "net_w", "best_gain_w", "n_candidates", "ref_spec", "ref_fuel_w", "n_valid",
    "ref_evals", "fault_type", "fault_types", "completion", "validation",
    "trajectory", "answer",
)

# Answer-derived VALUES whose canonical JSON/number strings must not leak into
# any serialized output row (guards against accidental embedding in a field).
ANSWER_KEYS = {
    "repair": ("plant_original",),
    "audit": ("ed_json", "fuel_w", "best_gain_w"),
    "design": ("ref_spec", "ref_fuel_w"),
    "ledger": ("fault_types",),
}

NOTE = "inputs only; grading is deferred to local ingest against data/frontier_export"


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _canonical_strings(value) -> set[str]:
    """Distinct strings that could represent an answer value in a serialized row."""
    out: set[str] = set()
    if isinstance(value, str):
        out.add(value)
        try:
            out.add(json.dumps(json.loads(value), sort_keys=True, separators=(",", ":")))
        except (ValueError, TypeError):
            pass
    else:
        out.add(json.dumps(value, sort_keys=True, separators=(",", ":")))
        out.add(json.dumps(value, sort_keys=True))
        out.add(repr(value))
        out.add(str(value))
    return {s for s in out if s}


def _haystacks(row: dict) -> list[str]:
    """Raw string values (incl. nested prompt messages) + the compact JSON form."""
    found: list[str] = []

    def walk(obj):
        if isinstance(obj, str):
            found.append(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                found.append(k)
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(row)
    found.append(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return found


def project(row: dict, task: str) -> dict:
    """Copy exactly the allowlisted input keys, in whitelist order."""
    keys = FIELD_ALLOWLIST[task]
    out = {}
    for k in keys:
        if k not in row:
            raise KeyError(f"{task} row {row.get('id', '?')} is missing required input key {k!r}")
        out[k] = row[k]  # prompt is carried verbatim (system + user messages)
    if set(out) != set(keys):
        raise AssertionError(f"projection for {task} produced unexpected keys: {sorted(out)}")
    return out


def leak_check(pairs: list[tuple[str, dict, dict]]) -> dict:
    """Fail loudly on forbidden keys or on an answer value embedded in an output row."""
    forbidden_hits = []
    value_hits = []
    for task, src, out in pairs:
        bad_keys = sorted(set(out) & set(FORBIDDEN_KEYS))
        if bad_keys:
            forbidden_hits.append({"task": task, "id": src.get("id"), "keys": bad_keys})
        hay = _haystacks(out)
        for key in ANSWER_KEYS[task]:
            for needle in _canonical_strings(src[key]):
                if any(needle in text for text in hay):
                    value_hits.append({"task": task, "id": src.get("id"),
                                       "key": key, "needle": needle[:80]})
                    break
    if forbidden_hits or value_hits:
        raise AssertionError(json.dumps({
            "answer_leak_check": "FAIL",
            "forbidden_keys": forbidden_hits[:5],
            "answer_values": value_hits[:5],
            "n_forbidden": len(forbidden_hits),
            "n_values": len(value_hits),
        }, indent=1))
    return {
        "result": "PASS",
        "checked_rows": len(pairs),
        "forbidden_keys": list(FORBIDDEN_KEYS),
        "answer_keys_checked": {t: list(k) for t, k in ANSWER_KEYS.items()},
    }


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(export_dir: pathlib.Path, out_dir: pathlib.Path) -> dict:
    sources: dict[str, list[dict]] = {}
    source_hashes = {}
    for task in TASK_ORDER:
        path = export_dir / f"{task}_train.jsonl"
        sources[task] = read_jsonl(path)
        source_hashes[path.name] = {"sha256": sha256(path), "rows": len(sources[task])}

    pairs: list[tuple[str, dict, dict]] = []
    for task in TASK_ORDER:
        pairs.extend((task, src, project(src, task)) for src in sources[task])

    leak = leak_check(pairs)

    # Deterministic: fixed concatenation order + seeded shuffle + round-robin.
    rng = random.Random(SEED)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    shards: list[list[tuple[str, dict, dict]]] = [[] for _ in range(N_SHARDS)]
    for i, pair in enumerate(shuffled):
        shards[i % N_SHARDS].append(pair)

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_names = [f"prompts_s{i:02d}.jsonl" for i in range(N_SHARDS)]
    for name, shard in zip(shard_names, shards):
        (out_dir / name).write_text(
            "".join(json.dumps(out, ensure_ascii=False) + "\n" for _, _, out in shard),
            encoding="utf-8",
        )

    by_task = Counter(task for task, _, _ in pairs)
    per_shard = {name: Counter(task for task, _, _ in shard)
                 for name, shard in zip(shard_names, shards)}
    manifest = {
        "generator": "scripts/make_hosted_prompts.py",
        "note": NOTE,
        "seed": SEED,
        "n_shards": N_SHARDS,
        "shards": shard_names,
        "total_rows": len(pairs),
        "per_task_counts": {t: by_task[t] for t in TASK_ORDER},
        "per_shard_counts": {name: sum(c.values()) for name, c in per_shard.items()},
        "per_task_per_shard_counts": {
            name: {t: c[t] for t in TASK_ORDER} for name, c in per_shard.items()
        },
        "field_allowlist": {t: list(k) for t, k in FIELD_ALLOWLIST.items()},
        "leak_check": leak,
        "source_hashes": source_hashes,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    metadata = {
        "title": "thermo-agent big-model teacher prompts",
        "id": "rezanadimi/thermo-teacher-prompts-big",
        "description": "Tool-ready task inputs only (no answers). " + NOTE + ".",
        "licenses": [{"name": "MIT"}],
        "files": [{"name": name, "description": "inputs only (no answers)"}
                  for name in shard_names],
    }
    (out_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "out": str(out_dir),
        "total_rows": len(pairs),
        "per_task": {t: by_task[t] for t in TASK_ORDER},
        "per_shard": {name: sum(c.values()) for name, c in per_shard.items()},
        "per_task_per_shard": {
            name: {t: c[t] for t in TASK_ORDER} for name, c in per_shard.items()
        },
    }, indent=1, ensure_ascii=False))
    print(f"answer_leak_check: {leak['result']}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-dir", type=pathlib.Path, default=EXPORT,
                    help=f"frontier export directory (default: {EXPORT})")
    ap.add_argument("--out", type=pathlib.Path, default=OUT,
                    help=f"upload staging directory to (over)write (default: {OUT})")
    args = ap.parse_args(argv)
    build(args.export_dir, args.out)


if __name__ == "__main__":
    main()
