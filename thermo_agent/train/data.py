"""Corpus loading shared by SFT and GRPO: read ``task`` AND ``tier`` on every row.

Reza's directive (DECISIONS 2026-11-19 section 4.2) makes ``tier`` a real column,
equal thirds per task, derived from :mod:`thermo_agent.synth.tiers`.  The old GRPO
loader only looked for ``tier``/``difficulty``/``level`` on the raw row and quietly
put every real row (which had none of them) into a single ``<missing>`` bucket, so
the advertised uniform (task x tier) sampler was really a task-only sampler.
This module closes that hole once, for both stages.

Rules (additive: corpus files are never rewritten here):

* An explicit ``tier``/``difficulty``/``level`` column wins; its value must be one
  of ``synth.tiers.TIERS``.
* Otherwise the tier is *derived* with ``synth.tiers.assign_tiers_grouped`` over the
  whole task slice, so the three tiers are equal thirds and the two variants of one
  ground-truth plant (``direct``/``recover``) stay in the same tier.
* A row with no measurable artefact stays ``<missing>`` and is counted.  Real
  corpora must have zero such rows; ``joint_rows`` refuses them loudly by default.
* ``frontier_<task>_<split>.jsonl`` is read in addition to ``<task>_<split>.jsonl``:
  DECISIONS E1 puts accepted frontier tool traces first in the SFT and GRPO mix.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter
from typing import Iterable

from ..synth import tiers as tier_mod

TIER_FIELDS = ("tier", "difficulty", "level")
UNKNOWN_TIER = tier_mod.UNKNOWN_TIER


def read_jsonl(path: pathlib.Path | str) -> list[dict]:
    """Read one JSONL shard; blank lines are ignored (a trailing newline is normal)."""
    with pathlib.Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def shard_paths(data_dir, task: str, split: str) -> list[pathlib.Path]:
    """Deterministic, frontier-first shard order for one task/split.

    Frontier tool rows are emitted as ``frontier_<task>_<split>.jsonl`` by the
    ingest scripts; the deterministic tool corpus and the A-slice use the plain
    name.  Both are part of the decided mix, so a loader that reads only one of
    them silently trains on half the corpus.
    """
    root = pathlib.Path(data_dir)
    paths = []
    for name in (f"frontier_{task}_{split}.jsonl", f"{task}_{split}.jsonl"):
        p = root / name
        if p.is_file():
            paths.append(p)
    return paths


def explicit_tier(row: dict) -> str | None:
    """The row's own tier column, normalised; None when it has none.

    An unknown/typo'd tier is a hard error rather than a new bucket: a silent
    ``TierX`` cell would break equal-thirds sampling and the eval breakdown.
    """
    for key in TIER_FIELDS:
        value = row.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            tier = str(value).strip().lower()
            if tier not in tier_mod.TIERS:
                raise ValueError(
                    f"invalid tier {value!r} in field {key!r}; expected one of {tier_mod.TIERS}"
                )
            return tier
    return None


def row_tier(row: dict) -> str:
    """Tier for an already-loaded row (explicit, else ``<missing>``); no derivation."""
    return explicit_tier(row) or UNKNOWN_TIER


def _group_key(row: dict) -> str:
    """Ground-truth key: paraphrase variants (``<id>#vN``) share their plant's tier."""
    return str(row.get("record_id") or row.get("id") or "").split("#v", 1)[0]


def attach_task_tier(rows: Iterable[dict], task: str, *, derive: bool = True) -> tuple[list[dict], dict]:
    """Return copies of ``rows`` carrying ``task`` and ``tier`` plus a tier census.

    ``derive=False`` is only for callers that deliberately keep an unresolved tier
    (e.g. a unit test with content-less rows); production loaders use the default.
    """
    if task not in tier_mod.TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {tier_mod.TASKS}")
    rows = list(rows)
    pending = [r for r in rows if explicit_tier(r) is None]
    derived: dict[str, str] = {}
    if derive and pending:
        derived = tier_mod.assign_tiers_grouped(task, pending)
    out: list[dict] = []
    counts: Counter = Counter()
    unresolved = []
    for raw in rows:
        row = dict(raw)
        row["task"] = raw.get("task") or task
        tier = explicit_tier(raw)
        if tier is None:
            tier = derived.get(_group_key(raw)) or UNKNOWN_TIER
            if tier == UNKNOWN_TIER:
                unresolved.append(_group_key(raw))
        row["tier"] = tier
        counts[tier] += 1
        out.append(row)
    return out, {"counts": dict(counts), "unresolved": unresolved, "derived": derived}


def load_shards(data_dir, task: str, split: str, *, derive: bool = True) -> tuple[list[dict], dict]:
    """Load every shard of one task/split, with ``task`` and ``tier`` attached."""
    rows: list[dict] = []
    census: Counter = Counter()
    unresolved: list[str] = []
    derived: dict[str, str] = {}
    for path in shard_paths(data_dir, task, split):
        loaded, info = attach_task_tier(read_jsonl(path), task, derive=derive)
        rows.extend(loaded)
        census.update(info["counts"])
        unresolved.extend(info["unresolved"])
        derived.update(info["derived"])
    if not rows:
        return [], {"counts": {}, "unresolved": [], "derived": {}}
    return rows, {"counts": dict(census), "unresolved": unresolved, "derived": derived}


def assert_tiers_resolved(task: str, info: dict) -> None:
    """Fail before a training cell starts if a corpus has no real tier for some rows."""
    n = len(info["unresolved"])
    if n:
        raise ValueError(
            f"{task}: {n} row(s) have no measurable artefact and no explicit tier "
            f"(e.g. {info['unresolved'][:3]}). Tiers must be real equal thirds "
            "(DECISIONS section 4.2); rebuild this corpus with thermo_agent.synth.tiers."
        )
