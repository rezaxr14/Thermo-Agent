"""GRPO rewards for Design (TRL contract). Dataset column: site (JSON string).

The scoring-only reference answers (``ref_fuel_w``, ``ref_spec``, ``ref_evals``) are
NOT dataset columns and never travel with a training row or a prompt. They are
joined from ``data/frontier_export/design_train.jsonl`` at SCORING time into an
ephemeral kwargs copy, keyed by the row's ``record_id``/``id``. A missing export
file or an unresolvable id is a hard error: a design rollout must never be scored
against a zero (or silently default) fuel target.

Legacy callers (reward diagnostics, direct unit tests) that pass an explicit
positive ``ref_fuel_w`` column keep the old behaviour and are not joined.
"""
from __future__ import annotations
from functools import lru_cache, wraps
import json
import os
import pathlib
import re
from ..core.schema import Site
from ..verify.grounding import check_numbers
from .spec import parse_design, DesignError
from .build import evaluate_design


def completion_text(c) -> str:
    return c if isinstance(c, str) else c[-1]["content"]


@lru_cache(maxsize=8192)
def evaluate(text: str, site_js: str) -> dict:
    out = {"parsed": False, "built": False, "valid": False, "fuel_w": None, "grounded": 0.0, "has_rationale": False, "reason": ""}
    site = Site.model_validate_json(site_js)
    m = re.search(r"<rationale>(.*?)</rationale>", text, flags=re.S); out["has_rationale"] = m is not None
    try: spec = parse_design(text, needs_heat=site.heat_demand_w > 0); out["parsed"] = True
    except DesignError as e: out["reason"] = str(e); return out
    ev = evaluate_design(site, spec)
    out.update(built=ev.built, valid=ev.valid, fuel_w=ev.fuel_w, reason=ev.reason)
    if m and ev.result is not None and ev.result.ok: out["grounded"] = check_numbers(m.group(1), ev.plant, ev.result, rel=0.03)["fraction"]
    return out


def _each(completions, kw, fn):
    return [fn(evaluate(completion_text(c), kw["site"][i]), i, kw) for i, c in enumerate(completions)]


# --------------------------------------------------------------------------- #
# Scoring-only reference join. Nothing here writes back into a row/kwargs: the
# reference fields are copied into a NEW kwargs dict that only the reward sees.
# --------------------------------------------------------------------------- #
DESIGN_REFS_ENV = "THERMO_DESIGN_REFS"
DEFAULT_DESIGN_REFS = (pathlib.Path(__file__).resolve().parents[2]
                       / "data" / "frontier_export" / "design_train.jsonl")
REF_FIELDS = ("ref_fuel_w", "ref_spec", "ref_evals")


class DesignReferenceError(RuntimeError):
    """The scoring-only design reference answer is absent or cannot be resolved."""


def reference_path() -> pathlib.Path:
    """Path of the scoring-only design reference export (env override for staging)."""
    return pathlib.Path(os.environ.get(DESIGN_REFS_ENV) or DEFAULT_DESIGN_REFS)


def clear_reference_cache() -> None:
    """Forget the joined reference index (used when the export file is replaced)."""
    _load_references.cache_clear()


def identity_keys(row: dict) -> tuple[str, ...]:
    """Lookup keys for one row: record_id/id plus their ground-truth base.

    Mix ids carry a teacher/model suffix (``<id>-t-<model>-...``) and paraphrase
    variants carry ``#vN``; the export is keyed by the ground-truth id. Both the
    exact id and its base are tried, with ``record_id`` first.
    """
    keys: list[str] = []
    for field in ("record_id", "id"):
        value = row.get(field)
        if isinstance(value, (str, int)) and str(value).strip():
            key = str(value).strip()
            keys.append(key)
            base = re.split(r"-t-|#v", key, maxsplit=1)[0]
            if base and base != key:
                keys.append(base)
    return tuple(dict.fromkeys(keys))


@lru_cache(maxsize=8)
def _load_references(path_str: str) -> dict[str, dict]:
    """Index the export by every identity key. Fail loudly on a missing/broken file."""
    path = pathlib.Path(path_str)
    if not path.is_file():
        raise DesignReferenceError(
            f"design scoring reference file is missing: {path}. Design rewards join "
            f"ref_fuel_w/ref_spec/ref_evals from this scoring-only export at scoring "
            f"time; set {DESIGN_REFS_ENV} if it is staged elsewhere."
        )
    index: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise DesignReferenceError(
                    f"invalid JSON in design scoring reference file {path}:{lineno}: {exc}"
                ) from exc
            if not isinstance(row, dict) or not all(k in row for k in REF_FIELDS):
                continue
            for key in identity_keys(row):
                index.setdefault(key, row)
    if not index:
        raise DesignReferenceError(
            f"design scoring reference file {path} has no row carrying {REF_FIELDS}"
        )
    return index


def _explicit_reference(kw: dict, n: int) -> bool:
    """True when the caller supplied a real reference column (legacy path)."""
    values = kw.get("ref_fuel_w")
    if not isinstance(values, (list, tuple)) or len(values) != n:
        return False
    try:
        return all(v is not None and str(v).strip() and float(v) > 0 for v in values)
    except (TypeError, ValueError):
        return False


def _joined_reference_fields(kw: dict, n: int) -> dict[str, list]:
    """Join reference fields for a whole batch into a fresh dict (never mutates kw)."""
    path = reference_path()
    index = _load_references(str(path))
    ids, rids = kw.get("id"), kw.get("record_id")
    ids = list(ids) if isinstance(ids, (list, tuple)) else [None] * n
    rids = list(rids) if isinstance(rids, (list, tuple)) else [None] * n
    if len(ids) != n or len(rids) != n:
        raise DesignReferenceError(
            "design scoring join needs one 'id'/'record_id' per scored row "
            f"(got {len(ids)} ids, {len(rids)} record_ids for {n} completions)"
        )
    out: dict[str, list] = {name: [] for name in REF_FIELDS}
    for i in range(n):
        candidates: list[str] = []
        for value in (rids[i], ids[i]):
            if value is None:
                continue
            key = str(value).strip()
            if not key:
                continue
            candidates.append(key)
            base = re.split(r"-t-|#v", key, maxsplit=1)[0]
            if base and base != key:
                candidates.append(base)
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            raise DesignReferenceError(
                f"design scoring row {i} has neither 'record_id' nor 'id'; cannot join "
                f"the reference answer from {path} (no zero-default fallback)"
            )
        match = next((key for key in candidates if key in index), None)
        if match is None:
            raise DesignReferenceError(
                f"no design reference answer for scoring row {i} "
                f"(record_id={rids[i]!r}, id={ids[i]!r}, lookup keys={candidates}) in "
                f"{path}; scoring with a zero/default fuel target is forbidden"
            )
        ref = index[match]
        for name in REF_FIELDS:
            if name not in ref:
                raise DesignReferenceError(
                    f"reference row {match!r} in {path} is missing {name!r}"
                )
            out[name].append(ref[name])
    return out


def assert_references_for_rows(rows) -> None:
    """Preflight a design batch: every row must resolve before scoring starts.

    GRPO calls this once at startup so a partially exported reference file fails
    with one actionable message instead of mid-run on the first design rollout.
    Rows that explicitly carry a real ``ref_fuel_w`` column keep the legacy path.
    """
    rows = list(rows)
    if not rows:
        return
    if all(_explicit_reference({"ref_fuel_w": [r.get("ref_fuel_w")]}, 1) for r in rows):
        return
    index = _load_references(str(reference_path()))
    missing = []
    for row in rows:
        keys = identity_keys(row)
        if not any(k in index for k in keys):
            missing.append(keys[0] if keys else "<no id>")
    if missing:
        raise DesignReferenceError(
            f"{len(missing)} of {len(rows)} design rows have no scoring reference in "
            f"{reference_path()} (e.g. {missing[:5]}); regenerate the scoring-only "
            "export with full coverage before GRPO (no zero/default fallback)"
        )


def with_scoring_references(fn):
    """Wrap a design reward so reference answers arrive by join, not by dataset."""
    @wraps(fn)  # TRL maps reward_weights by __name__; C9 dispatch keys on it too
    def wrapped(prompts=None, completions=None, **kw):
        n = len(completions) if isinstance(completions, (list, tuple)) else 1
        if _explicit_reference(kw, n):
            return fn(prompts, completions, **kw)
        scoring_kw = dict(kw)  # ephemeral copy: the caller's row/kwargs stay untouched
        scoring_kw.update(_joined_reference_fields(kw, n))
        return fn(prompts, completions, **scoring_kw)
    return wrapped


def d_format(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: 1.0 if (e["parsed"] and e["has_rationale"]) else (0.3 if e["parsed"] else 0.0))

def d_valid(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: 1.0 if e["valid"] else (0.2 if e["built"] else 0.0))

def d_quality(prompts, completions, **kw):
    """1.0 at the reference fuel, falling linearly to 0 at 10 % more fuel; up to 1.05 for beating the reference."""
    def f(e, i, k):
        if not e["valid"]: return 0.0
        ref = float(k["ref_fuel_w"][i]); return max(0.0, min(1.05, 1.0 - (e["fuel_w"] - ref) / (0.1 * ref)))
    return _each(completions, kw, f)

def d_grounded(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: e["grounded"] if e["valid"] else 0.0)


# All four public rewards join the scoring-only references; functools.wraps keeps
# d_format/d_valid/d_quality/d_grounded as their __name__ for TRL weights + C9 dispatch.
d_format = with_scoring_references(d_format)
d_valid = with_scoring_references(d_valid)
d_quality = with_scoring_references(d_quality)
d_grounded = with_scoring_references(d_grounded)

REWARD_FUNCS = [d_format, d_valid, d_quality, d_grounded]
REWARD_WEIGHTS = [0.5, 2.0, 2.0, 0.5]
