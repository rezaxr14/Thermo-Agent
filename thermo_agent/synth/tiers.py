"""Task difficulty tiers: `easy` / `medium` / `hard`, equal thirds within a task.

Reza's directive (2026-11-19, `docs/DECISIONS_2026-11-19.md` section 4): "we will
include tiers equal thirds ... make the training data and test data properly".

How a tier is decided
---------------------
`complexity(task, row)` is the size of the artefact the task is about, measured on
data that already exists in the row (no new hyper-parameter, no manual label):

* repair  - number of components in the broken plant (`plant_broken`);
* audit   - number of components in the plant (`plant`);
* ledger  - number of components in the ledger topology (`ledger["data"]["topo"]`);
* design  - number of components of the reference design, built from the row's
            `site` + `ref_spec` with `thermo_agent.design.build.build_design`.

`assign_tiers` sorts the rows of one task by that score, breaks ties by the sha256
of the row id (deterministic, no RNG state), and cuts them into three groups of
`floor(n/3)`, `floor(n/3)`, `n - 2*floor(n/3)`.  The remainder therefore goes to
the higher tier, exactly as the directive requires; the caller must record the
resulting counts.

Why a rank and not fixed thresholds
-----------------------------------
The scores are coarse integers (repair has component counts 4,5,6,7,9).  Fixed
thresholds cannot produce equal thirds on such a distribution; a rank can, and
its meaning is stated honestly: "easy" is the smallest third of *this* set, not an
absolute property.  Every caller therefore writes, next to the tiers, the score
range each tier covers, so a reader can compare the training and test cuts.
"""
from __future__ import annotations

import hashlib
import json

TIERS = ("easy", "medium", "hard")
UNKNOWN_TIER = "<missing>"
TASKS = ("repair", "design", "audit", "ledger")


def _json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def _components(obj) -> int | None:
    if obj is None:
        return None
    comps = getattr(obj, "components", None)
    if comps is None and isinstance(obj, dict):
        comps = obj.get("components")
    return len(comps) if comps is not None else None


def _plant_components(row: dict, key: str) -> int | None:
    return _components(_json(row.get(key)))


def _ledger_components(row: dict) -> int | None:
    ledger = _json(row.get("ledger"))
    if not isinstance(ledger, dict):
        return None
    data = ledger.get("data") or {}
    topo = data.get("topo")
    return len(topo) if isinstance(topo, (dict, list)) else None


def _design_components(row: dict) -> int | None:
    spec = _json(row.get("ref_spec"))
    site_raw = row.get("site")
    if not isinstance(spec, dict) or site_raw is None:
        return None
    from ..core.schema import Site
    from ..design.build import build_design
    try:
        site = Site.model_validate_json(site_raw) if isinstance(site_raw, str) else Site.model_validate(site_raw)
        return _components(build_design(site, spec))
    except Exception:
        # A reference design that cannot be rebuilt has no measured size; the row is
        # scored later (see `unscored`) rather than given an invented tier.
        return None


_MEASURES = {
    "repair": lambda row: _plant_components(row, "plant_broken"),
    "audit": lambda row: _plant_components(row, "plant"),
    "ledger": _ledger_components,
    "design": _design_components,
}


def complexity(task: str, row: dict) -> int | None:
    """Measured size of the artefact the task is about; None when the row cannot supply it."""
    if task not in _MEASURES:
        raise KeyError(f"unknown task {task!r}; expected one of {TASKS}")
    return _MEASURES[task](row)


def _rank_key(task: str, row: dict) -> tuple:
    score = complexity(task, row)
    rid = str(row.get("record_id") or row.get("id") or "")
    # Unscored rows sort with the median group so they cannot silently become "hard";
    # the caller counts them.
    return (score is None, score if score is not None else -1, hashlib.sha256(rid.encode()).hexdigest())


def tier_sizes(n: int) -> tuple[int, int, int]:
    """(easy, medium, hard) sizes for n rows: equal thirds, remainder to the higher tier."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    q = n // 3
    return q, q, n - 2 * q


def assign_tiers(task: str, rows: list[dict], id_key: str | None = None) -> dict[str, str]:
    """id -> tier for one task's rows. Deterministic; equal thirds by measured size."""
    # A row with no measurable artefact is NOT given an invented tier: it stays
    # out of the mapping and the caller counts it as unresolved. Real corpora
    # must have zero such rows (enforced by the corpus builders).
    rows = [r for r in rows if complexity(task, r) is not None]
    n = len(rows)
    if n == 0:
        return {}
    ordered = sorted(rows, key=lambda r: _rank_key(task, r))
    sizes = tier_sizes(n)
    out: dict[str, str] = {}
    for tier, size in zip(TIERS, sizes):
        for row in ordered[len(out):len(out) + size]:
            key = str(row.get(id_key)) if id_key else str(row.get("record_id") or row.get("id"))
            out[key] = tier
    return out


def assign_tiers_grouped(task: str, rows: list[dict], key_field: str = "record_id") -> dict[str, str]:
    """key -> tier where every row of one ground-truth plant shares its tier.

    The tool corpora emit two variants (`direct`, `recover`) per plant. Tiering
    each emitted row separately could split a plant's pair across a tier boundary
    even though both rows describe the same artefact; grouping by the source id
    keeps the pair together and is deterministic for the same reason as
    `assign_tiers`.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        # A-slice ids are paraphrases (`<plant-id>#vN`) of a ground-truth plant;
        # they must share the tier of that plant's tool rows.
        key = str(row.get(key_field) or row.get("id") or "").split("#v", 1)[0]
        groups.setdefault(key, []).append(row)
    representatives = [group[0] for group in groups.values()]
    per_key = assign_tiers(task, representatives)
    return {key: per_key[key] for key in groups if key in per_key}


def tier_ranges(task: str, rows: list[dict], tiers: dict[str, str]) -> dict[str, dict]:
    """Per-tier score range and count, so training and test cuts can be compared."""
    out = {t: {"n": 0, "min": None, "max": None} for t in TIERS}
    out["unscored"] = 0
    for row in rows:
        key = str(row.get("record_id") or row.get("id")).split("#v", 1)[0]
        tier = tiers.get(key)
        if tier is None:
            out["unscored"] += 1
            continue
        score = complexity(task, row)
        if score is None:
            continue
        cell = out[tier]
        cell["n"] += 1
        cell["min"] = score if cell["min"] is None else min(cell["min"], score)
        cell["max"] = score if cell["max"] is None else max(cell["max"], score)
    return out
