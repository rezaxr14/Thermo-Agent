"""Global diagram-assignment ledger for the frontier cohort campaign.

Policy (Reza, 2026-11-19)
-------------------------
A diagram is keyed by its #vN-stripped row id (the ground-truth plant / site /
ledger identity).  Each diagram is assigned to AT MOST ONE model for the whole
campaign.  The only exception is a retry: if a model attempted a diagram but no
verifier-accepted tool trace exists for it, the diagram stays eligible for a
different model.  Once any model has a verifier-accepted tool trace, the diagram
is claimed permanently and must never be sent again.  Per-model journal dedupe
remains for resume; this ledger is the cross-model authority.

Commands::

    .venv/Scripts/python.exe scripts/frontier_cohort_assign.py init
    .venv/Scripts/python.exe scripts/frontier_cohort_assign.py plan proof
    .venv/Scripts/python.exe scripts/frontier_cohort_assign.py plan bulk
    .venv/Scripts/python.exe scripts/frontier_cohort_assign.py sync
    .venv/Scripts/python.exe scripts/frontier_cohort_assign.py stats
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mix_tool_corpora import is_verifier_accepted  # noqa: E402

INDEX = ROOT / "data" / "frontier_tools_cohort_prompts" / "diagram_index.json"
LEDGER = ROOT / "data" / "frontier_tools_cohort_prompts" / "diagram_ledger.json"
PLAN_DIR = ROOT / "data" / "frontier_tools_cohort_prompts"

MODELS = ("gpt-6-luna", "claude-haiku-5-5", "gpt-6.2-sol",
          "claude-sonnet-5-5", "claude-opus-5-5-default", "gemini-4-argon")
PREMIUM = MODELS[2:]
# Reza's revised authorisation: BUDGET ONLY.  No premium/mid/flagship model is
# planned, pushed, run or downloaded.  Existing premium proof assignments in the
# ledger are deliberately left reserved and untouched.
BUDGET_MODELS = ("gpt-6-luna", "claude-haiku-5-5")
PLAN_MODELS = BUDGET_MODELS
ALL_CELLS = tuple(f"{t}/{ti}" for t in ("repair", "design", "audit", "ledger")
                  for ti in ("easy", "medium", "hard"))
BULK_CELLS = {
    "gpt-6-luna": ("audit/easy", "audit/medium", "audit/hard"),
    "claude-haiku-5-5": ("audit/easy", "audit/medium", "ledger/easy", "ledger/medium"),
    **{m: ALL_CELLS for m in PREMIUM},
}
PER_CELL = 3  # diagrams allocated per model per task x tier cell
PROOF_N = 5

POLICY = (
    "one diagram per model (key = row id with #vN removed); a diagram with only "
    "verifier-failed attempts stays eligible for a retry by another model; a "
    "verifier-accepted tool trace claims the diagram permanently"
)


def _read(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _base(rid: str) -> str:
    return str(rid or "").split("#v", 1)[0]


def _full_id(base: str, index: dict) -> str:
    return index[base]["id"]


def _empty_ledger(index: dict) -> dict:
    return {
        "policy": POLICY,
        "key": "row id with #vN removed (plant/site/ledger identity)",
        "diagrams": {
            base: {
                "id": info["id"], "task": info["task"], "tier": info["tier"],
                "cell": info["cell"], "assigned_model": None,
                "attempted_by": [], "accepted_by": [], "retried_by": [],
                "claimed": False,
            }
            for base, info in index.items()
        },
        "plans": {},
        "events": [],
        "counters": {
            "cross_model_duplicates_avoided": 0,
            "failed_then_retried": 0,
            "pre_policy_duplicate_diagrams": 0,
        },
    }


def load_ledger() -> tuple[dict, dict]:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    if not LEDGER.exists():
        return _empty_ledger(index), index
    return json.loads(LEDGER.read_text(encoding="utf-8")), index


def save_ledger(ledger: dict) -> None:
    LEDGER.write_text(json.dumps(ledger, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                      encoding="utf-8")


def _journal_paths() -> list[pathlib.Path]:
    out = []
    base = ROOT / "data" / "teacher_runs" / "kaggle_hosted"
    for pattern in ("budget_pilot/**/harness_completions.jsonl",
                    "cohort/**/harness_completions.jsonl"):
        out += sorted(base.glob(pattern))
    return out


def _corpus_paths() -> list[pathlib.Path]:
    out = []
    for dirname in ("frontier_tools_pilot", "frontier_tools_cohort"):
        base = ROOT / "data" / dirname
        out += sorted(base.glob("*/*.jsonl"))
    return out


def _model_key(model: str) -> str | None:
    m = str(model or "").lower()
    for canonical in MODELS:
        if canonical.lower() in m or m in canonical.lower():
            return canonical
    return None


def _sync_state(ledger: dict, index: dict) -> dict:
    """Refresh attempted_by / accepted_by / claimed from journals and corpora."""
    diagrams = ledger["diagrams"]
    for base, info in diagrams.items():
        info["attempted_by"] = []
        info["accepted_by"] = []
        info["claimed"] = False
    # Attempts (journals): every row with an id, including API-error rows that
    # still represent a diagram the model saw.
    for path in _journal_paths():
        for rec in _read(path):
            base = _base(rec.get("id"))
            if base not in diagrams:
                continue
            model = _model_key(rec.get("model")) or _model_key(path.parent.parent.name)
            if model and model not in diagrams[base]["attempted_by"]:
                diagrams[base]["attempted_by"].append(model)
    # Accepted tool traces (ingested corpora) are the ONLY claim source.
    for path in _corpus_paths():
        model = _model_key(path.parent.name)
        if not model:
            continue
        for row in _read(path):
            base = _base(row.get("record_id") or row.get("id"))
            if base not in diagrams:
                continue
            blob = json.dumps(row, ensure_ascii=False)
            if "<tool_call" not in blob or "<tool_result" not in blob:
                continue
            if not is_verifier_accepted(row):
                continue
            info = diagrams[base]
            if model not in info["accepted_by"]:
                info["accepted_by"].append(model)
            info["claimed"] = True
    # Pre-policy pilot duplication (the 40-row pilots sent the SAME ids to both
    # budget models).  Count once per diagram that has two accepted_by entries.
    ledger["counters"]["pre_policy_duplicate_diagrams"] = sum(
        1 for info in diagrams.values() if len(info["accepted_by"]) > 1)
    return ledger


def cmd_expand() -> None:
    """Add newly-built candidate diagrams to an existing ledger without losing assignments."""
    ledger, index = load_ledger()
    diagrams = ledger["diagrams"]
    added = 0
    for base, info in index.items():
        if base not in diagrams:
            diagrams[base] = {
                "id": info["id"], "task": info["task"], "tier": info["tier"],
                "cell": info["cell"], "assigned_model": None,
                "attempted_by": [], "accepted_by": [], "retried_by": [],
                "claimed": False,
            }
            added += 1
    save_ledger(ledger)
    print(json.dumps({"candidate_diagrams": len(diagrams), "added": added}, indent=1))


def cmd_init() -> None:
    ledger, index = load_ledger()
    ledger = _sync_state(ledger, index)
    save_ledger(ledger)
    print(json.dumps(stats(ledger), indent=1))


def cmd_sync() -> None:
    ledger, index = load_ledger()
    ledger = _sync_state(ledger, index)
    save_ledger(ledger)
    print(json.dumps(stats(ledger), indent=1))


def stats(ledger: dict) -> dict:
    diagrams = ledger["diagrams"]
    claimed = [b for b, i in diagrams.items() if i["claimed"]]
    by_model = collections.Counter()
    for base in claimed:
        for m in diagrams[base]["accepted_by"]:
            by_model[m] += 1
    retried = [b for b, i in diagrams.items() if i["retried_by"]]
    return {
        "policy": POLICY,
        "candidate_diagrams": len(diagrams),
        "unique_diagrams_covered": len(claimed),
        "accepted_by_model": dict(by_model),
        "cross_model_duplicates_avoided":
            ledger["counters"]["cross_model_duplicates_avoided"],
        "failed_then_retried": len(retried),
        "failed_then_retried_ids": retried,
        "pre_policy_duplicate_diagrams":
            ledger["counters"]["pre_policy_duplicate_diagrams"],
    }


def cmd_stats() -> None:
    ledger, _ = load_ledger()
    print(json.dumps(stats(ledger), indent=1, ensure_ascii=False))


def _eligible(info: dict, model: str) -> tuple[bool, str]:
    """(eligible, reason) for sending a diagram to `model`."""
    if info["claimed"]:
        return False, "claimed"
    current = info["assigned_model"]
    if current is None or current == model:
        return True, "free" if current is None else "resume"
    # A different model owns it.  Only a verifier FAILED attempt may be retried.
    if current in info["attempted_by"] and current not in info["accepted_by"]:
        return True, "retry"
    return False, "reserved"


def cmd_plan(phase: str, quota_file: str | None = None) -> None:
    ledger, index = load_ledger()
    ledger = _sync_state(ledger, index)
    diagrams = ledger["diagrams"]
    quota = None
    if quota_file:
        quota = json.loads(pathlib.Path(quota_file).read_text(encoding="utf-8"))
    plans: dict[str, list[dict]] = {}
    for model in PLAN_MODELS:
        plans[model] = []
    if phase == "proof":
        models = list(PLAN_MODELS)
        used: set[str] = set()
        # One diagram from each task first, then extra, all distinct across models.
        order = [f"{t}/{ti}" for ti in ("easy", "medium", "hard")
                 for t in ("repair", "design", "audit", "ledger")]
        for model in models:
            chosen = []
            for cell in order:
                if len(chosen) >= PROOF_N:
                    break
                for base in sorted(index, key=lambda b: (index[b]["cell"], b)):
                    if index[base]["cell"] != cell or base in used:
                        continue
                    info = diagrams[base]
                    ok, reason = _eligible(info, model)
                    if not ok:
                        if reason in ("claimed", "reserved"):
                            ledger["counters"]["cross_model_duplicates_avoided"] += 1
                        continue
                    chosen.append(base)
                    used.add(base)
                    info["assigned_model"] = model
                    if reason == "retry":
                        info["retried_by"].append(model)
                        ledger["counters"]["failed_then_retried"] += 1
                        ledger["events"].append({
                            "base": base, "from": info["attempted_by"][-1],
                            "to": model, "reason": "failed-then-retried"})
                    break
            plans[model] = [
                {"id": _full_id(b, index), "base": b, "cell": diagrams[b]["cell"]}
                for b in chosen
            ]
    elif phase == "bulk":
        for model in PLAN_MODELS:
            cells = quota[model] if quota and model in quota else BULK_CELLS[model]
            for cell in cells:
                per_cell = int(cells[cell]) if isinstance(cells, dict) else PER_CELL
                picked = 0
                # Deterministic order: sha256 of the base id, matching the shard builder.
                import hashlib
                candidates = sorted(
                    (b for b in index if index[b]["cell"] == cell),
                    key=lambda b: hashlib.sha256(b.encode()).hexdigest())
                for base in candidates:
                    if picked >= per_cell:
                        break
                    info = diagrams[base]
                    ok, reason = _eligible(info, model)
                    if not ok:
                        if reason in ("claimed", "reserved"):
                            ledger["counters"]["cross_model_duplicates_avoided"] += 1
                        continue
                    # Already reserved for this model on an earlier plan: keep it,
                    # but don't double count retries/events.
                    if info["assigned_model"] != model:
                        info["assigned_model"] = model
                        if reason == "retry":
                            if model not in info["retried_by"]:
                                info["retried_by"].append(model)
                                ledger["counters"]["failed_then_retried"] += 1
                            ledger["events"].append({
                                "base": base,
                                "from": next((m for m in info["attempted_by"]
                                              if m != model), "unknown"),
                                "to": model, "reason": "failed-then-retried"})
                    picked += 1
                plans[model] += [
                    {"id": _full_id(b, index), "base": b, "cell": diagrams[b]["cell"]}
                    for b in candidates
                    if diagrams[b]["assigned_model"] == model
                    and diagrams[b]["cell"] == cell
                ][:per_cell]
    else:
        raise SystemExit(f"unknown phase {phase!r}; use proof or bulk")

    ledger["plans"][phase] = plans
    save_ledger(ledger)
    for model, rows in plans.items():
        path = PLAN_DIR / f"plan_{phase}_{model}.json"
        path.write_text(json.dumps({
            "phase": phase, "model": model, "policy": POLICY,
            "rows": rows,
        }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "phase": phase,
        "per_model": {m: len(rows) for m, rows in plans.items()},
        "distinct_diagrams": len({r["base"] for rows in plans.values() for r in rows}),
        **stats(ledger),
    }, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("init", "expand", "plan", "sync", "stats"))
    ap.add_argument("phase", nargs="?", default=None,
                    help="plan phase (proof/bulk), or path to a quota JSON file for bulk")
    ap.add_argument("quota", nargs="?", default=None,
                    help="optional path to batch quota JSON: {model: {task/tier: N}}")
    args = ap.parse_args()
    if args.action == "init":
        cmd_init()
    elif args.action == "expand":
        cmd_expand()
    elif args.action == "sync":
        cmd_sync()
    elif args.action == "stats":
        cmd_stats()
    elif args.action == "plan":
        if args.phase not in ("proof", "bulk"):
            raise SystemExit("plan needs a phase: proof or bulk")
        cmd_plan(args.phase, args.quota)


if __name__ == "__main__":
    main()
