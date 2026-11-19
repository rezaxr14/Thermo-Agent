"""Build the plain-prompt A-slice (decision E2 / J1) from stream A.

Additive-only: reads ``data/sft_all/*_train.jsonl`` (and ``data/heldout`` ids)
and writes a *new* directory (default ``data/a_slice``).  No protected corpus
is ever written, no kaggle command is run, and nothing is truncated.

    python scripts/build_a_slice.py --stats      # dry run: counts only, no writes
    python scripts/build_a_slice.py              # writes data/a_slice

Ground-truth plant
------------------
``thermo_agent.synth.repair_dataset.make_records`` assigns one id per generated
plant, ``rep-<seed>-<i:05d>``.  ``thermo_agent.synth.combine._emit`` emits the
training paraphrases of that row as ``<id>#v1`` ... ``<id>#v<n-1>`` (v0 keeps the
bare id), with ``paraphrase: {variant, n}``.  All of them share the same prompt
and the same plant; only the diagnosis completion is re-phrased.  The ground-
truth plant key is therefore the row id with a trailing ``#vN`` tag removed.
For the three non-paraphrased tasks every row is its own plant.

Selection rule (exact, seeded, deterministic)
---------------------------------------------
1. target = ``round(--tool-rows * --ratio)``; quota per task = ``target // T``
   plus one for the first ``target % T`` tasks in ``--tasks`` order.
2. A row is eligible when it is a plain (``harness in {none, absent}``) train
   row, its exact id / record id and its plant key are not in ``--heldout``,
   and its measured prompt+completion token count is <= 8192.  Over-gate rows
   are dropped and counted, never truncated; the next eligible rows backfill.
3. Eligible rows are shuffled with ``random.Random(--seed)`` (tasks processed in
   ``--tasks`` order) and the first row of each plant key wins, up to the
   task quota.  One paraphrase per plant, task-balanced.
4. The stream-A ``tokens`` column (recorded by the same tokenizer at corpus
   build time) is a cheap pre-filter; every row that can be selected is then
   re-measured on the rendered prompt + completion with the real Qwen3.5
   tokenizer (sum of message contents + completion + the same 96-token chat
   reserve as ``synth.combine``) and dropped/backfilled if it is over the gate.
   If the tokenizer is unavailable the recorded tokens are the measurement;
   nothing is ever estimated under the gate or truncated.

Output rows keep the completion, task and all other stream-A columns; only
``prompt_variant="plain"`` and ``harness="none"`` are set, so the mix can keep
them distinct from the ``harness_tools`` tool rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_agent.synth import combine  # noqa: E402

TASKS = ("repair", "design", "audit", "ledger")
SCRIPT_TAG = "build_a_slice.py"
MANIFEST_NAME = "MANIFEST.json"
GATE = 8192
DEFAULT_TOOL_ROWS = 4800
DEFAULT_RATIO = 0.3333
PARAPHRASE_TAG = re.compile(r"#v\d+$")
TOOL_MARKERS = ("<tool_call", "<tool_result")

# Additive-only contract: only a fresh output directory is written. `data/a_slice` is the default
# output, so it is not protected here; the non-empty check below is what stops an overwrite.
PROTECTED = (
    "data/sft_all", "data/grpo_all", "data/sft", "data/grpo",
    "data/sft_tools", "data/grpo_tools", "data/sft_tools_h", "data/grpo_tools_h",
    "data/sft_mix", "data/grpo_mix", "data/sft_mix_h", "data/grpo_mix_h",
    "data/heldout", "data/heldout_v2", "data/frontier",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _resolve(value: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else (ROOT / path)


def _rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plant_key(row: dict) -> str:
    """Ground-truth plant id: stream-A row id without a trailing ``#vN`` tag."""
    value = row.get("id") or row.get("record_id") or ""
    return PARAPHRASE_TAG.sub("", str(value))


def _tokenizer():
    """The real tokenizer when available; None makes :func:`measure_tokens` use
    the stream-A ``tokens`` column instead (the same real-tokenizer measurement,
    recorded by ``synth.combine`` at corpus build time)."""
    return combine.tokenizer()


def _completion_text(row: dict) -> str:
    value = row.get("completion")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def measure_tokens(row: dict, tok=None) -> int | None:
    """Prompt + completion tokens of the rendered row, with the chat reserve.

    Never truncates.  ``None`` means the row has no usable measurement and is
    treated as ineligible instead of being guessed under the gate.
    """
    if tok is not None:
        prompt = "".join(str(m.get("content", ""))
                         for m in row.get("prompt", []) if isinstance(m, dict))
        completion = _completion_text(row)
        return (len(tok.encode(prompt, add_special_tokens=False).ids)
                + len(tok.encode(completion, add_special_tokens=False).ids)
                + combine.TOKEN_RESERVE)
    n = row.get("tokens")
    if isinstance(n, int) and not isinstance(n, bool) and n >= 0:
        return n
    return None


def is_plain_row(row: dict) -> bool:
    """Stream-A plain arm: no harness block and no tool turns in the prompt."""
    if str(row.get("harness") or "none") != "none":
        return False
    prompt = "".join(str(m.get("content", ""))
                     for m in row.get("prompt", []) if isinstance(m, dict))
    return not any(marker in prompt for marker in TOOL_MARKERS)


def heldout_ids(heldout_dir: pathlib.Path | str) -> tuple[set[str], list[str]]:
    """Ids / record ids / plant ids from every ``data/heldout`` jsonl file."""
    directory = _resolve(heldout_dir)
    files = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
    ids: set[str] = set()
    for path in files:
        for row in _read_jsonl(path):
            for key in ("id", "record_id"):
                value = row.get(key)
                if value:
                    ids.add(str(value))
                    ids.add(plant_key({"id": str(value)}))
    return ids, [_rel(p) for p in files]


def _quota(target: int, tasks) -> dict[str, int]:
    """Split the target evenly; the remainder goes to the first tasks."""
    if not tasks:
        return {}
    base, remainder = divmod(target, len(tasks))
    return {task: base + (1 if i < remainder else 0) for i, task in enumerate(tasks)}


def _pct(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))]


def _length_stats(values: list[int]) -> dict:
    return {"n": len(values), "p50": _pct(values, 0.50),
            "p95": _pct(values, 0.95), "max": max(values) if values else None}


SELECTION_RULE = (
    "target=round(tool_rows*ratio); quota=target//T plus 1 for the first "
    "target%T tasks; eligible=plain (harness none/absent, no tool markers) "
    "train rows with exact id/record_id and plant key absent from --heldout "
    "and recorded-token count <= 8192; over-gate rows dropped (never "
    "truncated) and counted, next eligible rows backfill; eligible rows "
    "shuffled with random.Random(seed) in --tasks order and the first row of "
    "each ground-truth plant (id with trailing #vN removed) wins until the "
    "task quota is met; every selected row re-measured with the real Qwen3.5 "
    "tokenizer as sum(prompt message contents)+completion+96 chat reserve and "
    "dropped/backfilled if that measurement is over the gate (stream-A tokens "
    "column only when the tokenizer is unavailable)."
)


# --------------------------------------------------------------------------- #
# planning (read-only)
# --------------------------------------------------------------------------- #
def plan_slice(sft_all: pathlib.Path | str, tool_rows: int = DEFAULT_TOOL_ROWS,
               ratio: float = DEFAULT_RATIO, seed: int = 0, tasks=TASKS,
               heldout: pathlib.Path | str = "data/heldout",
               gate: int = GATE, tok=...) -> dict:
    """Read all inputs and return the planned slice without writing anything.

    ``tok`` defaults to the real tokenizer; pass ``None`` to force the stored
    stream-A ``tokens`` column (used by the tests with tiny fixtures).
    """
    if tool_rows < 0:
        raise SystemExit("--tool-rows must be >= 0")
    if not 0 <= ratio <= 1:
        raise SystemExit("--ratio must be in [0, 1]")
    src = _resolve(sft_all)
    tok = _tokenizer() if tok is ... else tok
    target = round(tool_rows * ratio)
    quota = _quota(target, tasks)
    heldout_set, heldout_files = heldout_ids(heldout)

    selected: dict[str, list[dict]] = {task: [] for task in tasks}
    plants: dict[str, list[str]] = {task: [] for task in tasks}
    tokens: dict[str, list[int]] = {task: [] for task in tasks}
    dropped = {task: 0 for task in tasks}
    heldout_skipped = {task: 0 for task in tasks}
    ineligible = {task: 0 for task in tasks}
    shortfall = {task: 0 for task in tasks}
    input_hashes: dict[str, str] = {}
    missing: list[str] = []
    rng = random.Random(seed)

    for path in heldout_files:
        input_hashes[path] = _sha256_file(_resolve(path))

    for task in tasks:
        path = src / f"{task}_train.jsonl"
        if not path.is_file():
            missing.append(_rel(path))
            shortfall[task] = quota[task]
            continue
        input_hashes[_rel(path)] = _sha256_file(path)
        eligible: list[tuple[dict, str, int]] = []
        for row in _read_jsonl(path):
            if not is_plain_row(row):
                ineligible[task] += 1
                continue
            key = plant_key(row)
            if not key or key in heldout_set or str(row.get("id") or "") in heldout_set:
                heldout_skipped[task] += 1
                continue
            # Cheap pre-filter: stream-A tokens were recorded by the same
            # tokenizer/convention; the rows that can be selected are measured
            # again on the rendered text below.
            n = measure_tokens(row, None)
            if n is None or n > gate:
                dropped[task] += 1
                continue
            eligible.append((row, key, n))

        rng.shuffle(eligible)
        seen: set[str] = set()
        for row, key, _stored in eligible:
            if len(selected[task]) >= quota[task]:
                break
            if key in seen:
                continue
            n = measure_tokens(row, tok)
            if n is None or n > gate:
                dropped[task] += 1
                continue
            seen.add(key)
            out = dict(row)
            out["prompt_variant"] = "plain"
            out["harness"] = "none"
            selected[task].append(out)
            plants[task].append(key)
            tokens[task].append(n)
        shortfall[task] = quota[task] - len(selected[task])

    by_task = {task: len(selected[task]) for task in tasks}
    total = sum(by_task.values())
    all_tokens = [n for task in tasks for n in tokens[task]]
    all_plants = sorted({p for task in tasks for p in plants[task]})
    plan = {
        "generator": SCRIPT_TAG,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "sft_all": _rel(src), "out": None, "tool_rows": tool_rows,
            "ratio": ratio, "seed": seed, "tasks": list(tasks),
            "heldout": str(heldout), "gate": gate,
            "tokenizer": str(combine.tokenizer_path()) if tok is not None else None,
        },
        "target": target, "selected": total, "met_target": total == target,
        "counts_by_task": by_task,
        "quota_by_task": quota,
        "plants_represented_by_task": {task: len(plants[task]) for task in tasks},
        "plants_represented": all_plants,
        "tokens_by_task": {task: tokens[task] for task in tasks},
        "token_length": _length_stats(all_tokens),
        "token_length_by_task": {task: _length_stats(tokens[task]) for task in tasks},
        "dropped_over_gate": sum(dropped.values()),
        "dropped_over_gate_by_task": dropped,
        "heldout_excluded": sum(heldout_skipped.values()),
        "heldout_excluded_by_task": heldout_skipped,
        "ineligible_non_plain": sum(ineligible.values()),
        "shortfall_by_task": shortfall,
        "heldout_files": heldout_files,
        "input_file_hashes": input_hashes,
        "inputs_missing": missing,
        "selection_rule": SELECTION_RULE,
        "rows": selected,
    }
    return plan


def print_stats(plan: dict, stream=None) -> None:
    """Print the per-task table (no writes)."""
    if stream is None:
        stream = sys.stdout
    tasks = plan["config"]["tasks"]
    print(f"{'task':<10} {'quota':>7} {'selected':>9} {'plants':>8} "
          f"{'dropped':>8} {'tokens-p50':>11} {'tokens-p95':>11} {'max':>7}", file=stream)
    for task in tasks:
        st = plan["token_length_by_task"][task]
        print(f"{task:<10} {plan['quota_by_task'][task]:>7} "
              f"{plan['counts_by_task'][task]:>9} "
              f"{plan['plants_represented_by_task'][task]:>8} "
              f"{plan['dropped_over_gate_by_task'][task]:>8} "
              f"{str(st['p50']):>11} {str(st['p95']):>11} {str(st['max']):>7}",
              file=stream)
    st = plan["token_length"]
    print(f"{'TOTAL':<10} {plan['target']:>7} {plan['selected']:>9} "
          f"{len(plan['plants_represented']):>8} {plan['dropped_over_gate']:>8} "
          f"{str(st['p50']):>11} {str(st['p95']):>11} {str(st['max']):>7}", file=stream)
    print(f"seed={plan['config']['seed']} target={plan['target']} "
          f"met_target={plan['met_target']} heldout_excluded={plan['heldout_excluded']}",
          file=stream)


# --------------------------------------------------------------------------- #
# writing (only a fresh output dir)
# --------------------------------------------------------------------------- #
def _check_destination(out: pathlib.Path, manifest: pathlib.Path) -> None:
    for name in PROTECTED:
        if out.resolve() == (ROOT / name).resolve():
            raise SystemExit(f"refusing to write into protected corpus: {out}")
    for dest in (out.resolve(), manifest.resolve().parent):
        if dest in (ROOT.resolve(), (ROOT / "data").resolve()):
            raise SystemExit(f"refusing to write into the repo/data root: {dest}")


def prepare_dir(out: pathlib.Path) -> None:
    """Refuse a non-empty output dir; never delete or overwrite foreign files."""
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to write into non-empty output dir {out}; "
                         f"choose a fresh path")
    out.mkdir(parents=True, exist_ok=True)


def write_slice(plan: dict, out: pathlib.Path) -> dict:
    """Write per-task jsonl and the manifest; return the manifest dict."""
    if not plan["met_target"]:
        raise SystemExit(f"target not met ({plan['selected']}/{plan['target']}); "
                         f"refusing to write a partial A-slice")
    prepare_dir(out)
    owned: list[str] = []
    for task in plan["config"]["tasks"]:
        path = out / f"{task}_train.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in plan["rows"][task]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        owned.append(_rel(path))
    manifest = dict(plan["config"])
    manifest["out"] = _rel(out)
    manifest.update({
        "generator": SCRIPT_TAG,
        "generated_at": plan["generated_at"],
        "target": plan["target"], "selected": plan["selected"],
        "met_target": plan["met_target"],
        "counts_by_task": plan["counts_by_task"],
        "quota_by_task": plan["quota_by_task"],
        "plants_represented_by_task": plan["plants_represented_by_task"],
        "plants_represented": plan["plants_represented"],
        "token_length": plan["token_length"],
        "token_length_by_task": plan["token_length_by_task"],
        "dropped_over_gate": plan["dropped_over_gate"],
        "dropped_over_gate_by_task": plan["dropped_over_gate_by_task"],
        "heldout_excluded": plan["heldout_excluded"],
        "heldout_excluded_by_task": plan["heldout_excluded_by_task"],
        "ineligible_non_plain": plan["ineligible_non_plain"],
        "shortfall_by_task": plan["shortfall_by_task"],
        "heldout_files": plan["heldout_files"],
        "input_file_hashes": plan["input_file_hashes"],
        "inputs_missing": plan["inputs_missing"],
        "selection_rule": plan["selection_rule"],
        "owned_files": owned + [_rel(out / MANIFEST_NAME)],
    })
    manifest_path = out / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-all", default="data/sft_all",
                    help="stream A dir with <task>_train.jsonl (default: data/sft_all)")
    ap.add_argument("--out", default="data/a_slice",
                    help="new A-slice output dir (default: data/a_slice); must be empty")
    ap.add_argument("--tool-rows", type=int, default=DEFAULT_TOOL_ROWS,
                    help=f"R_tool used to derive the target (default: {DEFAULT_TOOL_ROWS})")
    ap.add_argument("--ratio", type=float, default=DEFAULT_RATIO,
                    help=f"A-slice fraction of the final mix (default: {DEFAULT_RATIO})")
    ap.add_argument("--seed", type=int, default=0, help="selection seed (default: 0)")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS),
                    help=f"tasks to slice (default: {' '.join(TASKS)})")
    ap.add_argument("--heldout", default="data/heldout",
                    help="dev-set ids that must never enter the slice (default: data/heldout)")
    ap.add_argument("--gate", type=int, default=GATE,
                    help=f"drop rows over this many rendered tokens (default: {GATE})")
    ap.add_argument("--stats", "--dry-run", dest="stats", action="store_true",
                    help="print the plan and write nothing")
    args = ap.parse_args(argv)

    tasks = tuple(t.strip() for t in args.tasks if t.strip())
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        ap.error(f"unknown task(s): {', '.join(unknown)}; choose from {' '.join(TASKS)}")
    if args.gate <= 0:
        ap.error("--gate must be > 0")

    out = _resolve(args.out)
    manifest_path = out / MANIFEST_NAME
    if not args.stats:
        _check_destination(out, manifest_path)

    plan = plan_slice(args.sft_all, tool_rows=args.tool_rows, ratio=args.ratio,
                      seed=args.seed, tasks=tasks, heldout=args.heldout,
                      gate=args.gate)
    print_stats(plan)
    if plan["inputs_missing"]:
        print(f"WARNING: missing input files: {plan['inputs_missing']}", file=sys.stderr)
    if not plan["met_target"]:
        print(f"WARNING: target not met: {plan['selected']}/{plan['target']}", file=sys.stderr)
    if args.stats:
        return 0

    manifest = write_slice(plan, out)
    print(json.dumps({
        "out": _rel(out), "target": manifest["target"],
        "selected": manifest["selected"], "counts_by_task": manifest["counts_by_task"],
        "dropped_over_gate": manifest["dropped_over_gate"],
        "heldout_excluded": manifest["heldout_excluded"],
        "token_length": manifest["token_length"],
        "input_files_hashed": len(manifest["input_file_hashes"]),
        "manifest": _rel(manifest_path),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
