#!/usr/bin/env python3
"""Audit the F4 Z/H prompt separation of every arm in ``kaggle/eval_runner.py``.

The runner is notebook source that executes Kaggle bootstrap at import time, so its
``SPECS`` dict is read with ``ast`` (never imported or executed). For each arm the
audit renders the *same* prompt the evaluator builds -- compact harness for H/legacy
arms with tools, compact harness without the tool contract for harness-text arms, a
plain prompt for Z -- and checks what the model actually sees:

* ``mode: Z``  -> no harness marker and no tool contract in the prompt.
* ``mode: H``  -> both the harness marker and the tool contract are present.
* legacy/placeholder arms -> the prompt matches their declared ``harness`` /
  ``tool_access`` (harness text iff harness != none, contract iff tools).

A mismatch is a hard error (nonzero exit): a mislabeled arm would silently corrupt
the paper's Z-vs-H contrast.

    .venv/Scripts/python.exe scripts/audit_prompt_separation.py
    .venv/Scripts/python.exe scripts/audit_prompt_separation.py --runner path/to/eval_runner.py --out audit.json
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_agent.agent import tool_prompt  # noqa: E402
from thermo_agent.eval import model_eval as ME  # noqa: E402

DEFAULT_RUNNER = ROOT / "kaggle" / "eval_runner.py"

# The probe task has a selected skill ("exergy-audit"), so a harness-text prompt can
# be recognised without relying on component skills or a real plant.
PROBE_TASK = "audit"
PROBE_MESSAGES = [{"role": "system", "content": "You are a plant engineer."},
                  {"role": "user", "content": "Audit the plant."}]

# The compact and the historical full harness both start with this heading; both
# tool-contract builders carry these four strings.
HARNESS_MARKERS = ("# AGENTS.md",)
TOOL_MARKERS = ("MULTI-TURN TOOL ENVIRONMENT", "inspect_plant", "answer-oracle")

_UNSET = object()
_REQUIRED_FIELDS = ("backend", "harness", "tool_access", "zero_shot")


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _UNSET  # env vars / paths: not needed for prompt separation


def read_specs(runner: pathlib.Path) -> dict[str, dict[str, Any]]:
    """Parse the SPECS dict literal out of the runner source without importing it."""
    tree = ast.parse(runner.read_text(encoding="utf-8"), filename=str(runner))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SPECS" for t in node.targets):
            if not isinstance(node.value, ast.Dict):
                raise ValueError("SPECS must be a dict literal")
            specs = {}
            for key, val in zip(node.value.keys, node.value.values):
                name = _literal(key)
                if not isinstance(name, str):
                    continue
                if not isinstance(val, ast.Dict):
                    raise ValueError(f"SPECS[{name}] must be a dict literal")
                specs[name] = {_literal(fk): _literal(fv) for fk, fv in zip(val.keys, val.values)}
            return specs
    raise ValueError(f"SPECS assignment not found in {runner}")


def prompt_for(spec: dict[str, Any]) -> str:
    """Render exactly the messages the evaluator would hand to this arm's generator."""
    mode = ME.normalize_harness(spec.get("harness"))
    tools = spec.get("tool_access") == "tools"
    if tools:
        # ToolLoop._base_messages: compact harness + tool contract.
        msgs = tool_prompt.build_harness_tool_messages(PROBE_MESSAGES, PROBE_TASK, None, mode=mode)
    else:
        # _wrap: compact harness WITHOUT the contract (the no-tools arm).
        msgs = tool_prompt.build_compact_harness_messages(
            PROBE_MESSAGES, PROBE_TASK, None, mode=mode, include_contract=False)
    return "\n".join(str(m.get("content", "")) for m in msgs)


def audit(specs: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows, failures = [], []
    for name, spec in specs.items():
        mode = spec.get("mode", _UNSET)
        missing = [f for f in _REQUIRED_FIELDS if f not in spec or spec[f] is _UNSET]
        if missing:
            failures.append(f"{name}: missing required spec field(s): {', '.join(missing)}")
        if mode not in ("Z", "H", "legacy", "placeholder"):
            failures.append(f"{name}: unknown F4 mode {mode!r} (expected Z/H/legacy/placeholder)")
        prompt = prompt_for(spec)
        has_harness = all(m in prompt for m in HARNESS_MARKERS)
        has_contract = all(m in prompt for m in TOOL_MARKERS)

        if mode == "Z":
            expect_harness = expect_contract = False
            problems = []
            if has_harness:
                problems.append("Z prompt contains the harness marker")
            if has_contract:
                problems.append("Z prompt contains the tool contract")
        elif mode == "H":
            expect_harness = expect_contract = True
            problems = []
            if not has_harness:
                problems.append("H prompt is missing the harness marker")
            if not has_contract:
                problems.append("H prompt is missing the tool contract")
        else:
            # Legacy/placeholder: check the prompt against the declared config.
            normalized = ME.normalize_harness(spec.get("harness"))
            expect_harness = normalized != "none"
            expect_contract = spec.get("tool_access") == "tools"
            problems = []
            if has_harness != expect_harness:
                problems.append(f"harness marker present={has_harness}, expected {expect_harness}")
            if has_contract != expect_contract:
                problems.append(f"tool contract present={has_contract}, expected {expect_contract}")

        status = "OK" if not problems else "FAIL: " + "; ".join(problems)
        if problems:
            failures.append(f"{name}: {status}")
        rows.append({"arm": name, "mode": mode if isinstance(mode, str) else str(mode),
                     "backend": spec.get("backend"), "harness": spec.get("harness"),
                     "tool_access": spec.get("tool_access"), "zero_shot": spec.get("zero_shot"),
                     "harness_marker": has_harness, "tool_contract": has_contract,
                     "expect_harness": expect_harness, "expect_tool_contract": expect_contract,
                     "status": status})
    return rows, failures


def print_table(rows: list[dict[str, Any]]) -> None:
    header = (f"{'arm':>24} {'mode':>11} {'backend':>8} {'harness':>14} {'tools':>6} "
              f"{'H.mark':>7} {'contract':>8}  status")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['arm']:>24} {str(r['mode']):>11} {str(r['backend']):>8} "
              f"{str(r['harness']):>14} {str(r['tool_access']):>6} "
              f"{str(r['harness_marker']):>7} {str(r['tool_contract']):>8}  {r['status']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runner", type=pathlib.Path, default=DEFAULT_RUNNER,
                    help="path to eval_runner.py (default: this repo's kaggle/eval_runner.py)")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="optional JSON report path")
    args = ap.parse_args(argv)

    try:
        specs = read_specs(args.runner)
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"[audit] cannot read SPECS from {args.runner}: {exc}", file=sys.stderr)
        return 2
    rows, failures = audit(specs)
    print_table(rows)
    print(f"\n[audit] {len(rows)} arm(s), {len(failures)} mismatch(es)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"runner": str(args.runner), "rows": rows,
                                        "failures": failures}, indent=2) + "\n", encoding="utf-8")
        print(f"[audit] wrote {args.out}")
    if failures:
        for f in failures:
            print("  FAIL " + f, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
