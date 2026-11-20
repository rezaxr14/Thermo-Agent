#!/usr/bin/env python3
"""Zero-cost preflight: prove a batch would produce rows BEFORE any model call.

Every batch failure so far came from a shard the hosted runner would filter to
zero rows (stale dataset copy, missing tier field, or tier balancing against an
empty tier). Each one was only visible in the Kaggle log *after* a run had been
scheduled. This script replays the runner's own selection code locally, so a bad
batch is rejected for free.

It uses the EXACT runner module (`notebooks/kaggle_model_runner.py`) rather than a
copy of its rules, so the two cannot drift.

Usage
    python scripts/preflight_budget_batch.py                 # local shards
    python scripts/preflight_budget_batch.py --dataset       # download from Kaggle and check those bytes
    python scripts/preflight_budget_batch.py --model gpt-6-luna
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data" / "frontier_tools_cohort_prompts" / "harness_config.json"
PROMPTS = CONFIG.parent
RUNNER = ROOT / "notebooks" / "kaggle_model_runner.py"
KAGGLE = ROOT / "scripts" / "kaggle_project.ps1"
HARD_BLOCKED = ("gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini")
LEAK_FIELDS = ("ref_spec", "ref_evals", "ref_fuel_w", "completion", "sft_completion")


def load_runner():
    spec = importlib.util.spec_from_file_location("runner_under_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing shard: {path}")
    rows = []
    for lineno, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            raise SystemExit(f"{path.name}:{lineno}: invalid JSON ({exc})")
    return rows


def check_model(runner, cfg, model: str, base: pathlib.Path) -> dict:
    problems: list[str] = []
    shard = cfg.shard_for(model)
    path = base / shard
    if not path.is_file():
        return {"model": model, "shard": shard, "problems": [f"shard not found: {path}"]}

    rows = read_jsonl(path)
    if not rows:
        problems.append("shard is empty")

    ids = [str(r.get("id") or "") for r in rows]
    if any(not rid for rid in ids):
        problems.append(f"{sum(1 for i in ids if not i)} rows have no id")
    duplicates = [rid for rid, n in collections.Counter(ids).items() if n > 1]
    if duplicates:
        problems.append(f"duplicate ids: {duplicates[:3]}")

    leaked = sorted({field for row in rows for field in LEAK_FIELDS if field in row})
    if leaked:
        problems.append(f"answer-bearing fields present in shard: {leaked}")

    selected = cfg.select_rows(rows)
    tiers = collections.Counter(cfg.tier_of(r) for r in selected)
    if not selected:
        # This is the exact failure that produced 0-byte journals: the runner
        # journals __error__ and never calls the model.
        problems.append(
            "runner selection yields 0 rows "
            f"(tiers in shard: {dict(collections.Counter(cfg.tier_of(r) for r in rows))})")
    unknown = [r for r in rows if cfg.tier_of(r) not in cfg.tiers]
    if unknown:
        problems.append(f"{len(unknown)} rows have no usable tier/cell")

    if cfg.limit and len(selected) > cfg.limit:
        problems.append(f"selection {len(selected)} exceeds limit {cfg.limit}")

    return {"model": model, "shard": shard, "rows": len(rows),
            "selected": len(selected), "tiers": dict(tiers), "problems": problems}


def check_config(cfg) -> list[str]:
    problems: list[str] = []
    for blocked in HARD_BLOCKED:
        if any(blocked in m for m in cfg.allowed_models):
            problems.append(f"hard-blocked model allowed: {blocked}")
    if not cfg.allowed_models:
        problems.append("no allowed_models configured")
    if cfg.rounds <= 0:
        problems.append("rounds must be > 0")
    if cfg.k_memory < 0:
        problems.append("k_memory must be >= 0")
    return problems


def download_dataset_files(base: pathlib.Path, files: list[str]) -> None:
    """Fetch only the config + shard files (no model call, no training data)."""
    for name in files:
        rel = f"data/frontier_tools_cohort_prompts/{name}"
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-File", str(KAGGLE), "datasets", "download",
             "rezanadimi/thermo-agent-data", "--file", rel, "--path", str(base), "--unzip"],
            cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"dataset download failed for {rel}:\n{proc.stdout}\n{proc.stderr}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--dataset", action="store_true",
                        help="validate the bytes in the Kaggle dataset, not the local shards")
    parser.add_argument("--model", action="append", choices=["gpt-6-luna", "claude-haiku-5-5"])
    args = parser.parse_args(argv)

    config_path = pathlib.Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"missing harness config: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    runner = load_runner()
    cfg = runner.Config.from_dict(raw)
    models = args.model or list(cfg.allowed_models)

    problems = check_config(cfg)
    base = config_path.parent
    if args.dataset:
        files = [config_path.name] + [cfg.shard_for(m) for m in models]
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="preflight_"))
        download_dataset_files(tmp, files)
        base = tmp / "data" / "frontier_tools_cohort_prompts"

    results = [check_model(runner, cfg, m, base) for m in models]
    print(f"preflight: config={config_path if not args.dataset else 'dataset copy'} "
          f"mode={cfg.mode} rounds={cfg.rounds} k_memory={cfg.k_memory} "
          f"limit={cfg.limit} tiers={list(cfg.tiers)}")
    for result in results:
        print(f"  {result['model']}: shard={result.get('shard')} rows={result.get('rows')} "
              f"selected={result.get('selected')} tiers={result.get('tiers')}")
        for problem in result["problems"]:
            print(f"    FAIL {problem}")

    failed = bool(problems) or any(r["problems"] for r in results)
    for problem in problems:
        print(f"  FAIL {problem}")
    print("PREFLIGHT " + ("FAILED" if failed else "OK"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
