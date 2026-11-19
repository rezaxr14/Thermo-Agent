"""Render the frozen evaluation protocol from data/heldout_v2/MANIFEST.json.

Deterministic: re-running after a manifest change updates every number in the protocol.
The protocol records BOTH roles: test = ``data/heldout_v2`` (F1), dev = old ``data/heldout``
(214 ids).  Default output is ``docs/EVAL_PROTOCOL_HELD_OUT_DEV.md``; pass ``--out`` to render
to another path (the canonical ``docs/EVAL_PROTOCOL.md`` can be produced that way by the owner
of that file).
"""
from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/heldout_v2/MANIFEST.json"
DEFAULT_OUT = ROOT / "docs/EVAL_PROTOCOL_HELD_OUT_DEV.md"

# Frozen facts of the OLD dev set (docs/EVAL_PROTOCOL.md at freeze, eval_set_version below).
DEV_HASH = "8b409359e7a2dd441d0be99c89c00f0f3c61320f9d06e9695cc22569f8ef4c6e"
DEV_CELLS = (
    ("repair", "test_id", 6), ("repair", "test_ood_fault", 70), ("repair", "test_ood_topology", 15),
    ("audit", "test_id", 16), ("audit", "test_ood_topology", 6),
    ("ledger", "test_id", 11), ("ledger", "test_ood_fault", 32), ("ledger", "test_ood_topology", 7),
    ("design", "test_id", 4), ("design", "test_ood_climate", 20),
    ("design", "test_ood_load", 13), ("design", "test_ood_scale", 14),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = pathlib.Path(args.out)

    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cells = m["cells"]
    total = sum(v["rows"] for v in cells.values())
    tasks = ("repair", "audit", "ledger", "design")

    lines: list[str] = []
    A = lines.append
    A("# Evaluation protocol — FROZEN test set `heldout_v2` (dev: `data/heldout`)")
    A("")
    A(f"Status: **frozen {m['created_utc']}**. This file is generated from "
      f"`data/heldout_v2/MANIFEST.json` by `scripts/finalize_heldout_v2_protocol.py`; do not edit by hand. "
      f"It supersedes the test-set role of `docs/EVAL_PROTOCOL.md`; the old dev-set protocol's cell "
      f"counts are preserved in §6.")
    A("")
    A("## 1. The two evaluation sets")
    A("")
    A(f"- **TEST = `data/heldout_v2`**, seed **{m['seed']}** (DECISIONS F1 / plan 0.6). "
      f"Every reported paper number uses this set. It contains test files only — no `train`/`val` "
      f"row is written into it and none may be scored or used as a GRPO prompt.")
    A(f"- **DEV = `data/heldout`** (seed 987654, 214 ids). Tuning/model selection only — never a "
      f"reported paper number. Its frozen version hash is `{DEV_HASH}` (details in §6).")
    A("")
    A(f"Test JSONL files: **{len(m['jsonl_sha256'])}** plus `MANIFEST.json`; scored rows: "
      f"**{total}** across **{len(cells)}** reported (task, split) cells.")
    A(f"Test-set hash (`meta.eval_set_version`): `{m['set_hash']}`")
    A(f"Hash rule: {m['set_hash_rule']}")
    A("")
    A("Verify with:")
    A("")
    A("```powershell")
    A("cd C:\\Users\\rezax\\Home\\Code\\thermo_agent\\Repo")
    A("@'")
    A("import hashlib, json, pathlib")
    A("d = pathlib.Path('data/heldout_v2')")
    A("h = hashlib.sha256()")
    A("for p in sorted(d.glob('*')):")
    A("    if not p.is_file() or p.name == 'MANIFEST.json': continue")
    A("    h.update(p.name.encode()); h.update(p.read_bytes())")
    A("print(h.hexdigest())")
    A("'@ | .venv/Scripts/python.exe -")
    A("```")
    A("")
    A("## 2. Scored cells, counts and difficulty tiers")
    A("")
    A("| Task | Split | Rows | Tier counts |")
    A("|---|---|---:|---|")
    for task in tasks:
        for key, v in cells.items():
            if not key.startswith(task + "/"):
                continue
            split = key.split("/", 1)[1]
            tiers = ", ".join(f"{k} {n}" for k, n in sorted(v["tiers"].items()))
            A(f"| {task} | `{split}` | {v['rows']} | {tiers} |")
    A("")
    A("The `tier` field is REAL and derived from the documented per-task size measure in "
      "`thermo_agent/synth/tiers.py` — the same measure used for training. Per task, all rows "
      "(pooled across that task's splits) are ranked by that measure, ties broken by the sha256 of "
      "the row id, and cut into `floor(n/3)`, `floor(n/3)`, `remainder` for easy/medium/hard "
      "(remainder to hard). No random tiers, no assistant-chosen threshold.")
    A("")
    A("| Task | Measure | n | easy | medium | hard | easy range | medium range | hard range |")
    A("|---|---|---:|---:|---:|---:|---|---|---|")
    for task in tasks:
        t = m["tiers"][task]
        c = t["counts"]
        r = t["ranges"]
        def rng(x):
            return "—" if x["n"] == 0 else f"[{x['min']}, {x['max']}]"
        A(f"| {task} | {t['measure']} | {t['n']} | {c['easy']} | {c['medium']} | {c['hard']} | "
          f"{rng(r['easy'])} | {rng(r['medium'])} | {rng(r['hard'])} |")
    A("")
    A("Hard = the largest third by the measured size for each task. For repair/audit this is the "
      "19-component `big_cchp_steam2p_reheat_hp` cycle where the generators support it; design and "
      "ledger have no big generator, so their hard tier is the largest available measured size. "
      "The exact counts and ranges above are frozen in the manifest.")
    A("")
    A("## 3. Merged cells")
    A("")
    if m.get("merged_cells"):
        A("Some cells could not reach the target from a single generator. They were merged only "
          "within a task and with the same success metric; no rows were fabricated.")
        A("")
        A("| Task | Members | Label | Rows | Reason |")
        A("|---|---|---|---:|---|")
        for g in m["merged_cells"]:
            A(f"| {g['task']} | {', '.join('`'+x+'`' for x in g['members'])} | {g['label']} | "
              f"{g['rows']} | {g['reason']} |")
    else:
        A("**None.** Every reported cell reached the "
          f"{m['target_rows_per_cell']}-row target directly from a generator; no cells were pooled. "
          "Split families that do not exist in a generator (audit `test_ood_fault`, design topology "
          "families) are not cells and were not merged.")
    A("")
    A("## 4. Scoring rule")
    A("")
    A(f"- **`LIMIT = 0`**. `model_eval.load_rows` returns all rows when `limit` is falsy, so every "
      f"row of every file in §2 is scored. No truncation, no sampling, no per-arm cap (DECISIONS F2).")
    A("- One identical `DATA_DIRS=heldout_v2`, one identical split list and one identical `LIMIT=0` "
      "for every arm (DECISIONS F4). An arm whose split ids or per-split `n` differ from §2 is "
      "invalid: discard and re-run it.")
    A("- Files are read separately by `model_eval.load_rows`; concatenating splits into one "
      "reported cell is a mix and is forbidden, except the explicitly labelled merges in §3.")
    A(f"- Reports must record `meta.eval_set_version = {m['set_hash']}` and the per-split "
      "`n`. Missing or mismatched values void the report.")
    A("- Decoding is the frozen DECISIONS table: greedy primary; pass@8 secondary "
      "(T 0.6, top_p 0.95, top_k 20, seeds 0-7); no repetition/presence penalty; "
      "4,096 new tokens per turn, 8 tool calls, 6 turns, max_length 8,192.")
    A("")
    A("## 5. Contamination gate")
    A("")
    c = m["contamination"]
    A(f"- Result: **{'PASS' if c['passed'] else 'FAIL'}** — overlap **{c['overlap']}** (zero required).")
    A(f"- Index: {c['heldout_v2_rows_indexed']} rows, {c['signatures']} "
      f"plant/site/spec/ledger/id signatures.")
    A(f"- Scanned {c['files_scanned']} files / {c['lines_scanned']} lines in "
      f"{', '.join('`'+d+'`' for d in c['target_dirs'])}.")
    A(f"- Self-check (planted-row detection): {c['self_check_detects_planted_row']}.")
    A("- `data/heldout` is the dev set, not a training dir; it is excluded from the scan by design. "
      "A failure voids every report scored on this set; regenerate v2 with a new seed, never delete "
      "training data.")
    A("")
    A("## 6. Dev set (`data/heldout`, 214 ids) — tuning only")
    A("")
    A(f"- Eval-set version: `{DEV_HASH}` (12 test files, 1,531,809 bytes).")
    A("- These counts describe the DEV set only; they are not test cells and are never used for a "
      "reported paper number.")
    A("")
    A("| Task | Split | Rows |")
    A("|---|---|---:|")
    for task, split, n in DEV_CELLS:
        A(f"| {task} | `{split}` | {n} |")
    A(f"| **all** | | **{sum(n for _t, _s, n in DEV_CELLS)}** |")
    A("")
    A("## 7. Generation provenance")
    A("")
    A(f"Seed {m['seed']}, {m['workers']} workers, `--max-evals {m['max_evals']}`. "
      f"Total generation + freeze time {m.get('generation_seconds')} s. "
      f"Generation command: `{m['generation_command_str']}`. "
      f"Every generator invocation is listed with its full command, requested n and yield in "
      f"`data/heldout_v2/MANIFEST.json` under `generator_runs`; cells were topped up from real "
      f"generator batches until they met the {m['target_rows_per_cell']}-row target, and per-cell "
      f"yields before/after top-up are in the manifest. Non-test generator rows are dropped "
      f"(counted in `dropped_non_test_rows`).")
    A("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines, set_hash={m['set_hash']})")


if __name__ == "__main__":
    main()
