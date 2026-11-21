# Evaluation protocol — FROZEN test set `heldout_v2` (dev: `data/heldout`)

Status: **frozen 2026-11-19T18:26:51+00:00**. This file is generated from `data/heldout_v2/MANIFEST.json` by `scripts/finalize_heldout_v2_protocol.py`; do not edit by hand. It supersedes the test-set role of `docs/EVAL_PROTOCOL.md`; the old dev-set protocol's cell counts are preserved in §6.

## 1. The two evaluation sets

- **TEST = `data/heldout_v2`**, seed **20261119** (DECISIONS F1 / plan 0.6). Every reported paper number uses this set. It contains test files only — no `train`/`val` row is written into it and none may be scored or used as a GRPO prompt.
- **DEV = `data/heldout`** (seed 987654, 214 ids). Tuning/model selection only — never a reported paper number. Its frozen version hash is `8b409359e7a2dd441d0be99c89c00f0f3c61320f9d06e9695cc22569f8ef4c6e` (details in §6).

Test JSONL files: **12** plus `MANIFEST.json`; scored rows: **3476** across **12** reported (task, split) cells.
Test-set hash (`meta.eval_set_version`): `3cc51773128724b1dbb2e21a70a3b4d24da86f3231adaed312b797677211054a`
Hash rule: sha256 over sorted(file_name_bytes + file_bytes) for every file in data/heldout_v2 except MANIFEST.json

Verify with:

```powershell
cd C:\Users\rezax\Home\Code\thermo_agent\Repo
@'
import hashlib, json, pathlib
d = pathlib.Path('data/heldout_v2')
h = hashlib.sha256()
for p in sorted(d.glob('*')):
    if not p.is_file() or p.name == 'MANIFEST.json': continue
    h.update(p.name.encode()); h.update(p.read_bytes())
print(h.hexdigest())
'@ | .venv/Scripts/python.exe -
```

## 2. Scored cells, counts and difficulty tiers

| Task | Split | Rows | Tier counts |
|---|---|---:|---|
| repair | `test_id` | 130 | easy 21, hard 44, medium 65 |
| repair | `test_ood_fault` | 797 | easy 173, hard 285, medium 339 |
| repair | `test_ood_topology` | 289 | easy 211, hard 77, medium 1 |
| audit | `test_id` | 254 | easy 28, hard 86, medium 140 |
| audit | `test_ood_topology` | 226 | easy 132, hard 74, medium 20 |
| ledger | `test_id` | 150 | easy 26, hard 66, medium 58 |
| ledger | `test_ood_fault` | 501 | easy 84, hard 170, medium 247 |
| ledger | `test_ood_topology` | 303 | easy 208, hard 82, medium 13 |
| design | `test_id` | 112 | easy 19, hard 22, medium 71 |
| design | `test_ood_climate` | 316 | easy 108, hard 93, medium 115 |
| design | `test_ood_load` | 202 | easy 99, hard 103 |
| design | `test_ood_scale` | 196 | easy 49, hard 58, medium 89 |

The `tier` field is REAL and derived from the documented per-task size measure in `thermo_agent/synth/tiers.py` — the same measure used for training. Per task, all rows (pooled across that task's splits) are ranked by that measure, ties broken by the sha256 of the row id, and cut into `floor(n/3)`, `floor(n/3)`, `remainder` for easy/medium/hard (remainder to hard). No random tiers, no assistant-chosen threshold.

| Task | Measure | n | easy | medium | hard | easy range | medium range | hard range |
|---|---|---:|---:|---:|---:|---|---|---|
| repair | number of components in plant_broken | 1216 | 405 | 405 | 406 | [3, 5] | [5, 7] | [7, 19] |
| audit | number of components in plant | 480 | 160 | 160 | 160 | [3, 5] | [5, 8] | [8, 19] |
| ledger | number of components in ledger.data.topo | 954 | 318 | 318 | 318 | [3, 5] | [5, 7] | [7, 13] |
| design | number of components of the reference design built from site + ref_spec | 826 | 275 | 275 | 276 | [4, 10] | [10, 11] | [11, 13] |

Hard = the largest third by the measured size for each task. For repair/audit this is the 19-component `big_cchp_steam2p_reheat_hp` cycle where the generators support it; design and ledger have no big generator, so their hard tier is the largest available measured size. The exact counts and ranges above are frozen in the manifest.

## 3. Merged cells

**None.** Every reported cell reached the 85-row target directly from a generator; no cells were pooled. Split families that do not exist in a generator (audit `test_ood_fault`, design topology families) are not cells and were not merged.

## 4. Scoring rule

- **`LIMIT = 0`**. `model_eval.load_rows` returns all rows when `limit` is falsy, so every row of every file in §2 is scored. No truncation, no sampling, no per-arm cap (DECISIONS F2).
- One identical `DATA_DIRS=heldout_v2`, one identical split list and one identical `LIMIT=0` for every arm (DECISIONS F4). An arm whose split ids or per-split `n` differ from §2 is invalid: discard and re-run it.
- Files are read separately by `model_eval.load_rows`; concatenating splits into one reported cell is a mix and is forbidden, except the explicitly labelled merges in §3.
- Reports must record `meta.eval_set_version = 3cc51773128724b1dbb2e21a70a3b4d24da86f3231adaed312b797677211054a` and the per-split `n`. Missing or mismatched values void the report.
- Decoding is the frozen DECISIONS table: greedy primary; pass@8 secondary (T 0.6, top_p 0.95, top_k 20, seeds 0-7); no repetition/presence penalty; 4,096 new tokens per turn, 8 tool calls, 6 turns, max_length 8,192.

## 5. Contamination gate

- Result: **PASS** — overlap **0** (zero required).
- Index: 3476 rows, 10578 plant/site/spec/ledger/id signatures.
- Scanned 298 files / 454302 lines in `data\_pre_leakfix`, `data\_pre_leakfix\processed`, `data\_pre_leakfix\processed_arch`, `data\a_slice`, `data\frontier`, `data\frontier_eval_export`, `data\frontier_export`, `data\frontier_prov`, `data\frontier_tools`, `data\frontier_tools_clean`, `data\frontier_tools_clean_v2`, `data\frontier_tools_pilot\claude-haiku-5-5`, `data\frontier_tools_pilot\gpt-6-luna`, `data\grpo`, `data\grpo_all`, `data\grpo_tools`, `data\grpo_tools_h`, `data\grpo_tools_v2`, `data\grpo_tools_v2_h`, `data\kaggle_datasets\teacher_big_prompts`, `data\kaggle_datasets\thermo-agent-eval-replays`, `data\memory`, `data\processed`, `data\processed_arch`, `data\processed_big`, `data\provenance`, `data\sft`, `data\sft_all`, `data\sft_tools`, `data\sft_tools_h`, `data\sft_tools_v2`, `data\sft_tools_v2_h`, `data\teacher_runs`, `data\teacher_runs\kaggle_hosted\budget_pilot\claude-haiku-5-5\thermo-teacher-budget-pilot-s00\2\claude-haiku-5-5\7218726`, `data\teacher_runs\kaggle_hosted\budget_pilot\gpt-6-luna\thermo-teacher-budget-pilot-s00\2\gpt-6-luna\7218725`, `data\teacher_runs\kaggle_hosted\budget_pilot\proof\claude-haiku-5-5\thermo-teacher-budget-pilot-s00\1\claude-haiku-5-5\7214497`, `data\teacher_runs\kaggle_hosted\budget_pilot\proof\gpt-6-luna\thermo-teacher-budget-pilot-s00\1\gpt-6-luna\7214496`, `data\teacher_runs\kaggle_hosted\gpt-6-luna\thermo-teacher-tools-s00\5\gpt-6-luna\7179244`, `data\teacher_runs\kaggle_hosted\thermo-teacher-big-s00\2\claude-sonnet-5-5-default\7086306`, `data\teacher_runs\kaggle_hosted\thermo-teacher-big-s00\2\gemini-3.8-flash\7085914`, `data\teacher_runs\kaggle_hosted\thermo-teacher-tools-s00\6\gemini-3.8-flash\7179338`.
- Self-check (planted-row detection): True.
- `data/heldout` is the dev set, not a training dir; it is excluded from the scan by design. A failure voids every report scored on this set; regenerate v2 with a new seed, never delete training data.

## 6. Dev set (`data/heldout`, 214 ids) — tuning only

- Eval-set version: `8b409359e7a2dd441d0be99c89c00f0f3c61320f9d06e9695cc22569f8ef4c6e` (12 test files, 1,531,809 bytes).
- These counts describe the DEV set only; they are not test cells and are never used for a reported paper number.

| Task | Split | Rows |
|---|---|---:|
| repair | `test_id` | 6 |
| repair | `test_ood_fault` | 70 |
| repair | `test_ood_topology` | 15 |
| audit | `test_id` | 16 |
| audit | `test_ood_topology` | 6 |
| ledger | `test_id` | 11 |
| ledger | `test_ood_fault` | 32 |
| ledger | `test_ood_topology` | 7 |
| design | `test_id` | 4 |
| design | `test_ood_climate` | 20 |
| design | `test_ood_load` | 13 |
| design | `test_ood_scale` | 14 |
| **all** | | **214** |

## 7. Generation provenance

Seed 20261119, 12 workers, `--max-evals 60`. Total generation + freeze time 3637.5 s. Generation command: `C:\Users\rezax\Home\Code\thermo_agent\Repo\.venv\Scripts\python.exe scripts/build_heldout_v2.py --seed 20261119 --out data/heldout_v2 --workers 12 --max-evals 60 --target 85`. Every generator invocation is listed with its full command, requested n and yield in `data/heldout_v2/MANIFEST.json` under `generator_runs`; cells were topped up from real generator batches until they met the 85-row target, and per-cell yields before/after top-up are in the manifest. Non-test generator rows are dropped (counted in `dropped_non_test_rows`).
