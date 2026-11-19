# Hosted teacher prompt audit — 2026-11-19

Scope: the dataset actually read by hosted task `thermo-teacher-tools-s00`
(`kaggle/teacher_harness_task_s00.py`, `SPEC["dataset"]=rezanadimi/thermo-teacher-prompts-big`,
`SPEC["shard"]=prompts_s00.jsonl`, lines 23–34).

**Path note (verified):** there is no local `data/kaggle_datasets/thermo-teacher-prompts-big/`.
That string is the *Kaggle dataset id*. The local staging dir is
`data/kaggle_datasets/teacher_big_prompts/` (`scripts/make_hosted_prompts.py:28–30`,
`kaggle/teacher_harness_task_s00.py:201`). All shards were produced by
`scripts/make_hosted_prompts.py`, per `data/kaggle_datasets/teacher_big_prompts/manifest.json:2`
(`"generator": "scripts/make_hosted_prompts.py"`). The older compact generator
`scripts/make_teacher_big_shards.py` (SHARD_SIZE=200, `system`/`user` strings only) is **not** what
is on disk now.

---

## 1. Shard shape and inputs-only property

| Item | Value | Status |
|---|---|---|
| Shards | 5 (`prompts_s00.jsonl` … `prompts_s04.jsonl`) | verified |
| Rows / shard | 320, 320, 320, 320, 320 | verified |
| Total rows | 1,600 (400 repair / 400 audit / 400 ledger / 400 design) | verified |
| Message roles | system=1600, user=1600 (one each per row) | verified |
| Duplicate ids | 0 (within and across shards) | verified |

Observed keys (counts = rows carrying key):
`id` 1600, `split` 1600, `task` 1600, `prompt` 1600, `site` 1200, `topology` 1200,
`plant_broken` 400, `plant` 400, `ledger` 400. Per-task keysets match `FIELD_ALLOWLIST`
(`scripts/make_hosted_prompts.py:39–44`) exactly.

Forbidden/answer fields present in shards:
`completion`, `plant_original`, `min_edits`, `deducible`, `detected`, `ed_json`, `fuel_w`,
`net_w`, `best_gain_w`, `n_candidates`, `ref_spec`, `ref_fuel_w`, `n_valid`, `ref_evals`,
`fault_type`, `fault_types`, `validation`, `trajectory`, `answer`, `sft_completion`,
`verdict`, `graded`, `usage`, `model` — **0 occurrences (verified)**.

Independent answer-*value* leak check (replicating `make_hosted_prompts.py:121–149` against
`data/frontier_export`): **0 hits across all 1,600 rows (verified)**. Manifest also records
`leak_check.result = PASS`, `checked_rows = 1600`
(`data/kaggle_datasets/teacher_big_prompts/manifest.json:94–96`).

Conclusion: **every row is inputs-only — verified.**

## 2. Shard ids vs `data/frontier_export`

| Measure | Count | Status |
|---|---|---|
| `frontier_export` unique ids (4 × `*_train.jsonl`) | 1,600 | verified |
| Shard unique ids | 1,600 | verified |
| Missing (in export, not in shards) | 0 | verified |
| Extra (in shards, not in export) | 0 | verified |
| Duplicates | 0 | verified |
| Exact set equality | **True** | verified |

Source hashes in `manifest.json:136–153` match the live export files (all 4 MATCH, verified).
Conclusion: **exact 1:1 id correspondence — verified.**

## 3. Does the hosted task read heldout or answer fields?

**`data/heldout`: NOT read — verified.** A grep of `kaggle/teacher_harness_task_s00.py` finds no
`heldout` reference; the only data-root references are:

- `_find_shard()` — `kaggle/teacher_harness_task_s00.py:196–212`: searches `/kaggle/input`,
  bundle root, `.`, then `root/data/kaggle_datasets/teacher_big_prompts` (line 201).
- `_find_export_dir()` — lines 215–226: searches for `repair_train.jsonl` + all four
  `*_train.jsonl` under `/kaggle/input`/root, else `root/data/frontier_export` (lines 223–225).
- `_load_rows()` — lines 278–290: **prefers** `load_exports` when such a dir is found, and only
  falls back to the shard (`_find_shard`, line 288) when no export dir / no rows.

On Kaggle, the two attached datasets are `thermo-teacher-prompts-big` and `thermo-agent-code`;
`kaggle_bundle.zip` contains `data/memory/*` but **no `data/frontier_export` and no `heldout`**
(verified by listing the 178 zip entries). Therefore `_find_export_dir()` returns `None` on
Kaggle and the task reads `prompts_s00.jsonl` via `_find_shard()` → `shard` source path.

**Answer fields:** the shard rows carry none (§1). The task does name answer-derived columns in
`GRADABLE_KEYS` (lines 54–60: `plant_original`, `ed_json`, `fuel_w`, `net_w`) but only to test
gradability in `_row_is_gradable()` (lines 337–342); since shards lack them, rows get
`DEFERRED_VERDICT` and local `transfer.ingest` re-grades by id (lines 345–368, 475–484).
No ground-truth answer is ever loaded from the shard.

**Caveat (verified, local-only):** when a `frontier_export` dir *is* reachable (local dev:
`data/frontier_export` exists), `_load_rows()` lines 280–287 returns **full export rows** via
`transfer.load_exports` (`thermo_agent/teacher/transfer.py:99–104`, which copies every column —
those rows *do* contain `plant_original`, `ed_json`, `fuel_w`, `ref_spec`, `fault_types`, etc.).
This path is not reachable with the Kaggle bundle as configured. Also
`build_context()` parses `plant_original` if present (`thermo_agent/teacher/agent_graph.py:87`),
irrelevant for shards.

## 4. Prompt text / tool contract / bookkeeping

Scan of all 1,600 row `prompt` message contents (verified counts):

| Pattern | Occurrences |
|---|---|
| `<tool_call` | 0 |
| `tool_output` / `tool result` | 0 |
| `model_id` / `total_cost` / `nanodollar` | 0 |
| `timestamp` / `record_id` | 0 |
| `AGENTS.md` | 321 |
| `system` (as literal word) | 91 |

What is actually present:
- Each row is just a `system` instruction (task rules + answer XML-tag format) and a `user`
  message with the physical inputs (`SITE …`, `PLANT …`, `LEDGER …`).
- 321 system messages (repair 71, ledger 103, design 147 — exactly the `agents+skills`
  harness rows) embed an `# AGENTS.md: energy-plant tasks` guidance section plus component/task
  skill text. This is **guidance/task documentation inherited from the export prompt**, not tool
  bookkeeping — no model ids, costs, timestamps, record ids, or tool outputs.
- The literal `<tool_call …>` string exists in the codebase only at
  `thermo_agent/teacher/agent_graph.py:207`, inside `finalize()`, where it builds the **SFT
  completion (training target)**, not the model prompt.
- Runtime (not stored in the shard): `build_context()` (`agent_graph.py:82–129`) appends a compact
  guidance block (`compact_memory.py:83`), compact failure memory with model id / cost / ts /
  record id / run id stripped (`compact_memory.py:3–8, 43–58`), and a short tool contract naming
  tools via `tool_names()` (`agent_graph.py:113–120`). Tools are executed through kbench native
  tool calling (`make_respond`, `run_direct`, harness lines 463–469), so no literal tool contract
  is needed in the row prompt — consistent with the 0 `<tool_call` count.

Conclusion: **no literal tool contract and no tool-output/memory bookkeeping in the prompt text —
verified.**

## 5. Determinism and resumability

**Generation deterministic — verified.** `make_hosted_prompts.py:32–35, 168–188`: fixed
`TASK_ORDER`, `SEED=20261119`, `random.Random(SEED).shuffle`, round-robin into 5 shards. An
in-memory rebuild from the live export reproduced all 1,600 rows in **identical order and content**
for every shard (byte-level diff is only Windows CRLF translation by `Path.write_text`; parsed
rows match 100%). Source hashes also match (§2).

**Resumable at run time — verified with caveats.**
- `JOURNAL = /kaggle/working/harness_completions.jsonl` (lines 36–38); a `done` set of ids is read
  from it at lines 427–433; rows whose id is in `done` are skipped (lines 437–441).
- One JSON line is appended and `flush()`ed per row (lines 501–502), so a dead worker loses at
  most one answer. Malformed/partial journal lines are ignored by the `try/except` at lines
  428–433. Rerun after a partial journal resumes at the first un-journaled id in the fixed shard
  order; nothing is recomputed for done ids.
- Stops (deadline / cost cap / 5 errors with 0 answers) write `partial`/`paused` manifest status
  (lines 509–517) and are meant to be resumed by rerun.

Caveats:
- `LIMIT=40` is applied **before** the done-filter (lines 416–419). With the first 40 ids already
  journaled, a rerun at the default limit skips all 40 and answers 0; progress past the pilot
  requires raising `HARNESS_LIMIT`. (verified by code path; not executed end-to-end — **not
  verified** empirically.)
- Shard selection order flips if a `frontier_export` dir is attached (`load_exports`, lines
  280–287), which returns full answer-carrying rows; not applicable to the configured Kaggle
  inputs (§3).
- End-to-end hosted resume behavior was **not verified** (no Kaggle run inspected).

## 6. Training/eval contamination

| Check | Count | Status |
|---|---|---|
| Unique ids in `data/heldout` (20 `.jsonl` files) | 542 | verified |
| Shard ∩ heldout ids | **0** | verified |
| `frontier_export` train ∩ heldout ids | **0** | verified |
| `data/frontier_eval_export` unique ids | 360 | verified |
| Shard ∩ `frontier_eval_export` ids | **0** | verified |

Conclusion: **no contamination from these shards — verified.** The shards are exactly the
`frontier_export` *train* ids (intended teacher training set) and are disjoint from the held-out
set and from the separate `make_eval_export.py` export (`scripts/make_eval_export.py:1–8, 20–41`,
which strips `completion` and, per its default, sources `data/heldout`; the on-disk manifest
currently lists `data/processed`, `data/processed_arch`, limit 30).

## Summary table

| Question | Result | Status |
|---|---|---|
| Shards/rows/inputs-only | 5 shards × 320 = 1600; no answer fields or values | verified |
| Id match to frontier_export | exact 1:1 (0 missing/extra/dup) | verified |
| Reads heldout / answer fields | never reads heldout; no answers in shards (local export path caveat) | verified |
| Literal tool contract / bookkeeping in prompt | none; AGENTS.md guidance present, bookkeeping stripped | verified |
| Deterministic + resumable | deterministic; journal-resumable; LIMIT-before-done caveat | verified (code/rebuild); hosted rerun not verified |
| Contamination | 0 overlap with heldout or eval export | verified |
