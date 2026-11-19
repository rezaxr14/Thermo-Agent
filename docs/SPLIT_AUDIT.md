# Split audit: is the `topology_label` split-label fix actually in the on-disk data?

Audit date: 2026-11-19. Read-only analysis. Scope: `IMPLEMENTATION_PLAN.md` P0-16 — verify whether the
`synth.splits.topology_label` fix (ideal-rejection engines must go to `test_ood_topology`, not `train`/`test_id`)
is reflected in the data, and quantify any leakage.

## Bottom line

**The current data is POST-FIX and CLEAN. A full `pwsh scripts/gen_data.ps1` regeneration is NOT required
to remove the ideal-rejection-engine leak: every live directory (`data/processed`, `data/processed_arch`,
`data/processed_big`, `data/heldout`) and both corpora (`data/sft_all`, `data/grpo_all`) already carry the fix.**

The pre-fix data still exists, but only as the deliberately preserved backup `data/_pre_leakfix/`, which is
**STALE** and must never be used for training/eval. The leak that P0-16 warned about is fully reproducible in
that backup (e.g. 412 ideal-rejection engines in `repair_train`, 40 in `repair_test_id`, 111 in `audit_train`),
and **zero** such rows exist in the live directories.

Corroboration: the fix is commit `c566543` "fix(splits): hold out ideal-rejection engines from the MIX
generators" at **2026-11-17 19:32 UTC**. Every current `meta_*.json` was created *after* that
(`data/processed` 19:46–21:20 UTC, `data/processed_arch` 21:21–21:25 UTC, `data/processed_big`
2026-11-18 16:08 UTC, `data/heldout` 2026-11-18 18:04–18:11 UTC).

## Post-fix rules used as the reference (from code)

- `thermo_agent/synth/splits.py`:
  - `is_ood_topology(name)`: OOD iff name (with `_`→`-`, `_sampled` stripped) equals `cgam-steam-orc`
    (steam/HRSG export) **or** contains both parts `engine` and `ext`.
  - `topology_label(plant)`: if the name is already OOD, return it; otherwise append `-ext` when the plant
    has a `gas_engine` component **and** a stream `ext:* -> CH.rej_in*` (ideal chiller rejection).
- Split assignment (uses the fixed label): `synth/sft_data.py::assign_split` (repair),
  `audit/data.py::split_of`, `ledger/data.py::split_of`, `synth/architectures.py::split_for` —
  `is_ood_topology(topology)` → `test_ood_topology` first; then fault-based `test_ood_fault`; then md5(id)%10
  → `val` (0) / `test_id` (1) / `train` (else). Held-out ledger faults = `{wrong_stream, dropped_work, wrong_ref}`.
- `synth/verify.py` checks structural/token integrity of corpora but does **not** check topology/split
  semantics, so it cannot detect this leak on its own.

Leak detection method (re-implements `topology_label` on the stored plant JSON): a row is an
**ideal-rejection-engine leak** iff its plant has a `gas_engine` component and an `ext: -> CH.rej_in*`
stream (so post-fix label is `...-ext`, i.e. OOD) **but** the row sits in `train`/`val`/`test_id`.
A **stale label** is a stored `topology` that differs from the graph-derived `topology_label`. I also
recomputed each row's expected split from `topology` + `id` + fault type; all mismatches were 0 once ledger's
correct held-out fault set was used (an earlier non-zero count was my own check bug, not a data problem).

## Sampling note

For topology/label/leak counts I read the **first 20,000 lines of each JSONL file**. Every file in
`data/processed`, `data/processed_arch`, `data/processed_big` and `data/heldout` has < 20,000 rows, so those
are effectively **full** scans. Only `data/sft_all/repair_train.jsonl` (110,635 rows) was sampled for the label
check (20,000 rows); its row count and source histogram were computed over the **whole** file. No data was
modified. No network/Kaggle/API calls, no full test suite. Runtime well under 5 minutes.

---

## Table 1 — `data/processed` (MIX generators, seed 0) — **verdict: CLEAN**

`|rows|` = exact line count. `D` = distinct `topology` values. `eng` = engine topologies without `-ext`
(legitimate dry/wet rejection, allowed in train). `OOD` = topology that post-fix rules hold out
(`*-ext` engine / `cgam_steam_orc`). `stale` / `leak` = counts defined above; all zero.

| task | split | rows | D | engine topologies (non-ext) | OOD topologies | stale | leak |
|---|---|---:|---:|---|---|---:|---:|
| repair | train | 5458 | 5 | `dc_engine_absorption`, `dc_engine_electric` | none | 0 | 0 |
| repair | val | 663 | 5 | same two | none | 0 | 0 |
| repair | test_id | 649 | 5 | same two | none | 0 | 0 |
| repair | test_ood_fault | 4203 | 5 | same two | none | 0 | 0 |
| repair | **test_ood_topology** | **837** | 2 | — | `dc_engine_absorption-ext`, `dc_engine_electric-ext` | 0 | 0 |
| ledger | train | 4846 | 5 | same two | none | 0 | n/a¹ |
| ledger | val | 672 | 5 | same two | none | 0 | n/a¹ |
| ledger | test_id | 612 | 5 | same two | none | 0 | n/a¹ |
| ledger | test_ood_fault | 2229 | 5 | same two | none | 0 | n/a¹ |
| ledger | **test_ood_topology** | **640** | 2 | — | both `dc_engine_*-ext` | 0 | n/a¹ |
| audit | train | 2192 | 5 | same two | none | 0 | 0 |
| audit | val | 290 | 5 | same two | none | 0 | 0 |
| audit | test_id | 299 | 5 | same two | none | 0 | 0 |
| audit | **test_ood_topology** | **135** | 2 | — | both `dc_engine_*-ext` | 0 | 0 |
| design | train / val / test_id / test_ood_climate / test_ood_load / test_ood_scale | 1768 / 235 / 204 / 615 / 411 / 460 | — | none (no `topology` field) | n/a² | n/a | n/a |

¹ Ledger rows store no plant graph, so the graph-based leak count is not computable; instead the stored
`topology` is checked (0 OOD topologies in train/val/test_id) and the split recomputed from `topology`+`id`+
`fault_types` (0/8999 mismatches). Pre-fix, this directory had **no** `test_ood_topology` file; the 640 rows now
in it are exactly the ideal-rejection-engine rows that were previously misrouted.
² Design has no `topology` field; its OOD splits are climate/load/scale, not topology, so the topology leak
does not apply.

## Table 2 — `data/processed_arch` (65-diagram catalog, seed 0) — **verdict: CLEAN**

Engine topologies in train/test_id are `engine-absorption-dry/wet*`, `engine-electric-dry/wet*` (legit); the few
`*-ext` values in train are `turbine-*-ext` (no `engine` part ⇒ not OOD). All `engine-*-ext*` and
`cgam_steam_orc` rows are confined to `test_ood_topology`.

| task | split | rows | distinct topologies | OOD topologies in split | stale | leak |
|---|---|---:|---:|---|---:|---:|
| repair | train | 818 | 45 | none | 0 | 0 |
| repair | val | 110 | 39 | none | 0 | 0 |
| repair | test_id | 95 | 37 | none | 0 | 0 |
| repair | test_ood_fault | 761 | 44 | none | 0 | 0 |
| repair | **test_ood_topology** | **279** | 8 | `cgam_steam_orc`, `engine-absorption-ext*`, `engine-electric-ext*` | 0 | 0 |
| ledger | train | 1097 | 45 | none | 0 | n/a¹ |
| ledger | val | 110 | 38 | none | 0 | n/a¹ |
| ledger | test_id | 152 | 41 | none | 0 | n/a¹ |
| ledger | test_ood_fault | 464 | 44 | none | 0 | n/a¹ |
| ledger | **test_ood_topology** | **292** | 8 | as above | 0 | n/a¹ |
| audit | train | 996 | 45 | none | 0 | 0 |
| audit | val | 113 | 41 | none | 0 | 0 |
| audit | test_id | 116 | 42 | none | 0 | 0 |
| audit | **test_ood_topology** | **141** | 6 | `cgam_steam_orc`, `engine-*-ext*` | 0 | 0 |

Note: this directory is byte-identical between `data/_pre_leakfix/processed_arch` and `data/processed_arch`
(same counts, 0 stale/leak). Reason: catalog architecture names already spell out `ext`, so `is_ood_topology`
caught them even before the `topology_label` fix; the bug only affected the MIX names (`dc_engine_*`).

## Table 3 — `data/processed_big` (big cycle / seeds 100, 101) — **verdict: CLEAN**

| task | split | rows | distinct topologies | stale | leak |
|---|---|---:|---:|---:|---:|
| repair | train | 57 | 1 (`big_cchp_steam2p_reheat_hp`, no engine) | 0 | 0 |
| repair | val | 6 | 1 (same) | 0 | 0 |
| repair | test_id | 5 | 1 (same) | 0 | 0 |
| repair | test_ood_fault | 50 | 1 (same) | 0 | 0 |
| audit | train | 66 | 5 | 0 | 0 |
| audit | val | 6 | 3 | 0 | 0 |
| audit | test_id | 3 | 3 | 0 | 0 |
| audit | **test_ood_topology** | **1** | 1 (`dc_engine_absorption-ext`) | 0 | 0 |

No ledger/design task exists here (as the plan notes). Repair uses only the 19-component cycle, which has no
gas engine, so the ideal-rejection leak cannot occur.

## Table 4 — `data/heldout` (seed 987654) — **verdict: CLEAN**

| task | split | rows | distinct topologies | OOD topologies in split | stale | leak |
|---|---|---:|---:|---|---:|---:|
| repair | train | 94 | 5 | none | 0 | 0 |
| repair | val | 13 | 3 | none | 0 | 0 |
| repair | test_id | 6 | 4 | none | 0 | 0 |
| repair | test_ood_fault | 70 | 5 | none | 0 | 0 |
| repair | **test_ood_topology** | **15** | 2 | both `dc_engine_*-ext` | 0 | 0 |
| ledger | train | 60 | 5 | none | 0 | n/a¹ |
| ledger | val | 10 | 4 | none | 0 | n/a¹ |
| ledger | test_id | 11 | 5 | none | 0 | n/a¹ |
| ledger | test_ood_fault | 32 | 5 | none | 0 | n/a¹ |
| ledger | **test_ood_topology** | **7** | 2 | both `dc_engine_*-ext` | 0 | n/a¹ |
| audit | train | 84 | 5 | none | 0 | 0 |
| audit | val | 9 | 4 | none | 0 | 0 |
| audit | test_id | 16 | 4 | none | 0 | 0 |
| audit | **test_ood_topology** | **6** | 2 | both `dc_engine_*-ext` | 0 | 0 |
| design | train / val / test_id / climate / load / scale | 54 / 4 / 4 / 20 / 13 / 14 | — | n/a² | n/a | n/a |

## Control — `data/_pre_leakfix/` — **verdict: STALE (pre-fix backup; do not use)**

Same detection code on the preserved pre-fix snapshot quantifies exactly what P0-16 described:

| pre-fix file | rows | stale labels | ideal-rejection engines in train/val/test_id |
|---|---:|---:|---:|
| `processed/repair_train.jsonl` | 5870 | **412** | **412** |
| `processed/repair_test_id.jsonl` | 689 | 40 | **40** |
| `processed/repair_val.jsonl` | 717 | 54 | **54** |
| `processed/repair_test_ood_fault.jsonl` | 4534 | 331 | 0 (already an OOD split) |
| `processed/audit_train.jsonl` | 2303 | **111** | **111** |
| `processed/audit_test_id.jsonl` | 309 | 10 | **10** |
| `processed/audit_val.jsonl` | 304 | 14 | **14** |
| `processed_arch/*` | — | 0 | 0 |

Totals reconcile exactly: pre-fix `processed` repair = 11,810 rows with **no** `test_ood_topology` file;
post-fix has 11,810 rows with 837 `test_ood_topology` rows (506 graph-confirmed leaked into
train/val/test_id + 331 mislabeled inside `test_ood_fault`). Audit: 2,916 → 135 OOD rows (111+10+14).
Ledger: 8,999 → 640 OOD rows (ledger has no plant graph, so enumerated by split presence rather than graph).

## Seed cross-check (from `meta_*.json`)

| directory | meta file(s) | seed | expected | result |
|---|---|---|---|---|
| `data/heldout` | `meta_repair.json`, `meta_audit.json`, `meta_ledger.json`, `meta_design.json` | **987654** (all four) | 987654 | ✅ match |
| `data/processed` | `meta_repair.json`, `meta_audit.json`, `meta_ledger.json`, `meta_design.json` | **0** (all four) | — | recorded |
| `data/processed_arch` | `meta_repair_arch.json`, `meta_audit_arch.json`, `meta_ledger_arch.json` | **0** (all three) | plan Gate 1 says 100/101 | ⚠️ doc/code mismatch, not a leak |
| `data/processed_big` | `meta_repair_arch.json`, `meta_audit.json` | **100** (repair), **101** (audit) | 100/101 | ✅ match |

The held-out seed is correct. The only seed discrepancy is documentation: `IMPLEMENTATION_PLAN.md` Gate 1.1
expects `processed_arch` at seeds 100/101, but its meta files say seed 0 (and `processed_big` holds 100/101).

## Corpora: `data/sft_all` vs `data/grpo_all` (task 5)

`scripts/build_corpus.ps1` builds both from `data/processed` + `--extra-processed data/processed_arch`
(**not** `processed_big`, not `heldout`), which are post-fix. Both corpora contain post-fix
`test_ood_topology` files with only `-ext` engine topologies, and the 20k-row label/leak scan found
**0 stale labels, 0 ideal-engine leaks, 0 OOD topologies in train/val/test_id** in every corpus file.
**Neither corpus was built from pre-fix data.**

Actual `source` field histogram (full files): the only values are `ground_truth` and `frontier`. **There is no
`source`/`engine` value meaning "ideal-rejection engine"** — no `engine` field exists in any row (verified by
inspecting keys of the first row of every JSONL). Ideal-rejection engines are identified solely by
`topology` + the plant graph, as done above; `frontier` is just teacher provenance.

| corpus | train rows | total rows | `frontier` rows (train) | `ground_truth` rows (train) | post-fix `test_ood_topology` rows (repair/ledger/audit) | leak scan |
|---|---:|---:|---|---|---:|---|
| `data/sft_all` | 119,807 | 133,032 | repair 335, ledger 160, design 140 | repair 110,300, ledger 4,846, design 1,768, audit 2,258 | 837 / 640 / 136 | 0 |
| `data/grpo_all` | 15,022 | 28,247 | repair 335, ledger 160, design 140 | repair 5,515, ledger 4,846, design 1,768, audit 2,258 | 837 / 640 / 136 | 0 |

(Train totals reconcile with `sft_all/sft_report.json`: 119,807; and with the plan's 15,022 for GRPO.
`audit_test_ood_fault.jsonl` is empty, 0 bytes, in both corpora — expected, as audit has no fault OOD.)

## Exact commands used

All run from `C:\Users\rezax\Home\Code\thermo_agent\Repo`, Python = `.venv/Scripts/python.exe`.

```powershell
# File inventory (sizes/names), per data directory
Get-ChildItem data/processed,data/processed_arch,data/processed_big,data/heldout,data/sft_all,data/grpo_all -File
Get-ChildItem data/_pre_leakfix -Recurse -File

# Meta/seeds
Get-Content data/processed/meta_*.json -Raw   # (likewise for processed_arch, processed_big, heldout)

# Row schema keys for one row of every JSONL (confirmed only 'source' provenance, no 'engine' field)
.venv/Scripts/python.exe -c "import json,glob,os; ... print(sorted(json.loads(open(p).readline()).keys())) ..."

# Core analysis (scripts written to %TEMP%/opencode, never into the repo):
.venv/Scripts/python.exe %TEMP%/opencode/split_audit.py    > %TEMP%/opencode/split_audit_out.json
.venv/Scripts/python.exe %TEMP%/opencode/split_audit2.py   > %TEMP%/opencode/split_audit2_out.txt
.venv/Scripts/python.exe %TEMP%/opencode/split_audit3.py
```

The scripts (a) re-implement `is_ood_topology` / `topology_label` from `synth/splits.py` and apply them to the
stored plant JSON (`plant_original` for repair, `plant` for audit), (b) count rows/topologies per task/split,
(c) recompute expected split from `topology`+`id`+fault type, (d) apply the same checks to
`data/_pre_leakfix` as a control, and (e) count lines and histogram `source` over the corpora (full files).

Source of rules: `thermo_agent/synth/splits.py`, `thermo_agent/synth/verify.py`,
`thermo_agent/synth/sft_data.py`, `thermo_agent/audit/data.py`, `thermo_agent/ledger/data.py`,
`thermo_agent/ledger/faults.py` (`HELD_OUT = {wrong_stream, dropped_work, wrong_ref}`),
`thermo_agent/synth/architectures.py`, `scripts/build_corpus.ps1`, `git log -- thermo_agent/synth/splits.py`.

## Regeneration decision

- **Required to remove the ideal-rejection leak: NO.** The fix is already present in `data/processed`,
  `data/processed_arch`, `data/processed_big`, `data/heldout`, `data/sft_all` and `data/grpo_all`; all
  post-fix OOD topology rows are isolated in `test_ood_topology`, with zero leaks into `train`/`val`/`test_id`.
- **Must NOT be used:** `data/_pre_leakfix/` (STALE backup). No training or eval job should point at it. If any
  job does, either repoint it to the live directories or regenerate that directory — there is nothing else to
  regenerate for this specific bug.
- **Out of scope of this leak:** Gate 1 of the plan may still call for *larger/targeted* regeneration (e.g.
  ~85 paired rows per held-out cell, new dev set, seed changes for `processed_arch`) to meet statistical power
  and doc targets. Those are size/power decisions, not a leak fix; this audit does not recommend against them,
  only states that a full regeneration is **not needed to fix the split-label leak**.

---

## 2026-11-19 — independent leakage re-check

Independent integrity subagent, read-only. Scope: re-verify that the frozen held-out set does not leak into any
training artifact. **No data, code, config or paper file was modified; only this document was appended to.** Every
scan below is a **full** scan (all JSONL lines parsed, no `MAX_LINES` sampling).

### Commands (exact, run from `C:\Users\rezax\Home\Code\thermo_agent\Repo`, Python `.venv/Scripts/python.exe`)

```powershell
$env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe -m pytest tests -q -k "contamination or heldout or held_out or leak" 2>&1 | Select-Object -Last 20
$env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe -m pytest tests/test_contamination.py tests/test_cohort_coverage.py -v 2>&1 | Select-Object -Last 40
$env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe -m pytest tests/test_contamination.py -v -s 2>&1 | Select-Object -First 30
# Independent full-scan script (written to %TEMP%/opencode/leak_recheck.py, NOT into the repo):
$env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe C:\Users\rezax\AppData\Local\Temp\opencode\leak_recheck.py
```

### Test results (exact ids)

- Broad filter `-k "contamination or heldout or held_out or leak"`: **2 failed, 37 passed, 594 deselected in 110.24 s**.
  The 2 failures are **not leakage failures** and touch no real data:
  - `tests/test_eval_runner_arms.py::test_b_data_dirs_default_is_the_frozen_heldout_set` — `SystemExit` from
    `require_test_corpus`: the `tmp_path` fixture root has no `*_test_*.jsonl`. Fixture/environment issue.
  - `tests/test_verify.py::test_a_duplicate_id_across_splits_is_a_leak` — `IndexError` reading row 0 of an empty
    generated `repair_train.jsonl` in `tmp_path`. Fixture/environment issue.
- Focused, real-data contamination suites: `tests/test_contamination.py` + `tests/test_cohort_coverage.py`
  → **89 passed in 48.98 s** (0 failed), including:
  - `tests/test_contamination.py::test_training_artifacts_are_free_of_heldout_signatures` — **PASSED**
    (scanner print: `contamination scan: 214 held-out ids, 675 signatures, 136 files, 117651 lines sampled`).
  - `tests/test_contamination.py::test_scanner_detects_planted_heldout_leak` — **PASSED** (self-check: a planted
    held-out row IS flagged, so the scanner can fail).
  - `tests/test_cohort_coverage.py::test_heldout_ids_are_unseen_by_every_cohort_and_seed_is_fresh` — **PASSED**.

### Independent counts (full scan; distinct ids per artifact)

Held-out side: **all 20 `data/heldout/*.jsonl` = 542 ids / 4,040,644 bytes**; **frozen 12 `*_test_*.jsonl` = 214
ids / 1,531,809 bytes** (matches EVAL_PROTOCOL §1's 214). Per the protocol, only `test_*` is the frozen set; both
comparisons are reported.

| artifact | files | rows | distinct ids | id ∩ frozen 214 | id ∩ all heldout 542 | payload digest collisions | cross-field collisions |
|---|---:|---:|---:|---:|---:|---:|---:|
| `data/sft_tools` | 4 | 960 | 960 | **0** | **0** | **0** | **0** |
| `data/grpo_tools` | 4 | 960 | 960 | **0** | **0** | **0** | **0** |
| `data/sft_all` | 21 | 133,032 | 133,032 | **0** | **0** | **0** | **0** |
| `data/grpo_all` | 21 | 28,247 | 28,247 | **0** | **0** | **0** | **0** |
| `data/sft` | 21 | 123,024 | 123,024 | **0** | **0** | **0** | **0** |
| `data/grpo` | 21 | 28,214 | 28,214 | **0** | **0** | **0** | **0** |
| `data/frontier*` (4 dirs, 36 files) | 36 | 3,296 | 2,628 | **0** | **0** | **0** | **0** |
| `data/processed*` (corroboration, protocol §7) | 42 | 33,156 | 33,156 | **0** | **0** | **0** | **0** |
| `data/kaggle_datasets/teacher_big_prompts` (corroboration) | 5 | 1,600 | 1,600 | **0** | **0** | **0** | **0** |

Overlapping ids: **none** in any set. `data/frontier_heldout_export` and `thermo-teacher-prompts*` do not exist on
disk at audit time; `data/_pre_leakfix` is the explicitly STALE backup and is not a training artifact.

### Payload hash-normalisation method and field presence

For every row carrying any of `plant`, `plant_broken`, `plant_original`, `ledger`, `site`, `ed_json` the value is
sha256'd as `json.dumps(value, sort_keys=True, separators=(',',':'), ensure_ascii=False)`; JSON stored as a string
is parsed first; blank/empty values are skipped (identical to the repo scanner's `_parse`, which returns `None`
for `not value.strip()`). Collisions are counted per field between held-out `test_*` and each artifact, plus a
cross-field union check.

All six fields are **present** in the frozen held-out rows (row counts: `plant` 22, `plant_broken` 91,
`plant_original` 91, `ledger` 50, `site` 156, `ed_json` 22) and in **every** artifact scanned above — **no field is
absent** from any set. All collision counts are **0**.

**Transparently recorded non-zero intermediate finding (investigated, not a leak):** a first supplemental sweep of
the 5 teacher-prompt files used a snippet that did not skip blank values and reported one `site` digest collision
(`12ae32cb1ec02d01eda3581b127c1fee3b0dc53572ed6baf239721a03d82e126`). Inspection showed it is the empty-string
placeholder `site = ''`, in 8 held-out rows (e.g. `aud-987654-00015`, `rep-98765400000-00034`) and 71 unrelated
prompt rows with different seeds (`rep-77-00078`, `aud-0-02436`, …) — no identifying content. Re-running with the
blank-value filter used by the repo scanner and the main scan returns **0** id overlap and **0** collisions. This
was a false positive of the throwaway snippet, not data leakage.

### Eval-set hash

The definition IS clear in `docs/EVAL_PROTOCOL.md` §1 and Appendix A: SHA-256 over sorted test-file **names**
followed by each file's **bytes**, over the 12 `*_test_*.jsonl` files. Recomputed here:

```
protocol definition (12 test files): 8b409359e7a2dd441d0be99c89c00f0f3c61320f9d06e9695cc22569f8ef4c6e  (12 files, 1,531,809 bytes)
```

This **matches** the value recorded in `docs/EVAL_PROTOCOL.md` §1 and Appendix A exactly. For completeness, this
worker's own variant over **all 20** held-out JSONL files (same name-then-bytes rule) is
`12cb731a062486e51e39582e3c1de708e7d8276295cba6f3994051314b12be3e` (20 files, 4,040,644 bytes) — informational
only; it is **not** the protocol eval-set hash.

### Verdict

**CLEAN.** The frozen held-out set has zero id overlap and zero payload-digest collisions with every training,
prompt, frontier and processed artifact scanned, and the recorded eval-set hash matches the current on-disk data.
The only non-zero result encountered was an empty-string placeholder investigated above and shown not to be a leak.
The two failures in the broad `-k` run are unrelated `tmp_path` fixture issues, not contamination.
