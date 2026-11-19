# Evaluation protocol — FROZEN (Gate 1)

Status: **frozen 2026-11-19**. This file fixes the evaluation set, the arms, the splits, the statistics and the
evidence rule before any model result is read. No arm may be re-scored under different rules. Where a rule is not yet
implemented in code, the gap is named in §4 rather than papered over. Repo head at freeze:
`93744d5ee7c3257cb6b893460769cc8087141824`.

## 1. Evaluation set and version

The evaluation set is **`data/heldout`, seed 987654**, generated 2026-11-18 with the standard generators
(`build_repair_data`, `audit.data`, `ledger.data`, `design.data --max-evals 60`). Only the `test_*` rows are the frozen
set. The `train`/`val` `.jsonl` files that also sit in `data/heldout` are **not** part of it: they are never scored and
must never be used for training, prompt selection or analysis.

Eval-set version (must be written into every report as `meta.eval_set_version`):

```
eval_set_version = 8b409359e7a2dd441d0be99c89c00f0f3c61320f9d06e9695cc22569f8ef4c6e
```

This is SHA-256 over the sorted test-file names followed by each file's bytes (12 files, 1,531,809 bytes total); the
command is in Appendix A. A change to any test row changes this hash and invalidates every report scored against it.

Measured cells (unique ids per file; `split` field checked to equal the filename split). **Total = 214 test ids.**

| Task | `test_id` | `test_ood_fault` | `test_ood_topology` | `test_ood_climate` | `test_ood_load` | `test_ood_scale` | task total |
|---|---:|---:|---:|---:|---:|---:|---:|
| repair | 6 | 70 | 15 | — | — | — | 91 |
| audit | 16 | — | 6 | — | — | — | 22 |
| ledger | 11 | 32 | 7 | — | — | — | 50 |
| design | 4 | — | — | 20 | 13 | 14 | 51 |
| **all** | **37** | **102** | **28** | **20** | **13** | **14** | **214** |

Topology cross-table (only splits carrying a `topology` field; design rows have no topology field):

| Task / split | n | topology counts |
|---|---:|---|
| repair / test_id | 6 | cgam_orc 2, dc_electric 1, dc_engine_absorption 2, dc_engine_electric 1 |
| repair / test_ood_fault | 70 | cgam_orc 4, dc_absorption 28, dc_electric 28, dc_engine_absorption 6, dc_engine_electric 4 |
| repair / test_ood_topology | 15 | dc_engine_absorption-ext 5, dc_engine_electric-ext 10 |
| audit / test_id | 16 | cgam_orc 2, dc_absorption 4, dc_electric 8, dc_engine_absorption 2 |
| audit / test_ood_topology | 6 | dc_engine_absorption-ext 1, dc_engine_electric-ext 5 |
| ledger / test_id | 11 | cgam_orc 1, dc_absorption 3, dc_electric 3, dc_engine_absorption 3, dc_engine_electric 1 |
| ledger / test_ood_fault | 32 | cgam_orc 3, dc_absorption 9, dc_electric 16, dc_engine_absorption 2, dc_engine_electric 2 |
| ledger / test_ood_topology | 7 | dc_engine_absorption-ext 2, dc_engine_electric-ext 5 |

## 2. Splits scored, and the identical-splits rule

Scored splits are exactly the twelve `test_*` cells in §1: `test_id` for all four tasks; `test_ood_fault` for repair
and ledger; `test_ood_topology` for repair, audit and ledger; `test_ood_climate`, `test_ood_load`, `test_ood_scale`
for design. `train` and `val` are never scored.

`kaggle/eval_runner.py` passes one `DATA_DIRS` list and one `SPLITS` list to every arm (its `run_arm` builds the same
`model_eval` command for all of them, and the code comment explicitly removes the old `frontier_tools`/`test_id`
restriction). An arm whose split ids or per-split `n` differ from §1 is invalid and its report must be discarded and
re-run. Split files are read separately by `model_eval.load_rows`; concatenating splits into one reported cell is a
mix and is forbidden, except for the explicitly labelled aggregates in §6.

`LIMIT` defaults to 30 in `eval_runner.py`, which would truncate every cell over 30 rows (including the 70-row
repair fault cell). It must be set to **at least 70** for this frozen set; use one large value for all arms and record
it as `meta.limit`. The same value applies to every arm, every task and every split.

## 3. Arms and their exact configuration

These are the ten arms in the `ARMS` default of `kaggle/eval_runner.py`. Config fields are the `SPECS` entries;
`harness` values are `model_eval` modes, `tool_access` is a report flag.

| Arm | backend | base model | adapter | harness | tool_access | zero_shot | Extra |
|---|---|---|---|---|---|---|---|
| `ours_base` | hf | Qwen/Qwen3.5-4B | none | none | none | yes | base (Z-style) arm |
| `ours_sft` | hf | Qwen/Qwen3.5-4B | `ADAPTER` (SFT) | agents+skills | none | no | our SFT |
| `ours_harness` | hf | Qwen/Qwen3.5-4B | `ADAPTER` (SFT) | agents+skills | none | no | see note below |
| `ours_sft_grpo_k4` | hf | `SFT_MERGED` | `GRPO_K_ADAPTERS[4]` | agents+skills | none | no | `grpo_k=4`, `train_task=TASK` |
| `ours_sft_grpo_k8` | hf | `SFT_MERGED` | `GRPO_K_ADAPTERS[8]` | agents+skills | none | no | `grpo_k=8`, `train_task=TASK` |
| `ours_sft_grpo_k16` | hf | `SFT_MERGED` | `GRPO_K_ADAPTERS[16]` | agents+skills | none | no | `grpo_k=16`, `train_task=TASK` |
| `frontier_zero` (Z) | replay | `eval_all.jsonl` | n/a | none | none | yes | frontier zero-shot |
| `frontier_tools` | replay | `harness_eval_completions2.jsonl` | n/a | tools | tools | yes | frontier + verifier tools |
| `frontier_harness` (H) | replay | `harness_completions.jsonl` | n/a | tools | tools | yes | frontier in LangGraph harness |
| `other_model` | replay | `eval_gptnano_replay.jsonl` | n/a | none | none | yes | domain-model stand-in |

Two further specs exist but are not in the default `ARMS`: `ours_grpo` (hf on `SFT_MERGED` with `GRPO_ADAPTER`,
`train_task=TASK`) and `other_model_api` (openai backend, `API_MODEL` at `API_BASE_URL`, default
`http://127.0.0.1:1234/v1`, harness agents+skills). They may only be reported if explicitly added to `ARMS` for a run
and their config recorded in `arm_config`.

Constraints on this table:

- **GRPO arms require real weights.** `require_adapter()` is a hard error on a missing or empty adapter; a GRPO arm
  may never be reported as base-model scoring. Each k has its own checkpoint dir `runs/grpo_<task>_k<k>`, and the base
  is the merged SFT policy (`SFT_MERGED`), never raw Qwen. `train_task` is written into the report and manifest.
- **`ours_sft` and `ours_harness` are currently configured identically** (same weights, harness agents+skills,
  `tool_access: none`). They are not a Z-vs-H contrast. Do not present `ours_harness` as a distinct result while its
  spec is unchanged; the H-vs-Z contrasts are `ours_base` vs `ours_sft`/GRPO, and `frontier_zero` vs
  `frontier_harness`. A change that makes `ours_harness` distinct must be recorded in a new frozen version of this
  file, not silently in a report.
- Replay arms score saved answers through the same verifiers and prompt rows; a replay result is evidence for that
  saved answer set only, and the replay file is recorded as `meta.replay_file`.

## 4. How a report is produced

1. `kaggle/eval_runner.py` (source of `kaggle/eval.ipynb`, built by `kaggle/make_notebook.py`) runs each arm with
   `python -m thermo_agent.eval.model_eval ... --out reports/<arm>.json`, on `DATA_DIRS=heldout`, with the common
   `SPLITS`/`LIMIT`. Existing reports are not overwritten unless `FORCE=1`.
2. `thermo_agent/eval/model_eval.py` writes `reports/<arm>.json` (per-task, per-split metrics with Wilson intervals
   and per-example arrays) and a `reports/<arm>.completions.jsonl` sidecar.
3. `eval_runner.py` writes `arm_config` into each report, builds `paper/assets/model_results_manifest.json`, and calls
   `thermo_agent.eval.comparison.main` to produce `paper/assets/model_results.json`.
4. Paired tests use `python -m thermo_agent.eval.compare A.json B.json`; statistics are defined in §5.

Required fields on every report used as evidence:

- `meta.eval_set_version` — the hash in §1 (**not currently written by the code — see gap below**).
- `meta.harness_config_hash` — a hash of the harness/limits config (AGENTS.md + skills + tool list + max turns +
  tool-call cap + `LIMIT`); identical for every arm sharing a harness mode (**not currently written — see gap**).
- `arm_config` — written by `eval_runner.py`: display name, family, harness, tool_access, zero_shot, backend, model,
  train_task, grpo_k, data_dirs, splits.
- Split ids and per-split `n` for all four tasks; `meta.data`, `meta.harness`, `meta.limit`, `meta.adapter`,
  `meta.backend`, `meta.model_family`, `meta.zero_shot`, `meta.tool_access`, python and package versions.
- `meta.completions` sidecar present for the arm, so any reported example can be re-checked.

**Code gap (do not ignore):** `eval_set_version` and `harness_config_hash` appear nowhere in `kaggle/` or
`thermo_agent/` at freeze time, and `model_eval.main` writes neither into `meta`. Until the code owner adds them,
the harness/eval owner must attach them when assembling the manifest, and any report lacking them is **not** valid
evidence for a Gate 1/9 number. This protocol does not waive the requirement.

## 5. Statistics

- Every reported proportion carries a **Wilson 95% interval** (`thermo_agent.eval.model_eval.wilson`, `z = 1.96`),
  reported with the same `n` as the cell. No point estimate without its interval.
- Comparisons are **paired by id** on identical splits: exact McNemar for binary metrics and sign-flip permutation
  (20,000 flips) plus a 4,000-replicate paired bootstrap interval for continuous metrics
  (`thermo_agent/eval/compare.py`). Pairs with a missing id or metric are dropped and the count reported.
- **Holm correction** is applied across the four headline comparison families of Gate 9: (1) harness uplift H − Z per
  model; (2) SFT over base in the same mode; (3) GRPO over SFT by k; (4) final 4B versus base 9B, other LLMs and
  frontier Z/H. `p_holm` is recorded next to the raw `p`. Note that `compare.py` currently Holm-corrects every row it
  is given, so the analysis must restrict each Holm family to the headline comparisons (or re-apply Holm to them)
  and say which it did.
- Significance is claimed only in a cell that meets the power requirement in §6. Smaller cells are reported as
  descriptive (point estimate + Wilson interval + raw and corrected p where computed), explicitly labelled
  underpowered. A negative result stays in the paper.

## 6. Power statement (computed from the measured counts)

Two-proportion comparison, normal approximation, 80% power, two-sided α = 0.05:
`MDD = (z₀.₉₇₅ + z₀.₈₀) · sqrt(2·p·(1−p)/n)` with `z₀.₉₇₅ = 1.9600`, `z₀.₈₀ = 0.8416` (sum 2.8016). With no model
results yet, `p = 0.5` is the conservative variance; `p = 0.3` and `p = 0.2` are shown for context.

| Cell | n | MDD @ p=.5 | MDD @ p=.3 | MDD @ p=.2 |
|---|---:|---:|---:|---:|
| repair / test_id | 6 | 0.809 | 0.741 | 0.647 |
| repair / test_ood_fault | 70 | 0.237 | 0.217 | 0.189 |
| repair / test_ood_topology | 15 | 0.511 | 0.469 | 0.409 |
| audit / test_id | 16 | 0.495 | 0.454 | 0.396 |
| audit / test_ood_topology | 6 | 0.809 | 0.741 | 0.647 |
| ledger / test_id | 11 | 0.597 | 0.547 | 0.478 |
| ledger / test_ood_fault | 32 | 0.350 | 0.321 | 0.280 |
| ledger / test_ood_topology | 7 | 0.749 | 0.686 | 0.599 |
| design / test_id | 4 | 0.991 | 0.908 | 0.792 |
| design / test_ood_climate | 20 | 0.443 | 0.406 | 0.354 |
| design / test_ood_load | 13 | 0.549 | 0.504 | 0.440 |
| design / test_ood_scale | 14 | 0.529 | 0.485 | 0.424 |

Plain statement: **not one reported cell reaches 85 rows.** The largest is `repair/test_ood_fault` (n = 70): it can
only detect a difference of about **24 percentage points** at p = 0.5 (19 points even at p = 0.2). Every other cell is
below 32 rows and detects **35–99 points**. To detect the plan's 15-point difference at 80% power needs about
**175 paired rows per cell** at p = 0.5 (147 at p = 0.3, 112 at p = 0.2); the plan's "85 rows ≈ 15 points" is only
reachable under a paired-correlation assumption (ρ ≈ 0.5) that cannot be asserted before results exist. The paired
McNemar tests in §5 can be more powerful when the two arms' errors are correlated, but no such gain is claimed here.

**Underpowered comparisons and defensible merges.** All twelve cells are underpowered for the paper's pairwise
claims; `repair/test_ood_fault` is the least bad but still below 85. Pooling may be reported **only** as a labelled
aggregate, never as a way to claim per-family significance:

- Defensible: `repair/test_ood_fault` + `repair/test_ood_topology` → one **"repair structural-OOD"** cell, n = 85,
  MDD @ p=.5 = 0.215. Both splits are structural OOD for the same task and the same success metric.
- Defensible with the same caveat: `ledger/test_ood_fault` + `ledger/test_ood_topology` → **"ledger structural-OOD"**,
  n = 39, MDD 0.317; and `design` climate + load + scale → **"design OOD-tail"**, n = 47, MDD 0.289. These remain
  underpowered for 15 points; keep the per-split numbers visible.
- **Not defensible:** merging `test_id` with an OOD split (destroys the in-distribution vs OOD distinction); pooling
  across tasks (different metric definitions); pooling audit's 16 id rows with its 6 topology rows (n = 22, still
  tiny). Audit stays descriptive-only; no merge rescues it.

## 7. Contamination gate

Before any arm is scored, `tests/test_cohort_coverage.py` and `tests/test_contamination.py` must pass with **zero**
held-out/dev overlap against every `data/sft*`, `data/grpo*`, `data/processed*`, `data/frontier_export`,
`data/frontier_heldout_export`, `thermo-teacher-prompts*` and memory note. The held-out export
(`data/frontier_heldout_export`) and the training export (`data/frontier_export`) must never share ids. The gate result
and the §1 eval-set hash are recorded in the run log; a failure voids every report scored on that set.

## 8. Explicitly not allowed

- Scoring, reporting or tuning on `train`/`val` (in `data/heldout` or any other directory), or using held-out rows as
  GRPO prompts.
- Mixing splits, changing `SPLITS`, `LIMIT` or `DATA_DIRS` between arms, or restricting a frontier arm (including
  `frontier_tools`) to `test_id`.
- Reporting a GRPO arm without its own verified adapter, or on raw Qwen weights instead of `SFT_MERGED`.
- Reporting `ours_harness` as a distinct arm while its config is identical to `ours_sft` (§3); reporting a number that
  does not trace to a `reports/*.json` with the §4 fields and the §1 hash.
- Claiming significance in a cell below the §6 threshold, or presenting a labelled merge as a per-split result.

## Appendix A — measurement commands and output (run 2026-11-19)

Cells, topology cross-table, totals and MDD were produced by:

```powershell
cd C:\Users\rezax\Home\Code\thermo_agent\Repo
Get-ChildItem data/heldout/*_test_*.jsonl | Sort-Object Name | ForEach-Object {
  $n = (Get-Content $_.FullName | Where-Object { $_.Trim() -ne '' }).Count
  "{0,-42} {1,5}" -f $_.Name, $n }
```

```
audit_test_id.jsonl                           16
audit_test_ood_topology.jsonl                  6
design_test_id.jsonl                           4
design_test_ood_climate.jsonl                 20
design_test_ood_load.jsonl                    13
design_test_ood_scale.jsonl                   14
ledger_test_id.jsonl                          11
ledger_test_ood_fault.jsonl                   32
ledger_test_ood_topology.jsonl                 7
repair_test_id.jsonl                           6
repair_test_ood_fault.jsonl                   70
repair_test_ood_topology.jsonl                15
```

A Python pass over the same files verified unique ids per file, `row.split == filename split`, the topology counts and
the total 214; it also printed the MDD table above with `z₀.₉₇₅ = 1.959963984540054`,
`z₀.₈₀ = 0.8416212335729143`, `MDD = zsum·sqrt(2p(1−p)/n)`. The eval-set hash was computed by:

```powershell
@'
import hashlib, glob, os
files = sorted(glob.glob('data/heldout/*_test_*.jsonl'))
h = hashlib.sha256()
for f in files:
    h.update(os.path.basename(f).encode()); h.update(open(f,'rb').read())
print(h.hexdigest(), len(files), sum(os.path.getsize(f) for f in files))
'@ | .venv/Scripts/python.exe -
# 8b409359e7a2dd441d0be99c89c00f0f3c61320f9d06e9695cc22569f8ef4c6e 12 1531809
```
