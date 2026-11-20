# Dataset cards

All four datasets (repair, ledger, audit, design) are synthetic: plants and sites are sampled by the code in this repository and labelled by the simulator and verifiers. They are not measurements. (The unrelated "Frontier energy dataset" can set site specifications through `thermo_agent.datasets.frontier`; no measured data is inside these files, and it is distinct from the *frontier tool traces* described below.) Files are written as `<task>_<split>.jsonl` plus `meta_<task>.json` (seed, settings, CoolProp and schema versions). The data files are not in the repository; generate them with the commands in `docs/RUNBOOK.md`. Licence: not yet chosen by the authors.

## Evaluation set and dev set (F1, 2026-11-19)

Per DECISIONS F1 (2026-11-19), which supersedes the earlier frozen protocol:
the **test set is `data/heldout_v2`** (new seed; manifest must be frozen before any model is scored on it),
and **`data/heldout` (214 `test_*` ids, seed 987654) is now the dev/tuning set** — it is never reported as the
headline test result. The older frozen protocol `docs/EVAL_PROTOCOL_HELD_OUT_DEV.md` was written while
`data/heldout` was still the evaluation set; its role as *test* is superseded by F1, while its cell counts below
remain an accurate description of the dev set. `data/heldout_v2` is **generated and FROZEN** (3,476 scored rows in
12 cells, equal thirds per task, contamination scan PASS with overlap 0; test hash `3cc51773…` recorded in
`docs/EVAL_PROTOCOL.md`). No arm has been scored on it.

Counts below (dev set `data/heldout`) were measured by counting non-empty lines in each `*_test_*.jsonl`, not copied
from prose; they agree with the `counts` in `meta_<task>.json`. **214 dev ids in 12 cells.**

| Task | `test_id` | `test_ood_fault` | `test_ood_topology` | `test_ood_climate` | `test_ood_load` | `test_ood_scale` | task total |
|---|---:|---:|---:|---:|---:|---:|---:|
| repair | 6 | 70 | 15 | — | — | — | 91 |
| audit | 16 | — | 6 | — | — | — | 22 |
| ledger | 11 | 32 | 7 | — | — | — | 50 |
| design | 4 | — | — | 20 | 13 | 14 | 51 |

Measured topology composition of the held-out `test_ood_topology` cells: repair 15 rows = `dc_engine_absorption-ext` 5
+ `dc_engine_electric-ext` 10; audit 6 rows = 1 + 5; ledger 7 rows = 2 + 5. This corrects the generic card wording
below ("steam-ORC plant and gas-engine plants"): the frozen held-out topology cells contain **only the two extended
gas-engine topologies**, no steam-ORC. Design rows have no `topology` field. Audit has no `test_ood_fault` cell.
The repair/ledger `test_ood_fault` cells mix the in-training topologies (repair: cgam_orc 4, dc_absorption 28,
dc_electric 28, dc_engine_absorption 6, dc_engine_electric 4; ledger: 3, 9, 16, 2, 2).

Every dev cell is below 85 rows (largest: repair `test_ood_fault`, n = 70), so no cell supports a 15-point pairwise
claim at 80% power; see `docs/EVAL_PROTOCOL_HELD_OUT_DEV.md` §6 for the computed minimum detectable differences and
the only defensible merges. This is precisely why F1 mandates `data/heldout_v2` with >= 85 rows per reported cell
where a generator can reach it, equal thirds per task (easy/medium/hard), and a frozen manifest before scoring.

## Data provenance and corpus status (2026-11-19)

This section is the stable mirror of the hand-maintained section in
`docs/PROVENANCE.md`. See that file for the per-row field definitions. Every row
states whether it is deterministic or AI-generated and, if AI, which model
produced it: `source` (`ground_truth` / `deterministic_tools` / frontier value
with `source_detail: frontier_tools`), `provenance` (deterministic rows:
`{generator, variant, source_row_id, source_dir}`), `model` (AI rows only), and
`paraphrase` (A-slice rows). Frontier rows do not carry a `provenance` dict;
they carry `model`, `provider`, `run_id` and `source_detail`.

| Source family | Producer | Model (AI rows) | Where stored | Rows |
| --- | --- | --- | --- | ---: |
| Deterministic ground truth | synth generators + simulator/verifiers; **nothing deleted or overwritten** | none (deterministic) | `data/processed` | **27,418** across all splits; train split repair 5,458 / ledger 4,846 / audit 2,192 / design 1,768 |
| Plain A-slice (COPY of ground truth) | `build_a_slice.py`, seeded, one paraphrase per plant, plain prompt, no tool markers | none (deterministic copy) | `data/a_slice` | **1,600** (400/task) = round(4,800 × 0.3333) |
| Deterministic tool traces | `synth.tool_traces`, 600 plants × 2 variants (`direct`, `recover`) × 4 tasks | none (deterministic) | `data/sft_tools_v2*`, `data/grpo_tools_v2*` (and `*_h`, `sft_mix_v2`) | **DONE: 4,800 per task family**, 100% verifier-accepted, real tool results, equal-thirds tiers |
| Frontier harness CoT (AI) | generic Kaggle runner, verifier-accepted | `gpt-6-luna`, `claude-haiku-5-5` | per-model frontier trace dirs fed from the HF corpus repo | **IN PROGRESS**: each model's 5-row proof (5/5 accepted) and one 100-row batch completed; journals at gate/ingest/audit. Target **thousands**, no cap, equal thirds. |

Notes on the two trace families:

- **Deterministic tool traces are multi-turn and complete.** In `data/sft_tools_v2*`
  the full `tool_call` / `tool_result` round sequence is stored in one SFT
  `completion` (a JSON list of assistant turns ending in the verified answer);
  in `data/grpo_tools_v2*` the same rounds are in `trajectory.turns`. The corpus is
  built at the decided 4,800 rows per task family = 600 plants/task × 2 variants ×
  4 tasks; tiers are equal thirds per task from `thermo_agent/synth/tiers.py`.
- **Frontier harness-CoT traces are being collected now.** Only `gpt-6-luna` and
  `claude-haiku-5-5` are authorized. Each accepted trace carries its exact `model`
  id, tool calls and verifier verdict; rows are gated at 8,192 tokens (over-gate
  rows dropped, never truncated). A small set of legacy pilot rows recorded in the
  provenance ledger comes from a superseded early campaign and is not part of the
  current authorized campaign.

**Frontier models NOT to be run:** `gpt-6.2-sol`, `claude-sonnet-5-5`,
`claude-opus-5-5-default`, `gemini-4-argon`, `gemini-3.8-flash` and
`gemini-3.5-flash-lite`. None was run; no traces, costs, acceptance counts or
scores exist for them. The old 72-cell premium target is discarded. The A-slice
and all deterministic corpora are AI-free. Data flows local → private HF repo
`Reza-Nadimi/thermo-agent-corpora` → Kaggle kernel → one Kaggle dataset; no local
uploads to Kaggle, credentials only in the repository-root `.env`, one generic
reusable Kaggle notebook for all workflows.

## Ledger
Purpose: repair an equation-level fault in the exergy ledger of a solved plant (stream exergy lines, component destruction lines, the ambient constant). These faults leave a plant that solves and passes parameter-range checks; only the ledger's own checks (closure, negative destruction) can reveal them, and some are invisible even to those. Generator: `thermo_agent.ledger.data`; 13 fault types in three layers (stream lines, component lines, the T0 constant); one fault (70 percent) or two. No code is executed: lines are arithmetic expressions checked by a restricted evaluator.
Columns: `id`, `split`, `topology`, `fault_types`, `prompt`, `completion`, `ledger`, `min_edits`.
Splits: `test_ood_topology` (steam-ORC plant and gas-engine plants), `test_ood_fault` (wrong_stream, dropped_work, wrong_ref), then `train`, `val`, `test_id` by hash.
Notes: `ledger` is a JSON string with the tables, topology, broken lines, truth and the numbers shown in the prompt, so rewards need no other column. A program that rebuilds every line from the topology solves every row by construction (`thermo_agent.ledger.run_eval` reports this); the benchmark measures whether a model applies that rule and localises the fault from text.

## Repair
Purpose: fix a plant that fails the solver or the verifiers. Generator: `thermo_agent.synth.build_repair_data`; one fault of five types (unit slip, sign flip, efficiency above 1, swapped ports, wrong fluid) injected into a sampled valid plant of one of five topologies. Only faults the solver or the verifiers detect are kept.
Columns: `id`, `split`, `topology`, `fault_type`, `prompt`, `completion`, `plant_broken`, `plant_original`, `site`, `deducible`, `min_edits`, `detected`.
Splits: `test_ood_topology` (steam-ORC plant and gas-engine plants), `test_ood_fault` (sign flip, swapped ports), then `train`, `val`, `test_id` by hash of the id.
Notes: `deducible` is false when the original value cannot be recovered from the broken plant (an efficiency above 1 that is not a percentage); exact-match reward applies only when it is true. `completion` is a programmatically written expert answer with a number-grounded diagnosis. Turbine plants are drawn three times as often as the held-out topologies.

## Design
Purpose: choose a plant for a site. Generator: `thermo_agent.design.data`; sites are sampled (IT load 5 to 60 MW log-uniform, chilled fraction 0.3 to 0.95, ambient 283 to 310 K, humidity 0.2 to 0.9, optional heat demand and water limit). The reference is the best of enumeration plus a Nelder-Mead polish; use `--max-evals 60` or more.
Columns: `id`, `split`, `prompt`, `completion`, `site`, `ref_fuel_w`, `ref_spec`, `n_valid`, `ref_evals`.
Splits: `test_ood_scale` (IT load >= 45 MW), `test_ood_climate` (ambient >= 305 K), `test_ood_load` (chilled fraction >= 0.85), then `train`, `val`, `test_id` by hash.
Notes: the reference fuel is the best design found by this search, not a proven optimum. Outcomes depend on assumed technology constants (see `paper/methods_draft.md`, last section).

## Audit
Purpose: rank exergy destroyers, propose one allowed change, predict its fuel effect. Generator: `thermo_agent.audit.data`; sampled valid plants (some with ORC bottoming and district heat). Only plants where at least one allowed change saves fuel are kept.
Columns: `id`, `split`, `topology`, `prompt`, `completion`, `plant`, `site`, `ed_json`, `fuel_w`, `net_w`, `best_gain_w`, `n_candidates`.
Splits: `test_ood_topology` (steam-ORC plant and gas-engine plants), then `train`, `val`, `test_id` by hash.
Notes: `ed_json` holds exergy destruction per component (the IT load excluded) and is hidden from the prompt. The ranking reward ignores the combustor or gas engine, which was the largest destroyer in every plant generated so far. `best_gain_w` is the largest fuel saving among the allowed changes after re-sizing the prime mover to the same net electricity.
