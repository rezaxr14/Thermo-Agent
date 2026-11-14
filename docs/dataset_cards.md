# Dataset cards

All three datasets are synthetic: plants and sites are sampled by the code in this repository and labelled by the simulator and verifiers. They are not measurements. Measured data (the Frontier energy dataset) can set site specifications through `thermo_agent.datasets.frontier`, but no measured data is inside these files. Files are written as `<task>_<split>.jsonl` plus `meta_<task>.json` (seed, settings, CoolProp and schema versions). The data files are not in the repository; generate them with the commands in `docs/RUNBOOK.md`. Licence: not yet chosen by the authors.

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
