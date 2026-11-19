# Held-out test set — frozen, never-seen evaluation set

Seed **987654**, generated 2026-11-18 into `data/heldout/` with the standard generators
(`build_repair_data`, `audit.data`, `ledger.data`, `design.data --max-evals 60`).

**This set is excluded from every training corpus** (`sft`, `sft_all`, `grpo`, `grpo_all`), every
teacher prompt shard (`data/frontier_export`, `thermo-teacher-prompts*`), and every provenance
source. It is attached to Kaggle as `heldout.zip` in the corpus dataset and is the single set on
which ALL arms are scored (so the zero-shot frontier baseline and our final model are directly
comparable on identical, never-seen rows).

**Evaluation protocol is frozen in `docs/EVAL_PROTOCOL.md`**: only the `test_*` rows below are scored, with identical
splits and limits for every arm; the complete arm/config table and the power statement live there.

## Splits (D8 / the user's cohort-coverage decision)

| Task | splits present | rows (sampled from generation) |
|---|---|---|
| repair | train, val, test_id, test_ood_fault, test_ood_topology | 94 / 13 / 6 / 70 / 15 |
| audit  | train, val, test_id, test_ood_topology | 84 / 9 / 16 / 6 |
| ledger | train, val, test_id, test_ood_fault, test_ood_topology | 60 / 10 / 11 / 32 / 7 |
| design | train, val, test_id, test_ood_climate, test_ood_load, test_ood_scale | 54 / 4 / 4 / 20 / 13 / 14 |
| **test total** | 12 `test_*` files | **214 ids** (repair 91, audit 22, ledger 50, design 51) |

Counts re-measured 2026-11-19 by counting non-empty lines in every `*_test_*.jsonl`; all ids are unique per file and
each row's `split` matches its filename. These agree with the `counts` in `meta_<task>.json`. Evaluation uses the
`test_*` splits only (train/val are never scored). The twelve cells cover every OOD family the generators support:
`test_ood_fault` for repair and ledger, `test_ood_topology` for repair, audit and ledger, and climate/load/scale for
design. Audit has no `test_ood_fault` cell; design rows carry no `topology` field. `LIMIT` must be at least 70 (the
largest cell) so no cell is truncated.

## Difficulty balance (D-cohort decision)

- **simple** — `data/processed` MIX family (seed 0), all four tasks.
- **medium** — `data/processed_arch` 65-diagram catalog (structural OOD), repair/audit/ledger.
- **hard** — `data/processed_big` 19-component / 34-stream two-pressure steam-reheat cycle, repair/audit.
- **never-seen** — this held-out set (seed 987654), all four tasks.

`tests/test_cohort_coverage.py` asserts this coverage and that held-out ids share zero overlap with
the seen corpora (seed-namespace + first-line-sample proof).

## Arms scored here (in order, each recorded separately)

The authoritative arm list and exact configs are the frozen `docs/EVAL_PROTOCOL.md` §3 (which also includes
`ours_base`, `ours_harness` and the frontier Z/H distinction). The original objective list was:

1. `frontier_zero` — frontier zero-shot baseline.
2. `frontier_tools` — frontier + our verifier tool (replay).
3. `frontier_harness` — frontier in the LangGraph harness with real tools + skills + memory (tool-using CoT).
4. `other_model` — other thermo LLM (GPT-nano stand-in until a domain id is supplied).
5. `ours_sft` — our 4B after SFT alone.
6. `ours_sft_grpo_k4` / `_k8` / `_k16` — SFT + GRPO at group sizes 4, 8, 16, each from its own
   checkpoint dir (`runs/grpo_<task>_k4`, `_k8`, `_k16`) so they never overwrite each other.

Score with `kaggle/eval_runner.py` setting `DATA_DIRS=heldout` (the notebook resolves it to
`/kaggle/working/data/heldout`). Reports are written to `reports/<arm>.json` and assembled into
`paper/assets/model_results.json` by `thermo_agent.eval.comparison`.
