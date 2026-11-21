# Data samples (redacted, tiny) — for orientation only

These are **representative single rows** copied locally from each corpus so another
model can see the exact JSON shapes without loading full corpora. Each `*.jsonl`
file holds **one** JSON object on one line. Long strings are capped and marked
`...[TRUNCATED ...]`; long `trajectory.turns` lists keep the first 2 turns. Each
row carries a `__SAMPLE__` object with its `source_path` and a `note` describing
what was truncated. No credentials or token-like strings appear in any sample
(scanned: none found); no full corpus was copied. All samples are < 5 KB.

| Sample file | Source path | Schema / why it is here |
|---|---|---|
| `raw_prompt_shard.jsonl` | `data/frontier_tools_cohort_prompts/prompts_batch11_gpt-6-luna.jsonl` | Raw tracegen shard row: `id`, `split`, `topology`, `prompt` (system+user messages), `plant`, `site`, `task`, `cell`, `tier`. |
| `a_slice_row.jsonl` | `data/a_slice/design_train.jsonl` | Plain A-slice row — a seeded COPY of ground truth, one paraphrase per plant, `prompt_variant: plain`, no tool markers. |
| `heldout_v2_row.jsonl` | `data/heldout_v2/design_test_id.jsonl` | Frozen test-set row: `id`, `prompt`, `completion`, `site`, `ref_fuel_w`, `ref_spec`, `n_valid`, `ref_evals`, `tier`, `gen_seed`. Equal-thirds tiers per task. |
| `memory_note.jsonl` | `data/memory/frontier_notes.jsonl` | RAG memory note: short success/failure recap (`accepted`, `lesson`, `fix`, `symptoms`, `task`, `record_id`, `source`); model ids/timestamps/cost are stripped before serving. |
| `design_scoring_export.jsonl` | `data/frontier_export/design_train.jsonl` | Design scoring export row — same prompt/answer shape as the mix rows but **includes the scoring-only fields** `ref_fuel_w`, `ref_spec`, `ref_evals`. |

| `tool_outputs.json` | `thermo_agent.teacher.toolkit.make_tools` on `data/heldout_v2` | What `inspect_plant` returns for each task (shapes, units, per-line ledger residuals). |
| `det_trace_v3.json` | `thermo_agent/synth/tool_traces.py` on `data/heldout_v2` | Deterministic tool trace v3, one per task: reasoning built from the previous tool result's numbers. |
| `skills_on_demand.json` | `toolkit.list_skills` / `load_skill` over `thermo_agent/agent/skills/` | Description-only catalog vs full skill body loaded on demand. |


| `tool_outputs.json` | `thermo_agent.teacher.toolkit.make_tools` on `data/heldout_v2` | What `inspect_plant` returns for each task (shapes, units, per-line ledger residuals). |
| `skills_on_demand.json` | `toolkit.list_skills` / `load_skill` over `thermo_agent/agent/skills/` | Description-only catalog vs full skill body loaded on demand. |


**Scoring-only `ref_*` fields exist ONLY in the design scoring export sample**
(`design_scoring_export.jsonl`) among the mix/export rows. They were stripped from
the `sft_mix_v2` / `grpo_mix_v2` training rows and are read back at scoring time by
the GRPO design-reward join from `data/frontier_export/design_train.jsonl`
(1,216/1,216 design rows resolved). The ground-truth-derived samples (`a_slice`,
`det_tool`, `heldout_v2`) naturally carry their own reference values as part of the
ground truth; those are not the scoring join.

Dynamic tools (9 continuous-input, CoolProp-backed tools) are opt-in via
`include_dynamic`; the old 5-tool trace rows above remain valid and are not
migrated (DECISIONS §9).


No AI trace with reasoning exists yet: the 237 earlier ones had none and are archived in `data/archive/2026-11-20_pre_cot/`. The first reasoning-gated batch adds `frontier_trace_with_reasoning.json` here.
