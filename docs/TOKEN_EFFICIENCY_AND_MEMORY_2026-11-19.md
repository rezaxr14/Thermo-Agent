# Token efficiency and RAG-memory serving — 2026-11-19

Scope: **model-visible text only**. On-disk ledgers, manifests, journals and raw
verifier logs stay intact for provenance and papers. This note records the prompt
paths that were mapped, the bookkeeping surfaces found, the fix, and the measured
token reduction. No training/eval files were edited.

## 1. Prompt paths mapped

| # | Serving path | Builder before | Model saw | Bookkeeping surface found |
|---|---|---|---|---|
| 1 | Kaggle hosted tracegen (`kaggle/teacher_harness*.py` → `teacher/agent_graph.py::build_context` → `teacher/kbench_adapter.py::make_respond`) | full `AGENTS.md` + *all* component skills (`agent/harness.py::augment`), then D10 guidance, compact memory, and a ~180-token tool-name paragraph | 3,028–3,882 tokens of system text on one real row per task | row fields available but not serialised; harness text not shared with training/eval |
| 2 | Compact harness (`agent/tool_prompt.py`) — training `*_h` corpora (`scripts/build_tool_corpus.py`, `scripts/mix_tool_corpora.py`), eval H (`eval/model_eval.py::ToolLoop`), GRPO | already the single shared builder: compact AGENTS.md + the *one* selected skill + the five-tool contract | correct; 661–1,385 tokens per real row | none |
| 3 | Eval Z arm / harness-text arms (`eval/model_eval.py::_wrap`) | `build_compact_harness_messages(..., include_contract=False)` | task + artefact + compact AGENTS/skill, no tool contract | none |
| 4 | RAG memory (`teacher/memory.py::retrieve` → `teacher/compact_memory.py::compact_notes`) | retriever filters task, excludes the current `record_id`, reserves a failure slot; renderer emits one `OK`/`FAIL` recap line per note | fault class + reusable fix pattern only | **dropped:** `id`, `record_id`, `source`, `extra`, `architecture`, `symptoms`, model slugs, timestamps, `ts=`/`cost=`/`tokens=`/`latency=` |
| 5 | `agent/harness.py::augment_with_memory` (shared agent-layer memory builder) | `memory.augment_with_memory` → verbose `format_notes` | verbose `Lesson:/Fix:` block; note text was passed through verbatim | could carry bookkeeping from `lesson`/`fix`/`symptoms` |
| 6 | `teacher/sample.py::build_messages`, `synth/combine.py::_augment` (corpus building, **not owned**) | compact/verbose harness + `memory.augment_with_memory` | see §5 | verbose `format_notes` leak risk (reported, not edited) |

Bookkeeping on the **rows** themselves — `id`, `run_id`, `manifest`,
`export_manifest`, `journal`, `source`, `model`, `ts`, `record_id` — is never read
by any of the builders: they render only `row["prompt"]` (task statement +
artefact) plus the harness and the retrieved recaps. This is now pinned by a
sentinel-injection test on all four tasks and all three serving paths.

## 2. Changes made (files owned)

- **`thermo_agent/teacher/agent_graph.py`** — `build_context` now appends the
  harness through the shared `build_harness_tool_messages` (same function training,
  eval and GRPO use), passing the enforced round count as `max_turns`. The verbose
  full AGENTS.md / component-skill block is gone. The trailing ~180-token paragraph
  that re-listed every tool name was replaced by one short reminder that keeps the
  enforced round count in the prompt (D4). Retrieved memory is still rendered once
  by `compact_notes`, and `notes_used` still matches exactly the notes served.
- **`thermo_agent/agent/harness.py`** — `augment_with_memory` now retrieves once
  and serves `compact_notes` recaps instead of the verbose `format_notes` ledger;
  falls back to the task text as the retrieval query when there is no plant.
- **`tests/test_rag_memory_serving.py`** (new, 11 tests) — sentinel bookkeeping
  injected into rows and notes never reaches the model; OK/FAIL success/failure
  recaps appear when relevant; the hosted harness is byte-identical to the shared
  builder; the agent memory layer serves recaps, not the verbose ledger.
- **`tests/test_tool_prompt_token_audit.py`** (new, 1 test) — reproducible p50/max
  before/after measurement with the cached Qwen3.5 tokenizer and the 2,500-token gate.
- **`docs/TOKEN_EFFICIENCY_AND_MEMORY_2026-11-19.md`** — this document.

Physics content, tool results in SFT completions, final-answer tags and the
`<tool_call>`/`<tool_result>` format are unchanged. The compact harness builder's
text was **not** modified, so eval prompts and `harness_config_hash` are unaffected.

## 3. Token measurements (cached real Qwen3.5 tokenizer)

### 3a. Harness block, 8 real rows per task (32 rows)

| task | n | compact p50 / max | verbose p50 / max | saved p50 |
|---|---:|---:|---:|---:|
| repair | 8 | 661 / 661 | 3,921 / 3,921 | 3,260 |
| design | 8 | 1,041 / 1,041 | 2,894 / 2,894 | 1,853 |
| audit | 8 | 1,385 / 1,385 | 4,869 / 5,516 | 3,484 |
| ledger | 8 | 1,204 / 1,204 | 3,046 / 3,046 | 1,842 |
| **ALL** | **32** | **1,122 / 1,385** | **3,046 / 5,516** | **1,924** |

p50 reduction ≈ **63 %** (1,924 / 3,046). **Compact harness max = 1,385 ≤ 2,500**
(DECISIONS J2/J3 satisfied with 1,115 tokens of headroom).

### 3b. Hosted tracegen system prompt (one real row per task, same rows before/after)

| task | before | after | saved |
|---|---:|---:|---:|
| repair | 3,882 | 1,067 | 2,815 |
| design | 3,028 | 1,619 | 1,409 |
| audit | 3,676 | 1,813 | 1,863 |
| ledger | 3,047 | 1,649 | 1,398 |

Across 8 rows per task after the fix: p50 **1,634**, max **1,813** tokens.
The remaining ~220 tokens are the D10 method guidance (kept because a non-owned
test pins `"FIRST inventory"`), not bookkeeping.

## 4. Tests run (exact results)

- Changed-path suite (memory / agent / tool-prompt / RAG / harness / hosted /
  eval-arm guards / transfer / validate, including tests I do not own):
  **163 passed in 46.51 s** (re-run; earlier identical run: 58.22 s).
- New files only: `test_rag_memory_serving.py` **11 passed in 4.29 s**;
  `test_tool_prompt_token_audit.py` **1 passed in 10.47 s**.
- `scripts/audit_prompt_separation.py`: **24 arms, 0 mismatches** (Z/H separation intact).

No test file owned by the eval-arm work was edited.

## 5. Conflicts / gaps outside this agent's ownership (reported, not edited)

- **Eval-arm work** (`thermo_agent/eval/model_eval.py`, `kaggle/eval_runner.py`,
  `tests/test_eval_arm_prompts.py`, `tests/test_eval_runner_arms.py`,
  `tests/test_eval_f4_arms.py`, …) is concurrently modified in the working tree.
  This task made **no** edits there. The change in `agent_graph.py` affects only the
  hosted tracegen path; all eval arms still call the unchanged shared compact builder.
  The separation audit is green (24/24), so there is no prompt-contract conflict.
- **`thermo_agent/teacher/sample.py::build_messages`** uses the compact harness but
  still appends `memory.augment_with_memory`, which renders the verbose `format_notes`
  block verbatim. If a mining step ever writes a model id, timestamp or `run_id=` into
  `lesson`/`fix`, that text would reach the hosted teacher. Recommend routing it through
  `teacher.compact_memory.compact_notes` (a one-line change, outside owned files).
- **`thermo_agent/synth/combine.py::_augment`** still uses `harness.augment` (verbose)
  plus verbose memory for the A-slice. DECISIONS E4 intends the A-slice to be plain,
  so this is a lower priority, but it is a second non-compact serving surface.
- `tests/test_kbench_adapter.py`, `tests/test_sample.py` and `tests/test_combine.py`
  (not owned) assert the verbose markers `"FIRST inventory"` / `"KNOWN FAILURE MODES"`.
  They were deliberately kept green: the D10 guidance block remains in `build_context`,
  and the global `memory.format_notes` was not changed. Removing the guidance would
  save another ~220 tokens per hosted prompt but requires updating those three tests
  in the eval/teaching workstream first.
