# Audits (dated entries)

## Native protocol + 8,192 gate audit - 2026-11-22 (step 3 rebuild)

After the step-3 prompt additions the harness corpora were rebuilt and re-measured:
**10,404 rows, 0 over 8,192 (rendered), 0 over 8,192 (pipeline), 0 anomalies**.
Per-task rendered p50/p95/max: audit 5,709/8,073/8,096, design 4,819/8,070/8,096,
ledger 5,586/8,055/8,096, repair 5,320/7,131/7,451. The extra audit-id line cost
539 deterministic rows; they were dropped by the gate, never truncated.

## Native protocol + 8,192 gate audit - 2026-11-22

`python -m scripts.audit_harness_corpus --dirs data/sft_tools_v3_h
data/grpo_tools_v3_h data/sft_mix_v3 data/grpo_mix_v3 --out
reports/harness_corpus_audit_native.json` with the cached Qwen3.5 chat tokenizer:
**12,560 rows, 0 over 8,192 (rendered), 0 over 8,192 (pipeline), 0 anomalies**.
Per-task rendered p50/p95/max: audit 5,671/8,057/8,096; design 4,819/8,070/8,096;
ledger 5,982/8,003/8,096; repair 5,320/7,131/7,451. The earlier build (before the
exact gate) had 3,270 over-gate rows, confirming the old plain-text gate was not a
valid measurement. The five remaining over-gate frontier rows were removed by an
exact gate at mix time; legacy frontier traces were migrated to structured messages.

Every audit of traces, training path, memory and directive compliance. Add new audits here as dated sections.

## Training-trace quality audit — 2026-11-19

*(merged from `docs/AUDITS.md` on 2026-11-20; content kept, headings demoted one level)*

**Scope.** Read-only inspection of every training trace produced so far: SFT multi-turn deterministic,
GRPO rows, and AI/frontier harness traces. No data, corpora, models or checkpoints were modified. No
network, Kaggle, training or scoring was used. Analyzer scripts were written only under
`C:\Users\rezax\AppData\Local\Temp\trace_audit\` and read from `data/` with plain `open()`.

**Verdict up front: the data is NOT ready for training.** The deterministic SFT/GRPO corpora are clean and
well-instrumented, but the frontier six-model cohort is incomplete (4 of 6 models absent, several
task×tier cells empty) and `scripts/check_datagen_complete.py` exits **1** (`passed=false`). In addition,
the luna/haiku pilot traces are not gate-cleaned (13/28 haiku and 2/27 luna rows exceed 8,192 tokens)
and 10 haiku rows called "tool traces" contain zero tool calls.

---

### 0. Commands / tooling used (all read-only)

Interpreter: `.venv/Scripts/python.exe` (Python 3.12.3). Working directory: repo root.

| Purpose | Command / tool |
|---|---|
| Manifest summary | `$TEMP\trace_audit\summarize_manifests.py` (parses the 6 `MANIFEST.json` files, abbreviates large id lists) |
| Row schemas | `$TEMP\trace_audit\inspect_schemas.py` |
| SFT markers + GRPO structure | `$TEMP\trace_audit\sft_grpo_checks.py` |
| Final-answer tails + AI traces | `$TEMP\trace_audit\ai_traces.py` |
| Tool-bearing counts + SFT/GRPO prompt compare | `$TEMP\trace_audit\prompt_compare.py` |
| Overlap / leakage | `$TEMP\trace_audit\overlaps.py`, `$TEMP\trace_audit\prompt_overlap.py` |
| Token lengths (real Qwen tokenizer) | `$TEMP\trace_audit\token_lengths.py` (`AutoTokenizer.from_pretrained(C:\Users\rezax\.cache\thermo\qwen3.5-4b)`, `len(tok.encode(prompt+completion))+96`) |
| Tool-result JSON + token spot-check | `$TEMP\trace_audit\verify_fields.py` |
| GRPO reward detail | `$TEMP\trace_audit\grpo_rewards.py` |
| Masking test | `.venv/Scripts/python.exe -m pytest tests/test_tool_loss_masking.py -q` → **8 passed** |
| Gate | `.venv/Scripts/python.exe scripts/check_datagen_complete.py` → **exit 1**, report saved to `$TEMP\trace_audit\gate_report.json` |
| Docs | `reports/teacher_campaign_log.md`, `docs/DECISIONS.md` §7, `docs/DECISIONS.md`, `docs/RUNBOOK.md`, `thermo_agent/train/{sft,grpo}.py`, `scripts/check_datagen_complete.py` |

---

### 1. Inventory — row counts per task/source/tier and the role of each source

All counts are measured by counting JSONL lines (cross-checked against manifests).

| Directory | Total | repair | design | audit | ledger | Composition | Prompt variant |
|---|---:|---:|---:|---:|---:|---|---|
| `data/sft_tools_v2` | **4,832** | 1,209 | 1,207 | 1,206 | 1,210 | 4,800 deterministic + 32 frontier | `plain` (system 773 chars) |
| `data/grpo_tools_v2` | **4,832** | 1,209 | 1,207 | 1,206 | 1,210 | 4,800 deterministic + 32 frontier | `plain` |
| `data/sft_tools_v2_h` | **4,831** | 1,209 | 1,207 | 1,206 | 1,209 | 4,800 deterministic + 31 frontier | `harness_tools` (system 3,248 chars) |
| `data/grpo_tools_v2_h` | **4,831** | 1,209 | 1,207 | 1,206 | 1,209 | 4,800 deterministic + 31 frontier | `harness_tools` |
| `data/sft_mix_v2` | **6,431** | 1,609 | 1,607 | 1,606 | 1,609 | 4,831 harness tool rows + 1,600 plain-A | `harness_tools` + `plain` |
| `data/grpo_mix_v2` | **4,831** | 1,209 | 1,207 | 1,206 | 1,209 | 4,800 deterministic + 31 frontier | `harness_tools` |
| `data/a_slice` | **1,600** | 400 | 400 | 400 | 400 | 1,600 plain ground-truth (no tools) | `plain` |
| `data/frontier_tools_clean_v2` | **32** | 9 | 7 | 6 | 10 | cleaned Gemini traces | `plain` |
| `data/frontier_tools_pilot/gpt-6-luna` | **27** | 8 | 8 | **0** | 11 | luna pilot (all tool-bearing) | harness |
| `data/frontier_tools_pilot/claude-haiku-5-5` | **28** | 11 | 4 | 2 | 11 | 18 tool-bearing + 10 zero-shot | harness |
| `data/heldout_v2` (test only) | **3,476** | 1,216 | 826 | 480 | 954 | 12 cells, frozen test set | `plain` |

**Tier distribution.** `sft_tools_v2`/`sft_tools_v2_h` use near-equal thirds by component-count rank
(e.g. repair 400/404/405 easy/medium/hard). `sft_mix_v2` re-tiers all emitted rows to exact thirds:
repair 939/939/940, design 938/938/938, audit 937/937/938, ledger 939/939/940. `a_slice` rows carry
**no `tier` field** (the mix builder re-derives it). `heldout_v2` per-task tiers: repair 405/405/406,
audit 160/160/160, ledger 318/318/318, design 275/275/276 — the gate confirms equal thirds.

**Exact role of each source:**

- **Deterministic tool rows (4,800 per corpus, 600 plants × 2 variants `direct`/`recover` per task).**
  Generated by `synth.tool_traces` from `data/processed`, `plant_seed=0`, `max_tool_result_chars=4500`.
  These are the backbone of SFT and GRPO: full `<think>/<tool_call>/<tool_result>/answer` multi-turn
  traces (SFT) and the matching prompts + verified references (GRPO). `max_tokens=8000`, `limit=0`.
- **Plain A slice (1,600, 400/task).** Selected from `data/sft_all` with "first row per ground-truth
  plant wins", no tool markers, `harness=none`, `prompt_variant=plain`, `ratio=0.3333` of the 4,800
  tool rows; 8,192 gate applied (0 dropped). It adds direct-answer coverage without tool scaffolding.
  Note `sft_mix_v2` manifest records `a_slice_ratio: 0.0` but with `cap_precedence` "0/0 means no cap",
  so all 1,600 A rows were included — the `0.0` is a cap setting, not zero inclusion.
- **AI / frontier traces.** Only `google-gemini-3-8-flash` traces are in the training corpora: 32 in
  `sft_tools_v2`, 31 in the `harness_tools` corpora (one dropped for exceeding the gate). The luna and
  haiku pilots live **only** in `data/frontier_tools_pilot/` and are not mixed into any training dir.
- **`data/heldout_v2`** is the frozen paper test set; its manifest forbids using any of its rows as a
  GRPO prompt.

---

### 2. SFT quality

All four SFT dirs were checked row by row (`sft_grpo_checks.py`, `ai_traces.py`, `verify_fields.py`).

**Good — completions end with the verified final-answer markup.** 100% of rows in every dir end with the
task's tail tag: repair `</edits>`, design `</design>`, audit `</prediction>`, ledger `</fix>`.
`data/sft_tools_v2`: repair 1,209/1,209, design 1,207/1,207, audit 1,206/1,206, ledger 1,210/1,210.
`data/sft_tools_v2_h`, `data/sft_mix_v2`, `data/a_slice`: likewise 100% (0 bad in each task).

**Good — tool markers present exactly where expected.** Every tool row has both `<tool_call` and
`<tool_result` in the completion: `sft_tools_v2` 4,832/4,832; `sft_tools_v2_h` 4,831/4,831;
`sft_mix_v2` 4,831/4,831 (the 1,600 A rows have 0, by design); `a_slice` 0/1,600 (by design).

**Good — tool results are verbatim physics output (structurally verified).**
`sft_tools_v2`: 13,926 `<tool_result>` blocks, **13,926 valid JSON dicts**; top keys `ok` (13,926),
`failed_checks` (12,726), `valid`/`checks` (8,443 each), `exergy_closure_residual_W` (6,072),
`total_exergy_destruction_MW` (5,451), `unit_range_flags` (5,441), `n_streams` (4,823).
`sft_tools_v2_h`: 13,922/13,922 valid JSON dicts. `frontier_tools_clean_v2`: 128/128 valid JSON dicts
with `tool`, `ok`, `failed_checks`. We could not re-run the physics engine (read-only, no scoring), so
this verifies structure/provenance, not a fresh numerical re-derivation; the generator is recorded as
`synth.tool_traces (deterministic)` in the manifests.

**Good — no `<tool_result>` leaks into labels (verified in code and by test, not guessed).**
`thermo_agent/train/sft.py` defines `_tool_result_spans` (lines 42–71) and `mask_tool_result_spans`
(74–94); both XML `<tool_result>` and Qwen-native `<tool_response>` spans are set to `-100`. This is wired
into the real training path: `preprocess_row` (107–127) → `preprocess_rows` (130–133) → called in
`sft.main()` at line 216, with `completion_only_loss=True` at line 229. Tool *calls* and the final answer
remain trainable. `pytest tests/test_tool_loss_masking.py -q` → **8 passed**, including exact span
boundaries, multiple/unclosed spans, and a loss/gradient equivalence check.

**Good — token length distribution and the 8,192 gate.** The stored `tokens` field is
`rendered prompt + completion + 96-token chat reserve`; a 200-row spot-check per dir against the real
Qwen3.5 tokenizer gave mean differences of −0.2 / 0.0 / −0.0 tokens (max |diff| = 1), so the field is
trustworthy.

| Dir | n | p50 | p95 | max | rows > 8,192 |
|---|---:|---:|---:|---:|---:|
| `sft_tools_v2` (manifest) | 4,832 | 3,753 | 4,731 | 7,268 | 0 (`dropped_over_8k=0`) |
| `sft_tools_v2_h` (manifest) | 4,831 | 4,938 | 5,953 | 7,600 | 0 (`dropped_over_8k=1`) |
| `sft_mix_v2` (measured `tokens`) | 6,431 | — | — | 7,600 | 0 |
| `a_slice` (manifest) | 1,600 | 1,454 | 2,492 | 4,065 | 0 |

`sft_tools_v2_h` dropped one frontier row for being over budget (`frontier.over_budget=1`, ledger
frontier 10 → 9); `sft_tools_v2` dropped 2 rows with oversized tool results. Rows are dropped, never
truncated — correct behavior.

**Good — no heldout contamination.** `sft_tools_v2`/`sft_tools_v2_h` manifests record a
`contamination_guard` against `data/heldout` (dev) with `overlap=0, leaks=[], passed=true` over 4,832
and 4,831 rows. `data/heldout_v2/MANIFEST.json` records a broader check: `overlap=0`, `leaks=[]`,
`passed=true`, `files_scanned=298`, `lines_scanned=454,302`, `signatures=10,578`,
`self_check_detects_planted_row=true`, covering all v2 SFT/GRPO/frontier/pilot dirs. Our independent
exact prompt-hash comparison (system+user content SHA-256) found **0** overlap between every train set
and every `heldout_v2` test cell, for all four tasks (§5).

**Weak / missing.** The `tokens` count is a post-hoc field, and only the frontier-clean dir has a true
tokenizer-measured before/after manifest. `a_slice` has no `tier` column and no `record_id` in some
rows; provenance labels are inconsistent for A rows (`source` is `ground_truth` for 399 repair but
`frontier` for 1, 34 design, 10 ledger — inherited from `sft_all`). This is documented-ish but messy.

---

### 3. GRPO quality, and why GRPO has its own data

#### 3.1 Structural check (all rows, `sft_grpo_checks.py`)

| Dir | n | missing `task`/`tier`/`prompt` | empty `completion` | empty `trajectory.turns` | no `trajectory.final` | `validation.reward > 0` |
|---|---:|---:|---:|---:|---:|---:|
| `grpo_tools_v2` | 4,832 | 0 / 0 / 0 | 0 | 0 | 0 | 4,832 (100%) |
| `grpo_tools_v2_h` | 4,831 | 0 / 0 / 0 | 0 | 0 | 0 | 4,831 (100%) |
| `grpo_mix_v2` | 4,831 | 0 / 0 / 0 | 0 | 0 | 0 | 4,831 (100%) |

Every row carries `task`, `tier`, `prompt` (system+user), a non-empty **final completion used as the
reference** (`completion[0].content`, duplicated in `trajectory.final`), non-empty `trajectory.turns`
(the sampled tool rollout), `trajectory.sft_completion`, and `validation = {reward, reasons}`. Rewards:
1.0 for essentially all repair/design/ledger rows; audit rows are continuous ≈0.9996–0.99995 with
reason `audit accepted`; **6 repair rows (3 plants × direct/recover) have reward 0.45** with reason
`restored a valid plant exactly`; 4 Gemini audit rows ≈0.9978–0.9998. All are accepted (reward > 0).

**Pure deterministic vs frontier:** `grpo_tools_v2` 4,800 deterministic / 32 frontier (repair 9,
design 7, audit 6, ledger 10); `grpo_tools_v2_h` and `grpo_mix_v2` 4,800 / 31 (ledger 9). Worthy of
note: `grpo_mix_v2/MANIFEST.json` `counts` says `frontier: 0` but `provenance` says
`ai_generated:google-gemini-3-8-flash: 31`. This is a labeling artifact — the mix builder treated the
whole `grpo_tools_v2_h` dir as "deterministic_grpo", so the 31 Gemini rows arrived inside that bucket.
The data is present and correctly labeled per-row; only the manifest's source counter is misleading.

**Weak — prompts are duplicated ~2×.** The 1,200 deterministic rows per task are 600 plants ×
`direct`/`recover`, and the two rows share an identical prompt. By prompt hash, each GRPO task has only
~604–609 unique prompts (repair 609, design 604, audit 605, ledger 608/609), not 1,209; up to 3 rows
share one prompt in design/audit/ledger (a frontier trace colliding with a deterministic pair).
For GRPO this means duplicated rollout starts with different reference completions. It may be deliberate,
but it is not documented as such and double-weights those plants during sampling.

#### 3.2 Why GRPO has its own training data (conceptual answer)

GRPO is initialized from the **merged best SFT checkpoint** (`runs/sft_merged`, decision **C10**) and
trains a *separate* adapter (`grpo_<task>_k<K>`, r8/alpha16 on top of the merged SFT model). The SFT
adapter is the **starting policy only** — GRPO never rewrites it.

What SFT and GRPO consume are different things:

- **SFT** teaches the policy the answer/tool format by supervised loss on the expert completions (with
  tool results masked). Its rows = prompt + completion.
- **GRPO** needs **prompts**, not answers. It samples G rollouts per prompt from the current policy and
  scores each rollout with the task's physics verifiers/rewards. The reference completion stored in the
  GRPO file exists for auditing/reward input; it is **not** the object of supervised loss.

This is confirmed in code: `thermo_agent/train/grpo.py::load_rows` (lines 361–366) keeps only the reward
columns declared in `TASKS[task][2]` — e.g. repair `("prompt", "plant_broken", "plant_original",
"site", "deducible", "min_edits")` plus `task`, `tier`, `id`. `completion` and `trajectory` are never
loaded. `docs/DECISIONS.md` §7 states: "SFT rows teach the policy the answer/tool format;
GRPO needs the same task prompts to sample rollouts and score them with the task verifiers/rewards. The
reference answer is a reward input, not the object of supervised loss. **Different prompt sets are
allowed only when documented in the manifests; the SFT adapter is only the starting policy for GRPO.**"

**Are our prompt sets aligned?** Yes, and the difference is documented. Comparing prompt hashes and
record ids between `sft_mix_v2` and `grpo_mix_v2`:

| Task | SFT rows | GRPO rows | unique prompts SFT / GRPO | prompt overlap | SFT-only | GRPO-only | record-id overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| repair | 1,609 | 1,209 | 1,009 / 609 | **609** | 400 | 0 | 609 |
| design | 1,607 | 1,207 | 993 / 604 | **604** | 389 | 0 | 607 |
| audit | 1,606 | 1,206 | 1,005 / 605 | **605** | 400 | 0 | 606 |
| ledger | 1,609 | 1,209 | 1,007 / 608 | **608** | 399 | 0 | 609 |

Every GRPO prompt appears verbatim in SFT; the SFT-only prompts (400 per task, 389 design, 399 ledger)
are exactly the plain-A rows, which by design have no tool harness and are not GRPO prompts. There are
**zero GRPO-only prompts**. Normalized ids (`-t-` removed) overlap on all 1,200 deterministic rows per
task; the remaining 9/7/6/9 GRPO-only ids are the frontier rows whose id suffix differs. So the two
sets are aligned, not divergent. The only caveat is the 2× within-GRPO prompt duplication noted above.

---

### 4. AI / frontier trace quality

#### 4.1 Measured per trace set

| Set | Rows | Task split | Tool-bearing | Verifier accepted | Provenance (model / provider / run) | Token length (real tokenizer, +96 reserve) |
|---|---:|---|---:|---:|---|---|
| `frontier_tools_clean_v2` (Gemini) | 32 | 9 / 7 / 6 / 10 | **32/32** (exactly 4 `<tool_call>` + 4 `<tool_result>`, 5 turns each) | 32 (30×1.0; 2 audit 0.998/0.999) | `google-gemini-3-8-flash` / `google` / `hosted-v6-7179338` | p50 3,147, p95 6,395, max 7,268, **0 over 8,192** |
| `frontier_tools_pilot/gpt-6-luna` | 27 | 8 / 8 / **0** / 11 | **27/27** (1–16 calls, 2–7 turns) | 27 (all 1.0) | `gpt-6-luna` / `openai` / `budget-pilot-7218725` | p50 4,939, p95 8,344, max 9,761, **2/27 over 8,192** |
| `frontier_tools_pilot/claude-haiku-5-5` | 28 | 11 / 4 / 2 / 11 | **18/28** (11 repair, 4 design, 2 audit, **1 ledger**) | 28 (27×1.0; 1 audit 0.992) | `claude-haiku-5-5` / `anthropic` / `budget-pilot-7218726` | p50 8,073, p95 16,424, max 17,746, **13/28 over 8,192** |

The "18 haiku tool traces" figure in the task is correct for **tool-bearing** traces; the directory
contains 28 accepted rows because 10 haiku ledger rows are direct single-turn, zero-tool answers
(kept as zero-shot rows, not padded into fake traces). Per `reports/teacher_campaign_log.md` lines 589–591,
the tool-bearing breakdown is haiku 11 repair / 1 ledger / 4 design / 2 audit, which our measurement
reproduces exactly.

`frontier_tools_clean_v2/MANIFEST.json` documents the cleaning that makes Gemini traces usable:
tokens reduced from max 17,212 / p50 7,191 to max 7,268 / p50 3,285 across 32 rows, 256 rewritten
blocks, `leftover_call_ids=0`, 32/32 rows changed. **The luna and haiku pilots have not received this
cleaning** — that is why they breach the gate. All pilot rows carry `model`, `provider`, `run_id`; the
clean Gemini rows carry `model`, `provider`, `run_id`, `source_detail=frontier_tools`.

#### 4.2 Failure reasons and the incomplete six-model cohort

From `reports/teacher_campaign_log.md` (lines 577–609):

- 40-row pilots (run ids 7218725 / 7218726): luna 40 answered, **27 accepted (all 27 tool-bearing)**;
  haiku 40 answered, **28 accepted but only 18 tool-bearing**, 10 zero-shot. No API/auth errors.
- **Failure taxonomy (verifier, after ingest):** luna 12 rejected — 8 `format` (7× "unknown component
  ids ['DC']", 1× empty `<edits>`), 2 `grounding`, 1 `verifier`, 1 `edit`; haiku 11 rejected — 5 `format`
  (all "unknown component ids ['DC']"), 6 `grounding` (70–79% of rationale numbers present in the
  plant). One journal id per model (`rep-130-00000#v5`) is unresolvable in the export dirs and yields no
  corpus row.
- The campaign-wide 18.2% acceptance figure covers only 11 graded rows (2 accepted / 9 rejected) out
  of 5,677 — it is not a campaign-level rate; 99.8% of rows are ungraded.
- **Binding rule (DECISIONS §6):** the gate stays RED until *every* model in the six-model cohort —
  budget `gpt-6-luna`, `claude-haiku-5-5`, then premium `gpt-6.2-sol`, `claude-sonnet-5-5`,
  `claude-opus-5-5-default`, `gemini-4-argon` — has verifier-accepted tool traces in **all four** tasks
  and **all three** tiers. Premium runs were never started, training was never started.
- **Gate result (just re-run, read-only):** `scripts/check_datagen_complete.py` → `passed=false`, exit
  1. All other items pass (v2 tool corpora row counts, A-slice 1600, frontier-clean 32, both mix dirs,
  heldout_v2 frozen + equal thirds, provenance/tier audit **11,262 rows / 0 problems**). The single RED
  item is `frontier CoT cohort (models x tasks x tiers)`: **77 accepted tool traces** (32 Gemini + 27
  luna + 18 haiku), with observed coverage:
  - luna: repair 2/2/4, design 2/2/4, **audit 0/0/0**, ledger 3/3/5 (easy/medium/hard)
  - haiku: repair 3/3/5, design 1/1/2, audit 0/0/2, ledger 0/0/1
  - all four premium models: every task/tier cell 0
  - Note: `gemini-3.8-flash` is not in `FRONTIER_MODELS`, so its 32 traces count toward the 77 but do
    not satisfy any cohort cell. Including it would still leave luna/audit and most premium cells empty.

**Conclusion: the six-model cohort is 2/6 present and far from the 6×4×3=72 cell coverage floor.**

---

### 5. Duplication / leakage risk (measured overlap counts)

1. **Same diagram in train and test — no leak.** Exact prompt-hash overlap (system+user content) between
   every train set and `heldout_v2` test = **0** for all four tasks, across `sft_tools_v2`,
   `sft_tools_v2_h`, `sft_mix_v2`, `a_slice`, and all three frontier trace sets. Exact `plant_broken`
   JSON-hash overlap (repair has that field) = **0**, exact id overlap = **0**. The structural signature
   (plant name + component ids/types) overlaps 8–9 repair plants per train dir with test — these are
   the same *architecture families* with different sampled parameters, not duplicated diagrams; the
   exact-parameter hash is 0, so this is expected, not a leak. `heldout_v2` manifest independently
   reports `overlap=0` across 298 files / 454,302 lines / 10,578 signatures with a planted-row
   self-check that passes.
2. **Same question answered by multiple models — real, and intentional.** Across the three frontier
   trace sets, **30 distinct `record_id`s** are answered by more than one model: 18 by all three
   (Gemini + luna + haiku), 5 by Gemini+haiku, 4 by Gemini+luna, 3 by luna+haiku. Frontier dirs are kept
   per-model precisely so these shared ids are not silently deduplicated. This is useful for diversity,
   but any future merged training mix must apply an explicit policy (keep-all vs dedup) and document it.
3. **A-slice overlap with tool rows — soft duplication.** `a_slice` shares **38 exact `plant_broken`
   hashes** and **10 `record_id`s** with `sft_tools_v2` (same for `sft_tools_v2_h`), though exact `id`
   strings overlap 0. The A-slice selection enforced "one row per plant" *within* the slice but not
   against the tool corpora, so the same underlying plant appears both as a plain direct-answer SFT row
   and as a tool-rollout SFT row (and in GRPO). This is not test leakage but is worth a documented rule.
4. **GRPO prompt duplication** — see §3.1: each deterministic prompt is emitted twice (direct/recover),
   giving ~600 unique prompts per task instead of ~1,200 rows.

---

### 6. Readiness verdict and top 3 issues to fix

**NOT ready for training.** The gate is red (`check_datagen_complete.py` exit 1). The deterministic
SFT/GRPO data, A-slice, heldout_v2, masking, and contamination checks are in good shape; the blocker is
the frontier cohort and the uncleaned pilot traces.

**Top 3 quality issues to fix (in priority order):**

1. **Complete the six-model frontier cohort / turn the gate green.** Four premium models
   (`gpt-6.2-sol`, `claude-sonnet-5-5`, `claude-opus-5-5-default`, `gemini-4-argon`) have zero traces;
   luna has 0 audit rows and haiku has 0 easy/medium ledger rows. Until every model×task×tier cell has
   a verifier-accepted tool trace, training must not start. No gate weakening.
2. **Clean and gate the luna/haiku pilot traces before using them.** 13/28 haiku rows and 2/27 luna
   rows exceed 8,192 tokens (haiku max 17,746); 10 haiku rows have zero tool calls and must be labeled
   zero-shot, not counted as tool traces. Apply the same `scripts/clean_frontier_traces.py` treatment
   that produced the 32-token-clean Gemini traces (which went from max 17,212 to 7,268).
3. **Document and fix duplication/prompt-set hygiene.** (a) A-slice shares 38 plants / 10 record_ids
   with the tool corpora; (b) GRPO emits each deterministic prompt twice (direct/recover), ~600 unique
   prompts per task; (c) 30 frontier questions are answered by multiple models. Publish an explicit
   dedup/inclusion rule in the manifests, and correct `grpo_mix_v2/MANIFEST.json`'s `frontier: 0`
   counter (reality: 31 Gemini rows) and `sft_mix_v2`'s `a_slice_ratio: 0.0` ambiguity.

**What is genuinely good:** 100% verified final-answer markup, 100% tool markers on tool rows, 13,926+
13,922+128 valid-JSON verbatim physics tool results, masking proven by an 8-test suite wired into the
SFT trainer, 0 rows over the 8,192 gate in all SFT corpora with stored token counts accurate to ±1,
0 heldout overlap by independent measurement, complete GRPO structure with 100% verifier-accepted rows,
and a fully aligned GRPO prompt set that is a strict subset of SFT (400 plain-A prompts/task are the
only difference).

### Addendum — the pilot traces are NOT a chain of thought yet (main agent, 2026-11-19)

Measured on `data/frontier_tools_pilot/{gpt-6-luna,claude-haiku-5-5}/*.jsonl` with a read-only script:

| model | accepted rows | rows with any reasoning text | note |
|---|---:|---:|---|
| `gpt-6-luna` | 24 (after over-gate drops) | **0** | API rejects tools + reasoning; `reasoning_effort="none"` is the documented deviation |
| `claude-haiku-5-5` | 28 | **0** | no `reasoning_traces` exposed on the native-tool path with backend default |

The completion text contains genuine multi-round tool calls (luna 24/24, haiku 18/28) with real physics
results, but the visible "thought" is a short assistant sentence, not a reasoning trace. Tool results still
carry raw `ToolInvocationResult(name=..., call_id=..., output=...)` wrappers — and haiku `call_id`s even embed
hidden-thought blobs — which is bookkeeping that must never reach training. **Verdict: not yet a useful CoT
dataset.** Before sending more batches: (1) strip the wrappers to the JSON payload; (2) get an explicit
haiku reasoning probe; (3) decide with Reza whether luna's tool-revision-only chains count as CoT, since the
provider hard-refuses tools + reasoning on this route. No further batch may be sent until this is resolved.

## Training-path audit and fixes — 2026-11-19

*(merged from `docs/AUDITS.md` on 2026-11-20; content kept, headings demoted one level)*

Scope: Phase 0.7 + Phase 1.1 of `docs/IMPLEMENTATION_PLAN.md`, checklist DECISIONS §4.4 items 1–7.
No training run was started (datagen is incomplete), no Kaggle command pushed/ran anything, no
token was printed or written to a file. All work is CPU/local and additive.

### Method

* Read the decided values in `docs/DECISIONS.md` and the binding order in
  `docs/IMPLEMENTATION_PLAN.md`.
* Read every training module under `thermo_agent/train/**` and the Kaggle train runner/smoke kernel.
* Inspected the *actual* corpus shards under `data/sft*`/`data/grpo*` for the columns and prompt
  text the loaders claim to read.
* Fetched the v28 kernel log **read-only** with `kaggle kernels logs rezanadimi/thermo-train-smoke`
  (proxy set only for that command; `PYTHONIOENCODING=utf-8` to avoid the console `charmap`
  failure; the token was read from `.env` into an env var and never printed).
* Reproduced each suspected defect with a CPU test, fixed it, and re-ran the tests.

### Findings and fixes

#### 1. Data loading reads `task` AND `tier`; uniform task × tier GRPO sampling — **DEFECT, FIXED**

**Root cause.** No real corpus row has a `tier`, `difficulty` or `level` column (verified on
`sft_all`, `sft_tools`, `sft_tools_h`, `grpo_all`, `grpo_tools`, `grpo_tools_h`). `grpo.row_tier`
therefore returned `<missing>` for 100% of rows and `sample_joint` visited 4 cells (one per task)
rather than 12 (task × tier). The docstring advertised uniform task × tier sampling; the code did
task-only sampling. Separately, `sft.read_rows` and `grpo.joint_rows` read only
`<task>_<split>.jsonl`, silently skipping `frontier_<task>_<split>.jsonl` (DECISIONS E1 puts accepted
frontier tool traces first in the mix).

**Fix.** New `thermo_agent/train/data.py` is the single loader for both stages:

* explicit `tier`/`difficulty`/`level` field wins and is validated against `synth.tiers.TIERS`
  (a typo is a hard error, not a new bucket);
* otherwise tiers are derived with `synth.tiers.assign_tiers_grouped` over the whole task slice —
  equal thirds, remainder to the higher tier, the two variants of a plant stay together;
* unscorable rows stay `<missing>` and are counted; `joint_rows`/`read_rows` call
  `assert_tiers_resolved` and refuse to train on them;
* `frontier_*` shards are loaded alongside the deterministic shards.

`thermo_agent/train/grpo.py` (`joint_rows`, `load_rows`, `row_tier`) and `thermo_agent/train/sft.py`
(`read_rows`) now delegate to it. `grpo.row_tier` and `grpo.UNKNOWN_TIER` remain as aliases so older
callers keep working.

**Tests.** `tests/test_train_data.py` (7 tests): equal thirds without a column, plant-pair grouping,
explicit-wins/typo-raises, unscorable-row refusal, 12-cell uniform sampling, frontier shard loading,
SFT task/tier attachment. Existing `tests/test_grpo_joint_and_plumbing.py` still passes unchanged.

#### 2. Single compact harness; no bookkeeping in model prompts — **DEFECT, FIXED**

**Root cause.** `render_pair` fed `row["prompt"]` straight to the chat template with no check that
it came from the one J2/J3 builder. Inspection showed the checked-in `sft_tools_h` / `grpo_tools_h`
shards are **stale**: their system prompt contains `## Components` (the retired verbose builder)
and not the compact marker `# AGENTS.md: energy-plant tasks (compact)`. Training on them would
teach a prompt different from eval's compact-harness prompt and would push 8,192-token rows over the
T4 memory budget. Nothing rejected a message dict carrying extra bookkeeping keys either.

**Fix.** `thermo_agent/train/render.py` now has `model_messages()` (only `role`/`content` reach the
tokenizer; malformed messages are a hard error) and `assert_compact_harness()` (a harness row must
carry the compact marker and no verbose marker). `render_pair` and `prefix_report` use
`model_messages`; both `sft.main` and `grpo.main` validate every loaded row before training, so a
stale shard fails loudly instead of silently training the wrong prompt. The datagen rebuild of the
`*_h` corpora (owned by the datagen agent) is still required to make those shards pass.

**Tests.** `tests/test_train_render.py` (7 tests), including a fresh compact prompt from the shared
`build_harness_tool_messages(compact=True)` for all four tasks, rejection of a verbose prompt,
metadata stripping, and wiring checks in both trainers.

#### 3. SFT label masking — **VERIFIED, no code change**

`preprocess_row` masks the prompt by token-id prefix and every `<tool_result>`/`<tool_response>`
span by character offsets (tags included, nested/unclosed handled), keeps tool *calls* trainable,
sets `completion_only_loss=True`, and the corpus gate stays 8,192 (dropped rows never truncated).
Existing `tests/test_tool_loss_masking.py` plus a new multi-turn integration assertion in
`tests/test_train_collator.py` cover this.

#### 4. Collator, batching and length stats — **VERIFIED + TESTS ADDED**

Inspected TRL 1.18.1 `DataCollatorForLanguageModeling`: it pads `input_ids` with the pad id, builds
`attention_mask`, and pads `labels` with -100. `tests/test_train_collator.py` now checks a batch of
two variable-length multi-turn traces end-to-end (prompt and tool spans masked, padding -100,
attention mask zero on pads) and that `token_audit.audit` reports prompt/completion/total length
statistics plus a new per-file `tier_counts` breakdown.

#### 5. Checkpoint/resume and stage names — **DEFECT, FIXED**

`checkpoint.latest()` picked the highest `checkpoint-N` directory without checking that a save had
finished; a session killed mid-save could resume from a directory with no optimizer, RNG or weights,
silently restarting those states under a "resumed" label. Added `checkpoint.is_complete()` and
`latest(out, complete=True)`; both `sft.main` and `grpo.main` now use the strict form. Save/eval
cadence is already 50/50 (`SftSettings`), best-model loading and early stopping are configured, and
`grpo.stage_name` writes distinct readable `grpo_k4`/`grpo_k8`/`grpo_k16` names (legacy names are
read-only aliases).

**Tests.** `tests/test_checkpoint.py` extended with completeness and interrupted-save cases; stage
names remain covered by `tests/test_grpo_joint_and_plumbing.py` and `tests/test_grpo_stage_names.py`.

#### 6. Eval/save cadence and missing adapter — **VERIFIED, no code change**

`SftSettings.save_steps == eval_steps == 50`; GRPO checkpoints on `save_steps`. `kaggle/eval_runner.py`
already defines `_has_adapter_weights`/`require_adapter` and calls them for every trained arm, so a
missing or empty adapter is a `SystemExit` rather than a base-model score under a GRPO label.
Those files are outside this subagent's ownership and were read/verified, not edited.

#### 7. Smoke kernel v28 (read-only triage) — **OOM DEFECT, LOCAL FIX; RERUN BLOCKED**

Fetched log via `kaggle kernels logs` (read-only). Timeline: setup and env-pin OK (torch 2.11.0+cu130,
transformers 5.23.0, TRL 1.18.1, 2×T4, sdpa resolved), SFT launched with
`--max-steps 20 --limit-rows 64 --per-device-batch 1 --grad-accum 8 --loss-token-chunk 512
--loss-vocab-chunk 8192`, then at step 0/20 the first backward failed:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.10 GiB. GPU 0 has a total
capacity of 14.56 GiB of which 1.01 GiB is free. ... 13.55 GiB memory in use; 12.52 GiB
allocated by PyTorch ... PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ...
thermo_agent.train.sft FAILED   (PapermillExecutionError at In [10])
```

The chunked LM head was active (`[loss] chunked cross-entropy: 512 x 8192 ...`), so the failing
1.10 GiB allocation is the per-layer activation working set of an 8,192-token row during backward,
not the full-vocabulary logits. `transformers` 5.23 supports pinning those saved activations to
host memory (`activation_checkpointing_kwargs={"offload": True}`), reached through the deprecated
alias already used by both trainers.

**Fix.** `config.activation_checkpointing_kwargs()` returns
`{"use_reentrant": False, "offload": True}`; both `sft.py` and `grpo.py` pass it as
`gradient_checkpointing_kwargs`. No change to weights, labels, loss, gate or batch settings.
**Tests.** `tests/test_train_memory.py` (3 tests).

**Remaining blocker for a green re-run (not fixable in this task without touching out-of-scope
files):** the checked-in smoke notebook still selects `SFT_DATA=sft_tools` / `GRPO_DATA=grpo_tools`,
i.e. plain-prompt corpora, not compact-harness tool rows as G1 requires; and the available `*_h`
shards are still the stale verbose-harness build described in finding 2. The smoke notebook's env
cell is pinned byte-for-byte by `tests/test_smoke_kernel_config.py` (not owned here) and the `*_h`
rebuild belongs to datagen, so neither can be changed by this subagent. With this fix the OOM is
addressed locally; the actual green re-run must wait for datagen and an authorised Kaggle push.

### Test results

All CPU-only. Per-file counts from the final run:

| File | Result |
|---|---|
| `tests/test_train_data.py` (new) | 7 passed |
| `tests/test_train_render.py` (new) | 7 passed |
| `tests/test_train_collator.py` (new) | 3 passed |
| `tests/test_train_memory.py` (new) | 3 passed |
| `tests/test_checkpoint.py` | 9 passed |
| `tests/test_token_audit.py` | 4 passed |
| `tests/test_grpo_joint_and_plumbing.py` | 15 passed |
| `tests/test_grpo_multiturn_wiring.py` | 13 passed |
| `tests/test_grpo_prompt_consistency.py` | 8 passed |
| `tests/test_grpo_stage_names.py` | 6 passed |
| `tests/test_tool_loss_masking.py` | 8 passed |
| `tests/test_render.py` | 2 passed |
| `tests/test_smoke_kernel_config.py` | 2 passed |
| `tests/test_ddp.py` | 6 passed |
| `tests/test_chunked_loss.py` | 7 passed |
| `tests/test_decisions_applied.py` | 4 passed |
| `tests/test_eval_runner_arms.py::test_h_missing_grpo_adapter_raises` | 1 passed |
| **Combined run of the 16 train-related files above** | **104 passed** |
| **All 23 test files importing `thermo_agent.train`** | **174 passed** |

Full-repository `pytest -q`: **15 failed, 845 passed, 9 skipped in 1468 s**. All 15 failures are in
files owned by the concurrent datagen work (not this audit): `test_check_datagen_complete.py` (1),
`test_heldout_v2_freeze.py` (3), `test_heldout_v2_overlap.py` (2), `test_heldout_v2_tiers.py` (8),
and `test_kaggle_notebook.py::test_notebook_writes_one_line_per_attempt_and_records_errors` (1,
which picked up the real 4,149-attempt shared-progress corpus instead of the two fixture rows).
No failure is in any train-path file changed here.

### Files changed

* `thermo_agent/train/data.py` (new) — shared task+tier loader, frontier shards, tier guard.
* `thermo_agent/train/grpo.py` — delegate loading to `data.py`; compact-harness guard; offload; strict resume.
* `thermo_agent/train/sft.py` — delegate loading to `data.py`; compact-harness guard; offload; strict resume.
* `thermo_agent/train/config.py` — `activation_checkpointing_kwargs()` (v28 OOM fix).
* `thermo_agent/train/render.py` — `model_messages`, `assert_compact_harness`, sanitised rendering.
* `thermo_agent/train/checkpoint.py` — `is_complete`, `latest(complete=True)`.
* `thermo_agent/train/token_audit.py` — per-file `tier_counts` length-stat breakdown.
* `tests/test_train_data.py`, `tests/test_train_render.py`, `tests/test_train_collator.py`,
  `tests/test_train_memory.py` (new) — one regression test per fix.
* `tests/test_checkpoint.py`, `tests/test_token_audit.py` — extended for the new behaviours.

No file outside the owned set was edited. No commit was made.

## Directive compliance audit — 2026-11-19

*(merged from `docs/AUDITS.md` on 2026-11-20; content kept, headings demoted one level)*

Purpose: an independent model (e.g. Claude) can read this file, run the commands, and decide for itself
whether the work matches what the user actually asked for. It is written to make deviations easy to find,
not to make the project look good.

Conventions:
- **[VERIFIED]** = re-checked by command in this session; the command and output shape are given.
- **[RECORDED]** = taken from an earlier doc/journal; verify before trusting.
- **[OPEN]** = known to be incomplete or broken right now.
- Status vocabulary: `DONE`, `PARTIAL`, `NOT DONE`, `DEVIATED` (did something the user did not ask for).

---

### 1. How to audit this repo in 10 minutes

```pwsh
# 1. what changed since the last commit
git log --oneline -3
git status --short

# 2. what data actually exists (counts only, no tokens)
python -c "import glob,json;[print(d, sum(1 for p in glob.glob(d+'/*.jsonl') for l in open(p,encoding='utf-8'))) for d in ('data/sft_all','data/sft_tools','data/sft_tools_h','data/grpo_tools_h')]"

# 3. are there ANY frontier tool-using traces? (the central claim)
python scripts/mix_tool_corpora.py --stats

# 4. eval-arm semantics (harness / tools / zero-shot per arm)
python -m pytest tests/test_eval_arm_separation.py -q

# 5. prompt separation in training corpora
python -c "import glob,json;rows=[json.loads(l) for p in glob.glob('data/sft_tools_h/*_train.jsonl') for l in open(p,encoding='utf-8')];print(len(rows), sum(r.get('prompt_variant')=='harness_tools' for r in rows))"

# 6. memory size actually served to the model (must be a short recap, not a doc dump)
python -m pytest tests/test_memory*.py -q
```

---

### 2. Compliance matrix

| ID | Directive (short) | Status | Evidence |
|---|---|---|---|
| D1 | Real thermo tools, no fake wrappers | DONE | `thermo_agent/teacher/toolkit.py`; numbers produced by `core.solve.solve` |
| D2 | Full harness loop: propose → tools → revise → answer | DONE (builder) / [OPEN] exercised end-to-end only when a hosted run completes | `thermo_agent/teacher/agent_graph.py` |
| D3 | Skills + fresh memory exposed to the model | DONE | `agent_graph.build_context`, `data/memory/*.jsonl` |
| D4 | Record frontier traces and train on the behaviour | **NOT DONE** | frontier tool traces = 0; see §4 and deviation V1 |
| D5 | LangGraph state machine | DONE | `StateGraph(AgentState)` |
| D6 | Stop faulty AI calls, don't waste quota | DONE at the time; still true today (no runaway local calls) | journals + quota |
| D7 | Budget tier first (luna → haiku); sonnet deferred | PARTIAL | user has now authorised a premium subset (D14); gemini-3.5-flash-lite still excluded |
| D8 | Zero-shot → tool CoT → train → eval on `data/heldout` | PARTIAL | training corpus not built yet (correct per D13); eval arms defined |
| D9 | Update notebooks/docs to the new objective | PARTIAL | docs updated; notebook parity not fully re-checked |
| D10 | Agent looks for skills/tools first, then memory | DONE in prompt text | `tool_prompt.py`, `AGENTS.md` |
| D11 | Token-efficient memory (what failed/worked only) | DONE | `docs/AUDITS.md`: served k=3 p50 538 chars / 80 tokens, p95 892 / 102 |
| D12 | Two sources across 4 task families (yes/no) | **answer: no, three streams; frontier stream empty** | §4 |
| D13 | Traces first, training later | DONE (no training started) | no `runs/sft*` checkpoint exists |
| D14 | Many more traces, multi-model | **NOT DONE** | 0 accepted tool traces; plan in §6 |
| D15 | Prompt separation in tracegen/training/evals | PARTIAL | matrix §5; two leaks fixed today, one open |
| D16 | Smoke-test before big run | **OPEN** | smoke kernel v28 = `ERROR`, log not triaged |
| D17 | Sync to HF + Kaggle datasets | **NOT DONE** | scripts written, never executed |
| D18 | Free subagents for bulk work | DONE | 3× `opencode/hy4-flash-free` sessions |
| D19 | Document instructions for an auditor | DONE | this file + `docs/DECISIONS.md` |
| D20 | Epoch count is the user's decision | DEVIATED then corrected | `config.py` still holds `epochs = 1` (placeholder, unauthorised) |

---

### 3. Data reality today [VERIFIED]

| Corpus | Rows | What it is |
|---|---|---|
| `data/sft_all` | 133,032 | 20-variant zero-shot corpus + 635 accepted zero-shot frontier rows + expert-scale rows |
| `data/grpo_all` | 28,247 | matching GRPO prompts |
| `data/sft_tools` | 960 | deterministic multi-turn tool traces, plain prompt (4 tasks × 2 variants × 120) |
| `data/grpo_tools` | 960 | matching GRPO rows |
| `data/sft_tools_h` | 960 | same completions, harness system prompt; 960/960 rows `prompt_variant="harness_tools"` |
| `data/grpo_tools_h` | 960 | same, GRPO side; 960/960 marked |
| `data/frontier` | 668 scanned / 635 accepted | **zero** rows with `<tool_call>` + `<tool_result>` |

Held-out: `data/heldout`, 214 test ids, hash `8b409359…f8ef4c6e` [RECORDED: `docs/PROVENANCE.md`].

`python scripts/mix_tool_corpora.py --stats` today prints:

```
WARNING: 0 verifier-accepted frontier rows containing BOTH <tool_call> and <tool_result> were selected
(scanned=668, accepted=668, with tool turns=0). The output is a deterministic-only corpus.
```

That warning is the single most important fact in this repo: **there is no frontier tool CoT training data yet.**

---

### 4. Self-reported deviations (read this section first)

#### V1 — Frontier CoT was described as if it existed (HIGH)
Early claims of "frontier CoT training data" were only zero-shot answers (`harness="none"`, `tools="none"`).
The 635 accepted rows contain no tool calls. Any statement implying the student was learning frontier tool
behaviour from them was wrong.

Check: `python scripts/mix_tool_corpora.py --stats` → `with tool turns=0`.

#### V2 — `epochs = 1` was set by the assistant, not the user (HIGH)
`thermo_agent/train/config.py:23`. Sized against quota (~30 h/epoch on 2×T4), never authorised.
Options: 1 epoch ≈ 30 h ≈ 3 sessions; 2 epochs does not fit before the 2026-11-21 refill; step-based
(`--max-steps`) is the fixed-cost alternative. **Status: undecided — user sets it.**

#### V3 — GRPO arms were scored without tools (HIGH, fixed today)
`kaggle/eval_runner.py`: `ours_grpo`, `ours_sft_grpo_k4/k8/k16` had `tool_access="none"` although GRPO is
trained inside the tool loop. Now `"tools"`. Guarded by `tests/test_eval_arm_separation.py`.

#### V4 — Eval rendered a different chat template than training (HIGH, fixed today)
`HFGenerator` hardcoded `enable_thinking=False`; the SFT/GRPO corpora are rendered with `enable_thinking=True`.
Now defaults to `True` with an explicit `--no-thinking` escape, and a test asserts the default.

#### V5 — `--harness-prompt` was unreachable (MEDIUM, fixed today)
`scripts/mix_tool_corpora.py` had `build_mix(..., harness_prompt=...)` but `main()` never registered the flag.
Fixed; 14/14 tests in `tests/test_mix_tool_corpora.py` pass. Without this, the "re-prompt frontier traces with
the eval prompt" decision would have been silently dead code.

#### V6 — Trace caps were the assistant's numbers (MEDIUM, removed)
480, then 2,880: neither was authorised. Both are gone; the hosted task now defaults to every row in the shard
with no cost cap and no deadline, and the mix cap defaults to unlimited. See §6.

#### V7 — Smoke kernel v28 is red and untriaged (HIGH, open)
`rezanadimi/thermo-train-smoke` → `KernelWorkerStatus.ERROR`. It was pushed to test sdpa/allocation with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Gate stays closed until the log is read.

#### V8 — HF/Kaggle sync was written but not run (MEDIUM, open)
`scripts/sync_corpora_hf.py` (private `Reza-Nadimi/thermo-agent-corpora`, allow-list, dry-run first) and
`scripts/upload_corpus_dataset.ps1` (adds `sft_tools_h`/`grpo_tools_h` zips). Nothing uploaded.

#### V9 — Replay arms can re-execute stored tool calls (LOW, open)
`frontier_tools`/`frontier_harness` replay with `tool_access="tools"`, so `ToolLoop` may re-run a stored
`<tool_call>` against a fresh environment. Historical behaviour, documented in source, still unchecked.

#### V10 — "Two sources" framing was wrong (LOW)
No single "two things". Three streams, four task families. See §3.

---

### 5. Prompt separation matrix

| Stage | Arm / stream | System prompt | Tools | Zero-shot |
|---|---|---|---|---|
| tracegen | zero-shot baseline shards (`thermo-teacher-big-s00`) | plain task prompt | none | yes |
| tracegen | harness pass (`thermo-teacher-tools-s00`) | AGENTS.md + skills + shared tool contract (`tool_prompt.py`) | 5 real physics tools, 4 rounds, memory notes | no |
| training | `data/sft_tools` / `grpo_tools` | plain (`harness="none"`) | in-completion tool calls | no |
| training | `data/sft_tools_h` / `grpo_tools_h` | `harness_tools` (960/960 marked) | in-completion tool calls | no |
| training | frontier rows in a harness mix | re-prompted at mix time (`--harness-prompt`), completion untouched | teacher's real calls | no |
| eval | `ours_base` | none | none | yes |
| eval | `ours_sft` | agents+skills | none | no |
| eval | `ours_harness` | agents+skills | tools | no |
| eval | all GRPO arms | agents+skills | tools (fixed today) | no |
| eval | `frontier_zero` | none | none | yes |
| eval | `frontier_tools` / `frontier_harness` | stored (already augmented) | label `tools` | replay |

Rules that must hold and are now tested:
1. A harness-trained model is never scored with the plain zero-shot prompt.
2. A zero-shot run never receives the harness/tool contract.
3. Frontier zero-shot rows cannot enter a tool mix: they fail the `<tool_call>` + `<tool_result>` gate.

---

### 6. Trace campaign v2 (no assistant-invented cap)

User instruction: many traces from luna and haiku, some from sol / opus / sonnet / gemini-4.
The assistant previously wrote a 480 cap and then a 2,880 cap. **Both were removed on 2026-11-19**: the user
never set either number. There is now no row cap, no cost cap and no deadline inside the hosted task
(`SPEC["limit"]`, `max_cost_usd`, `deadline_seconds` all default to "no stop"), and the mix cap default is
unlimited (`--frontier-per-task 0`). The account-level Kaggle quotas are the only ceilings.

| Tier | Model | Hosted slug status |
|---|---|---|
| bulk | `openai/gpt-6-luna` | confirmed from the v5 journal |
| bulk | `anthropic/claude-haiku-5-5` | to confirm on its first run |
| premium | gemini-4 | declared `gemini-4-argon` in `EXPECTED_MODELS`; confirm on first run |
| premium | `anthropic/claude-sonnet-5-5` | declared; user-authorised on 2026-11-19 |
| premium | `gpt-6.2-sol` | declared; confirm on first run |
| premium | `claude-opus-5-5` | declared; confirm on first run |

Remaining gate: a model that rejects the payload (tools + reasoning) is dropped, not retried blindly — that is
a correctness check, not a quota limit. Per-model cost is read from each run's journal (`usage`), not estimated
or capped by the assistant.

---

### 7. Open blockers (in order)

1. `thermo-teacher-tools-s00` v6 is `RUNNING` with `gemini-3.8-flash`; luna cannot start until a version
   completes ("Only completed tasks can be run").
2. Smoke v28 `ERROR` — log untriaged; training gate closed.
3. Zero accepted tool traces; mix is deterministic-only.
4. HF/Kaggle sync not executed.
5. `epochs` undecided; do not start a run with a placeholder value.

---

### 8. Things a reviewer should distrust

- Any sentence that says "frontier traces" without the `<tool_call>` count next to it.
- Any "passing" claim about the smoke kernel without the latest `kernels status` + log.
- Any training number (epochs, steps, hours) that is not the user's explicit choice.
- Any doc that predates 2026-11-19 14:00 UTC and talks about "480 traces".
- `docs/STATUS.md` timelines: they are append-only history, not current state. This file wins for state.

## RAG success/failure memory audit — 2026-11-19

*(merged from `docs/AUDITS.md` on 2026-11-20; content kept, headings demoted one level)*

**Scope.** Read-only audit of `thermo_agent/teacher/memory.py`, `thermo_agent/teacher/compact_memory.py`,
`data/memory/*.jsonl`, `data/frontier/frontier_*_train.jsonl` and `docs/PROVENANCE.md`,
followed by one additive write: a **new** file `data/memory/agent_success_patterns.jsonl` (7 notes). No
existing memory note, corpus, model, config or doc was deleted or overwritten. All counts below were measured
by loading each JSONL through `thermo_agent.teacher.memory.load` with `.venv/Scripts/python.exe`.

**Key result.** Before this audit the memory contained **zero verifier-accepted success notes**. Every one of
the 6 `accepted=true` notes was a hand-written `seed:method` note — guidance, *not* verifier evidence. The
635 verifier-accepted frontier rows had no corresponding "what worked" notes. I added 7 compact,
verifier-grounded success notes (capped well below the 12-note limit); audit remains uncovered because **no
accepted audit frontier row exists**.

---

### 1. Counts by source / task / fault type / acceptance

#### 1a. Pre-audit ledger — 407 notes across 3 files

| file | notes | accepted | rejected |
|---|---:|---:|---:|
| `frontier_notes.jsonl` | 369 | 0 | 369 |
| `adversarial_repair.jsonl` | 24 | 0 | 24 |
| `agent_patterns.jsonl` | 14 | 6 | 8 |
| **TOTAL (pre-audit)** | **407** | **6** | **401** |

#### 1b. By `source` (pre-audit)

| source | notes | evidence class | accepted |
|---|---:|---|---:|
| `frontier:google-gemini-3-1-flash-lite-preview` | 176 | verifier-mined rejections | 0 |
| `frontier:openai-gpt-5-4-nano-2026-03-17` | 141 | verifier-mined rejections | 0 |
| `frontier:google-gemini-3-8-flash` | 52 | verifier-mined rejections | 0 |
| `adversarial:echo_broken` | 13 | verifier-mined (verifier reason codes) | 0 |
| `adversarial:wrong_scale` | 8 | verifier-mined (verifier reason codes) | 0 |
| `adversarial:no_edits` | 3 | verifier-mined (verifier reason codes) | 0 |
| `seed:method` | **14** | **seed method notes — NOT verifier evidence** | 6 |

- Verifier-mined (frontier + adversarial): **393 notes, 0 accepted**.
- **Seed notes: 14 (all `source="seed:method"`, all `extra.seed=true`). Of these, 6 carry `accepted=true`,
  and those 6 are NOT verifier evidence** — they are documented method guidance ("verify before answer",
  "iterate on flag", …), not a verifier-accepted row. `_is_seed()` detects all 14 via both the flag and the
  `seed:` prefix, and `retrieve()` deliberately caps them to one slot per served block so they cannot crowd
  out mined evidence.

#### 1c. By task × acceptance (pre-audit)

| task | rejected | accepted | accepted source |
|---|---:|---:|---|
| repair | 70 | 2 | both `seed:method` |
| design | 180 | 2 | both `seed:method` |
| ledger | 54 | 1 | `seed:method` |
| audit | 97 | 1 | `seed:method` |

#### 1d. By fault type × acceptance (pre-audit)

Top classes (full distribution measured): empty fault type 273 (all rejected — most design/audit/ledger
frontier notes were never fault-classified), `unit_slip` 41, `wrong_fluid` 21, `efficiency_gt_1` 13,
`missing_m` 7, `sign_T0` 6, `wrong_factor` 6, `dropped_term` 5, `missing_ech` 5, `sign_flip` 5,
`t0_literal` 5, `t0_celsius` 4, `t0_offset` 3, plus 6 singleton failure classes.

**Every accepted fault type was a seed-only class**: `minimal_edit`, `verify_before_answer`,
`rank_by_evidence`, `closure_first`, `iterate_on_flag`, `closure_check` (one note each). No accepted note
had a verifier fault class such as `wrong_fluid` or `efficiency_gt_1`.

#### 1e. Frontier train rows (cross-check with the corpus inventory)

668 rows: 635 accepted (`reward == 1.0`), 33 rejected, **0 containing tool calls** — matches
`docs/PROVENANCE.md` §3. Accepted reasons are 4 generic strings: "restored a valid plant
exactly" (189), "ledger repaired" (160), "restored a valid plant" (146), "design accepted" (140). The 33
rejected rows are the audit rows, all with fractional reward (0.53–0.99) and a literal reason string
`"audit accepted"`; their detailed failure reasons (nDCG, prediction error, grounding %) live in the mined
notes, not in `validation.reasons`.

---

### 2. Served-memory size and leakage

Method: `memory.retrieve(q, notes, k=3, task=task)` → `compact_memory.compact_notes(top, k=3)` on the **final**
pool (414 notes, including the 7 new ones). Production (`teacher/agent_graph.py` lines 105–110) calls exactly
this pair with `task=task`. 5 representative queries, one per task plus a generic-units query. "Tokens" is a
regex word-token proxy `len([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)` — **no real tokenizer was run**, so treat it as
an approximation, not a model-specific count.

| # | task | chars | approx tokens |
|---|---|---:|---:|
| 1 | repair (fluid mismatch / efficiency / temperature cross) | 538 | 80 |
| 2 | design (gas turbine + absorption + dry rejection) | 892 | 102 |
| 3 | ledger (negative destruction / closure / T0 / 1000×) | 522 | 80 |
| 4 | audit (rank / change / prediction / grounding) | 552 | 82 |
| 5 | repair (unit slip / sign / second-law limits) | 439 | 61 |
| | **p50** | **538** | **80** |
| | **p95** | **892** | **102** |

(Unfiltered `retrieve(q, notes, k=3)` over the same queries gave chars 378–616, p50 586 / p95 616; tokens
56–93, p50 88 / p95 93. Either way a served block is well under ~1 KB / ~100 tokens.)

**Leakage scan** over all five served blocks — patterns for model slugs (`google-`, `openai-`, `gemini`,
`gpt-`, `claude`, `haiku`, `luna`), ISO timestamps, `cost`, `tokens=`, `latency`, `record_id` /
`rep-|des-|led-|aud-|arch-\d` / 8-hex digests, `run_id`, `ts=`, `seed:`, and `$`: **all clean**. No note id
leaked into any block text. `compact_memory._clean()` plus the fact that it only emits `fault_type`, `fix`
and `lesson` means `source`, `record_id`, `id`, `architecture`, `symptoms` and `extra` never reach the model.

**Full served block — example (query #1, repair):**

```
FAILURE MEMORY (what failed -> what works; OK=accepted, FAIL=rejected; classes only, not this row's answer):
- FAIL wrong_fluid: Attempt rejected (edit: unknown fluid FuelGas). Copying the broken label back, or inventing a third fluid, does not work; the destination component fixes the fluid class.
- OK   wrong_fluid: FLUID @s5 = FLUE
- FAIL wrong_fluid: Seen 24 times: Attempt rejected (verifier: invalid plant). Copying the broken label back, or inventing a third fluid, does not work; the destination component fixes the fluid class.
```

Note the `retrieve()` guarantees hold here: one seed note maximum, at least one FAIL slot reserved, and the
new `OK` note sits beside the failure that teaches the same class.

---

### 3. Success coverage

**Were there accepted-success notes?** Effectively no. All 6 pre-audit `accepted=true` notes were
`seed:method` (stated explicitly in §1b); **0 were verifier-mined**. So the harness could only ever show
generic method guidance as "OK", never an actual accepted frontier answer.

**Which tasks had none?** In verifier terms, *all four* had none: repair, design, ledger and audit each had
only seed "OK" notes. After the addition, repair (3), ledger (3) and design (1) have verifier-grounded OK
notes; **audit still has none, and cannot get one from this data** — there are 0 accepted audit frontier rows
(§1e).

**Do accepted frontier rows yield reusable success patterns?** Yes, but with a caveat. The acceptance
*reasons* are too generic on their own ("design accepted", "ledger repaired", "restored a valid plant");
they only say "this passed". However, each accepted row's **completion** contains a concrete, reusable edit
pattern that the verifier accepted, and `validate.success_note()` lifts exactly that into `fix` while keeping
the class-level statement in `lesson`. Measured distinct reusable patterns in the accepted pool:

- **repair (335 accepted):** `FLUID @stream = <class>` for fluid mismatches (110 rows use `FLUID`);
  `SET <comp>.eta_s = <fraction>` to bring an efficiency back into (0,1]; `SET DC.T_return = <above supply>`
  to clear a temperature cross. All 335 accepted repairs contain an `<edits>` block.
- **ledger (160 accepted):** `FIX T0 = <AMBIENT T0_K>` (wrong 273.15 literal), `FIX ED[DC] = E[chw_sup] -
  E[chw_ret] - W[DC]` (remove a spurious 1000× factor), `FIX ED[ENG] = E[air]+E[fuel]-E[s5]-W[ENG]`
  (missing outlet stream / closure). All 160 contain a `<fix>` block.
- **design (140 accepted):** gas-turbine prime mover + absorption cooling + dry rejection, turbine PR≈18 /
  T4≈1500 K, hot-water ≈363 K, all streams inside fluid limits, no grid import; only numbers quoted from the
  input. All 140 contain a `<rationale>` block.

So the patterns are **not too few** (335/160/140 accepted examples), but the *reason strings* are too generic
to stand alone — the reusable value is in the accepted completion. All 635 accepted rows are zero-shot
(`harness=none`, 0 tool calls), so they are success *answers*, not tool traces (consistent with the corpus
inventory §3d).

---

### 4. Duplication / conflicts / grounding

- **Exact duplicate ids:** 4 (`freq-audit-19493645` appears 3× — "Seen 344/36/57 times"; `freq-audit-f43891b0`,
  `freq-design-bab6218a`, `freq-ledger-f62c78a3` once each). These frequency notes were generated from
  different model sources but collide on an md5-of-lesson id. They are not content-identical (counts differ),
  but the id collision is a real ledger hygiene issue.
- **Exact duplicate content** (same task/fault/lesson/fix/accepted): **0 groups** — `dedupe()` keys are
  effectively unique.
- **"Contradictory fix" candidates:** one group shares the exact lesson "Attempt accepted (closure residual
  ~0, ledger repaired)" with two different fixes — `FIX T0 = 289.715…; FIX ED[DC] = …` vs just
  `FIX ED[DC] = …`. This is **not** a contradiction: one accepted repair needed both edits, the other only
  the ED fix (different plants). Similarly, 5 `(task, fault, param)` keys have multiple numeric targets
  (e.g. `SET GT.eta_s` = 0.843 vs 0.902) — these are **plant-specific values for different records**, not
  class-level conflicts, and `retrieve()` excludes a note from its own `record_id`.
- **Not grounded in a verifier:** the 6 accepted `seed:method` notes (no `record_id`, `extra.seed=true`) are
  method guidance, not verifier evidence — the single most important caveat in this audit. Separately, 24
  `frontier:` notes have an empty `record_id`: these are the `freq-*` aggregates whose "Seen N times"
  lessons are derived from verifier rejection counts but are not individually traceable to one record. They
  are verifier-derived statistics, not fabricated lessons, but should not be cited as a single verifier
  verdict.
- The 7 new notes are all grounded: each cites a real accepted row (`record_id`), a verifier reason
  (`symptoms`), and a `fix` lifted verbatim from that row's accepted completion (verified programmatically —
  `fix_from_completion=True` for all 7).

---

### 5. Recommendation

**Add now (done):** 7 compact success notes in the new `data/memory/agent_success_patterns.jsonl`, using the
canonical `validate.success_note()` format with `source="frontier:<model>"`, `record_id` and
`accepted=true`. They cover the three tasks that have accepted frontier rows and both frontier models:

| id (abbreviated) | task / source model | one-line lesson |
|---|---|---|
| `repair-rep-70-00009-…-ok-71b7256f` | repair / gemini-3-1-flash-lite | accepted repair fixes the fluid class on the destination port: `FLUID @s5 = FLUE` |
| `repair-rep-19-00067-…-ok-b98f3284` | repair / gemini-3-1-flash-lite | accepted repair rewrites an efficiency as a fraction (0,1]: `SET AC.eta_s = 0.848…` |
| `repair-rep-33-00085-…-ok-77f6f58d` | repair / gemini-3-1-flash-lite | accepted repair raises a below-ambient return above supply: `SET DC.T_return = 302.16…` |
| `ledger-led-0-04143-…-ok-9dac5ab2` | ledger / gemini-3-1-flash-lite | accepted ledger fixes both T0 and the DC exergy line to close the ledger |
| `ledger-led-0-05413-…-ok-2972682a` | ledger / gpt-5-4-nano | accepted ledger removes the spurious 1000× factor from `ED[DC]` |
| `ledger-led-0-06893-…-ok-57f57665` | ledger / gemini-3-1-flash-lite | accepted ledger adds the missing outlet stream `E[s5]` to `ED[ENG]` |
| `design-des-0-02514-…-ok-addb582a` | design / gpt-5-4-nano | accepted design: turbine + absorption + dry rejection, all numbers from the input |

**Why this is capped and safe.** 7 is ≤12 and is outnumbered ~57:1 by the 401 verifier-mined failure notes.
`retrieve()` already (a) caps seed notes at one slot, (b) reserves at least one FAIL slot for `k≥2`, and
(c) excludes notes from the same record. In a k=3 block an accepted note can occupy at most 2 of 3 slots,
and the measured served blocks above still contain FAIL lessons. Do **not** add more accepted notes: the
remaining 628 accepted rows mostly repeat these same 7 patterns with different plant-specific numbers, which
would inflate the pool without new class-level information and would risk crowding out failures.

**Refused to add, and why:**
- **No accepted audit success note.** There are 0 accepted audit frontier rows (all 33 audit rows have
  reward < 1.0), so there is no verifier acceptance to ground it in. The 52 `google-gemini-3-8-flash` audit
  notes are all rejections. Fabricating an "OK audit" note would violate the grounding requirement. The real
  fix is to obtain ≥1 accepted audit row (as the corpus inventory §5 notes) before mining an audit success.
- **No tool-trace / "how to call tools" success note.** All 635 accepted rows are zero-shot with 0 tool tags
  (§1e), so nothing verifier-grounded can be said about successful tool use.
- **No notes for the other 628 accepted rows.** Their reason strings are the same 4 generic sentences and
  their edits repeat the 7 patterns already added; adding them would be duplicate-by-lesson crowding, not new
  evidence.

---

### Validation performed

- Each new line loads through `thermo_agent.teacher.memory.load` as a `Note` with `accepted=True`, a non-empty
  `record_id`, and `source.startswith("frontier:")` — 7/7 pass.
- Each renders through `compact_memory.compact_notes` with an `OK` marker; no bookkeeping, model id, timestamp,
  cost, record id, run id, seed slug or `$` in served text (§2).
- Every new `fix` is a verbatim substring of its source row's accepted completion (checked programmatically).
- `pytest tests -k memory -q`: **34 passed, 624 deselected in 48.03s**.
- Existing files were not modified; the new file is gitignored along with the rest of `data/memory/` but is
  present on disk (3,335 bytes, 7 lines).

## 2026-11-20 — Claude review of the harness, traces and working method

Source: fresh clone at `8343dc7`, committed data and code. Raw hosted journals (`data/teacher_runs/`) and `data/memory/*.jsonl` are gitignored, so Kaggle yields are quoted from the repo's reports only. Every item below is FIXED in the 2026-11-20 patch unless marked.

| # | Finding | Evidence | Fix in this patch |
|---|---|---|---|
| F1 | The 237 accepted AI traces had **no reasoning text** (empty assistant prose on every non-final turn): tool-use traces, not chain of thought. The compact contract never asked for reasoning. | prose chars per turn `[0, 0, 0, 306]` | Archived to `data/archive/2026-11-20_pre_cot/`. The contract now REQUIRES 1-2 sentences before every tool call (default, no flag). `validate.reasoning_report` + the ingest gate reject any tool trace without usable reasoning (`rejected_no_reasoning`), and `--require-tools` rejects answers with no tool use. For models that take no reasoning parameter (gpt-6-luna with tools) the visible prose across the multi-round conversation IS the reasoning (`_turn_cot`). |
| F2 | Deterministic traces had boilerplate reasoning (18,598 blocks, 2,369 distinct, eight sentences = 45%) **and** the first tool call already contained the ground-truth answer, so the student learned to guess. | recount over `sft_tools_v2_h` | Archived. `tool_traces.py` v3: the first step inspects the broken plant; each later `<think>` is built from the numbers in the previous tool result plus the ground-truth explanation; recover variants start from a stated hypothesis. Tested: every later block cites an identifier or number from the previous result; >=95% of non-opener blocks distinct. Regenerate with `scripts/build_tool_corpus.py` (outputs `data/sft_tools_v3`, `data/grpo_tools_v3`). |
| F3 | AI yield almost all repair (214 of 237; audit 2, ledger 10, design 11); cohort rows had no `tier`. | cohort files | Ingest now copies `tier`/`topology` from the export row when present. Per-model x task yield is a gate before bulk runs (plan, step 3). |
| F4 | Ledger `inspect_plant` printed `truth_MW`, a per-line `ok` flag and `failed_lines`: the answer's location was a lookup. | tool output | Renamed `recomputed_MW`; `ok`, `failed_lines` and the line names in the solver message removed. The model must still read residuals and write the fix. |
| F5 | On-demand skills worked but were off everywhere (`include_dynamic=False`). | toolkit defaults | Default ON in `make_tools`, `tool_names`, `ThermoEnv`, `agent_graph`, the compact prompt (catalog of descriptions, bodies only via `load_skill`), and `list_skills("")` accepts a blank filter. The 16-tool schema adds prompt tokens: re-measure the 8,192 gate (plan step 2). |
| F6 | Docs were litter (35 files). | git | 10 files, `tests/test_docs_layout.py`. |
| F7 | `BUDGET_BATCH_REPORT` repeated one boilerplate block 4 times. | file | Deduped; loop logs a real tally once per batch tag. |
| F8 | Prompt 9 recorded truncated. | file | Completed; prompt 12 added. |
| F9 | `reports/eval_*.json`, significance files were scored on pre-fix data; no model has been scored on `heldout_v2`. | meta | Moved to `reports/archive_pre_heldout_v2/`. |
| F10 | Stale tests pinned the old defaults and "additive-only" hashes. | tests | Rewritten to the new defaults; corpus-dependent protection tests skip when the local corpora are absent. |
