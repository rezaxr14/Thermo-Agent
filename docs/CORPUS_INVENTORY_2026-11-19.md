# Corpus inventory — 2026-11-19

**Status:** dated, factual, read-only measurement. Only this file was written; no corpus, model, config or other
doc was modified. All counts below were measured this date by streaming each JSONL line-by-line (no corpus was
loaded into memory). "not measured" means the field is absent and no value was inferred.

Environment: `C:\Users\rezax\Home\Code\thermo_agent\Repo`, Python `.venv/Scripts/python.exe`.
Script and exact commands in the [Appendix](#appendix--script-and-commands).

---

## 1. Directory-level inventory (rows / harness / tool tags)

`completion` tool-tag counts = rows whose serialized `completion` contains the literal tag.
`traj` = `trajectory.sft_completion` (GRPO stores the multi-turn SFT target there, not in `completion`).

| dir (data/…) | files | rows | prompt present | completion present | `harness` values (rows) | `source` values (rows) | `<tool_call>` rows (completion) | `<tool_result>` rows (completion) | `<tool_response>` rows | traj tool-call rows |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| `processed` | 20 | 27,418 | 27,418 | 27,418 | field absent | field absent | 0 | 0 | 0 | 0 |
| `processed_arch` | 14 | 5,544 | 5,544 | 5,544 | field absent | field absent | 0 | 0 | 0 | 0 |
| `processed_big` | 8 | 194 | 194 | 194 | field absent | field absent | 0 | 0 | 0 | 0 |
| `sft_tools` | 4 | 960 | 960 | 960 | `none`: 960 | `deterministic_tools`: 960 | **960** | **960** | 0 | 0 |
| `grpo_tools` | 4 | 960 | 960 | 960 | field absent | `deterministic_tools`: 960 | 0 | 0 | 0 | **960** (`trajectory`) |
| `sft_all` | 21 | 133,032 | 133,032 | 133,032 | `none`: 87,219; `agents+skills`: 44,743; `skills`: 1,070 | `ground_truth`: 132,397; `frontier`: 635 | 0 | 0 | 0 | 0 |
| `grpo_all` | 21 | 28,247 | 28,247 | 28,247 | `none`: 18,268; `agents+skills`: 8,909; `skills`: 1,070 | `ground_truth`: 27,612; `frontier`: 635 | 0 | 0 | 0 | 0 |
| `frontier` | 4 | 668 | 668 | 668 | field absent | `frontier`: 668 | 0 | 0 | 0 | 0 |
| `frontier_prov` | 4 | 668 | 668 | 668 | field absent | `frontier:<model>` (see §3) | 0 | 0 | 0 | 0 |
| `frontier_export` | 4 | 1,600 | 1,600 | **0** (prompt-only export) | `none`: 1,213; `agents+skills`: 321; `skills`: 66 | `ground_truth`: 1,600 | 0 | 0 | 0 | 0 |
| `frontier_eval_export` | 24 | 360 | 360 | **0** (prompt-only export) | field absent | field absent | 0 | 0 | 0 | 0 |
| `heldout` | 20 | 542 | 542 | 542 | field absent | field absent | 0 | 0 | 0 | 0 |

Task field: present only in `sft_tools`, `grpo_tools`, `sft_all`, `grpo_all`, all `frontier*` task files and
`frontier_eval_export`; absent in `processed*`/`heldout` (task is taken from the filename there). Where present the
task histogram is given in the per-task tables below.

### 1a. Rows per task × split

`data/processed` (seed 0, MIX):

| task | train | val | test_id | test_ood_fault | test_ood_topology | test_ood_climate | test_ood_load | test_ood_scale | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| repair | 5,458 | 663 | 649 | 4,203 | 837 | 0 | 0 | 0 | 11,810 |
| design | 1,768 | 235 | 204 | 0 | 0 | 615 | 411 | 460 | 3,693 |
| audit | 2,192 | 290 | 299 | 0 | 135 | 0 | 0 | 0 | 2,916 |
| ledger | 4,846 | 672 | 612 | 2,229 | 640 | 0 | 0 | 0 | 8,999 |
| **TOTAL** | **14,264** | **1,860** | **1,764** | **6,432** | **1,612** | **615** | **411** | **460** | **27,418** |

`data/processed_arch` (65-diagram catalog; no design task):

| task | train | val | test_id | test_ood_fault | test_ood_topology | total |
|---|---:|---:|---:|---:|---:|---:|
| repair | 818 | 110 | 95 | 761 | 279 | 2,063 |
| audit | 996 | 113 | 116 | 0 | 141 | 1,366 |
| ledger | 1,097 | 110 | 152 | 464 | 292 | 2,115 |
| **TOTAL** | **2,911** | **333** | **363** | **1,225** | **712** | **5,544** |

`data/processed_big` (big cycle; only repair/audit):

| task | train | val | test_id | test_ood_fault | test_ood_topology | total |
|---|---:|---:|---:|---:|---:|---:|
| repair | 57 | 6 | 5 | 50 | 0 | 118 |
| audit | 66 | 6 | 3 | 0 | 1 | 76 |
| **TOTAL** | **123** | **12** | **8** | **50** | **1** | **194** |

`data/sft_all` (21 files; `audit_test_ood_fault.jsonl` is 0 bytes):

| task | train | val | test_id | test_ood_fault | test_ood_topology | climate | load | scale | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| repair | 110,635 | 669 | 654 | 4,253 | 837 | 0 | 0 | 0 | 117,048 |
| design | 1,908 | 235 | 204 | 0 | 0 | 615 | 411 | 460 | 3,833 |
| audit | 2,258 | 296 | 302 | 0 | 136 | 0 | 0 | 0 | 2,992 |
| ledger | 5,006 | 672 | 612 | 2,229 | 640 | 0 | 0 | 0 | 9,159 |
| **TOTAL** | **119,807** | **1,872** | **1,772** | **6,482** | **1,613** | **615** | **411** | **460** | **133,032** |

`data/grpo_all` (21 files; same non-train rows as `sft_all`, smaller train):

| task | train | val | test_id | test_ood_fault | test_ood_topology | climate | load | scale | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| repair | 5,850 | 669 | 654 | 4,253 | 837 | 0 | 0 | 0 | 12,263 |
| design | 1,908 | 235 | 204 | 0 | 0 | 615 | 411 | 460 | 3,833 |
| audit | 2,258 | 296 | 302 | 0 | 136 | 0 | 0 | 0 | 2,992 |
| ledger | 5,006 | 672 | 612 | 2,229 | 640 | 0 | 0 | 0 | 9,159 |
| **TOTAL** | **15,022** | **1,872** | **1,772** | **6,482** | **1,613** | **615** | **411** | **460** | **28,247** |

`data/sft_tools` and `data/grpo_tools` (4 files each, `*_train.jsonl` only):

| task | train | total |
|---|---:|---:|
| repair | 240 | 240 |
| design | 240 | 240 |
| audit | 240 | 240 |
| ledger | 240 | 240 |
| **TOTAL (each corpus)** | **960** | **960** |

`data/frontier*` task files — all `split=train`, 4 tasks:

| corpus | repair | design | audit | ledger | total |
|---|---:|---:|---:|---:|---:|
| `frontier` | 335 | 140 | 33 | 160 | 668 |
| `frontier_prov` | 335 | 140 | 33 | 160 | 668 |
| `frontier_export` (prompt-only) | 400 | 400 | 400 | 400 | 1,600 |
| `frontier_eval_export` (prompt-only) | 90 | 120 | 60 | 90 | 360 |

`frontier_eval_export` splits: `test_id` 120, `test_ood_fault` 60, `test_ood_topology` 90,
`test_ood_climate` 30, `test_ood_load` 30, `test_ood_scale` 30 (total 360).

`data/heldout` (20 JSONL):

| task | train | val | test_id | test_ood_fault | test_ood_topology | climate | load | scale | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| repair | 94 | 13 | 6 | 70 | 15 | 0 | 0 | 0 | 198 |
| design | 54 | 4 | 4 | 0 | 0 | 20 | 13 | 14 | 109 |
| audit | 84 | 9 | 16 | 0 | 6 | 0 | 0 | 0 | 115 |
| ledger | 60 | 10 | 11 | 32 | 7 | 0 | 0 | 0 | 120 |
| **TOTAL** | **292** | **36** | **37** | **102** | **28** | **20** | **13** | **14** | **542** |

---

## 2. `sft_tools` / `grpo_tools` deep metrics

### 2a. Multi-turn call-pair counts (per row, min / median / max)

For `sft_tools` the tags live in `completion[0].content`; for `grpo_tools` the `completion` is only the final answer
(**0** tool tags) and the multi-turn SFT target lives in `trajectory.sft_completion`. `<tool_call>` and
`<tool_result>` counts are equal per row (one result per call); rows with `<2` calls/results = **0** (measured).

| task | sft_tools calls (min/med/max) | sft_tools results | grpo_tools traj calls | grpo_tools traj results |
|---|---|---|---|---|
| repair | 3 / 3.0 / 3 | 3 / 3.0 / 3 | 3 / 3.0 / 3 | 3 / 3.0 / 3 |
| design | 2 / 2.5 / 3 | 2 / 2.5 / 3 | 2 / 2.5 / 3 | 2 / 2.5 / 3 |
| audit | 3 / 3.5 / 4 | 3 / 3.5 / 4 | 3 / 3.5 / 4 | 3 / 3.5 / 4 |
| ledger | 2 / 2.5 / 3 | 2 / 2.5 / 3 | 2 / 2.5 / 3 | 2 / 2.5 / 3 |

`<tool_response>` rows = **0** in both corpora; all tool blocks are XML `<tool_result>`.

### 2b. System-prompt char length (constant across rows of each task)

| task | system-prompt chars (sft_tools = grpo_tools) |
|---|---:|
| repair | 740 |
| design | 1,226 |
| audit | 914 |
| ledger | 838 |

### 2c. Token counts

- `sft_tools` has a stored per-row `tokens` field. Measured over all 960 stored values:
  **min 1,913 / median 3,778 / p95 4,645 / max 5,742** (matches `MANIFEST.json` token_length p50 3,778,
  max 5,742; manifest p95 4,620 uses a different quantile convention).
- `grpo_tools`: **not measured** — no per-row `tokens` field is stored. The real-tokenizer audit
  (`python -m thermo_agent.train.token_audit`, which needs the cached Qwen3.5 tokenizer) was **not** re-run here;
  it is not a cheap per-row read. The `sft_tools/MANIFEST.json` token_length block describes `sft_tools`, not
  `grpo_tools`.

### 2d. Stored `source` / `provenance.variant` mix

| corpus | `source` | `provenance.variant` (rows, per corpus) |
|---|---|---|
| `sft_tools` | `deterministic_tools`: 960 | `direct`: 480, `recover`: 480 |
| `grpo_tools` | `deterministic_tools`: 960 | `direct`: 480, `recover`: 480 |

Per task in each corpus: 240 rows = 120 `direct` + 120 `recover`. **`frontier_tools`: 0 rows** in both corpora
(corroborated by `data/sft_tools/MANIFEST.json`: counts `frontier_tools = 0` for all four tasks, `frontier.kept = 0`).
`harness`: `sft_tools` = `none` for all 960; `grpo_tools` has no `harness` field.

`grpo_tools` `validation.reward`: 718 rows at exactly `1.0`, 2 rows at `0.45`, and 240 rows at values in
`(0.9996, 1.0)` (non-unit floats) — i.e. all 960 rows are present as GRPO training rows regardless of reward.

---

## 3. Frontier (`data/frontier*`)

### 3a. Totals and acceptance

| corpus | rows | accepted (`validation.reward == 1.0`) | rejected (`reward != 1.0`) | rows with tool calls (`<tool_call>`/`<tool_result>`, completion or trajectory) |
|---|---:|---:|---:|---:|
| `frontier` | 668 | 635 | 33 | **0** |
| `frontier_prov` | 668 | 635 | 33 | **0** |
| `frontier_export` | 1,600 | not measured (no `validation`/`completion` field) | not measured | 0 (prompt-only) |
| `frontier_eval_export` | 360 | not measured (no `validation`/`completion` field) | not measured | 0 (prompt-only) |

All 33 rejected rows are the `google-gemini-3-8-flash` audit rows; every design/ledger/repair row has reward `1.0`.

### 3b. Rows by model and task (`frontier`; `frontier_prov` identical counts)

| task | rows | `google-gemini-3-1-flash-lite-preview` | `openai-gpt-5-4-nano-2026-03-17` | `google-gemini-3-8-flash` | accepted / rejected |
|---|---:|---:|---:|---:|---|
| repair | 335 | 311 | 24 | 0 | 335 / 0 |
| design | 140 | 114 | 26 | 0 | 140 / 0 |
| ledger | 160 | 88 | 72 | 0 | 160 / 0 |
| audit | 33 | 0 | 0 | 33 | 0 / 33 |
| **TOTAL** | **668** | **513** | **122** | **33** | **635 / 33** |

Model source: `frontier` carries no `model` field, so the model slug was parsed from the `id` suffix after `-t-`;
`frontier_prov` carries explicit `model`/`provider` fields and a `source` of `frontier:<model>`. Both give the same
per-task distribution. `frontier_prov/provenance_manifest.json` records the same histogram (513 / 122 / 33;
1, total 668) and SHA-256 per file.

### 3c. Prompts / responses present

| corpus | `prompt` | `completion` (response) | `validation` | note |
|---|---|---|---|---|
| `frontier` | yes, 668/668 | yes, 668/668 | yes, 668/668 | zero-shot answers, no tool turns |
| `frontier_prov` | yes, 668/668 | yes, 668/668 | yes, 668/668 | same rows + `model`/`provider` |
| `frontier_export` | yes, 1,600/1,600 | **no**, 0/1,600 | no | Kaggle teacher-prompt export (400/task) |
| `frontier_eval_export` | yes, 360/360 | **no**, 0/360 | no | held-out prompt export (30/row-cell, limit 30) |

### 3d. Usable frontier tool traces — explicit statement

**0 usable frontier tool traces exist on disk.** Measured: 0 of 668 `frontier` rows and 0 of 668 `frontier_prov`
rows contain any `<tool_call>` / `<tool_result>` tag in `completion` or `trajectory`; 0 rows in either tool corpus
have `source=frontier_tools`. The tool-trace builder's own manifest (`data/sft_tools/MANIFEST.json`,
generated 2026-11-19T09:27:59Z) records `frontier.scanned = 1,336` (= 668 + 668), `with_tool_turns = 0`,
`kept = 0`, `rejected = 0`, `over_budget = 0`, `deduplicated = 0`. The 635 accepted frontier rows that do exist are
zero-shot (`harness=none`, `tools=none`), with 0 tool calls, so they cannot serve as tool traces.

---

## 4. Held-out (`data/heldout`, seed 987654)

- 20 JSONL files / **542 distinct ids** total; of these, **12 frozen `*_test_*.jsonl` files / 214 ids / 4 tasks**
  are the scored set. Per-file counts (all ids unique per file; each row `split` matches its filename):

| file | split | task | rows |
|---|---|---|---:|
| `audit_test_id.jsonl` | test_id | audit | 16 |
| `audit_test_ood_topology.jsonl` | test_ood_topology | audit | 6 |
| `design_test_id.jsonl` | test_id | design | 4 |
| `design_test_ood_climate.jsonl` | test_ood_climate | design | 20 |
| `design_test_ood_load.jsonl` | test_ood_load | design | 13 |
| `design_test_ood_scale.jsonl` | test_ood_scale | design | 14 |
| `ledger_test_id.jsonl` | test_id | ledger | 11 |
| `ledger_test_ood_fault.jsonl` | test_ood_fault | ledger | 32 |
| `ledger_test_ood_topology.jsonl` | test_ood_topology | ledger | 7 |
| `repair_test_id.jsonl` | test_id | repair | 6 |
| `repair_test_ood_fault.jsonl` | test_ood_fault | repair | 70 |
| `repair_test_ood_topology.jsonl` | test_ood_topology | repair | 15 |
| **TOTAL** | — | **4 tasks** | **214** |

Task totals: repair 91, audit 22, ledger 50, design 51. Measured bytes of the 12 test files: **1,531,809** (this
matches the byte total quoted in the docs below).

**Frozen eval-set hash (quoted, not recomputed).** `docs/SPLIT_AUDIT.md` §"Eval-set hash" records, and
`docs/EVAL_PROTOCOL.md` §1 / Appendix A repeat, that the SHA-256 over sorted test-file names then each file's bytes
for the 12 `*_test_*.jsonl` files is:

```
8b409359e7a2dd441d0be99c89c00f0f3c61320f9d06e9695cc22569f8ef4c6e   (12 files, 1,531,809 bytes)
```

This report did **not** recompute the hash or guess the recipe; the value above is quoted from
`docs/SPLIT_AUDIT.md`, which states it matches the current on-disk data. (For reference only, the 20-file variant
quoted in the same doc is `12cb731a062486e51e39582e3c1de708e7d8276295cba6f3994051314b12be3e`; it is **not** the
protocol hash.) The `meta_<task>.json` files record seed 987654 and per-split `counts` that agree with the tables.

---

## 5. GAPS — what is missing for the Gate 4 mix (deterministic + accepted frontier traces)

Gate 4 (`IMPLEMENTATION_PLAN.md` §"Gate 4") requires `data/sft_tools` and `data/grpo_tools` to mix **deterministic
tool traces + accepted frontier tool traces** (single-turn rows optional as a documented ablation). Measured state:

| ingredient | sft_tools | grpo_tools | status |
|---|---:|---:|---|
| deterministic tool traces | 960 (240/task; 480 direct + 480 recover) | 960 (240/task; 480 direct + 480 recover) | present |
| accepted frontier tool traces | **0** | **0** | **missing** |
| frontier rows scanned by tool builder | 1,336 | 1,336 | scanned; 0 with tool turns |
| accepted frontier zero-shot rows (not tool traces) | 0 inside these corpora; 635 exist in `sft_all`/`grpo_all` | same | unusable as tool traces (0 tool calls) |

**Exact counts still needed:**

1. **Frontier tool traces: 960 per tool corpus (240 per task × 4) for a 1:1 frontier:deterministic mix, or a hard
   minimum of 1 per task × 4 = 4** so the "accepted frontier tool traces" ingredient is non-empty at all. Current
   measured count is **0 in both corpora**. To go from 960 → 1,920 rows in each corpus at 1:1 would require
   **+960 accepted frontier tool traces per corpus** (i.e. 960 for `sft_tools` and 960 for `grpo_tools`; if the same
   traces are reused for both, 960 unique accepted traces are needed).
2. **Frontier pool with tool turns: currently 0 of 1,336 scanned rows.** Need at least 1 accepted trace with a tool
   call per task; to match the deterministic tier, need 240 accepted tool-using traces per task (960 total). The
   build manifest shows `with_tool_turns=0`, so new harness generations (physics tools `inspect_plant`,
   `first_law_check`, `second_law_check`, `validity_check`, `solve_plant`) must be produced and ingested — none of
   the existing zero-shot rows qualify.
3. **Audit frontier: 0 accepted rows exist** (all 33 `google-gemini-3-8-flash` audit rows have reward < 1.0). A
   nonzero accepted audit yield is a prerequisite for any audit frontier tool trace; per `IMPLEMENTATION_PLAN.md`
   the audit prompt/tool fix (P0-5) should precede an audit pilot. Need **≥1 accepted audit frontier row**, and
   **240** to match the audit deterministic tier.
4. **Tokens for `grpo_tools`: not measured** (no stored `tokens` field). Need a real-tokenizer `token_audit` run
   (or a stored token field) to confirm the 8,000-token gate; not required to fix the frontier-trace gap but listed
   as an outstanding measurement.

Deterministic side is complete for Gate 4's mix; every remaining gap is on the accepted-frontier-tool-trace side.

---

## Appendix — script and commands

All reads were streaming `for line in open(path, encoding="utf-8")` over each `*.jsonl`; no corpus was loaded whole.
Scripts were written only to `%TEMP%/opencode`, never into the repo.

```
# from C:\Users\rezax\Home\Code\thermo_agent\Repo
$env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python.exe C:\Users\rezax\AppData\Local\Temp\opencode\corpus_inventory_measure.py   > ...\corpus_inventory_out.json
.venv\Scripts\python.exe C:\Users\rezax\AppData\Local\Temp\opencode\corpus_inventory_summary.py  > ...\corpus_inventory_summary.txt
.venv\Scripts\python.exe C:\Users\rezax\AppData\Local\Temp\opencode\corpus_inventory_crosstab.py > ...\corpus_inventory_crosstab.txt
```

`corpus_inventory_measure.py` computed, per file: row count; `split` histogram; `task`/`harness`/`source`/
`provenance.variant` histograms; `validation.reward` histogram; model (from `model` field or `id` suffix after
`-t-`); prompt/completion presence; counts of rows containing `<tool_call>` / `<tool_result>` / `<tool_response>`
in `completion` and in `trajectory.sft_completion`; per-row tag-count and system-prompt-char-length distributions;
and stored `tokens`. Companion scripts aggregated these into the cross-tabs and the frontier per-task/model/acceptance
tables. Quantiles use `xs[min(n-1, int(f*n))]` (p50/p95); medians in §2a use the mean of the two central values.
Held-out hash was quoted from `docs/SPLIT_AUDIT.md` ( corroborated in `docs/EVAL_PROTOCOL.md` §1/Appendix A ); not
recomputed here. No data, code, config or model was changed.
