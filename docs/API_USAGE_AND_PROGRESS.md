# API usage, data provenance, and project status

This is an audited record of which hosted models were called, what their outputs became, and how far the
project has come. Counts were obtained by streaming `json.loads` over the `.jsonl` files (see the commands in
§6); no number here is transcribed from a stale prose claim. The authoritative corpus report is
`data/sft/sft_report.json` / `data/grpo/sft_report.json` written by `synth/combine.py`.

## 1. Hosted models actually called

All frontier calls went through **Kaggle Benchmarks model access / quota**, not vendor keys. Prompt/answer journals
record the model id on every line, together with provider-reported token usage and nanodollar cost.

| Model (journal id) | Purpose | Prompts answered | Accepted by verifier | Acceptance | Recorded cost | Journal |
|---|---|---:|---:|---:|---:|---|
| `google/gemini-3.1-flash-lite-preview` | primary teacher + tool-harness arm | 1,600 full pass + 360 held-out + 444 tool answers (204 + 120 + 120) + 12 probe = **2,416** raw answers | 513 of 1,600 full-pass rows (plus held-out/tool answers) | **32.1 %** on the full pass | **$1.091** for the full pass | `data/teacher_runs/completions.jsonl`, `eval_*.jsonl`, `harness_*.jsonl`, `probe_gemini.jsonl` |
| `openai/gpt-5.4-nano-2026-03-17` | second teacher arm / domain-model stand-in | 1,600 full pass + 360 held-out + 12 probe = **1,972** raw answers | 122 of 1,600 full-pass rows | **7.6 %** on the full pass | **$0.658** for the full pass | `data/teacher_runs/completions_gptnano.jsonl`, `eval_gptnano_*.jsonl`, `probe_gptnano.jsonl` |

Full-pass spend ≈ **$1.75**. Counting every independent journal batch once (full passes, held-out eval answers,
both tool-harness versions, probes) the recorded API spend is **$2.752407**; excluding the superseded v1
tool-harness run it is **$2.561710**. Split journals and `eval_gptnano_replay.jsonl` are partitions/copies and
must not be added again. The decision rule, probe results and cost method are in
`docs/TEACHER_MODEL_CHOICES.md`; the cost fields come from `kaggle_benchmarks.usage.Usage` on each chat.

Held-out eval answers were generated for **360 prompts per model** (`data/frontier_eval_export/`, 30 per
task×split, 12 non-empty files): Gemini `eval_all.jsonl` ($0.184296), GPT-nano `eval_gptnano_all.jsonl` →
`eval_gptnano_replay.jsonl` ($0.131081). The tool arm is a further 120 Gemini answers with 242 `check_answer`
calls ($0.202697, `harness_eval_completions2.jsonl`).

**Models named but never successfully used for training data:** `anthropic/claude-sonnet-5-5@default` (the old
notebook default; 5 calls 400'd, now retired and deferred to a later round), `deepseek-r1-0528`, `qwen3-next-80b`,
`gpt-oss-120b`, `granite-4.0-h-small` (probe failures/no usable output), and the remaining Kaggle roster entries.
`google/gemini-3.8-flash` *was* called in the big-model campaign: 200 attempted, 33 accepted and ingested (see
`docs/TEACHER_MODEL_CHOICES.md`), so it is not in this list. No dedicated thermodynamics LLM was ever called: the
"other model" arm is GPT-5.4-nano's saved answers used as a stand-in, because no thermo-model id/endpoint was supplied.

## 2. Where each model's output went

```
1,600 exported ground-truth prompts (data/frontier_export, 400/task; no answers)
  -> raw teacher answers to those training prompts: 3,668 rows
       2,056 gemini  /  1,612 gpt-nano
     (held-out eval answers are a separate 360 prompts per model, scored as comparison arms)
       2,436 harness=none, 654 agents+skills, 444 tools, 134 skills
  -> verifier gate: 965 rejections -> 342 deduplicated failure-memory notes
                    635 accepts (data/frontier/*, reward=1.0)
  -> 8,000-token prompt gate (max_length 8,192): 635 accepts used in BOTH data/sft_all and data/grpo_all
      (legacy 4,000-token gate kept 460 accepts in the old data/sft and data/grpo)
```

| Artifact (training-prompt answers only) | Gemini | GPT-nano | Path |
|---|---:|---:|---|
| Raw answers | 2,056 | 1,612 | `data/teacher_runs/completions*.jsonl`, `harness_*.jsonl`, `eval_*.jsonl`, `probe_*.jsonl` |
| Accepted by verifier | 513 | 122 | `data/frontier/{repair,ledger,design,audit}_train.jsonl` (audit 0) |
| Actually in the legacy 4k-gated SFT/GRPO | 371 | 89 | `data/sft/*_train.jsonl`, `data/grpo/*_train.jsonl` (`source=frontier`) — all 635 are in the merged 8k-gated `data/sft_all` / `data/grpo_all` |
| Failure memory | 177 notes | 141 notes | `data/memory/frontier_notes.jsonl`, `data/memory/failure_reasons.json` |
| Frontier zero-shot eval arm | held-out answers | — | `reports/eval_frontier_zero.json`, `paper/assets/model_results.json` |
| Frontier + our tools arm | 120 tool answers | — | `reports/eval_frontier_tools.json`, `reports/tool_usage.json` |
| Domain-model arm | — | held-out answers | `reports/eval_other_model.json` |
| Paper examples / trace figure | 6 frontier + 3 harness examples | none promoted | `paper/assets/examples.json`, `paper/figures/svg/fig_trace_walkthrough.svg` |

All 635 accepted rows survive the 8,000-token prompt gate (max length 8,192) and are in training in the merged
corpora; none are dropped. (Under the superseded 4,000-token gate, 175 were dropped and the headline count of AI
traces used in the legacy `data/sft` / `data/grpo` corpora was **460**.) The 3,668 journal rows are raw captured
answers, not training rows; the headline count of AI traces actually used for training in the merged corpora is
**635**.

### 2.1 Tool-using CoT data generation (current objective)

The zero-shot journals above are **legacy training data**. The current objective is to teach the 4B student *tool
use*, not just final answers. Frontier models run inside a **LangGraph harness** (`thermo_agent/teacher/agent_graph.py`,
`build_context -> propose -> {call_tools -> reflect -> propose}* -> finalize -> END`) with **real physics tools**
from `thermo_agent/teacher/toolkit.py`: `first_law_check`, `second_law_check`, `validity_check`, `solve_plant` and
`check_answer`. These build the plant from the model's own candidate, run the sequential-modular solver and return
numbers with units, so the feedback the model sees is solver output, not keyword matching. The context node injects
**AGENTS.md + component/task SKILL.md** — the same skills the student trains on — and reloads **fresh failure memory**
from `data/memory/*.jsonl` on every row. The model can call tools over several rounds, read numeric feedback and
revise before finalizing.

When the verifiers accept a tool-using trace, the **full trajectory** (reasoning, tool calls, numeric tool results,
revisions) is persisted as the SFT completion with `source=frontier_tools`, so the student imitates the whole
tool-using behaviour. Ingest is additive and provenance is recorded per trace in
`data/provenance/trace_ledger.jsonl`.

Current model choice for generating these traces: `gpt-6-luna` then `claude-haiku-5-5`, reasoning ON
(`reasoning="low"`), run as the hosted task `thermo-teacher-tools-s00` via
`scripts/run_teacher_hosted.ps1 -m <slug>`; `gemini-3.5-flash-lite` is excluded and must not consume quota.
The expensive tier follows only after the budget tier reports accepted reasoning traces.
**REMINDER: revisit `claude-sonnet-5-5` later** (deferred as too expensive — needs Reza's explicit go-ahead). All
arms are scored on the held-out `data/heldout` set (fresh seed 987654, excluded from every corpus), with each
checkpoint in its own path (`runs/sft`, `runs/grpo_k4`, `runs/grpo_k8`, `runs/grpo_k16`).

## 3. Chain-of-thought and harness counts

**Synthetic (our own) reasoning traces** — every row carries a task-tagged reasoning block
(`<diagnosis>`,`<edits>` repair/ledger; `<analysis>` audit; `<design>` design):

| Source | repair | ledger | audit | design | total |
|---|---:|---:|---:|---:|---:|
| `data/processed` | 11,810 | 8,999 | 2,916 | 3,693 | 27,418 |
| `data/processed_arch` | 2,063 | 2,115 | 1,366 | — | 5,544 |
| **Unique synthetic base traces** | **13,873** | **11,114** | **4,282** | **3,693** | **32,962** |

Literal `<analysis>` exists only in audit rows: 2,916 + 1,366 = **4,282** unique synthetic audit traces.

Corpus rows with synthetic reasoning: **122,564 SFT rows** and **27,754 GRPO rows**. Repair train shows the
20-paraphrase expansion exactly: 4,990 base rows × 20 = **99,800** rows.

**AI-frontier traces captured in harness:**

| Stage | Count | Where |
|---|---:|---|
| Raw answers with a harness/inference-text block | 788 (654 agents+skills + 134 skills) | teacher journals |
| Raw answers with tools exposed | 444 | `harness_completions.jsonl`, `harness_eval_completions*.jsonl` |
| Accepted teacher rows with a harness block | 108 of 635 | `data/frontier/*` |
| Teacher rows in SFT/GRPO with `harness=agents+skills` | **80 of 460** | `data/sft`, `data/grpo` |

Do not confuse this with harness text injected into synthetic rows: 22,535 SFT rows and 6,442 GRPO rows have
`harness != none`, but almost all of those are ground-truth traces with AGENTS.md/skills text in the prompt.
The effective harness share is 18.3 % (SFT) / 22.8 % (GRPO) after paraphrase expansion and the token gate, not
the nominal 35 % applied per base row.

**Failure memory:** 965 graded rejections were deduplicated into **342 stored notes**
(`frontier_notes.jsonl` 318 + `adversarial_repair.jsonl` 24); by task design 178, repair 68, ledger 52, audit 44.

## 4. Measured effectiveness

- Full teacher pass: Gemini accepted 513/1,600 at $1.091 (~$0.00213/accepted row); GPT-nano 122/1,600 at
  $0.658 (~$0.00539/accepted row). The cheaper model is ~2.5× more expensive per accepted row.
- Frontier zero-shot eval (30/split): repair `test_id` **0.633**, `ood_fault` 0.467, `ood_topology` 0.433;
  design valid 0.167/0.2/0.067/0.1; audit rank 0.086/0.033; ledger all_correct 0/0/0.067.
- Frontier + our tools (replay, `test_id`): repair **0.667**, design valid **0.433**, audit 0.1, ledger
  **0.767**. Tool usage: 120 answer records, 242 `check_answer` calls, **$0.2027**, answer rate 1.0;
  calls/answer: repair 1.5, ledger 2.1, audit 1.73, design 2.73.
- Other model (GPT-nano stand-in): weaker than Gemini across repair/design/audit/ledger.

## 5. Project status

### Done
- Synthetic ground truth: 32,962 unique verifier-graded traces across four tasks and structural OOD splits;
  architecture catalog; 20 paraphrases per repair base row under an 8,000-token gate (`max_length` 8,192) in the
  merged corpora (the legacy 4,000-token gate applied only to `data/sft` / `data/grpo`).
- Verifier-gated teacher pass: 1,600 prompts × 2 models, 3,668 raw answers, 635 accepted, 342 memory notes,
  635 accepted traces in both merged training corpora (`data/sft_all` / `data/grpo_all`), all provenance preserved.
- Corpora: SFT **123,024** rows / 109,659 train; GRPO **28,214** rows / 14,849 train.
- Paper: 45-page draft, no overfull boxes, `verify_paper.py --no-build` clean, 242 `\val` keys, provenance
  for every generated number, static SVG figures preserved.
- Measured comparison arms: frontier zero-shot, frontier+tools, and domain-model stand-in (28 rows in the
  manifest); tool-usage accounting; deterministic paper examples and regenerable trace figure.
- Kaggle infrastructure: code/corpus/prompt/teacher-progress datasets, chunked cross-entropy that fixes the
  248k-vocab T4 OOM, smoke run that completed 20 training steps, cumulative logging/status, SOCKS proxy
  workflow, prebuilt fused-kernel wheel build, token-free cross-session resume relay.

### In progress
- Smoke v18 (eval-only memory fix) queued; fused-kernel wheel build running on Kaggle.
- Re-upload of the code bundle with resume/wheel support and the kernel metadata that attaches the new datasets.

### Not done
- **No SFT or GRPO weights exist yet.** The 4B model was never trained past the 20-step smoke; the full SFT
  epoch, merge and four GRPO stages are the remaining GPU work.
- `ours_base`, `ours_sft`, `ours_harness`, `ours_grpo` eval arms have not been run; their manifest rows are
  missing. The eval kernel and the `ours_sft`/`ours_grpo` specs are now prepared but unrun.
- GRPO-vs-SFT reward deltas, trained-policy zero-variance groups, memory ablation, GPU-hour totals,
  cost-quality frontier and significance tests are still pending (see `paper/Q1_CHECKLIST.md`).
- Three external figures (`fig_training`, `fig_corpus`, `fig_audit_first_vs_second_law`) still show pilot
  samples/split bars and must be redrawn from current assets before camera-ready.
- No thermo-specific external LLM was evaluated; no git push (invalid `gh` token).

## 6. Caveats found during the audit

- `reports/frontier_ingest.json` is cited in `docs/TEACHER_MODEL_CHOICES.md` but was never committed; cost is
  recoverable from the journals' `usage` fields and the acceptance counts from the id suffixes.
- Accepted rows in `data/frontier` have no `model` field: the model is encoded in the id suffix
  (`-t-google-gemini-...` / `-t-openai-gpt-...`).
- `data/memory/failure_reasons.json` grades 965 rejections, while `data/memory/frontier_notes.jsonl` stores 318
  deduplicated frontier notes (342 including adversarial); they are different stages, not a contradiction.
- `eval_frontier_tools.json` and `tool_usage.json` were computed from `harness_eval_completions2.jsonl` (242
  tool calls). The staged replay dataset previously shipped the superseded v1 (297 calls, 69 exhausted); it now
  ships v2 as well and `kaggle/eval_runner.py` points at v2.
- `paper/figures/figure_manifest.json` does not list `model_results.json` as a data source for
  `fig_trace_walkthrough`, although `figures_from_examples.py` consumes it.
- `docs/STATUS.md` still describes the unexecuted notebook path (Claude default) and some old counts; this report
  and `docs/TEACHER_MODEL_CHOICES.md` carry the measured values.

## 7. Additions after the original audit

- Synthetic traces grew from 32,962 to **33,156** unique base traces with the new expert-scale cycle
  (`data/processed_big`, 194 rows), and the per-trace ledger now covers **37,945** traces
  (`data/provenance/trace_ledger.jsonl`). The ledger auto-discovers new journals; big-model traces will be
  added without deleting the Gemini/GPT-nano records.
- The merged training corpora (`data/sft_all` / `data/grpo_all`) keep the old rows and add 1,140 / 57
  big-cycle training rows; the 8,000-token gate audited at max row 6,405 with zero drops.
- Kaggle datasets now attached to the train/smoke/eval kernels: `thermo-agent-code`, `thermo-agent-corpus`
  (with `sft_all.zip` / `grpo_all.zip`), `thermo-teacher-prompts`, `thermo-agent-resume`,
  `thermo-agent-eval-replays`; fused wheels come from kernel `thermo-kernel-wheels` via `kernel_sources`.
- Hosted-task campaign round 2 (`thermo-teacher-big-s00` v3, prompt dataset `thermo-teacher-prompts-big`):
  **`gpt-6-luna` then `claude-haiku-5-5`**, reasoning ON (`reasoning="low"`, 1,536 output tokens) on 974
  previously-unaccepted prompts, run as hosted task `thermo-teacher-tools-s00` via
  `scripts/run_teacher_hosted.ps1 -m <slug>`; `gemini-3.5-flash-lite` is excluded and must not consume quota.
  Additive to the old journals. **REMINDER:** revisit `claude-sonnet-5-5` later — the user deferred it as too
  expensive and it needs Reza's explicit go-ahead. No accepted rows from those models yet.

## 8. Reproduction commands

```powershell
# Corpus provenance and counts (authoritative)
.venv/Scripts/python.exe -c "import json;from collections import Counter;rows=[json.loads(l) for l in open('data/sft/sft_report.json')];print(rows)"

# Count frontier rows used in training
.venv/Scripts/python.exe -c "import json;from collections import Counter;c=Counter();[c.update({(json.loads(l)['task'],json.loads(l)['source']):1}) for f in __import__('glob').glob('data/sft/*_train.jsonl') for l in open(f,encoding='utf-8')];print(c)"

# Teacher cost and acceptance
.venv/Scripts/python.exe -c "import json,glob;from collections import Counter;a=Counter();[a.update({(r.get('model'),r.get('error') is None):1}) for f in glob.glob('data/teacher_runs/completions*.jsonl') for l in open(f,encoding='utf-8') for r in [json.loads(l)]];print(a)"
```
