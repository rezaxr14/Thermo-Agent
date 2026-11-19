# Directive compliance audit — 2026-11-19

Purpose: an independent model (e.g. Claude) can read this file, run the commands, and decide for itself
whether the work matches what the user actually asked for. It is written to make deviations easy to find,
not to make the project look good.

Conventions:
- **[VERIFIED]** = re-checked by command in this session; the command and output shape are given.
- **[RECORDED]** = taken from an earlier doc/journal; verify before trusting.
- **[OPEN]** = known to be incomplete or broken right now.
- Status vocabulary: `DONE`, `PARTIAL`, `NOT DONE`, `DEVIATED` (did something the user did not ask for).

---

## 1. How to audit this repo in 10 minutes

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

## 2. Compliance matrix

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
| D11 | Token-efficient memory (what failed/worked only) | DONE | `docs/MEMORY_AUDIT_2026-11-19.md`: served k=3 p50 538 chars / 80 tokens, p95 892 / 102 |
| D12 | Two sources across 4 task families (yes/no) | **answer: no, three streams; frontier stream empty** | §4 |
| D13 | Traces first, training later | DONE (no training started) | no `runs/sft*` checkpoint exists |
| D14 | Many more traces, multi-model | **NOT DONE** | 0 accepted tool traces; plan in §6 |
| D15 | Prompt separation in tracegen/training/evals | PARTIAL | matrix §5; two leaks fixed today, one open |
| D16 | Smoke-test before big run | **OPEN** | smoke kernel v28 = `ERROR`, log not triaged |
| D17 | Sync to HF + Kaggle datasets | **NOT DONE** | scripts written, never executed |
| D18 | Free subagents for bulk work | DONE | 3× `opencode/hy4-flash-free` sessions |
| D19 | Document instructions for an auditor | DONE | this file + `docs/USER_DIRECTIVES.md` |
| D20 | Epoch count is the user's decision | DEVIATED then corrected | `config.py` still holds `epochs = 1` (placeholder, unauthorised) |

---

## 3. Data reality today [VERIFIED]

| Corpus | Rows | What it is |
|---|---|---|
| `data/sft_all` | 133,032 | 20-variant zero-shot corpus + 635 accepted zero-shot frontier rows + expert-scale rows |
| `data/grpo_all` | 28,247 | matching GRPO prompts |
| `data/sft_tools` | 960 | deterministic multi-turn tool traces, plain prompt (4 tasks × 2 variants × 120) |
| `data/grpo_tools` | 960 | matching GRPO rows |
| `data/sft_tools_h` | 960 | same completions, harness system prompt; 960/960 rows `prompt_variant="harness_tools"` |
| `data/grpo_tools_h` | 960 | same, GRPO side; 960/960 marked |
| `data/frontier` | 668 scanned / 635 accepted | **zero** rows with `<tool_call>` + `<tool_result>` |

Held-out: `data/heldout`, 214 test ids, hash `8b409359…f8ef4c6e` [RECORDED: `docs/CORPUS_INVENTORY_2026-11-19.md`].

`python scripts/mix_tool_corpora.py --stats` today prints:

```
WARNING: 0 verifier-accepted frontier rows containing BOTH <tool_call> and <tool_result> were selected
(scanned=668, accepted=668, with tool turns=0). The output is a deterministic-only corpus.
```

That warning is the single most important fact in this repo: **there is no frontier tool CoT training data yet.**

---

## 4. Self-reported deviations (read this section first)

### V1 — Frontier CoT was described as if it existed (HIGH)
Early claims of "frontier CoT training data" were only zero-shot answers (`harness="none"`, `tools="none"`).
The 635 accepted rows contain no tool calls. Any statement implying the student was learning frontier tool
behaviour from them was wrong.

Check: `python scripts/mix_tool_corpora.py --stats` → `with tool turns=0`.

### V2 — `epochs = 1` was set by the assistant, not the user (HIGH)
`thermo_agent/train/config.py:23`. Sized against quota (~30 h/epoch on 2×T4), never authorised.
Options: 1 epoch ≈ 30 h ≈ 3 sessions; 2 epochs does not fit before the 2026-11-21 refill; step-based
(`--max-steps`) is the fixed-cost alternative. **Status: undecided — user sets it.**

### V3 — GRPO arms were scored without tools (HIGH, fixed today)
`kaggle/eval_runner.py`: `ours_grpo`, `ours_sft_grpo_k4/k8/k16` had `tool_access="none"` although GRPO is
trained inside the tool loop. Now `"tools"`. Guarded by `tests/test_eval_arm_separation.py`.

### V4 — Eval rendered a different chat template than training (HIGH, fixed today)
`HFGenerator` hardcoded `enable_thinking=False`; the SFT/GRPO corpora are rendered with `enable_thinking=True`.
Now defaults to `True` with an explicit `--no-thinking` escape, and a test asserts the default.

### V5 — `--harness-prompt` was unreachable (MEDIUM, fixed today)
`scripts/mix_tool_corpora.py` had `build_mix(..., harness_prompt=...)` but `main()` never registered the flag.
Fixed; 14/14 tests in `tests/test_mix_tool_corpora.py` pass. Without this, the "re-prompt frontier traces with
the eval prompt" decision would have been silently dead code.

### V6 — Trace caps were the assistant's numbers (MEDIUM, removed)
480, then 2,880: neither was authorised. Both are gone; the hosted task now defaults to every row in the shard
with no cost cap and no deadline, and the mix cap defaults to unlimited. See §6.

### V7 — Smoke kernel v28 is red and untriaged (HIGH, open)
`rezanadimi/thermo-train-smoke` → `KernelWorkerStatus.ERROR`. It was pushed to test sdpa/allocation with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Gate stays closed until the log is read.

### V8 — HF/Kaggle sync was written but not run (MEDIUM, open)
`scripts/sync_corpora_hf.py` (private `Reza-Nadimi/thermo-agent-corpora`, allow-list, dry-run first) and
`scripts/upload_corpus_dataset.ps1` (adds `sft_tools_h`/`grpo_tools_h` zips). Nothing uploaded.

### V9 — Replay arms can re-execute stored tool calls (LOW, open)
`frontier_tools`/`frontier_harness` replay with `tool_access="tools"`, so `ToolLoop` may re-run a stored
`<tool_call>` against a fresh environment. Historical behaviour, documented in source, still unchecked.

### V10 — "Two sources" framing was wrong (LOW)
No single "two things". Three streams, four task families. See §3.

---

## 5. Prompt separation matrix

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

## 6. Trace campaign v2 (no assistant-invented cap)

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

## 7. Open blockers (in order)

1. `thermo-teacher-tools-s00` v6 is `RUNNING` with `gemini-3.8-flash`; luna cannot start until a version
   completes ("Only completed tasks can be run").
2. Smoke v28 `ERROR` — log untriaged; training gate closed.
3. Zero accepted tool traces; mix is deterministic-only.
4. HF/Kaggle sync not executed.
5. `epochs` undecided; do not start a run with a placeholder value.

---

## 8. Things a reviewer should distrust

- Any sentence that says "frontier traces" without the `<tool_call>` count next to it.
- Any "passing" claim about the smoke kernel without the latest `kernels status` + log.
- Any training number (epochs, steps, hours) that is not the user's explicit choice.
- Any doc that predates 2026-11-19 14:00 UTC and talks about "480 traces".
- `docs/STATUS.md` timelines: they are append-only history, not current state. This file wins for state.
