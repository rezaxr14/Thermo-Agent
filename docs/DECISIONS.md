# Decisions

Decisions are verified against commands actually run on 2026-11-19 in this repo (`.venv/Scripts/python.exe`).
No secret values are reproduced. Where a number could not be verified it would be marked "unverified — omitted";
all numbers below were verified.

## 2026-11-19 — Held-out set is frozen at seed 987654, test splits only, zero contamination

**Decision.** The frozen evaluation set is `data/heldout`, generated with seed 987654. Only the `*_test_*.jsonl`
splits are evaluation rows (train/val there are generation helpers). It holds 214 ids / 675 signatures, and the
contamination guard reports 0 leaks into any training/prompt/memory artifact; the same test proves the scanner
actually fails on a planted leak.

**Evidence.** `.venv\Scripts\python.exe -m pytest tests/test_contamination.py -s -q` → `contamination scan: 214 held-out ids, 675 signatures, 127 files, 115091 lines sampled` then `2 passed in 39.67s` (no leak list printed). `Get-ChildItem data/heldout -Filter meta_*.json | ForEach-Object { $_.Name; Get-Content $_.FullName -TotalCount 30 }` → `meta_audit.json`, `meta_design.json`, `meta_ledger.json`, `meta_repair.json` each report `"seed": 987654`. `tests/test_contamination.py` line 108 (`files = sorted(heldout_dir.glob("*_test_*.jsonl"))`) confirms test splits only.

**Status.** active

## 2026-11-19 — Never swap numpy inside the Kaggle kernel; pin per child process

**Decision.** Do not reinstall numpy/scipy into the running kernel: Kaggle's ipykernel preloads numpy before cell 1, so the old compiled C extension stays in `sys.modules` and a new pure-Python layer cannot bind to it. Instead install `numpy==2.4.4` / `scipy==1.17.1` with `pip --target` into a side directory and put that directory first on `PYTHONPATH` for **every child process**; the parent never imports the numeric stack.

**Evidence.** `Select-String thermo_agent/kaggle_notebook.py -Pattern '^def (_pinned_env|_child|_install_pinned_numeric|setup|install_cached_kernels|run)\(|--target|numpy==2.4.4|scipy==1.17.1|stale C extension stays cached'` → `def _install_pinned_numeric` uses `pip install --no-deps --target <dir> numpy==2.4.4 scipy==1.17.1` and its docstring says "stale C extension stays cached in sys.modules"; `def _pinned_env` prepends `PINNED_SITE` to `PYTHONPATH`; `def _child` runs a fresh interpreter with that env; `def setup` sets `PYTHONPATH` so children import the pinned wheels from startup; `def run` spawns each training/eval subprocess. `Select-String logs/smoke_v23.log,logs/smoke_v24.log -Pattern '\[compat\]'` → both logs print `[compat] numpy 2.4.4 scipy 1.17.1 import OK`.

**Status.** active

## 2026-11-19 — Big-model CoT runs as a hosted task (`-m <slug>`), not as an SDK key

**Decision.** The Kaggle Benchmarks SDK exposes only 8 models in this account's roster. `luna`, `claude-haiku-5-5`, `gemini-3.5-flash-lite`, `sonnet-5-5`, `opus-5-5` and `gemini-4-argon` are NOT SDK keys, so big-model Chain-of-Thought must run as a hosted task with `-m <slug>`. The hosted catalog is documented, not probed offline.

**Evidence.** `.venv\Scripts\python.exe -c "... LLMS_AVAILABLE ..."` → `SDK LLMS_AVAILABLE count: 8` with slugs `claude-sonnet-5@default, deepseek-r1-0528, gemini-3-flash-preview, gemini-3.1-flash-lite-preview, granite-4.0-h-small, gpt-5.4-nano-2026-03-17, gpt-oss-120b, qwen3-next-80b-a3b-instruct`; `non-SDK (absent from LLMS_AVAILABLE): ['luna', 'claude-haiku-5-5', 'gemini-3.5-flash-lite', 'sonnet-5-5', 'opus-5-5', 'gemini-4-argon']`. `Select-String docs/TEACHER_MODEL_CHOICES.md -Pattern 'gpt-6-luna|claude-haiku-5-5|gemini-3\.5-flash-lite|claude-sonnet-5-5|gemini-4-argon|thermo-teacher-big-s00'` → lines 27-31 list the budget/expensive tiers and lines 58-72 list the hosted task `thermo-teacher-big-s00` with these slugs and the deferred-Sonnet reminder. `kaggle benchmarks tasks models` and the in-process `import kaggle_benchmarks` check were NOT run (no network): hosted membership is per that doc. Secret URL/API-key values were never printed.

**Status.** active

## 2026-11-19 — GRPO: k=4/8/16, stage `grpo_<task>_k<K>`, init `runs/sft_merged`, distinct dirs, hard-fail on missing adapter

**Decision.** GRPO group sizes are k=4, 8, 16. Each stage is named `grpo_<task>_k<K>` and uses its own `runs/grpo_<task>_k<K>` directory so checkpoints never share weights. All arms initialise from the merged SFT policy `runs/sft_merged`, never base weights. A missing/empty adapter is a hard error (never score base under a GRPO label).

**Evidence.** `Select-String thermo_agent/train/grpo.py -Pattern 'def stage_name|return f"grpo_|DEFAULT_INIT|k=4/8/16'` → `def stage_name(task, k): return f"grpo_{task}_k{k}"`, `DEFAULT_INIT = "runs/sft_merged"`, and comment "k=4/8/16 never share a remote prefix or checkpoint dir". `Select-String kaggle/eval_runner.py -Pattern 'for _k in \(4, 8, 16\)|def require_adapter|is missing or empty|grpo_k": (4|8|16)'` → `for _k in (4, 8, 16)`, `def require_adapter` raising on "GRPO adapter ... is missing or empty", and `grpo_k` 4/8/16 arm specs.

**Status.** active

## 2026-11-19 — Five physics tools; `check_answer` is not a tool

**Decision.** The teacher toolkit exposes exactly five physics tools. `check_answer` is deliberately absent: grading happens at acceptance (verifier ingest) and as the GRPO reward, not as a callable tool.

**Evidence.** `.venv\Scripts\python.exe -c "from thermo_agent.teacher import toolkit; n=toolkit.tool_names(); print(len(n), list(n), 'check_answer' in n)"` → `5 ['inspect_plant', 'first_law_check', 'second_law_check', 'validity_check', 'solve_plant'] False`.

**Status.** active

## 2026-11-19 — SFT loss: completion-only plus masked `<tool_result>` spans

**Decision.** SFT uses `completion_only_loss=True` (prompt tokens masked) and additionally masks every token whose character span intersects a `<tool_result>...</tool_result>` block, tags included. Rule: labels start as `-100` for the prompt; `mask_tool_result_spans` sets `out[i] = -100` when a token's `[start, end)` overlaps any tool-result span; nested or unclosed blocks mask to end-of-text; stray closing tags are ignored. The chunked cross-entropy ignores `-100` labels and normalises over unmasked tokens.

**Evidence.** `Select-String thermo_agent/train/sft.py -Pattern 'completion_only_loss=True|labels = \[-100\]|_TOOL_RESULT_OPEN|out\[i\] = -100|intersects any tool-result'` → line 199 `completion_only_loss=True`, line 108 `labels = [-100] * prompt_len + list(input_ids[prompt_len:])`, line 57 `def mask_tool_result_spans`, line 63 "masked when its half-open character span intersects any tool-result span", line 76 `out[i] = -100`; doc at lines 94-95 says the chunked loss ignores `-100`.

**Status.** active

## 2026-11-19 — Sequence gate: 8,000-token cap inside an 8,192 window; drop, never truncate

**Decision.** The model window is 8,192 tokens (`max_length`). Rows are gated at 8,000 tokens for prompt + completion, leaving a 192-token reserve for chat markers. Over-length rows are dropped and counted by `synth.combine`; nothing is silently truncated by the trainer.

**Evidence.** `Select-String thermo_agent/train/config.py -Pattern 'max_length|8000'` → `24: max_length: int = 8192  # prompt + completion, both gated at 8000 tokens by synth.combine`. `Select-String thermo_agent/synth/combine.py -Pattern 'max_tokens: int = 8000|silently truncated|8000 leaves 192'` → `91: def __init__(..., max_tokens: int = 8000)`, `14-15: "Every row is measured before it is written and dropped (and counted) ... so nothing is ever silently truncated by the trainer"`, `298: "8000 leaves 192 tokens under SFT max_length 8192"`.

**Status.** active

## 2026-11-19 — Precision: fp16 LoRA DDP on dual T4, GradScaler, fp32 reductions; no QLoRA/FSDP/ZeRO

**Decision.** Training is full fp16 LoRA with two DDP ranks on dual T4 (Turing, so bf16 is excluded). fp16 runs under the Trainer's `GradScaler` with gradient clipping; matmul reductions and the chunked cross-entropy/log-sum-exp are accumulated in fp32 to limit overflow. QLoRA is rejected under DDP, and FSDP/ZeRO are deliberately not used.

**Evidence.** `Select-String thermo_agent/train/{config,sft,grpo,ddp,chunked_loss}.py -Pattern 'Dual-T4 target|GradScaler|allow_fp16_reduced_precision_reduction = False|QLoRA \+ DDP is not supported|Why DDP and not FSDP/ZeRO|log-sum-exp in fp32'` → `config.py:4 "Dual-T4 target ... full fp16 LoRA, two DDP ranks, no 4-bit quantisation"`, `sft.py:9 "GradScaler"`, `sft.py:140` / `grpo.py:56` `allow_fp16_reduced_precision_reduction = False`, `sft.py:169` / `grpo.py:86` raise on `QLoRA + DDP`, `ddp.py:3 "Why DDP and not FSDP/ZeRO"`, `chunked_loss.py:9 "online log-sum-exp in fp32"`.

**Status.** active

## 2026-11-19 — Budget-first model order; ask Reza before `claude-sonnet-5-5`

**Decision.** The teacher campaign runs the budget tier first (`gpt-6-luna` then `claude-haiku-5-5`, reasoning ON, run as the hosted task `thermo-teacher-tools-s00` via `scripts/run_teacher_hosted.ps1 -m <slug>`); `gemini-3.5-flash-lite` is excluded and must not consume quota. Then the expensive tier (`gpt-6.2-sol`, `claude-sonnet-5-5`, `claude-opus-5-5-default`, `gemini-4-argon`) only after the budget tier reports verifier-accepted reasoning-trace yield, and only with `-AllowExpensive`; `claude-sonnet-5-5` additionally needs Reza's explicit go-ahead. Standing reminder: tell Reza / revisit `claude-sonnet-5-5` in a later round — it was deferred as too expensive; do not spend on it before the cheap-model check.

**Evidence.** `Select-String docs/TEACHER_MODEL_CHOICES.md -Pattern 'Budget tier|Expensive tier|Standing reminder|AllowExpensive|REMINDER'` → line 27 budget tier list, lines 28-31 expensive tier and `-AllowExpensive` gate, line 33 "Standing reminder: revisit `claude-sonnet-5-5` later — the user deferred it as too expensive", lines 71-72 repeat the reminder.

**Status.** active
