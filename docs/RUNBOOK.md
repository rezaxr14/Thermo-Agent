# Runbook: commands, Kaggle CLI, dual-T4, local model

How to run things.

## Runbook

*(merged from `docs/RUNBOOK.md` on 2026-11-20; content kept, headings demoted one level)*

Goal: generate data, check it, measure the base model, run SFT then GRPO on Qwen3.5-4B, and report back a small bundle of files.
Everything under "Generate data" and every test runs on CPU. The GPU steps (evaluation, SFT, GRPO, merge) target
Kaggle's dual T4 through `scripts/train_dual_t4.sh`; Section 10 lists what is most likely to need a fix. Run the
steps in order and stop at the first failure; send back the failing command and its output.

### Current plan: teacher campaign, Kaggle pushes, held-out scoring

The three commands below are the active plan. Sections 0–11 that follow are the older step-by-step procedure and
are kept for reference; the current state is summarised in `docs/STATUS.md`.

#### (a) Run the teacher campaign (budget models first)

```powershell
# Safe default: print the exact driver + ingest commands, no API calls and no writes.
pwsh -File scripts/run_teacher_campaign.ps1 -DryRun

# Actually run the budget tier (200 rows per model), resuming from shards already on disk.
pwsh -File scripts/run_teacher_campaign.ps1 -Execute -Rows 200 -Resume

# One task only:
pwsh -File scripts/run_teacher_campaign.ps1 -Execute -Task repair -Rows 200 -Resume

# Only after the budget tier has reported its accepted-CoT yield:
pwsh -File scripts/run_teacher_campaign.ps1 -Execute -Tier expensive -AllowExpensive -Rows 200 -Resume
```

The budget tier runs `gpt-6-luna` first, then `claude-haiku-5-5`, with reasoning ON; `gemini-3.5-flash-lite` is
excluded and must not consume quota. These are hosted-only slugs, so run them as the hosted task
`thermo-teacher-tools-s00` via `pwsh -File scripts/run_teacher_hosted.ps1 -Execute -m <slug>`, not on the local SDK
driver. The opt-in expensive tier is `gpt-6.2-sol`, `claude-sonnet-5-5`, `claude-opus-5-5-default` (`claude-opus`)
and `gemini-4-argon` (`gemini-4`) — only after the budget tier reports verifier-accepted reasoning-trace yield, and
`claude-sonnet-5-5` additionally needs Reza's explicit go-ahead. Any explicit mixed `-Models` list is reordered
budget-first, and `-Execute` on the expensive tier requires `-AllowExpensive`. **REMINDER:** revisit
`claude-sonnet-5-5` before spending on that tier — it was deferred as too expensive
(`docs/TEACHER_CAMPAIGN.md`). `-DryRun` is the default and wins if both switches are passed.
Preflight refuses to start unless `KAGGLE_API_TOKEN` and `HF_TOKEN` are available (from the environment or `.env`);
their values are never printed. Each model writes its own append-only shard
`data/teacher_runs/harness_<model>_<task>.jsonl`, then `transfer.ingest` adds accepted tool-CoT rows with
`source=frontier_tools`. Full contract in `docs/TEACHER_CAMPAIGN.md`.

#### (b) Push the smoke, train and eval kernels

```powershell
pwsh -File scripts/make_kaggle_bundle.ps1
pwsh scripts/upload_corpus_dataset.ps1            # includes heldout.zip, rebuilt every upload
$k = '.venv/Scripts/kaggle.exe'
foreach ($dir in @('kaggle/train_kernel_smoke', 'kaggle/train_kernel', 'kaggle/eval_kernel')) {
    $p = (Resolve-Path $dir).Path                  # absolute path required on Windows
    & $k kernels push -p "$p" --accelerator NvidiaTeslaT4
}
& $k kernels status rezanadimi/thermo-train-smoke
& $k kernels logs   rezanadimi/thermo-train-smoke
```

`kernels push` runs the kernel; add `--no-run` to only save a version. Fused wheels come from
`rezanadimi/thermo-kernel-wheels` through `kernel_sources`; no worker recompiles. The pushed smoke kernel carries the
PyPI-pinned numpy/scipy shim in `thermo_agent/kaggle_notebook.py::setup()` (`numpy==2.4.4`, `scipy==1.17.1`).
Confirm the step-20 evaluation no longer OOMs before any training session.

#### (c) Score every arm on `data/heldout`

```powershell
$env:DATA_DIRS = 'heldout'                         # resolves to the frozen data/heldout set in the worker
$env:ARMS = 'ours_base,ours_sft,ours_harness,ours_sft_grpo_k4,ours_sft_grpo_k8,ours_sft_grpo_k16,frontier_zero,frontier_tools,frontier_harness,other_model'
$env:LIMIT = '0'                                   # 0 = no cap: held-out cells reach 70 rows, so a positive cap truncates and model_eval refuses
$k = '.venv/Scripts/kaggle.exe'
$p = (Resolve-Path 'kaggle/eval_kernel').Path
& $k kernels push -p "$p" --accelerator NvidiaTeslaT4
& $k kernels logs rezanadimi/thermo-eval-run
```

`kaggle/eval_runner.py` writes one `reports/<arm>.json` per arm and assembles
`paper/assets/model_results_manifest.json` / `paper/assets/model_results.json`. Reports are not overwritten unless
`FORCE=1`. The GRPO arms use distinct checkpoint dirs (`runs/grpo_<task>_k4`, `_k8`, `_k16`), so k=4/8/16 never
share weights. `model_eval` scores only the `test_*` splits by default (`DEFAULT_SPLITS`), so train/val files in
`data/heldout` are never scored.

### 0. Hardware

| GPU | Use |
|---|---|
| Kaggle `GPU T4 x2` (2 x 16 GB) | the training target: `scripts/train_dual_t4.sh` (DDP, full fp16 LoRA) |
| RTX PRO 6000 (48 GB) or RTX 4090 (24 GB) | bf16 LoRA, default settings. Add `--vllm` to GRPO if vLLM is installed. |
| Kaggle T4 (16 GB), one GPU | `--dtype float16 --qlora`. T4 has no bf16. |
| Kaggle TPU v5e-8 | not supported by these scripts (PyTorch/TRL on GPU only). |
| RTX 4050 (6 GB) | data generation, tests, non-model baselines and the Kaggle teacher client only. |

Two T4s are not a single bigger GPU: they share PCIe. `scripts/train_dual_t4.sh` launches two DDP ranks with full
fp16 LoRA and no 4-bit weights, detects `NGPU` and the dataloader worker count from the machine, and sets
`NCCL_IB_DISABLE=1`. The reasoning and batch arithmetic are in `docs/RUNBOOK.md`.

### 1. Install

```
unzip thermo-agent.zip && cd thermo-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt && pip install -e ".[dev,train]"
python -m thermo_agent.agent.build_harness --check          # must print nothing and exit 0
pytest -q -x                                                # about 6 minutes on CPU; all tests must pass
```

### 2. Generate data (CPU, no model)

One script builds every dataset, mines the hindsight memory and combines the SFT corpus. It is resumable (a step is
skipped when its `meta_*.json` already records `n >= target`) and takes all sizes and worker counts from the
environment, so the same script covers a smoke run and the paper run:

| Variable | Default | Meaning |
|---|---|---|
| `THERMO_N_REPAIR` | 12000 | repair rows before splitting |
| `THERMO_N_AUDIT` | 3000 | audit plants |
| `THERMO_N_LEDGER` | 9000 | ledger rows |
| `THERMO_N_DESIGN` | 4000 | design sites |
| `THERMO_N_ARCH` | 3000 | architecture repair rows requested (about 70 % survive detection) |
| `THERMO_N_ARCH_AUDIT` | 2000 | architecture audit rows |
| `THERMO_N_ARCH_LEDGER` | 3000 | architecture ledger rows |
| `THERMO_MAX_EVALS` | 60 | reference-design polish budget (never below 60) |
| `THERMO_N_MEMORY` | 1000 | repair rows mined for hindsight notes |
| `THERMO_VARIANTS` | 20 | paraphrases per ground-truth repair row |
| `THERMO_WORKERS` | auto | `min(cores - 2, free_MB / 300)` |
| `THERMO_SEED` | 0 | generation seed |
| `THERMO_SKIP` | empty | comma list of steps to skip (`repair,audit,ledger,design,arch,memory,report,combine`) |

```powershell
pwsh scripts/gen_data.ps1
# smoke version:
$env:THERMO_N_REPAIR='400'; $env:THERMO_N_AUDIT='200'; $env:THERMO_N_LEDGER='400'
$env:THERMO_N_DESIGN='200'; $env:THERMO_N_ARCH='200'; $env:THERMO_N_MEMORY='100'
pwsh scripts/gen_data.ps1
```

Steps, in order: repair -> audit -> ledger -> design -> architecture corpus -> hindsight memory -> data report ->
combine. `reports/data_report.json` and `data/sft/sft_report.json` must both have an empty `problems` list. If a
report names a missing split, raise the matching `THERMO_N_*` value and rerun; the finished steps are skipped.

Measured on this 16-core machine with 14 workers: repair 12000 in about 2 minutes; audit 3000 in about 2 minutes;
ledger 9000 in about 3 minutes; design about 26 CPU-seconds per site (about 2 hours at 4000 sites and 60
evaluations); the architecture corpus 400 rows in 36 seconds (about 5 minutes at 3000). Design writes its split
files only when all sites finish, so a machine sleep during that step loses the whole step; the meta-based resume
simply reruns it.

The architecture corpus (`data/processed_arch/`) holds 65 diagrams: turbine and gas-engine prime movers, absorption
and electric cooling, wet/dry/extended rejection, with and without an ORC bottoming cycle, plus the
`cgam_steam_orc` steam cycle. **Diagram combinations** are held out for `test_ood_topology`: `cgam_steam_orc` and `engine-*-ext*` only. ORC, engines, turbines, district heat all train. Sign-flip / swapped-port faults go to `test_ood_fault`.

### 2b. Frontier teacher on Kaggle

The second half of the training data comes from a frontier model, validated by the same tool suite. Kaggle supplies
the model access (Kaggle Benchmarks with Kaggle's own quota); the physics checks run here, so nothing the teacher
writes enters the corpus unverified. The hand-off is three commands:

```powershell
python -m thermo_agent.teacher.transfer export data/sft data/frontier_export --per-task 400   # prompts, no answers
Compress-Archive -Path data/frontier_export -DestinationPath data/frontier_export.zip -Force
python kaggle/make_notebook.py                                                              # kaggle/teacher.ipynb
# upload the zip as a private Kaggle dataset, import kaggle/teacher.ipynb, Run all, download completions.jsonl
python -m thermo_agent.teacher.transfer ingest data/frontier_export completions.jsonl data/frontier `
    --memory-dir data/memory --model claude-haiku-5-5
```

This is the legacy zero-shot baseline arm (kept for the held-out baseline). The current tool-CoT generation runs in
the LangGraph harness (`kaggle/teacher_harness.py`), with `gpt-6-luna` then `claude-haiku-5-5` (reasoning ON) run as
the hosted task `thermo-teacher-tools-s00` via `pwsh -File scripts/run_teacher_hosted.ps1 -Execute -m <slug>`;
`gemini-3.5-flash-lite` is excluded and must not consume quota.
**REMINDER:** revisit `claude-sonnet-5-5` later — it was deferred as too expensive.

The Kaggle half needs only a browser: the local CLI returns 403 until the account's phone/identity verification is
done, and nothing about the hand-off depends on it. `kaggle/teacher.ipynb` is generated from
`kaggle/teacher_benchmark.py`, and a test fails if the checked-in notebook is stale.

Accepted answers are appended to `data/frontier/frontier_{task}_train.jsonl` with their reward and the verifier's
reasons; rejections become hindsight notes in `data/memory/frontier_notes.jsonl`. Ingest is resumable and never
duplicates a row, and `combine` already reads `data/frontier/`. See `kaggle/README.md`.

### 2c. Build the two corpora and prove they fit

`scripts/build_corpus.ps1` stages `data/sft` (harness on 35% of rows, memory where it retrieves) and
`data/grpo` (**same prompts as SFT**, one per plant, because GRPO samples G=4 completions per prompt and the model
already learned the harness behaviour in SFT), audits both with the real `Qwen/Qwen3.5-4B` tokenizer, verifies both
against the report, and only then swaps them into place. The GRPO trainer keeps only the prompt and the reward
columns, so the reference completion stays in the file for auditing but is never read.

```powershell
pwsh scripts/build_corpus.ps1                 # variants 20
# pwsh scripts/build_corpus.ps1 -Variants 20 -GrpoVariants 1   # overrides (default already correct)
python -m thermo_agent.train.token_audit data/sft --max-length 4096
python -m thermo_agent.synth.verify data/sft  # row counts, ids, splits, token gate
python -m thermo_agent.synth.verify data/grpo --allow-missing-completion
```

Every row is dropped and counted if it does not fit, never truncated. The tokenizer is `$THERMO_TOKENIZER` or the
cached `~/.cache/thermo/qwen3.5-4b/tokenizer.json`; without it the audit falls back to characters and says so.
`synth.verify` is the check that notices a corpus which no longer matches its report (a moved folder, a killed
copy); run it before a training session if anything touched `data/`.

### 3. Non-model baselines (the numbers a model has to beat)

`scripts/baselines.ps1` runs every split with environment-driven limits:
`THERMO_EVAL_REPAIR` (200), `THERMO_EVAL_DESIGN` (30), `THERMO_EVAL_AUDIT` (100), `THERMO_EVAL_LEDGER` (400),
`THERMO_DIAG_GROUPS` (20). Results land in `reports/baseline_*.txt` and `reports/reward_diagnostics.json`.

```powershell
pwsh scripts/baselines.ps1
```

Reference values measured on the ground-truth datasets: Repair, verifier-guided search repairs 83 to 100 percent of
the rows by fault class (58 of 65 efficiency-above-1 rows, all unit slips, wrong fluids, sign and port faults) at 8
to 283 solver calls per row depending on the class; exact restoration of the original plant is rarer, 4 of 65
efficiency-above-1 and 25 of 30 wrong-fluid rows on held-out diagrams. Design, enumerating the defaults takes 38 to
44 evaluations and lands 3.1 to 3.4 percent from the best known design, the polish reaches 0 at 126 to 148 evaluations,
and random search with 12 evaluations is 8.8 to 12.7 percent worse. Ledger, at least one second-law symptom appears in
90 to 100 percent of rows by fault type, closure alone is broken in 32 to 100 percent. Audit, first-law ranking has
nDCG@3 0.864 against 0.65 for random and top-1 0.63 against 0.401 (prime mover excluded); the fixed rule "raise
turbine inlet temperature 10 percent" is valid in 87 of 100 plants and captures 47 percent of the best saving on
average.

### 3b. Will GRPO get a learning signal? (CPU, before any GPU time)

```
python -m thermo_agent.train.reward_diagnostics data/processed data/processed_arch --groups 20 --generations 4 --out reports/reward_diagnostics.json
```

It simulates policies that answer correctly with probability 1 minus the error rate and draws groups of 4, then
reports the reward spread inside groups and the share of groups whose rewards are all equal (those groups teach
nothing, because GRPO normalises by the group's spread). Measured on small samples: with a 5 percent error rate
90 to 95 percent of groups are flat, at 10 percent 60 to 70 percent, at 30 percent 45 to 50 percent, and at 60
percent 0 to 10 percent. Practical reading: once an SFT model answers a task correctly more than about 90 percent of
the time, GRPO on that distribution has little to learn from; train GRPO on the tasks and splits where SFT is clearly
below that, and watch the share of zero-variance groups in the training log (stop or change the prompt mix when it
stays above about 50 percent). Fresh reward evaluation costs about 0.6 ms (ledger), 5 ms (repair), 11 ms (audit) and
23 ms (design) per completion, so rewards will not be the bottleneck next to generation.

### 4. Base model, no training

```
python -m thermo_agent.eval.model_eval data/processed data/processed_arch --model Qwen/Qwen3.5-4B --limit 30 --harness none          --out reports/base_none.json
python -m thermo_agent.eval.model_eval data/processed data/processed_arch --model Qwen/Qwen3.5-4B --limit 30 --harness agents+skills --out reports/base_harness.json
```

Evaluation always reads the ground-truth datasets — `data/processed` plus `data/processed_arch` for the architecture
diagrams — and never `data/sft`: the combined corpus's prompts already carry the harness on 35 percent of rows, so
evaluating there would double-augment and make the `--harness` ablation meaningless. `--limit 30` samples 30 rows
across both directories.

Add `--dtype float16 --load-4bit` on a T4. Check `reports/base_none.completions.jsonl`: are answers in the
requested format? If almost none parse, the model is not following the format (expected for the base model); if the
text starts with a thinking block, tell me.

### 5. Check the chat template before training

```
python -m thermo_agent.train.sft data/sft runs/tmp --tasks repair design audit ledger --print-sample
```

Checked on the authoring machine with the cached `Qwen/Qwen3.5-4B` tokenizer: `PREFIX CHECK` reports
`prefix_matches: true`, and `PROMPT TAIL` ends at `<|im_start|>assistant\n<think>\n\n</think>\n\n` — the generation
prompt, with the empty thinking block, exactly where the completion starts. If a future tokenizer changes that tail,
the check fails loudly here rather than silently at GRPO time.

### 6. SFT (dual T4)

`data/sft_all/` is the current combined corpus: ground-truth paraphrases (20 variants per repair row), validated
teacher traces (including the new tool-using CoT traces), the architecture corpus and retrieved hindsight notes,
with the harness injected on 35 percent of base rows. Files are `{task}_{split}.jsonl` (exactly what the trainers
read) plus `sft_report.json`. `data/grpo_all/` has the same prompts, one per plant; the GRPO loader drops everything
but the prompt and the reward columns. Two gates drop (never truncate) oversized rows: a character pre-filter and
**8,000 prompt tokens** measured with the real tokenizer (prompt + completion + a 96-token reserve for the chat
markers) under `max_length` **8,192**. The legacy `data/sft` / `data/grpo` corpora were built with the superseded
4,000-token gate. `python -m thermo_agent.train.token_audit data/sft_all` re-measures the written files and fails if
any row would be cut; `python -m thermo_agent.synth.verify data/sft_all` compares the files with the report.

Current merged SFT corpus: 119,807 train rows (including 1,140 big-cycle rows), with all 635 accepted frontier rows
present; the legacy 4k-gated `data/sft` had 123,024 rows / 109,659 train.

```bash
# smoke, then the full run (one adapter for all four tasks)
bash scripts/train_dual_t4.sh sft
```

On a single 24 GB GPU the equivalent is `python -m thermo_agent.train.sft data/sft runs/sft --tasks repair design
audit ledger` (add `--dtype float16 --qlora` on one T4). Note the seconds per step from the smoke log and that the
loss goes down. Then evaluate the last `checkpoint-*`:

```
python -m thermo_agent.eval.model_eval data/processed data/processed_arch --model Qwen/Qwen3.5-4B --adapter runs/sft/checkpoint-N --limit 30 --out reports/sft_none.json
python -m thermo_agent.eval.model_eval data/processed data/processed_arch --model Qwen/Qwen3.5-4B --adapter runs/sft/checkpoint-N --limit 30 --harness agents+skills --out reports/sft_harness.json
```

Expect Repair and Audit format and Design validity to be high after SFT. If the format is still poor, the problem is
the template or the data, not GRPO; stop and report.

**Resume**: SFT saves every 200 steps (`save_strategy="steps", save_steps=200`) and auto-resumes from
`latest(out_dir)` via `transformers.trainer_utils.get_last_checkpoint`. One SFT epoch ≈ 3,650 steps ≈ 30 h; plan
2–3 Kaggle sessions (12 h each). Since the local `/kaggle/working` tree is lost between sessions, every save is
also pushed to the single private Hugging Face repo `Reza-Nadimi/thermo-agent-checkpoints` (adapter, optimizer,
scheduler, scaler, RNG and trainer state). A new session pulls the highest checkpoint before the Trainer starts;
`python -m thermo_agent.train.hf_sync status --stage sft` compares local and remote steps. The token comes from
`HF_TOKEN` in `.env` or a Kaggle Secret; the sync refuses any other repo id and any non-private repo. Set
`THERMO_HF_SYNC=off` for a purely local run. `python scripts/smoke_hf_resume.py` exercises the whole round trip on
a tiny CPU model. See `docs/RUNBOOK.md` and `kaggle/README.md` for the notebook path.

### 7. GRPO (one task at a time, starting from the merged SFT model)

```bash
bash scripts/train_dual_t4.sh merge
bash scripts/train_dual_t4.sh grpo repair      # then ledger, audit, design
```

GRPO uses G = 4 with `per_device 4 x accum 2 x 2 ranks` = 16 prompts per step: 8 prompts x 8 rollouts does not fit
the T4 KV cache (about 9 GB of cache on top of the 8 GB of weights). Look in the log for the mean of each reward
function and the reward standard deviation. If the standard deviation is zero for most steps, every sample in a
group scores the same and nothing is learned: tell me which task and the reward means. Ledger is a cheap, exact
reward (pure arithmetic), so start GRPO there if time is short. Evaluate each result:

```
python -m thermo_agent.eval.model_eval data/processed data/processed_arch --model runs/sft_merged --adapter runs/grpo_repair/checkpoint-N --tasks repair --limit 30 --out reports/grpo_repair.json
```

**Resume**: GRPO saves every 50 steps and auto-resumes the same way.

### 8. Ablations and forgetting

```
# harness on/off for base, SFT, GRPO: repeat the model_eval commands with --harness none and --harness agents+skills
# the held-out new component (gas engine) is in test_ood_topology for Repair and Audit
python -m thermo_agent.eval.thermoqa describe                                  # send back the columns, then:
python -m thermo_agent.eval.thermoqa run --question-col <Q> --answer-col <A> --limit 100 --out reports/tqa_base.json
python -m thermo_agent.eval.thermoqa run --question-col <Q> --answer-col <A> --limit 100 --adapter runs/sft/checkpoint-N --out reports/tqa_sft.json
```

Once the Frontier file is downloaded (see `thermo_agent.datasets.frontier.describe`), rank the catalog on the real
trace; `--step-hours` is the sampling interval of the file and has no default on purpose. The result is a lower
bound (ideal part-load, plant re-sized per load bin):

```
python -m thermo_agent.design.trace_eval path/to/frontier.xlsx --columns it_power=<col> t_supply=<col> t_return=<col> --power-unit MW --temp-unit C --chilled-fraction 0.5 --step-hours <hours> --top 8
```

To see what a modular fleet costs on the same trace, add `--spec-json '{...}' --scan-units 8` (one design, 1 to 8
identical units; the penalty is fuel above the ideal part-load bound):

```
python -m thermo_agent.design.trace_eval path/to/frontier.xlsx --columns it_power=<col> t_supply=<col> t_return=<col> --step-hours <hours> \
    --spec-json '{"prime_mover":"engine","cooling":"absorption","rejection":"wet","bottoming":"none","dT_rej":8.0,"T_hw":368.15,"dT_hw":10.0}' --scan-units 8
```

Design sample efficiency (the central claim): run the anytime protocol, in which the model is told the fuel of each
valid design and may improve it, then plot best-result-after-b-evaluations against enumeration, random search and
the enumerate-then-polish reference:

```
python -m thermo_agent.eval.model_eval data/processed --model runs/sft_merged --adapter runs/grpo_design/checkpoint-N --tasks design --refine-turns 6 --limit 30 --out reports/grpo_design_refine.json
python -m thermo_agent.design.curves data/processed --split test_id --limit 20 --report reports/grpo_design_refine.json --out paper/generated/design_anytime.png
```

Paired comparisons for the ablations (same examples, exact McNemar test for binary metrics, permutation test
otherwise, Holm correction across everything listed):

```
python -m thermo_agent.eval.compare reports/base_none.json reports/base_harness.json
python -m thermo_agent.eval.compare reports/sft_none.json reports/grpo_repair.json --task repair
```

### 9. Bundle to send back

```
python -m thermo_agent.eval.summarize reports/*.json > reports/summary.md
python -m thermo_agent.eval.paper_tables reports/*.json --out paper/generated          # LaTeX tables and figures, for later
pip freeze | grep -iE "torch|transformers|peft|trl|bitsandbytes|accelerate|vllm|coolprop" > reports/versions.txt
tar czf bundle.tgz reports runs/*/trainer_state.json runs/*/checkpoint-*/trainer_state.json 2>/dev/null
```
Send `summary.md`, `versions.txt`, the data report, and any failing command with its output.

### 10. Most likely to need a fix (untested on GPU)

1. Loading `Qwen/Qwen3.5-4B` with `AutoModelForCausalLM` (the checkpoint is a vision-language model; if it fails, the image-text class has to be used).
2. TRL arguments written against TRL 1.17.1 source: `SFTConfig(completion_only_loss, max_length, model_init_kwargs)`, `GRPOConfig(reward_weights, chat_template_kwargs, loss_type, mask_truncated_completions, vllm_mode)`; a version mismatch shows up as an unexpected-keyword error.
3. 4-bit loading (`--qlora`, `--load-4bit`) and merging an adapter trained in 4-bit.
4. Greedy generation length (`max_new_tokens` 400 in the evaluator, `max_completion_length` 320 in GRPO): a truncated answer has no closing tag and scores zero.
5. Template: training uses `enable_thinking=True` (single constant `train.render.TRAIN_ON_REASONING`). Measured on the cached `Qwen/Qwen3.5-4B` tokenizer: with `True`, the generation prompt is a literal prefix of the full rendered conversation, so `completion_only_loss` masks exactly the prompt and the stored `<think>…</think>` reasoning stays trainable. With `False` the template injects an empty closed `<think>\n\n</think>\n\n`, the prefix check fails, and masking would train a duplicated `</think>`. `scripts/check_qwen_tool_template.py` re-checks this on any tokenizer.
6. Time per GRPO step on a T4 is unknown; use the smoke run to extrapolate before committing to 300 steps.

### 11. Qwen3.5-4B architecture for LoRA

**Text backbone** (32 layers, hybrid linear/full attention):
- `hidden_size=2560`, `intermediate_size=9216` (SwiGLU), `num_attention_heads=16`, `num_key_value_heads=4` (GQA)
- `head_dim=256`, `partial_rotary_factor=0.25`, `rope_theta=10_000_000`
- **Layer pattern** (every 4th layer is full attention, indices 3,7,11,15,19,23,27,31):
  ```
  linear_attention, linear_attention, linear_attention, full_attention,
  linear_attention, linear_attention, linear_attention, full_attention,
  ...
  ```
- Linear attention layers use **gated delta nets** (SSM-like): `linear_q_proj`, `linear_k_proj`, `linear_v_proj`, `linear_out_proj` with `linear_conv_kernel_dim=4`, `linear_num_key_heads=16`, `linear_num_value_heads=32`, `linear_key_head_dim=128`, `linear_value_head_dim=128`
- Full attention layers: standard `q_proj`, `k_proj`, `v_proj`, `o_proj`
- FFN (all 32 layers): `gate_proj`, `up_proj`, `down_proj`
- MTP (multi-token prediction): 1 extra layer `mtp_num_hidden_layers=1`

**Vision encoder** (24 layers, 1024 hidden, patch_size=16, out=2560):
- `qwen3_5_vision` config, `gelu_pytorch_tanh` act, axial RoPE
- We **do not train vision** — our data is text-only plant JSON.

**LoRA targets** (full fp16, rank 16 SFT / rank 8 GRPO, all-linear on text):
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `linear_q_proj`, `linear_k_proj`, `linear_v_proj`, `linear_out_proj`
- FFN: `gate_proj`, `up_proj`, `down_proj`
- (Skip `patch_embed.proj` and vision attention/FFN — not used)

### 2026-11-19 — exact hosted-pilot and smoke commands

Two paths are about to run. Every command below is copied from a script that exists in this repo. The scripts call
`.venv/Scripts/kaggle.exe` and `.venv/Scripts/python.exe`; bare `kaggle` / `python` are shorthand, not what the
scripts actually invoke. Keep each command on its own line.

#### 1. Environment preflight (once per shell)

All Kaggle traffic from this machine must go through the local SOCKS proxy (`scripts/run_teacher_hosted.ps1`,
`scripts/upload_corpus_dataset.ps1`, `docs/RUNBOOK.md`). The token line is the exact safe assignment used by
`scripts/upload_corpus_dataset.ps1` and `docs/RUNBOOK.md` — value read from `.env`, never printed:
```powershell
$env:HTTP_PROXY = 'socks5h://localhost:10808'
$env:HTTPS_PROXY = 'socks5h://localhost:10808'
$env:PYTHONIOENCODING = 'utf-8'
$env:KAGGLE_API_TOKEN = (& .venv/Scripts/python.exe -c "from pathlib import Path; print([l.split('=',1)[1].strip() for l in Path('.env').read_text().splitlines() if l.startswith('KAGGLE_API_TOKEN=')][0])")
```
`scripts/run_teacher_hosted.ps1` reads the same key through its `Get-Secret 'KAGGLE_API_TOKEN'` helper and also
refuses to start if it is missing.

#### 2. Code bundle (`rezanadimi/thermo-agent-code`)

```
pwsh -File scripts/make_kaggle_bundle.ps1
```
This writes `kaggle_bundle.zip` at the repo root. Stage the fresh zip into the dataset directory, then version it
(the dataset dir already holds `dataset-metadata.json` with id `rezanadimi/thermo-agent-code`):
```
$codeDir = (Resolve-Path 'data/kaggle_datasets/thermo-agent-code').Path
Copy-Item -Force kaggle_bundle.zip (Join-Path $codeDir 'kaggle_bundle.zip')
.venv/Scripts/kaggle.exe datasets version -p "$codeDir" -m "rebuild code bundle for hosted pilot" -t -r skip
```
Confirm `ready`:
```
.venv/Scripts/kaggle.exe datasets status rezanadimi/thermo-agent-code
```
`unverified`: the `Copy-Item` staging line and the `datasets status` command are not used by any script in this
repo. `make_kaggle_bundle.ps1` only writes the root zip; the repo has no staging step. The only existing
confirmation pattern is `kaggle datasets list -s <slug>` (existence only, does not prove `ready`). The `-m` message
text is a free-form version note, not a script-defined flag value.

#### 3. Hosted teacher pilot (`thermo-teacher-tools-s00`, `gpt-6-luna`)

Push both datasets (exact flags from `scripts/run_teacher_hosted.ps1`):
```
.venv/Scripts/kaggle.exe benchmarks tasks push thermo-teacher-tools-s00 -f kaggle/teacher_harness_task_s00.py -d rezanadimi/thermo-agent-code -d rezanadimi/thermo-teacher-prompts-big
```
Run:
```
.venv/Scripts/kaggle.exe benchmarks tasks run thermo-teacher-tools-s00 -m gpt-6-luna --wait 5400
```
Download (`-m` selects the run, `-o` only names the output directory; the downloaded journal filename is
model-independent — `harness_completions.jsonl`, fallback `completions.jsonl`):
```
.venv/Scripts/kaggle.exe benchmarks tasks download thermo-teacher-tools-s00 -m gpt-6-luna -o data/teacher_runs/kaggle_hosted/gpt-6-luna
```
Append-to-shard rule: APPEND the downloaded journal, id-deduped, to this model's own append-only shard
`data/teacher_runs/hosted_gpt-6-luna.jsonl`. Never overwrite another model's journal; each model has exactly one
shard. `scripts/run_teacher_hosted.ps1` performs this append and then ingest. Then run ingest explicitly:
```
.venv/Scripts/python.exe -m thermo_agent.teacher.transfer ingest data/frontier_export data/teacher_runs/hosted_gpt-6-luna.jsonl data/frontier --memory-dir data/memory --model gpt-6-luna
```
In-task hard stop: `SPEC["max_cost_usd"] = 2.0` in `kaggle/teacher_harness_task_s00.py` breaks the task with status
`paused` / `cost_cap` when spend reaches it; the override is the `HARNESS_MAX_COST_USD` environment variable (read
in the task). Deferred-to-ingest verdict rule: the compact prompt shard has no plant/spec columns, so rows are not
gradable inside the task and carry `verdict = {accepted: None, graded: False, reason: "deferred_to_ingest"}`.
The hosted verdict is never proof of acceptance; `transfer.ingest` re-grades against the full export by id and is
the only source of acceptance for those rows.

#### 4. Smoke kernel (`rezanadimi/thermo-train-smoke`)

Generate the notebook (all flags exist in `scripts/make_smoke_kernel.py`; note the default task is `ledger`, so set
`repair` explicitly):
```
python scripts/make_smoke_kernel.py --steps 20 --rows 64 --task repair --ngpu 1 --batch 1 --sft-data sft_tools --grpo-data grpo_tools
```
Push (an absolute path is required on Windows):
```
$smokeDir = (Resolve-Path 'kaggle/train_kernel_smoke').Path
.venv/Scripts/kaggle.exe kernels push -p "$smokeDir" --accelerator NvidiaTeslaT4
```
Status and logs:
```
.venv/Scripts/kaggle.exe kernels status rezanadimi/thermo-train-smoke
.venv/Scripts/kaggle.exe kernels logs rezanadimi/thermo-train-smoke
```
Required success criteria before any training session: setup stages `sft_tools` / `grpo_tools` (from `SFT_DATA` /
`GRPO_DATA`; no missing-corpus error), the training loss falls over the 20 steps, and the step-20 evaluation passes
(no OOM — the original reason for the smoke gate).

#### 5. Stats

```
python scripts/teacher_campaign_stats.py
```
It writes `data/teacher_runs/CAMPAIGN_STATS.json` (machine-readable snapshot) and
`reports/teacher_campaign_log.md` (append-friendly human log; prior snapshots are preserved under History).

### 2026-11-19 — optional-dependency fallback, smoke OOM, planned harness corpora

#### Hosted because of missing optional deps
The benchmarks venv (`/benchmarks/.venv`, Python 3.12) has no scipy and may have no CoolProp; the hosted task must
still run. Exact fallback behaviour today:

- scipy: `thermo_agent/core/_roots.py` uses scipy `brentq` when importable and a pure-Python bracketing Brent
  otherwise; `design/search.py` imports scipy lazily. `using_scipy` records which path is active.
- CoolProp: `core/fluids.py` / `core/props.py` import it optionally. The task tries
  `pip install --target /kaggle/working/pylibs CoolProp==6.8.0`; if that fails (network unavailable) it prints
  `[deps] CoolProp unavailable ...; gas-only plants will run` and continues with gas-only plants — it never fails
  the whole task for a missing real-fluid backend.

One-line check on this box (prints `False` only when scipy is absent; `True` on a normal dev box):

```
python -c "from thermo_agent.core._roots import using_scipy; print(using_scipy)"
```

#### Smoke v26 / v27 (short)
v26 OOM'd at the first backward on one T4 (13.55 GiB in use, 1.10 GiB requested); v27 pushed `sdpa` attention,
non-reentrant gradient checkpointing (SFT + GRPO) and `expandable_segments:True` in `kaggle/train_runner.py`.
**Result pending** — keep the step-20 no-OOM gate in section 4 before any training session.

#### Planned: harness-aligned tool corpora (not run)
Training tool rows still use only the plain `harness="none"` system prompt (740 chars for repair), while eval and
teacher tracegen already separate Z/H prompts. `--prompt-variant {plain,harness_tools}` work is in progress to emit
`*_h` corpora matching `ours_harness`; the gap stays **OPEN** until the code lands. Planned command shape —
**planned, verify against `--help` before running**:

```
python scripts/build_tool_corpus.py --prompt-variant harness_tools --out-harness data/sft_tools_h --grpo-out-harness data/grpo_tools_h
```

## Kaggle without a browser: CLI workflow actually used

*(merged from `docs/RUNBOOK.md` on 2026-11-20; content kept, headings demoted one level)*

`KAGGLE_SKILL.md` describes the Benchmarks surface; this file records the concrete commands that worked on this
account and the mistakes hit along the way. Credentials come from `KAGGLE_API_TOKEN` in `.env`; never print it.

### Credentials and quota

All Kaggle CLI/API traffic from this machine must go through the local SOCKS proxy:

```powershell
$env:HTTP_PROXY = 'socks5h://localhost:10808'
$env:HTTPS_PROXY = 'socks5h://localhost:10808'
# PySocks is installed in .venv; without it, requests raises "Missing dependencies for SOCKS support"
```

```powershell
$env:KAGGLE_API_TOKEN = (& .venv/Scripts/python.exe -c "from pathlib import Path; print([l.split('=',1)[1].strip() for l in Path('.env').read_text().splitlines() if l.startswith('KAGGLE_API_TOKEN=')][0])")
.venv/Scripts/kaggle.exe benchmarks quota --format json
.venv/Scripts/kaggle.exe benchmarks init -y --env-file .env
```

The proxy key lasts about an hour; `thermo_agent.teacher.cli_runner` refreshes it automatically.

### Teacher pass (hosted model, local driver)

```powershell
python -m thermo_agent.teacher.cli_runner probe
python -m thermo_agent.teacher.cli_runner run --model google/gemini-3.1-flash-lite-preview \
    --out data/teacher_runs/completions.jsonl --wait
python -m thermo_agent.teacher.transfer ingest data/frontier_export data/teacher_runs/completions.jsonl \
    data/frontier --memory-dir data/memory --model google/gemini-3.1-flash-lite-preview
```

The runner journals every success and error with token cost (nanodollars from Kaggle's `usage`), checks quota before
each batch, pauses with `--wait`, and mirrors the journal to `teacher/progress.jsonl` in the private HF repo. To
evaluate frontier arms on the *test* rows, first run `scripts/make_eval_export.py`, which exports exactly the rows
`model_eval` samples; answering an export built from the training split produces ids replay cannot find.

### Datasets and kernels

```powershell
pwsh scripts/upload_corpus_dataset.ps1
pwsh -File scripts/make_kaggle_bundle.ps1
# dataset slugs used by the kernel:
#   rezanadimi/thermo-agent-code, rezanadimi/thermo-agent-corpus, rezanadimi/thermo-teacher-prompts
#   rezanadimi/thermo-kernel-wheels (prebuilt fused kernels),
#   rezanadimi/thermo-agent-resume (LoRA resume state),
#   rezanadimi/thermo-agent-eval-replays (saved teacher answers)
$kdir = (Resolve-Path 'kaggle/train_kernel_smoke').Path
.venv/Scripts/kaggle.exe kernels push -p "$kdir" --accelerator NvidiaTeslaT4
.venv/Scripts/kaggle.exe kernels status rezanadimi/thermo-train-smoke
.venv/Scripts/kaggle.exe kernels logs rezanadimi/thermo-train-smoke
```

Gotchas that cost time:

- The metadata file is `dataset-metadata.json` (singular `dataset`), and it needs a `licenses` array.
- `kaggle datasets create` fails on relative `-p` paths on Windows; pass an absolute path.
- `Resolve-Path` returns a `PathInfo` that stringifies to a path but has no `.FullName`; cast to `[string]`.
- `kernels push` **runs** the kernel; use `--no-run` to only save a version.
- The GPU accelerator is selected on the push command (`--accelerator NvidiaTeslaT4`), not in metadata.
- Kernel title determines the slug; keep it identical to the id's last component.
- `PYTHONIOENCODING=utf-8` is required locally when the `kaggle_benchmarks` SDK prints emoji.
- Kaggle Secrets are unavailable in hosted benchmark tasks; use the shared HF repo for credentials and progress.

### Fused kernels: build once, never recompile

Qwen3.5-4B's linear attention uses the slow PyTorch reference kernels unless `fla` and `causal-conv1d` are
installed. Without them a T4 eval forward OOMs at 4096 tokens and a training step takes ~50 s. The exact pinned
environment is `kaggle/environment.lock.json`.

- Build the wheels **once** by pushing `kaggle/wheel_kernel/` (`rezanadimi/thermo-kernel-wheels`, source
  `kaggle/wheel_runner.py`) on a T4. It compiles `causal-conv1d` against Kaggle's own torch and verifies that
  CUDA torch survives.
- **Kaggle itself transfers the wheels**: the build kernel's output is attached to the train, smoke and eval
  kernels through `kernel_sources: ["rezanadimi/thermo-kernel-wheels"]` in their `kernel-metadata.json`. No local
  download or dataset upload is involved.
- `kaggle_notebook.setup()` locates `/kaggle/input/thermo-kernel-wheels/wheels/*.whl`, checks its `manifest.json`
  against the worker's torch/CUDA version, installs with `pip --no-deps`, and imports the kernels. A mismatch or
  missing wheel **fails loudly** — there is no source-build fallback in any worker. The only rebuild trigger is a
  change to Kaggle's torch/CUDA/Python image.

### Cross-session resume without the HF token

Kaggle's Secrets service was unreachable in the training worker, so the token cannot be relied on. Between
sessions, relay the newest checkpoint from the kernel output into a private dataset (no credentials in it):

```powershell
pwsh scripts/resume_kaggle_checkpoint.ps1 -Stage sft -Kernel rezanadimi/thermo-train-run
# ... then start the next session with `kernels push` on kaggle/train_kernel/
```

`restore_checkpoints()` in `thermo_agent/kaggle_notebook.py` copies the newest *complete* checkpoint into
`runs/<stage>/` at the start of a session; the trainer's existing `latest()` then resumes normally. GRPO uses
the locally merged SFT weights when no token is present. No HF token is copied into any notebook, bundle or
dataset.

## Training on two T4s (Kaggle `2x T4`)

*(merged from `docs/RUNBOOK.md` on 2026-11-20; content kept, headings demoted one level)*

Target: Qwen3.5-4B, **full fp16 LoRA** (no 4-bit), **two DDP ranks**, no NVLink. This file explains the
choices and gives the exact commands.

### Why DDP and not FSDP/ZeRO

The two T4s share one host over PCIe. Anything that crosses that link every step is the bottleneck.

| Strategy | What crosses PCIe per step | Verdict |
|---|---|---|
| **DDP (used here)** | LoRA gradients only, ~40–80 MB in fp16 | negligible |
| FSDP / ZeRO-3 | a shard of the 4B weights, ~8 GB | avoid |

DDP keeps a full copy of the frozen base model on each GPU and all-reduces only the adapter gradients.
Gradient accumulation is wrapped by Accelerate's `accumulate()` context (used by the HF Trainer), which calls
`no_sync()` on the micro-steps and communicates once per optimizer step, so the frequency is already minimal.
`gradient_as_bucket_view=True` and `ddp_bucket_cap_mb=64` keep the reducer from copying gradients into a
second buffer.

If NCCL hangs or reports an unsupported peer-to-peer setup, set `NCCL_P2P_DISABLE=1`. Host-staged all-reduce
is slightly slower, but the payload here is tens of MB, so the cost is a few milliseconds per step.

### Why fp16 and not bf16

The T4 is Turing (compute capability 7.5): it has no bf16 kernels. `--dtype bfloat16` is refused at startup by
`thermo_agent.train.ddp.check_dtype_supported`, so a wrong flag fails in the first second instead of after a
model load.

fp16 overflow is handled by the Trainer's AMP path (`fp16=True` -> `GradScaler` with dynamic loss scaling):
a step whose gradients contain inf/nan is skipped and the scale is reduced automatically. On top of that the
train path sets:

- `max_grad_norm=1.0` (`--grad-norm`), clipping before the optimizer step;
- `optim="adamw_torch"`, whose moments stay fp32 (bitsandbytes' fused optimizers are less forgiving in fp16);
- `allow_fp16_reduced_precision_reduction = False`, so matmuls accumulate in fp32;
- `logging_nan_inf_filter=True` (Trainer default), which keeps skipped steps out of the reported loss curve.

If `trainer_state.json` shows a large share of skipped steps, lower the learning rate before changing the dtype.

### Memory budget per T4

| Item | SFT | GRPO |
|---|---|---|
| base weights, fp16 | 8.0 GB | 8.0 GB |
| LoRA params + fp32 optimizer state | ~0.4 GB | ~0.2 GB |
| activations (checkpointing on) | ~2.8 GB at batch 2 × 8192 tokens | ~1 GB at 16 sequences × 320 tokens |
| cross-entropy tile (chunked, fp32) | 0.02 GB (512 × 8192) | — |
| KV cache | — | ~2 GB at 4 prompts × 4 completions × ~900 tokens |
| **total** | **~10 GB of 15 GB** | **~11 GB of 15 GB** |

Both fit with headroom. A 4-bit base is unnecessary and is not used (`--qlora` is refused under DDP).

### Chunked cross-entropy (the T4 fix)

Qwen3.5-4B has a **248,320-token** vocabulary, not 152k. Asking the model for `labels=` builds the whole
`(batch × sequence, 248320)` logits tensor: at batch 2 × 4096 that is **4.0 GiB in fp16**, and the fp32
`CrossEntropyLoss` upcast doubles it to **8.1 GiB**. Add 8 GiB of fp16 weights and the first attempt died with
13.61 GiB allocated and 126 MiB short.

`thermo_agent/train/chunked_loss.py` computes the same negative log-likelihood on `token_chunk × vocab_chunk`
tiles (512 × 8192 by default), folds each tile into an fp32 online log-sum-exp, and recomputes the tile in the
backward pass instead of storing it. Peak cross-entropy memory becomes one 16 MiB tile regardless of the
vocabulary size. No architecture, weight, label or objective changes: the tests compare the chunked loss and its
gradients against `F.cross_entropy` on the full logits.

```bash
--loss-token-chunk 512 --loss-vocab-chunk 8192   # defaults; smaller tiles use less memory, more loops
```

The `[loss]` line in the log states the tile size actually used. `[memory]` states the loaded weight dtype and
the GiB already allocated, so a silent fp32 load is visible before the first step.

### Batch arithmetic

`effective = per_device × grad_accum × ranks`.

| Stage | per device | accum | ranks | effective |
|---|---|---|---|---|
| SFT | 2 | 8 | 2 | 32 sequences |
| GRPO | 4 | 2 | 2 | 16 prompts, 4 completions each |

GRPO groups completions per prompt, so two divisibility readings matter across TRL versions: the per-device
prompt batch and the effective batch must both be multiples of `num_generations` (4). `--per-device-batch 4
--grad-accum 2` satisfies both; the trainer checks the effective batch before loading the model.

The group size is 4 rather than 8 because of the KV cache, not the weights. A 4B model with grouped-query
attention needs roughly 0.15 MB of KV per token; 8 prompts x 8 completions x (600-token prompt + 320 generated)
is about 9 GB of cache on top of the 8 GB of fp16 weights, which does not fit a 15 GB T4. At G=4 the same
arithmetic gives about 2 GB, leaving room for activations. If a future runtime has more memory, raise G and
keep the two divisibility conditions.

Memory for GRPO is dominated by the rollout: 8 prompts × 8 completions = 64 sequences of up to 320 tokens per
rank, which is why the KV cache is ~3 GB there. That is the line item to watch on a T4.

### Commands

```bash
# one-off preflight + SFT on both GPUs
scripts/train_dual_t4.sh sft

# merge, then GRPO on the merged SFT model, one task at a time
scripts/train_dual_t4.sh merge runs/sft/checkpoint-N runs/sft_merged
scripts/train_dual_t4.sh grpo ledger
scripts/train_dual_t4.sh grpo repair
```

Equivalent raw form:

```bash
torchrun --standalone --nproc_per_node=2 --module thermo_agent.train.sft \
    data/processed runs/sft --dtype float16 --per-device-batch 2 --grad-accum 8
```

Under DDP every rank prints a topology line at startup:

```
[rank 0/2] {"cuda": true, "world_size": 2, "gpus": [{"name": "Tesla T4", "capability": "7.5", ...}], ...}
```

If only rank 0 appears, or `nvidia-smi` shows one busy GPU, the launch did not distribute.

### What DDP does and does not parallelise

- **SFT**: shards the dataset across ranks; gradients are averaged each optimizer step. Wall time per step
  is roughly halved.
- **GRPO**: each rank draws its own prompts and generates their groups itself; the advantage computation
  stays within a rank, so no extra communication is introduced during rollouts. Generation dominates the
  step time on a T4, and that is the part that splits cleanly.
- **Not supported under DDP**: `--vllm` (colocate vLLM is a single-process mode; the script refuses it when
  `WORLD_SIZE > 1`).

### Resume across Kaggle sessions (private Hugging Face repo)

One SFT epoch is longer than a Kaggle session. Every Trainer save therefore pushes the **full resume state** —
LoRA adapter, optimizer moments, scheduler, GradScaler, RNG and `trainer_state.json` — to one private repo,
`Reza-Nadimi/thermo-agent-checkpoints`. The merged SFT model is pushed under `merged/sft` so GRPO can start in a
fresh session with `--init hf:merged/sft`.

- The token is read from `HF_TOKEN` (environment or `.env`); the Kaggle notebook reads a Kaggle Secret. Nothing
  is written to a notebook or a command line.
- `hf_sync` refuses any repo other than `<token owner>/thermo-agent-checkpoints` and refuses a repo that is not
  private, so no other Hugging Face repo is touched. Create it once with
  `python -m thermo_agent.train.hf_sync init`.
- `sft.py`, `grpo.py` and `merge.py` sync by default when a token is present (`--hf-sync auto`); `--hf-sync off`
  keeps a run local. A restarted session pulls the highest `checkpoint-N` before the Trainer is constructed, so
  the existing `latest(out)` logic resumes from it. Under DDP only rank 0 pulls and pushes.
- Inspect the state with `python -m thermo_agent.train.hf_sync status --stage sft` (also `push`, `pull`,
  `push-merged`, `pull-merged`). Stage names are `sft`, `grpo_repair`, `grpo_design`, `grpo_audit`, `grpo_ledger`.
- `python scripts/smoke_hf_resume.py` trains a tiny CPU model for two steps, pushes it to the private repo,
  restores it and resumes for one more step. It was run successfully before the first T4 session.
- Kaggle: import `kaggle/train.ipynb` (generated from `kaggle/train_runner.py`), attach the code and corpus
  datasets described there, add the `HF_TOKEN` secret, and run one stage per cell. Re-running a cell resumes.

### Expectations on T4

T4 fp16 is ~65 TFLOPS peak with a narrow memory bus. SFT on 4B with 3072-token sequences will be slow
(hours, not minutes) but the batch math above is conservative enough to run without OOM. Measure seconds per
step in the first 20 steps with `--max-steps 20` before committing to a full run.

## Running the model locally (LM Studio / any OpenAI-compatible server)

*(merged from `docs/RUNBOOK.md` on 2026-11-20; content kept, headings demoted one level)*

The evaluator does not require local weights. `--backend openai` sends the same messages the HF path would send to any
OpenAI-compatible chat endpoint: LM Studio, vLLM, llama.cpp server, Ollama.

### Start LM Studio

In LM Studio: **Developer** tab → select `qwen3.5-4b` → **Start Server** (default `http://127.0.0.1:1234/v1`).
Check it:

```powershell
Invoke-RestMethod http://127.0.0.1:1234/v1/models | Select-Object -ExpandProperty data | Select-Object id
```

### Evaluate

```bash
.venv/Scripts/python.exe -m thermo_agent.eval.model_eval data/processed \
    --backend openai --api-model qwen3.5-4b \
    --tasks ledger --limit 20 --out reports/local_ledger.json
```

Swap `--tasks` for `repair design audit ledger`. `--tasks design --refine-turns 6` runs the anytime protocol.

### Behaviour

- Greedy by default (`temperature 0`), matching the greedy evaluation protocol used for the paper.
- Reasoning is disabled through `chat_template_kwargs: {"enable_thinking": false}`. Pass `--thinking` to keep it on;
  a `<think>...</think>` block is stripped from the answer either way.
- `--max-new-tokens` defaults to 400, the same budget the HF path uses.
- Requests use `urllib` only, so the evaluation machine needs no extra package.
- `POST /v1/chat/completions` is retried 3 times with a short backoff, which absorbs LM Studio's first-call model load.

### Cost note

A 4B model at Q4 in 6 GB of VRAM generates roughly 20-40 tokens/s. With ~2k-token prompts and a 400-token budget,
`--limit 20` on the ledger task is minutes, not hours. Run small limits first, then scale.

### Reporting

`meta.backend` and `meta.model` record how a run was produced, so reports from this path are not confused with
`transformers` runs on a GPU box.
