# Runbook

Goal: generate data, check it, measure the base model, run SFT then GRPO on Qwen3.5-4B, and report back a small bundle of files.
Everything under "Generate data" and every test runs on CPU. The GPU steps (evaluation, SFT, GRPO, merge) target
Kaggle's dual T4 through `scripts/train_dual_t4.sh`; Section 10 lists what is most likely to need a fix. Run the
steps in order and stop at the first failure; send back the failing command and its output.

## 0. Hardware

| GPU | Use |
|---|---|
| Kaggle `GPU T4 x2` (2 x 16 GB) | the training target: `scripts/train_dual_t4.sh` (DDP, full fp16 LoRA) |
| RTX PRO 6000 (48 GB) or RTX 4090 (24 GB) | bf16 LoRA, default settings. Add `--vllm` to GRPO if vLLM is installed. |
| Kaggle T4 (16 GB), one GPU | `--dtype float16 --qlora`. T4 has no bf16. |
| Kaggle TPU v5e-8 | not supported by these scripts (PyTorch/TRL on GPU only). |
| RTX 4050 (6 GB) | data generation, tests, non-model baselines and the Kaggle teacher client only. |

Two T4s are not a single bigger GPU: they share PCIe. `scripts/train_dual_t4.sh` launches two DDP ranks with full
fp16 LoRA and no 4-bit weights, detects `NGPU` and the dataloader worker count from the machine, and sets
`NCCL_IB_DISABLE=1`. The reasoning and batch arithmetic are in `docs/DUAL_T4.md`.

## 1. Install

```
unzip thermo-agent.zip && cd thermo-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt && pip install -e ".[dev,train]"
python -m thermo_agent.agent.build_harness --check          # must print nothing and exit 0
pytest -q -x                                                # about 6 minutes on CPU; all tests must pass
```

## 2. Generate data (CPU, no model)

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

## 2b. Frontier teacher on Kaggle

The second half of the training data comes from a frontier model, validated by the same tool suite. Kaggle supplies
the model access (Kaggle Benchmarks with Kaggle's own quota); the physics checks run here, so nothing the teacher
writes enters the corpus unverified. The hand-off is three commands:

```powershell
python -m thermo_agent.teacher.transfer export data/sft data/frontier_export --per-task 400   # prompts, no answers
Compress-Archive -Path data/frontier_export -DestinationPath data/frontier_export.zip -Force
python kaggle/make_notebook.py                                                              # kaggle/teacher.ipynb
# upload the zip as a private Kaggle dataset, import kaggle/teacher.ipynb, Run all, download completions.jsonl
python -m thermo_agent.teacher.transfer ingest data/frontier_export completions.jsonl data/frontier `
    --memory-dir data/memory --model anthropic/claude-sonnet-5-5@default
```

The Kaggle half needs only a browser: the local CLI returns 403 until the account's phone/identity verification is
done, and nothing about the hand-off depends on it. `kaggle/teacher.ipynb` is generated from
`kaggle/teacher_benchmark.py`, and a test fails if the checked-in notebook is stale.

Accepted answers are appended to `data/frontier/frontier_{task}_train.jsonl` with their reward and the verifier's
reasons; rejections become hindsight notes in `data/memory/frontier_notes.jsonl`. Ingest is resumable and never
duplicates a row, and `combine` already reads `data/frontier/`. See `kaggle/README.md`.

## 2c. Build the two corpora and prove they fit

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

## 3. Non-model baselines (the numbers a model has to beat)

`scripts/baselines.ps1` runs every split with environment-driven limits:
`THERMO_EVAL_REPAIR` (200), `THERMO_EVAL_DESIGN` (30), `THERMO_EVAL_AUDIT` (100), `THERMO_EVAL_LEDGER` (400),
`THERMO_DIAG_GROUPS` (20). Results land in `reports/baseline_*.txt` and `reports/reward_diagnostics.json`.

```powershell
pwsh scripts/baselines.ps1
```

Reference values measured on small samples: Repair, verifier-guided search repairs 97 to 100 percent of the
numeric faults and all of the held-out sign and port faults, but at 114 to 124 solver calls per row (the 60-evaluation
budget leaves 1 to 3 unsolved rows in each of several classes); exact restoration of the original plant is rarer
(35 to 49 of 58 to 77 in-distribution rows). Design, enumerating the defaults takes 38 to 44 evaluations and lands
3.1 to 3.4 percent from the best known design, the polish reaches 0 at 126 to 148 evaluations, and random search with
12 evaluations is 8.8 to 12.7 percent worse. Ledger, at least one second-law symptom appears in 90 to 100 percent of
rows by fault type, closure alone is broken in 32 to 100 percent. Audit, first-law ranking has nDCG@3 0.864 against
0.65 for random and top-1 0.63 against 0.401 (prime mover excluded); the fixed rule "raise turbine inlet temperature
10 percent" is valid in 87 of 100 plants and captures 47 percent of the best saving on average.

## 3b. Will GRPO get a learning signal? (CPU, before any GPU time)

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

## 4. Base model, no training

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

## 5. Check the chat template before training

```
python -m thermo_agent.train.sft data/sft runs/tmp --tasks repair design audit ledger --print-sample
```

Checked on the authoring machine with the cached `Qwen/Qwen3.5-4B` tokenizer: `PREFIX CHECK` reports
`prefix_matches: true`, and `PROMPT TAIL` ends at `<|im_start|>assistant\n<think>\n\n</think>\n\n` — the generation
prompt, with the empty thinking block, exactly where the completion starts. If a future tokenizer changes that tail,
the check fails loudly here rather than silently at GRPO time.

## 6. SFT (dual T4)

`data/sft/` is the combined corpus: ground-truth paraphrases (20 variants per repair row), validated teacher traces,
the architecture corpus and retrieved hindsight notes, with the harness injected on 35 percent of rows. Files are
`{task}_{split}.jsonl` (exactly what the trainers read) plus `sft_report.json`. `data/grpo/` has the same prompts,
one per plant; the GRPO loader drops everything but the prompt and the reward columns. Two gates drop (never
truncate) oversized rows: 11500 characters, and 4000 tokens measured with the real tokenizer (prompt + completion + a
96-token reserve for the chat markers) under `max_length` 4096. `python -m thermo_agent.train.token_audit data/sft`
re-measures the written files and fails if any row would be cut; `python -m thermo_agent.synth.verify data/sft`
compares the files with the report.

Current SFT corpus: 129,684 rows, 116,809 in train (repair 106,420, ledger 5,602, audit 3,019, design 1,768).

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
2–3 Kaggle sessions (12 h each).

## 7. GRPO (one task at a time, starting from the merged SFT model)

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

## 8. Ablations and forgetting

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

## 9. Bundle to send back

```
python -m thermo_agent.eval.summarize reports/*.json > reports/summary.md
python -m thermo_agent.eval.paper_tables reports/*.json --out paper/generated          # LaTeX tables and figures, for later
pip freeze | grep -iE "torch|transformers|peft|trl|bitsandbytes|accelerate|vllm|coolprop" > reports/versions.txt
tar czf bundle.tgz reports runs/*/trainer_state.json runs/*/checkpoint-*/trainer_state.json 2>/dev/null
```
Send `summary.md`, `versions.txt`, the data report, and any failing command with its output.

## 10. Most likely to need a fix (untested on GPU)

1. Loading `Qwen/Qwen3.5-4B` with `AutoModelForCausalLM` (the checkpoint is a vision-language model; if it fails, the image-text class has to be used).
2. TRL arguments written against TRL 1.17.1 source: `SFTConfig(completion_only_loss, max_length, model_init_kwargs)`, `GRPOConfig(reward_weights, chat_template_kwargs, loss_type, mask_truncated_completions, vllm_mode)`; a version mismatch shows up as an unexpected-keyword error.
3. 4-bit loading (`--qlora`, `--load-4bit`) and merging an adapter trained in 4-bit.
4. Greedy generation length (`max_new_tokens` 400 in the evaluator, `max_completion_length` 320 in GRPO): a truncated answer has no closing tag and scores zero.
5. Template: `enable_thinking=False` is passed to `apply_chat_template`; if the template ignores it, SFT, GRPO and evaluation still agree with each other, but a thinking block may appear in answers (the evaluator strips ` thinking...`).
6. Time per GRPO step on a T4 is unknown; use the smoke run to extrapolate before committing to 300 steps.

## 11. Qwen3.5-4B architecture for LoRA

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