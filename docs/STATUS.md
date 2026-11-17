# Status and operating notes

Working copy: `C:\Users\rezax\Home\Code\thermo_agent\Repo` on REZA_PC (git, branch `main`). No SSH, no WSL: every
step below runs in this checkout.

## Environment

| Item | Value |
|---|---|
| Machine | REZA_PC, Windows 11 Home, 16 cores, 15.2 GB RAM, RTX 4050 6 GB |
| Python | 3.12.3 in `.venv` (created with `uv`), pinned by `requirements-lock.txt` |
| Package | installed editable (`pip install -e . --no-deps`) |
| Tools | `uv`, `ripgrep`, `fd`, `jq` in `%USERPROFILE%\tools\bin` and `%USERPROFILE%\.local\bin` |
| Local model | LM Studio serving `qwen3.5-4b` on `http://127.0.0.1:1234/v1` (format smoke checks only) |
| Tokenizer | `Qwen/Qwen3.5-4B` `tokenizer.json` cached in `~\.cache\thermo\qwen3.5-4b\` (248,070 tokens) |
| Training target | Kaggle `GPU T4 x2` via `scripts/train_dual_t4.sh` |
| Frontier teacher | Kaggle Benchmarks (`kaggle_benchmarks`) via `kaggle/teacher_benchmark.py` |

No WSL and no tmux: long jobs are detached Windows processes (`Win32_Process.Create`), so they survive an SSH
disconnect. `Start-Process` children do not. A machine sleep still kills them.

## Running long jobs

```powershell
pwsh scripts/long.ps1 start gen-data '& ./scripts/gen_data.ps1'   # start detached
pwsh scripts/long.ps1 status gen-data -Lines 40                    # state + log tail
pwsh scripts/long.ps1 list                                        # all jobs
pwsh scripts/long.ps1 kill gen-data
```

Logs: `logs/<name>.log`; the job writes `logs/<name>.pid`, which `status` uses to tell RUNNING from FINISHED.
Never give a job the same name as the script it runs.

## Data pipeline

The corpus has four sources, all deterministic or verifier-grounded (no free-form CoT labels):

1. **Ground-truth paraphrases** (`thermo_agent/synth/paraphrase.py`): every ground-truth repair row is re-emitted
   with `--variants 20` truthful phrasings built from the fault facts. A row picks its variants by a hash of its id,
   so the corpus as a whole covers the phrasing space, not one fixed wording: the current train split has
   **106,420 repair rows** (5,321 base plants × 20). The pool is 20 to 80 renderings per fault class
   (`unit_slip` 20, `efficiency_gt_1` 25, `wrong_fluid` 80), so 20 is always available and exhausts the sparsest
   class. Paraphrases are a train-split device only; the eval splits keep one row per plant so a model evaluation
   never counts the same plant twice.
2. **Architecture corpus** (`thermo_agent/synth/architectures.py`): 65 plant diagrams (turbine/engine prime mover,
   absorption/electric cooling, wet/dry/extended rejection, with/without ORC, plus `cgam_steam_orc`), each with ORC
   pressures chosen per working fluid. **Diagram combinations** are held out for the OOD splits: `cgam_steam_orc` and `engine-*-ext*` go to `test_ood_topology`. ORC, engines, turbines, district heat all train.
3. **Frontier teacher traces** (`thermo_agent/teacher/transfer.py` + `validate.py`): prompts are exported without
   their answers, answered on Kaggle Benchmarks, then every answer is re-run through the tool suite here. Only
   accepted answers become rows; rejections become hindsight memory.
4. **Hindsight failure memory** (`thermo_agent/teacher/memory.py`): lessons mined from rejected attempts are
   retrieved per prompt (k=3) and injected as "known failure modes". Only repair has notes today (24 from the
   adversarial miner), so only repair rows carry a memory block.

`scripts/build_corpus.ps1` stages, audits, verifies and swaps two corpora:

- `data/sft/{task}_{split}.jsonl` — harness on 35 percent of rows, memory where it retrieves;
- `data/grpo/{task}_{split}.jsonl` — **same prompts as SFT** (with harness/memory, `-GrpoVariants 1`), so the model sees at GRPO time exactly the input format it learned during SFT. The trainer (`train/grpo.py:load_rows`) keeps only the prompt and the reward columns; the reference completion stays in the file for auditing but is never read. GRPO samples G=4 completions per prompt, so 20 paraphrases of one plant would only repeat the same prompt.

### Token budget (the gate that matches the trainer)

Every candidate row is measured with the real Qwen3.5 tokenizer before it is written; rows that do not fit are
dropped and counted, never truncated. Two gates: 11,500 characters (cheap pre-filter) and 4,000 tokens
(prompt + completion + a 96-token reserve for the chat markers) under `max_length` 4096.

Measured harness cost per task (200 train rows, tokens, p50 / max):

| task | none | agents | skills | agents+skills | mode used |
|---|---|---|---|---|---|
| repair | 964 / 1129 | 2295 / 2460 | 1360 / 2630 | 2689 / 3959 | agents+skills |
| audit | 1362 / 2202 | 2692 / 3532 | 2854 / 4803 | 4183 / 6132 | skills |
| ledger | 1748 / 2983 | 3078 / 4313 | 2328 / 3563 | 3657 / 4892 | agents+skills |
| design | 502 / 519 | 1832 / 1849 | 944 / 961 | 2273 / 2290 | agents+skills |

Audit is the outlier: an audit plant can carry many components, so AGENTS.md plus its skills does not fit; audit
therefore trains with the skill block only. Ledger is the task whose harness rows hit the token ceiling: 985 ledger
candidates were dropped by the token gate (all with the agents+skills harness), and 28,808 repair candidates by the
character pre-filter.

### Current corpus (variants 20, harness 0.35, memory 3)

| corpus | rows | train rows | max row | rows over 4096 |
|---|---|---|---|---|
| `data/sft` | 129,684 | 116,809 | 4,000 tokens | 0 |
| `data/grpo` | 28,585 | 15,710 | 4,000 tokens | 0 |

Train split by task (SFT): repair 106,420 (18,920 harness, memory on every repair row, 49 architectures), ledger
5,602 (1,523 harness), audit 3,019 (892 harness), design 1,768 (635 harness). `sft_report.json` reports
`problems: []` for both corpora, and `reports/token_audit_{sft,grpo}.json` report `fits: true`. After the swap,
`python -m thermo_agent.synth.verify data/sft` re-reads every written file, compares the row counts with the report,
and rejects a duplicate id or a file the report does not know about; `build_corpus.ps1` runs it on the staged and
on the swapped corpus.

At 32 sequences per optimizer step (2 per device x 8 accumulation x 2 ranks) one SFT epoch is ~3,650 steps. A 4B fp16
LoRA step at 1.5k average tokens costs roughly 30 s on two T4s, so one epoch is about 30 hours: plan resume across
2–3 Kaggle sessions (12 h each). Checkpoints every 200 steps via `save_strategy="steps"`.

## Baselines (small-sample reference values)

`scripts/baselines.ps1` over `data/processed` + `data/processed_arch`: 65–106 repair rows per fault and split,
30 design sites per OOD split, 100 audit plants, 400 ledger rows.

- repair, verifier-guided search with a 60-evaluation budget (mean solver calls in brackets): `test_id` repairs
  63/65 `efficiency_gt_1` (123), 57/58 `unit_slip` (124) and 74/77 `wrong_fluid` (114); `test_ood_fault` repairs all
  106 `sign_flip` and all 94 `swapped_ports` (116); `test_ood_topology` repairs 40/41 `efficiency_gt_1`, 47/48
  `sign_flip`, 37/38 `swapped_ports`, 42/43 `unit_slip` and 29/30 `wrong_fluid`. Exact restoration of the original
  plant is rarer: 35–49 of 58–77 on `test_id`, 25–37 of 30–48 on held-out diagrams.
- design: enumerating the defaults lands 3.1–3.4 % from the best known design at 38–44 evaluations; the polish
  reaches 0.0 % at 126–148; random search with 12 evaluations is 8.8–12.7 % worse and invalid in 1–6 of 30 sites.
- audit, `test_ood_topology` (n=100): first-law nDCG@3 0.864 vs 0.65 for random, top-1 0.63 vs 0.401; the fixed rule
  (raise turbine inlet temperature 10 %) is valid in 87/100 and captures a mean 47.1 % of the best saving; change
  search needs 11.6 evaluations vs 1 for a model.
- ledger, `test_ood_fault` (n=400): no-op solves none, the regenerate rule solves all; at least one second-law
  symptom appears in 90–100 % of rows by fault type (closure residual or negative destruction), while closure alone
  is broken in 32–100 %.

## Qwen3.5-4B architecture (for LoRA)

`Qwen3_5ForConditionalGeneration` with separate text + vision configs:

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

## Training path (dual T4)

- `scripts/train_dual_t4.sh` defaults to `DATA=data/sft` for SFT and `GRPO_DATA=data/grpo` for GRPO, detects
  `NGPU` and the dataloader worker count, and disables the InfiniBand probe. `STEPS` / `ROWS` add smoke limits.
- two DDP ranks via `torchrun --nproc_per_node=$NGPU`;
- **full fp16 LoRA**, no 4-bit weights: the T4 is Turing and has no bf16 kernels, so `--dtype float16`, guarded by
  `thermo_agent/train/ddp.py`;
- DDP replicates the frozen 4B model; only LoRA gradients cross PCIe, once per optimizer step (`no_sync` on
  accumulation micro-steps, `gradient_as_bucket_view`, `ddp_bucket_cap_mb=64`);
- overflow handling: AMP `GradScaler`, `max_grad_norm=1.0`, `adamw_torch`, fp32 matmul reductions;
- batch arithmetic: SFT 2 x 8 x 2 = 32; GRPO G = 4 with 4 x 2 x 2 = 16 prompts per step. TRL requires the
  per-device batch to be a multiple of G, so 4 is the smallest legal choice;
- **Resume**: both trainers save every 200 steps (`save_strategy="steps", save_steps=200`, GRPO every 50) and
  auto-resume from `latest(out_dir)` via `transformers.trainer_utils.get_last_checkpoint`. One SFT epoch ≈ 3,650
  steps ≈ 30 h.
- SFT `max_length` 4096; rows are gated at 4,000 tokens when the corpus is built, so nothing is truncated.
- The chat template was checked against the real tokenizer: `prefix_matches: true`, and the prompt ends at
  `<|im_start|>assistant<think></think>`, so SFT, GRPO and evaluation render the same generation prompt.
- GRPO uses the **same prompts** as SFT (with harness/memory); its loader keeps only the prompt and the reward
  columns.

Trainings have not been run yet: the 4050 cannot hold a 4B model for training, and the Kaggle run is the next step.

## Tests

- Full CPU suite: **320 passed** in 5.0 minutes. `paper/methods_draft.md` exists, so its constant checks run instead
  of skipping at module level.
- New this round: `tests/test_verify.py` (the corpus verifier must notice a missing file, a row-count mismatch, a
  leak across splits and a row over the token gate), plus the earlier `test_token_audit.py`, `test_transfer.py`,
  `test_checkpoint.py`, `test_splits.py` and the per-task harness, train-only paraphrase and token-gate cases in
  `test_combine.py`.
- Run `pytest -q` before any commit that touches `thermo_agent/`; `paper/tools/verify_paper.py` checks that every
  `\val{...}` used in the sections exists in the generated assets (the LaTeX build itself needs a TeX installation).

## Kaggle (teacher pass)

`kaggle/README.md` has the three-step hand-off: export the prompts and zip them here, upload the zip as a private
dataset and import `kaggle/teacher.ipynb` (built by `kaggle/make_notebook.py` from `kaggle/teacher_benchmark.py`),
then ingest and validate here. The Kaggle half needs only a browser, so the local CLI's 403 (account verification
pending) does not block it. Nothing in the notebook decides what is true: `teacher.validate` does, and rejections
land in `data/memory/frontier_notes.jsonl`.

## Next

1. Run the Kaggle teacher pass: `data/frontier_export.zip` (1,600 prompts, 1.8 MB) is exported from the current
   corpus. Upload it as a private dataset, import `kaggle/teacher.ipynb`, download `completions.jsonl`, ingest.
   This is the only step that needs the user (a browser and a Kaggle session).
2. Smoke-test SFT/GRPO on Kaggle (`STEPS=20 ROWS=64 bash scripts/train_dual_t4.sh sft`, then `grpo ledger`).
3. The method section now carries the corpus sizes, the 20-variant policy and the held-out combinations; what is
   still missing is the teacher acceptance table, the model results and the trace/fleet section.