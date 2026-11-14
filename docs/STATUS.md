# Status and operating notes

Working copy on the second laptop: `C:\Users\rezax\Home\Code\thermo_agent\Repo` (git, branch `main`).

## Environment

| Item | Value |
|---|---|
| Machine | REZA_PC, Windows 11 Home, 16 cores, 15.2 GB RAM, RTX 4050 6 GB |
| Python | 3.12.3 in `.venv` (created with `uv`), pinned by `requirements-lock.txt` |
| Package | installed editable (`pip install -e . --no-deps`) |
| Tools | `uv`, `ripgrep`, `fd`, `jq` in `%USERPROFILE%\tools\bin` and `%USERPROFILE%\.local\bin` |
| Local model | LM Studio serving `qwen3.5-4b` on `http://127.0.0.1:1234/v1` (smoke checks only, never training data) |
| Remote access | SSH key auth, host alias `reza-pc` |
| Training target | Kaggle `GPU T4 x2` via `scripts/train_dual_t4.sh` |
| Teacher | Kaggle vLLM + cloudflared, `kaggle/README.md` |

No WSL and no tmux: long jobs are detached Windows processes (`Win32_Process.Create`), so they survive an SSH
disconnect. `Start-Process` children do not.

## Running long jobs

```powershell
pwsh scripts/long.ps1 start gen-data '& ./scripts/gen_data.ps1'   # start detached
pwsh scripts/long.ps1 status gen-data -Lines 40                    # state + log tail
pwsh scripts/long.ps1 list                                        # all jobs
pwsh scripts/long.ps1 kill gen-data
```

Logs: `logs/<name>.log`; the job writes `logs/<name>.pid`, which `status` uses to tell RUNNING from FINISHED.
Never give a job the same name as the script it runs.

## Pipeline

The corpus has four sources, all deterministic or verifier-grounded (no free-form CoT labels):

1. **Ground-truth paraphrases** (`thermo_agent/synth/paraphrase.py`): every ground-truth repair row is re-emitted
   with `--variants 4` truthful phrasings built from the fault facts, so the model cannot memorise one wording.
2. **Architecture corpus** (`thermo_agent/synth/architectures.py`): 65 plant diagrams (turbine/engine prime mover,
   absorption/electric cooling, wet/dry/extended rejection, with/without ORC, plus `cgam_steam_orc`), each with
   ORC pressures chosen per working fluid. Whole diagrams are held out for the OOD splits.
3. **Frontier teacher traces** (`thermo_agent/teacher/sample.py` + `validate.py`): a frontier open model answers
   the same prompts; every attempt is stored raw, and only answers that pass the tool suite (first law, second law,
   component-level feasibility) become SFT rows. Rejections are kept.
4. **Hindsight failure memory** (`thermo_agent/teacher/memory.py`): lessons mined from rejected attempts are
   retrieved per prompt (k=3) and injected as "known failure modes" so the model can re-check its own answer.

`thermo_agent/synth/combine.py` merges all four into `data/sft/sft_{task}_{split}.jsonl`, injects the harness on
35 percent of rows, and writes `sft_report.json` (dropped prompts, source counts, expert-reward check).

## Pipeline state

`scripts/gen_data.ps1` is env-driven and resumable. Current full run (`gen-data2`, 14 auto-selected workers):

- repair n=12000: done in about 2 minutes; splits train 3774 / test_ood_fault 3204 / val 463 / test_id 444 /
  test_ood_topology 3925.
- audit n=3000: running.
- ledger n=9000: queued (about 84 minutes at 7.8 CPU-s per row).
- design n=4000, max-evals 60: queued (about 2 hours at 26 CPU-s per site).
- architecture n=3000 requested: queued (about 70 percent survive detection; 400 rows took 36 s).
- memory: mine 1000 repair rows into `data/memory/`.
- data report + combine (variants 4, harness 0.35).

Earlier small run (kept for reference): repair 4000, audit 800, ledger 3000, design 1500; baselines in `reports/`.

## Baselines (small-sample reference values)

- repair, `test_id` and `test_ood_fault`: verifier-guided search repairs every row; 8 solver calls for sign faults,
  56 for unit slips, 245 to 290 for swapped ports and wrong fluid.
- audit, `test_ood_topology`: first-law top-1 15 % vs 32.8 % random; combined rule valid in 28/40 and captures a
  mean 10.7 % of the best saving; change search 11.9 evaluations vs 1 for a model.
- ledger, `test_ood_fault`: no-op solves none; the regenerate rule solves all; at least one second-law symptom in
  87 to 100 % of rows by fault type, closure broken in 26 to 100 %.
- design: pending the new split; `scripts/baselines.ps1` runs the OOD climate/load/scale splits.

## Training path (dual T4)

- two DDP ranks via `scripts/train_dual_t4.sh` (`torchrun --nproc_per_node=$NGPU`), which also detects the
  dataloader worker count and disables the InfiniBand probe;
- **full fp16 LoRA**, no 4-bit weights: the T4 is Turing and has no bf16 kernels, so `--dtype float16`, guarded by
  `thermo_agent/train/ddp.py`;
- DDP replicates the frozen 4B model; only LoRA gradients cross PCIe, once per optimizer step (`no_sync` on
  accumulation micro-steps, `gradient_as_bucket_view`, `ddp_bucket_cap_mb=64`). FSDP/ZeRO would stream ~8 GB per
  step over the same link;
- overflow handling: AMP `GradScaler` with dynamic loss scaling, `max_grad_norm=1.0`, `adamw_torch`, fp32
  accumulation in matmuls;
- batch arithmetic: SFT 2 x 8 x 2 = 32; GRPO G = 4 with 4 x 2 x 2 = 16 prompts (8 x 8 rollouts exceed the T4 KV
  cache: about 9 GB on top of the 8 GB of weights);
- SFT `max_length` 4096; prompts over 14000 characters are dropped and counted, never truncated.

Trainings have not been run yet: the 4050 cannot hold a 4B model for training, and the Kaggle notebook still needs
the user's credentials.

## Kaggle teacher

`kaggle/serve_teacher.py` runs vLLM (default `Qwen/Qwen2.5-32B-Instruct-AWQ`, TP=2, max-len 8192) and prints a
cloudflared `TEACHER_BASE_URL`. `scripts/kaggle_teacher.ps1 -Url ...` probes `/v1/models` and samples all tasks with
memory retrieval and the architecture corpus included. Sampling is resumable. Blocked on the user running the
notebook and supplying the tunnel URL.

## Tests

- Full CPU suite: 275 passing before the Kaggle/teacher additions; `tests/test_kaggle.py` (4) and
  `tests/test_sample.py` (4) pass; expect about 283 after the data jobs finish (the full suite is re-run then so
  generation is not starved).
- Run `pytest -q -x` before any commit that touches `thermo_agent/`.

## Next

1. Wait for `gen-data2`; confirm `reports/data_report.json` and `data/sft/sft_report.json` have empty `problems`.
2. Run `scripts/baselines.ps1` on the final splits and record the numbers above.
3. When the Kaggle notebook is up: `pwsh scripts/kaggle_teacher.ps1 -Url <url> -Samples 2 -Limit 50`, check
   acceptance rates, then scale the limit and rerun `combine`.
4. Smoke-test SFT/GRPO on Kaggle (`--max-steps 20`) before any full run.
5. Replace the small-sample numbers in `paper/outline.md` and `paper/methods_draft.md`; add the dual-T4 recipe and
   the four-source corpus to the methods.
