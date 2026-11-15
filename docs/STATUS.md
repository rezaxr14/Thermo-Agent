# Status and operating notes

Working copy on the second laptop: `C:\Users\rezax\Home\Code\thermo_agent\Repo` (git, branch `main`).

## Environment

| Item | Value |
|---|---|
| Machine | REZA_PC, Windows 11 Home, 16 cores, 15.2 GB RAM, RTX 4050 6 GB |
| Python | 3.12.3 in `.venv` (created with `uv`), pinned by `requirements-lock.txt` |
| Package | installed editable (`pip install -e . --no-deps`) |
| Tools | `uv`, `ripgrep`, `fd`, `jq` in `%USERPROFILE%\tools\bin` and `%USERPROFILE%\.local\bin` |
| Local model | LM Studio serving `qwen3.5-4b` on `http://127.0.0.1:1234/v1` (format smoke checks only) |
| Remote access | SSH key auth, host alias `reza-pc` |
| Training target | Kaggle `GPU T4 x2` via `scripts/train_dual_t4.sh` |
| Frontier teacher | Kaggle Benchmarks / CLI (Claude, OpenAI, Gemini); deferred until the local work is done |

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
   with `--variants 4` truthful phrasings built from the fault facts, so the model cannot memorise one wording.
2. **Architecture corpus** (`thermo_agent/synth/architectures.py`): 65 plant diagrams (turbine/engine prime mover,
   absorption/electric cooling, wet/dry/extended rejection, with/without ORC, plus `cgam_steam_orc`), each with
   ORC pressures chosen per working fluid. Whole diagrams are held out for the OOD splits.
3. **Frontier teacher traces** (`thermo_agent/teacher/sample.py` + `validate.py`): every attempt is stored raw
   with a timestamp, elapsed time and verdict; only answers that pass the tool suite become SFT rows. Rejections
   become hindsight memory. The Kaggle/CLI teacher pass is deferred; `combine` already reads its output.
4. **Hindsight failure memory** (`thermo_agent/teacher/memory.py`): lessons mined from rejected attempts are
   retrieved per prompt (k=3) and injected as "known failure modes".

`thermo_agent/synth/combine.py` merges all four into `data/sft/{task}_{split}.jsonl` (the filenames the SFT and
GRPO trainers read), injects the harness on 35 percent of rows, and writes `sft_report.json` (dropped prompts,
source counts, expert-reward check).

### Harness budget

The harness is `AGENTS.md` (4.7k chars) plus skills. Skills are selected by the components in the plant, but when
there is no plant yet (Design) only the task skill (`design-catalog`) is included: all eight component skills add
about 12k characters and pushed every Design and Ledger prompt over the 14k cap, which silently dropped their
harness rows. After the fix no row is dropped for length in those tasks.

### Current run

`gen-data2` died overnight during Design (the machine slept; Design writes only when every site finishes).
`gen-data3` resumed at 08:28 and skips the finished steps from their meta files:

- repair n=12000: done, about 2 minutes; train 3774 / test_ood_fault 3204 / val 463 / test_id 444 /
  test_ood_topology 3925.
- audit n=3000: done, about 2 minutes; train 1583 / test_ood_topology 915 / val 211 / test_id 207.
- ledger n=9000: done, about 3 minutes; train 3479 / test_ood_fault 1619 / test_ood_topology 2999 / test_id 426 /
  val 476.
- design n=4000, max-evals 60: running (about 2 hours).
- architecture n=3000 requested: queued (about 5 minutes; about 70 percent survive detection).
- memory: mine 1000 repair rows into `data/memory/`.
- data report + combine (variants 4, harness 0.35, check-limit 25).

A combine smoke on the partial data (variants 2) produced 0 problems: repair train 7664, ledger 3479, audit 1391,
design 652; all expert rows passed the verifier; mean prompt length 5.5k chars for repair (max 13.8k).

## Baselines (small-sample reference values)

- repair, `test_id` and `test_ood_fault`: verifier-guided search repairs every row; 8 solver calls for sign faults,
  56 for unit slips, 245 to 290 for swapped ports and wrong fluid.
- audit, `test_ood_topology`: first-law top-1 15 % vs 32.8 % random; combined rule valid in 28/40 and captures a
  mean 10.7 % of the best saving; change search 11.9 evaluations vs 1 for a model.
- ledger, `test_ood_fault`: no-op solves none; the regenerate rule solves all; at least one second-law symptom in
  87 to 100 % of rows by fault type, closure broken in 26 to 100 %.
- design: pending the new split; `scripts/baselines.ps1` runs the OOD climate/load/scale splits.

## Training path (dual T4)

- `scripts/train_dual_t4.sh` defaults to `DATA=data/sft`, detects `NGPU` and the dataloader worker count, and
  disables the InfiniBand probe. `STEPS` / `ROWS` add smoke limits.
- two DDP ranks via `torchrun --nproc_per_node=$NGPU`;
- **full fp16 LoRA**, no 4-bit weights: the T4 is Turing and has no bf16 kernels, so `--dtype float16`, guarded by
  `thermo_agent/train/ddp.py`;
- DDP replicates the frozen 4B model; only LoRA gradients cross PCIe, once per optimizer step (`no_sync` on
  accumulation micro-steps, `gradient_as_bucket_view`, `ddp_bucket_cap_mb=64`);
- overflow handling: AMP `GradScaler`, `max_grad_norm=1.0`, `adamw_torch`, fp32 matmul reductions;
- batch arithmetic: SFT 2 x 8 x 2 = 32; GRPO G = 4 with 4 x 2 x 2 = 16 prompts (a micro-batch is 4 prompts x 4
  rollouts = 16 sequences, about 9 GB of KV cache on top of the 8 GB of weights);
- SFT `max_length` 4096; prompts over 14000 characters are dropped and counted, never truncated;
- `all` runs SFT, merges the latest checkpoint to `runs/sft_merged`, then runs GRPO on each task.

Trainings have not been run yet: the 4050 cannot hold a 4B model for training, and the Kaggle run is the next step.

## Tests

- Full CPU suite: **283 passed, 1 skipped** in 10.6 minutes. The skip is `tests/test_methods_draft.py`: the
  markdown draft it quoted was replaced by `paper/sections/03_method.tex`.
- Run `pytest -q` before any commit that touches `thermo_agent/`.

## Kaggle (deferred)

The frontier pass will use Kaggle's own model access (Kaggle Benchmarks / CLI: Claude Sonnet/Haiku, OpenAI,
Gemini), not a self-hosted model behind a tunnel. The tunnel/vLLM scripts were removed. When that pass starts:
export prompts, run them in a Kaggle notebook, download the completions, validate with `teacher.validate`, and
merge with `combine`. See `kaggle/README.md`.

## Next

1. Wait for `gen-data3` (Design, architecture, memory, report, combine); confirm `reports/data_report.json` and
   `data/sft/sft_report.json` have empty `problems`.
2. Run `scripts/baselines.ps1` on the final splits.
3. Smoke-test SFT/GRPO on Kaggle (`STEPS=20 ROWS=64 bash scripts/train_dual_t4.sh sft`, then `grpo ledger`).
4. Start the Kaggle Benchmarks teacher pass when the user is ready.
5. Replace the small-sample numbers in `paper/sections/` and add the four-source corpus and dual-T4 recipe.
