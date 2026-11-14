# Status and operating notes

Working copy on the second laptop: `C:\Users\rezax\Home\Code\thermo_agent\Repo` (git, branch `main`).

## Environment

| Item | Value |
|---|---|
| Machine | REZA_PC, Windows 11 Home, 16 cores, 15.2 GB RAM, RTX 4050 6 GB |
| Python | 3.12.3 in `.venv` (created with `uv`), pinned by `requirements-lock.txt` |
| Package | installed editable (`pip install -e . --no-deps`) |
| Tools | `uv`, `ripgrep`, `fd`, `jq` in `%USERPROFILE%\tools\bin` and `%USERPROFILE%\.local\bin` |
| Local model | LM Studio serving `qwen3.5-4b` on `http://127.0.0.1:1234/v1` |
| Remote access | SSH key auth, host alias `reza-pc` |

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

## Pipeline state

1. `scripts/gen_data.ps1` is resumable: a step whose output already exists is skipped.
   - repair (n=4000): done, 61 s, splits 1248 / 149 / 161 / 1068 / 1310.
   - audit (n=800): done.
   - ledger (n=3000): done, splits 1166 / 155 / 145 / 535 / 998.
   - design (n=1500, max-evals 60, 14 workers): running. ~100 s of CPU per site on this machine, so the run
     is a few hours; the runbook requires 60 evaluations for reference designs.
   - data report: runs after design.
2. Baselines measured on the generated splits, saved under `reports/`:
   - repair, `test_id` and `test_ood_fault`: verifier-guided search repairs every row; 8 solver calls for
     sign faults, 56 for unit slips, 245 to 290 for swapped ports and wrong fluid.
   - audit, `test_ood_topology` (40 plants): first-law top-1 15 % vs 32.8 % random; combined rule valid in
     28/40 and captures a mean 10.7 % of the best saving; change search 11.9 evaluations vs 1 for a model.
   - ledger, `test_ood_fault` (200 rows): no-op solves none; the regenerate rule solves all; at least one
     second-law symptom in 87 to 100 % of rows by fault type, closure broken in 26 to 100 %.
   - design baselines: pending the design split.
3. Base model: `qwen3.5-4b` repair evaluation through LM Studio is running (`reports/local_repair.json`).

## Training path (dual T4)

No data parallelism in the original scripts: they call `.train()` in one process. The dual-T4 path is now
explicit and documented in `docs/DUAL_T4.md`:

- two DDP ranks via `scripts/train_dual_t4.sh` (`torchrun --nproc_per_node=2`);
- **full fp16 LoRA**, no 4-bit weights: the T4 is Turing and has no bf16 kernels, so `--dtype float16`,
  guarded by `thermo_agent/train/ddp.py`;
- DDP replicates the frozen 4B model; only LoRA gradients cross PCIe, once per optimizer step
  (`no_sync` on accumulation micro-steps, `gradient_as_bucket_view`, `ddp_bucket_cap_mb=64`). FSDP/ZeRO would
  stream ~8 GB per step over the same link;
- overflow handling: AMP `GradScaler` with dynamic loss scaling, `max_grad_norm=1.0`, `adamw_torch`,
  fp32 accumulation in matmuls;
- batch arithmetic: SFT 2 x 8 x 2 = 32; GRPO 8 x 2 x 2 = 32 prompts with G = 8.

Trainings have not been run yet: the 4050 cannot hold a 4B model for training, and Kaggle is not set up.

## Tests

- CPU suite: 234 passing before the dual-T4 work, plus 6 new `tests/test_ddp.py` tests (240).
- The full suite is re-run after the data jobs finish so the design generation is not starved.

## Next

1. Wait for design to finish, then confirm `reports/data_report.json` has an empty `problems` list.
2. Run the design baselines (`design/run_eval`) and reward diagnostics once the design split exists.
3. Finish the local-model repair eval; then ledger/audit/design limits as time allows.
4. Replace the "measured on small samples" numbers in `paper/outline.md` and `paper/methods_draft.md` with
   the values above, and add the dual-T4 recipe to the methods.
5. Smoke-test SFT/GRPO on Kaggle (`--max-steps 20`) before any full run.
