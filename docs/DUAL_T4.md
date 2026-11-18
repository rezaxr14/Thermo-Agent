# Training on two T4s (Kaggle `2x T4`)

Target: Qwen3.5-4B, **full fp16 LoRA** (no 4-bit), **two DDP ranks**, no NVLink. This file explains the
choices and gives the exact commands.

## Why DDP and not FSDP/ZeRO

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

## Why fp16 and not bf16

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

## Memory budget per T4

| Item | SFT | GRPO |
|---|---|---|
| base weights, fp16 | 8.0 GB | 8.0 GB |
| LoRA params + fp32 optimizer state | ~0.4 GB | ~0.2 GB |
| activations (checkpointing on) | ~2.8 GB at batch 2 × 8192 tokens | ~1 GB at 16 sequences × 320 tokens |
| cross-entropy tile (chunked, fp32) | 0.02 GB (512 × 8192) | — |
| KV cache | — | ~2 GB at 4 prompts × 4 completions × ~900 tokens |
| **total** | **~10 GB of 15 GB** | **~11 GB of 15 GB** |

Both fit with headroom. A 4-bit base is unnecessary and is not used (`--qlora` is refused under DDP).

## Chunked cross-entropy (the T4 fix)

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

## Batch arithmetic

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

## Commands

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

## What DDP does and does not parallelise

- **SFT**: shards the dataset across ranks; gradients are averaged each optimizer step. Wall time per step
  is roughly halved.
- **GRPO**: each rank draws its own prompts and generates their groups itself; the advantage computation
  stays within a rank, so no extra communication is introduced during rollouts. Generation dominates the
  step time on a T4, and that is the part that splits cleanly.
- **Not supported under DDP**: `--vllm` (colocate vLLM is a single-process mode; the script refuses it when
  `WORLD_SIZE > 1`).

## Resume across Kaggle sessions (private Hugging Face repo)

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

## Expectations on T4

T4 fp16 is ~65 TFLOPS peak with a narrow memory bus. SFT on 4B with 3072-token sequences will be slow
(hours, not minutes) but the batch math above is conservative enough to run without OOM. Measure seconds per
step in the first 20 steps with `--max-steps 20` before committing to a full run.
