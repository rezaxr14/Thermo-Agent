"""Stage 2: GRPO with the physics verifiers as reward.
    python -m thermo_agent.train.grpo DATA_DIR OUT_DIR --task repair|design|audit|ledger --init SFT_ADAPTER_OR_MERGED_MODEL
        [--dtype bfloat16|float16] [--qlora] [--max-steps N] [--limit-rows N]
        [--per-device-batch 2] [--grad-accum 8] [--grad-norm 1.0] [--num-workers 2]
Dual T4 (see docs/DUAL_T4.md):
    torchrun --standalone --nproc_per_node=2 --module thermo_agent.train.grpo DATA_DIR runs/grpo_ledger \
        --task ledger --init runs/sft_merged --dtype float16 --per-device-batch 2 --grad-accum 8
Each rank draws its own prompts, so data parallelism splits the rollout work instead of duplicating it. The
effective prompt batch (per-device x accum x ranks) must be a multiple of the group size G. --vllm is
single-process colocate mode and is refused under DDP.
Not run in the authoring environment (no GPU). Start from the SFT checkpoint: RLVR needs a policy that already
succeeds sometimes (Lambert et al. 2024). Log the fraction of groups with zero reward variance."""
from __future__ import annotations
import argparse, json, os, pathlib

from . import ddp
from .checkpoint import latest
from .config import BASE_MODEL, GrpoSettings
from . import hf_sync
from . import rewards as repair_rewards
from ..design import rewards as design_rewards
from ..audit import rewards as audit_rewards
from ..ledger import rewards as ledger_rewards

TASKS = {"repair": (repair_rewards.REWARD_FUNCS, repair_rewards.REWARD_WEIGHTS, ("prompt", "plant_broken", "plant_original", "site", "deducible", "min_edits")),
         "design": (design_rewards.REWARD_FUNCS, design_rewards.REWARD_WEIGHTS, ("prompt", "site", "ref_fuel_w")),
         "ledger": (ledger_rewards.REWARD_FUNCS, ledger_rewards.REWARD_WEIGHTS, ("prompt", "ledger")),
         "audit": (audit_rewards.REWARD_FUNCS, audit_rewards.REWARD_WEIGHTS, ("prompt", "plant", "site", "ed_json", "fuel_w", "net_w", "best_gain_w"))}


def load_rows(data_dir, split="train", task="repair"):
    cols = TASKS[task][2]
    return [{k: json.loads(l)[k] for k in cols} for l in open(pathlib.Path(data_dir) / f"{task}_{split}.jsonl", encoding="utf-8")]


def _report_topology(dtype: str, num_generations: int, s: GrpoSettings) -> None:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible; GRPO needs a GPU")
    ddp.check_dtype_supported(dtype, torch.cuda.get_device_capability(0))
    world = ddp.world_size_from_env()
    if torch.cuda.device_count() < world:
        raise SystemExit(f"torchrun started {world} processes but only {torch.cuda.device_count()} GPUs are visible")
    ddp.assert_grpo_batches(s.per_device_batch, s.grad_accum, world, num_generations)
    if dtype == "float16":
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    print(f"[rank {ddp.local_rank_from_env()}/{world}] {ddp.topology_line()} "
          f"effective_prompts_per_step={ddp.effective_batch(s.per_device_batch, s.grad_accum, world)}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--task", choices=list(TASKS), default="repair"); ap.add_argument("--init", default=BASE_MODEL); ap.add_argument("--dtype", default="bfloat16", choices=list(ddp.DTYPES))
    ap.add_argument("--vllm", action="store_true", help="colocate vLLM for generation (single process, needs a >=24 GB GPU)")
    ap.add_argument("--qlora", action="store_true"); ap.add_argument("--max-steps", type=int); ap.add_argument("--limit-rows", type=int)
    ap.add_argument("--per-device-batch", type=int); ap.add_argument("--grad-accum", type=int); ap.add_argument("--grad-norm", type=float)
    ap.add_argument("--save-steps", type=int, help="checkpoint interval; a rerun of the same command resumes automatically")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--hf-sync", choices=("auto", "on", "off"), default=os.environ.get("THERMO_HF_SYNC", "auto"))
    ap.add_argument("--hf-repo", default=None, help="must be <token owner>/thermo-agent-checkpoints")
    a = ap.parse_args(); g = GrpoSettings()
    if a.per_device_batch: g.per_device_batch = a.per_device_batch
    if a.grad_accum: g.grad_accum = a.grad_accum
    if a.grad_norm is not None: g.max_grad_norm = a.grad_norm
    if a.save_steps: g.save_steps = a.save_steps
    world = ddp.world_size_from_env()
    if a.vllm and world > 1:
        raise SystemExit("--vllm colocate mode is single-process; under DDP let each rank use HF generate")
    if a.qlora and world > 1:
        raise SystemExit("QLoRA + DDP is not supported here: use full fp16 LoRA (drop --qlora), as in docs/DUAL_T4.md")
    _report_topology(a.dtype, g.num_generations, g)

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    init = {"dtype": a.dtype}
    if a.qlora:
        import torch
        from transformers import BitsAndBytesConfig
        init["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, a.dtype))
    rows = load_rows(a.data, task=a.task)[: a.limit_rows]
    stage = f"grpo_{a.task}"
    repo, callback = hf_sync.prepare(a.hf_repo, stage, a.out, a.hf_sync)
    init_model = a.init
    if init_model.startswith("hf:"):
        if not repo:
            raise SystemExit("an hf: init needs a token (--hf-sync on/auto) to pull the merged model")
        prefix = init_model[3:].lstrip("/")
        local_init = os.environ.get("HF_INIT_DIR") or str(pathlib.Path(a.out).parent / "sft_merged")
        pulled = hf_sync.pull_merged(local_init, repo, prefix=prefix)
        if not pulled:
            raise SystemExit(f"no merged model at {repo}/{prefix}; run merge with --hf-sync on first")
        init_model = pulled
        print(f"[hf_sync] GRPO starts from {init_model}", flush=True)
    cfg = GRPOConfig(output_dir=a.out, learning_rate=g.lr, num_generations=g.num_generations,
                     max_completion_length=g.max_completion_length, temperature=g.temperature,
                     per_device_train_batch_size=g.per_device_batch, gradient_accumulation_steps=g.grad_accum,
                     max_steps=a.max_steps or g.max_steps, loss_type=g.loss_type, beta=g.beta, scale_rewards="group",
                     mask_truncated_completions=g.mask_truncated, reward_weights=TASKS[a.task][1],
                     remove_unused_columns=False, use_vllm=a.vllm, vllm_mode="colocate", log_completions=True,
                     chat_template_kwargs={"enable_thinking": False}, gradient_checkpointing=True,
                     max_grad_norm=g.max_grad_norm, dataloader_num_workers=a.num_workers,
                     gradient_as_bucket_view=True, ddp_find_unused_parameters=False, ddp_bucket_cap_mb=64,
                     optim="adamw_torch", save_total_limit=2, seed=0,
                     logging_steps=5, save_steps=g.save_steps, model_init_kwargs=init, report_to="none",
                     fp16=(a.dtype == "float16"), bf16=(a.dtype == "bfloat16"))
    lora = LoraConfig(r=g.lora.r, lora_alpha=g.lora.alpha, lora_dropout=g.lora.dropout,
                      target_modules=g.lora.target_modules, task_type="CAUSAL_LM")
    resume = latest(a.out)
    if resume:
        print(f"resuming GRPO from {resume}", flush=True)
    GRPOTrainer(model=init_model, reward_funcs=TASKS[a.task][0], args=cfg, train_dataset=Dataset.from_list(rows),
                peft_config=lora, callbacks=([callback] if callback else [])).train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    main()
