"""Stage 2: GRPO with the physics verifiers as reward.
    python -m thermo_agent.train.grpo DATA_DIR OUT_DIR --task repair|design|audit|ledger --init SFT_ADAPTER_OR_MERGED_MODEL
Reward functions: thermo_agent.train.rewards (format, validity, site, exact-match when deducible, minimality, grounding).
Not run in the authoring environment (no GPU). Start from the SFT checkpoint: RLVR needs a policy that already succeeds
sometimes (Lambert et al. 2024). Log the fraction of groups with zero reward variance; if it is high, add harder prompts."""
from __future__ import annotations
import argparse, json, pathlib
from .config import BASE_MODEL, GrpoSettings
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
    return [{k: json.loads(l)[k] for k in cols} for l in open(pathlib.Path(data_dir) / f"{task}_{split}.jsonl")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--task", choices=list(TASKS), default="repair"); ap.add_argument("--init", default=BASE_MODEL); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--vllm", action="store_true", help="colocate vLLM for generation (needs a >=24 GB GPU)")
    ap.add_argument("--qlora", action="store_true"); ap.add_argument("--max-steps", type=int); ap.add_argument("--limit-rows", type=int)
    a = ap.parse_args(); g = GrpoSettings()
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    init = {"dtype": a.dtype}
    if a.qlora:
        import torch
        from transformers import BitsAndBytesConfig
        init["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, a.dtype))
    rows = load_rows(a.data, task=a.task)[: a.limit_rows]
    cfg = GRPOConfig(output_dir=a.out, learning_rate=g.lr, num_generations=g.num_generations,
                     max_completion_length=g.max_completion_length, temperature=g.temperature,
                     per_device_train_batch_size=g.per_device_batch, gradient_accumulation_steps=g.grad_accum,
                     max_steps=a.max_steps or g.max_steps, loss_type=g.loss_type, beta=g.beta, scale_rewards="group",
                     mask_truncated_completions=g.mask_truncated, reward_weights=TASKS[a.task][1],
                     remove_unused_columns=False, use_vllm=a.vllm, vllm_mode="colocate", log_completions=True,
                     chat_template_kwargs={"enable_thinking": False}, gradient_checkpointing=True,
                     logging_steps=5, save_steps=50, model_init_kwargs=init, report_to="none", fp16=(a.dtype == "float16"), bf16=(a.dtype == "bfloat16"))
    lora = LoraConfig(r=g.lora.r, lora_alpha=g.lora.alpha, lora_dropout=g.lora.dropout,
                      target_modules=g.lora.target_modules, task_type="CAUSAL_LM")
    GRPOTrainer(model=a.init, reward_funcs=TASKS[a.task][0], args=cfg, train_dataset=Dataset.from_list(rows),
                peft_config=lora).train()


if __name__ == "__main__":
    main()
