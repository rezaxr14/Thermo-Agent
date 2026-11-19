"""Stage 2: GRPO with the physics verifiers as reward (multi-turn tool use, Gate 8).
    python -m thermo_agent.train.grpo DATA_DIR OUT_DIR --task repair|design|audit|ledger --init SFT_ADAPTER_OR_MERGED_MODEL
        [--dtype float16] [--max-steps N] [--limit-rows N]
        [--per-device-batch 2] [--grad-accum 8] [--grad-norm 1.0] [--num-workers 2]
        [--max-tool-calling-iterations 4] [--max-completion-length 2048]
Dual T4 (see docs/DUAL_T4.md):
    torchrun --standalone --nproc_per_node=2 --module thermo_agent.train.grpo DATA_DIR runs/grpo_ledger \
        --task ledger --init runs/sft_merged --dtype float16 --per-device-batch 4 --grad-accum 2
Each rank draws its own prompts, so data parallelism splits the rollout work instead of duplicating it. The
effective prompt batch (per-device x accum x ranks) must be a multiple of the group size G. --vllm is
single-process colocate mode and is refused under DDP.
Not run in the authoring environment (no GPU). Start from the SFT checkpoint: RLVR needs a policy that already
succeeds sometimes (Lambert et al. 2024). Log the fraction of groups with zero reward variance.

Multi-turn wiring (verified against the installed TRL 1.18.1 source):
  * `EnvironmentFactory = Callable[[], _SupportsReset]` and `reset(self, **kwargs)` — the factory is
    probed with NO arguments and `reset` receives every dataset column as kwargs
    (`grpo_trainer.py:136-140`, `:745`, `:2541-2554`). The returned string is appended to the last
    user message, so the wrapper returns the environment block, not the task prompt.
  * Tools are discovered with `inspect.getmembers(type(instance), predicate=inspect.isfunction)`,
    keeping every public method except `reset`/`get_reward` (`grpo_trainer.py:750-763`, `:2527-2531`).
    Hence the thin wrapper below exposes exactly the five physics tools and NOT `ThermoEnv.step` or
    `.observation_block`. Passing `tools=` as well would duplicate every tool schema (`:761`), so
    the tools are wired *through* `environment_factory`.
  * `tool_mask` marks tool-result tokens 0 and model tokens 1 (`grpo_trainer.py:2079`, `:2277-2285`);
    the loss uses `completion_mask * tool_mask` (`:2664`). That agrees with SFT's decision to mask
    `<tool_result>...</tool_result>` spans: neither stage teaches the model to predict tool output.
"""
from __future__ import annotations
import argparse, functools, json, logging, math, os, pathlib, warnings

from ..agent.env import ThermoEnv, row_has_harness_tools
from . import ddp
from .checkpoint import latest
from .config import GrpoSettings, supported
from .render import TRAIN_ON_REASONING  # single thinking decision, shared with SFT
from . import hf_sync
from . import rollout_log
from . import rewards as repair_rewards
from ..design import rewards as design_rewards
from ..audit import rewards as audit_rewards
from ..ledger import rewards as ledger_rewards

# The single thinking decision is defined in `render.py` (as TRAIN_ON_REASONING / CHAT_KW),
# already imported by SFT; GRPO imports the same constant rather than duplicating it. Measured on
# the cached Qwen3.5-4B tokenizer: `enable_thinking=True` ends the generation prompt with
# `<think>\n`, so reasoning-bearing teacher rows align with the prompt prefix; `False` pre-closes
# an empty think block and breaks that alignment. No architecture change is made.

# Multi-turn defaults. max_completion_length bounds the WHOLE rollout (all turns plus tool results),
# so the old single-turn 320 is far too small; it is CLI-configurable.
DEFAULT_MAX_TOOL_CALLING_ITERATIONS = 4
DEFAULT_MAX_COMPLETION_LENGTH = 2048
MAX_TOOL_CALLS = 6

TOOL_NAMES = ("inspect_plant", "first_law_check", "second_law_check", "validity_check", "solve_plant")

_log = logging.getLogger(__name__)


def multiturn_config(max_tool_calling_iterations: int = DEFAULT_MAX_TOOL_CALLING_ITERATIONS,
                     max_completion_length: int = DEFAULT_MAX_COMPLETION_LENGTH) -> dict:
    """The GRPOConfig values that make training multi-turn and stable.

    Keep this the single place these are set: `main` spreads the result into `GRPOConfig`.
    """
    return {
        "max_tool_calling_iterations": max_tool_calling_iterations,
        "max_completion_length": max_completion_length,
        "mask_truncated_completions": True,
        "chat_template_kwargs": {"enable_thinking": TRAIN_ON_REASONING},
    }


class GrpoThermoEnv:
    """Thin composition wrapper around :class:`ThermoEnv` for TRL's `environment_factory`.

    TRL calls `factory()` with no arguments and treats every public class method except `reset`
    and `get_reward` as a tool. `ThermoEnv.step` and `.observation_block` are therefore only
    reached *privately* from here; they are never exposed as tools. The five public tool methods
    have type hints and Google-style docstrings so transformers' `get_json_schema` can render
    them (`transformers/utils/chat_template_utils.py:289-423`, `:745-757`).
    """

    def __init__(self, row: dict | None = None, task: str = "repair",
                 max_turns: int = DEFAULT_MAX_TOOL_CALLING_ITERATIONS,
                 max_tool_calls: int = MAX_TOOL_CALLS) -> None:
        self.task = task
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self._env: ThermoEnv | None = None
        # Set by reset() from the row metadata/system prompt. When True the row already carries
        # the tool contract, so reset() returns "" instead of a second observation block.
        self.prompt_already_has_tools = False
        self._logged_prompt_already_has_tools = False
        if row is not None:
            self.reset(**row)

    def reset(self, **kwargs) -> str:
        """Bind the dataset row and start a fresh episode.

        TRL forwards every dataset column here (`grpo_trainer.py:2544-2545`), so this accepts
        arbitrary kwargs rather than ThermoEnv's `(row, task)` signature. The returned observation
        is appended to the prompt's last message (`:2554`); it must be the environment block, not
        the task prompt, which is already in the dataset's `prompt` column.

        Args:
            **kwargs: Dataset columns for this row (e.g. prompt, plant_broken, site, deducible).

        Returns:
            str: The multi-turn environment block (tool list, limits, final-answer format) for
            plain rows; an empty string for rows that already carry the tool contract, so TRL
            appends nothing and the model never sees two different contracts.
        """
        row = {k: v for k, v in kwargs.items() if k != "environment"}  # multi-env control field
        self._env = ThermoEnv(row, task=self.task, max_turns=self.max_turns,
                              max_tool_calls=self.max_tool_calls)
        self.prompt_already_has_tools = row_has_harness_tools(row)
        if self.prompt_already_has_tools:
            if not self._logged_prompt_already_has_tools:
                self._logged_prompt_already_has_tools = True
                _log.info("GRPO row already carries the harness tool prompt "
                          "(prompt_variant='harness_tools' or 'MULTI-TURN TOOL ENVIRONMENT' "
                          "in the system prompt); skipping observation_block() to avoid a "
                          "second tool contract")
            return ""
        return self._env.observation_block()

    # Exactly five public tool methods. All delegate to the owned ThermoEnv instance.
    def inspect_plant(self, candidate: str = "") -> str:
        """Solve and verify the plant or ledger as given, before any candidate exists.

        This is the only tool the student can call first: it needs no answer. The candidate
        argument is ignored so every tool shares one JSON schema.

        Args:
            candidate: Ignored; pass an empty string. Defaults to "".

        Returns:
            str: JSON with solver status, failed checks and key residuals (mass kg/s, energy and
            exergy W) for the row's plant/ledger.
        """
        return self._env.inspect_plant(candidate)

    def first_law_check(self, candidate: str) -> str:
        """Check the candidate against the first law using the real solver.

        Args:
            candidate: The candidate answer in the task's final XML format.

        Returns:
            str: JSON with per-line/per-component mass (kg/s) and energy (W) residuals, heat/power/
            fuel balance, convergence, tear residual, unit/range flags and the failed checks.
        """
        return self._env.first_law_check(candidate)

    def second_law_check(self, candidate: str) -> str:
        """Check the candidate against the second law using the real solver.

        Args:
            candidate: The candidate answer in the task's final XML format.

        Returns:
            str: JSON with exergy destruction (MW), exergy closure (MW), eta_I/eta_II, Carnot and
            COP-vs-reversible limits, and the failed checks.
        """
        return self._env.second_law_check(candidate)

    def validity_check(self, candidate: str) -> str:
        """Check the candidate's units, ranges, site constraints and hard verifiers.

        Args:
            candidate: The candidate answer in the task's final XML format.

        Returns:
            str: JSON with unit/range violations, site constraints, pinch approach (K), the
            verifier verdict/score and the failed checks.
        """
        return self._env.validity_check(candidate)

    def solve_plant(self, candidate: str) -> str:
        """Solve the candidate plant and return the full stream state table.

        Args:
            candidate: The candidate answer in the task's final XML format.

        Returns:
            str: JSON with one entry per stream (T in K, P in kPa, h and s in kJ/kg and kJ/kg-K,
            exergy in kJ/kg) so states can be reasoned about.
        """
        return self._env.solve_plant(candidate)


def make_env_factory(task: str, max_turns: int = DEFAULT_MAX_TOOL_CALLING_ITERATIONS,
                     max_tool_calls: int = MAX_TOOL_CALLS):
    """A zero-argument callable TRL can probe and use to draw one wrapper per rollout."""
    def factory() -> GrpoThermoEnv:
        return GrpoThermoEnv(task=task, max_turns=max_turns, max_tool_calls=max_tool_calls)
    factory.__name__ = f"thermo_env_factory_{task}"
    return factory


def wrapper_tool_methods(env: GrpoThermoEnv | None = None):
    """The five tool callables, for schema rendering/inspection (not passed as `tools=`)."""
    env = env if env is not None else GrpoThermoEnv()
    return [getattr(env, name) for name in TOOL_NAMES]


class RewardDiagnostics:
    """Log zero-variance group share and warn above 50%, with no training behaviour change.

    Each wrapped reward function records its per-completion values; after the last one is called
    for a batch the weighted group rewards are checked for variance. The share is published via
    TRL's `log_metric` hook (`grpo_trainer.py:1713-1714`, `:1685-1696`), so it is averaged over the
    logging step alongside the built-in metrics. TRL groups consecutive `num_generations` rewards
    (`:2935`), so the same `(batch, k)` reshape is used here.
    """

    def __init__(self, weights: list[float], num_generations: int, warn_above: float = 0.5) -> None:
        self.weights = weights
        self.k = num_generations
        self.warn_above = warn_above
        self._values: dict[int, list[float]] = {}
        self._warned = False
        self.latest: dict | None = None

    def wrap(self, funcs):
        n = len(funcs)

        def make(fn, index):
            @functools.wraps(fn)  # TRL maps reward_weights by __name__
            def inner(*args, **kwargs):
                values = fn(*args, **kwargs)
                self._values[index] = [float(v) if v is not None else math.nan for v in values]
                if len(self._values) == n:
                    self._publish(kwargs)
                    self._values = {}
                return values
            return inner

        return [make(fn, i) for i, fn in enumerate(funcs)]

    def _publish(self, kwargs) -> None:
        cols = [self._values[i] for i in range(len(self.weights))]
        n = len(cols[0]) if cols else 0
        totals = [sum(w * col[j] for w, col in zip(self.weights, cols) if not math.isnan(col[j]))
                  for j in range(n)]
        groups = [totals[j:j + self.k] for j in range(0, n, self.k)]
        complete = [g for g in groups if len(g) == self.k and all(not math.isnan(v) for v in g)]
        zero = sum(1 for g in complete if max(g) - min(g) < 1e-9)
        share = zero / max(len(complete), 1)
        self.latest = {"zero_variance_share": share, "groups": len(complete)}
        metric = kwargs.get("log_metric")
        if callable(metric):
            metric("reward_diagnostics/zero_variance_share", share)
        if share > self.warn_above and not self._warned:
            self._warned = True
            warnings.warn(
                f"reward_diagnostics: {share:.0%} of GRPO groups have zero reward variance "
                f"(>{self.warn_above:.0%}); those groups give no advantage signal — check the SFT policy "
                "and reward spread before spending more steps",
                stacklevel=2,
            )


# One stage name everywhere: the HF-sync prefix, the local runs/ dir and the eval_runner pull key.
# The group size k is part of the name so k=4/8/16 never share a remote prefix or checkpoint dir.
def stage_name(task: str, k: int) -> str:
    return f"grpo_{task}_k{k}"


# RLVR starts from the merged SFT policy, never from the base Qwen weights.
DEFAULT_INIT = "runs/sft_merged"

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
    ap.add_argument("--task", choices=list(TASKS), default="repair"); ap.add_argument("--init", default=DEFAULT_INIT); ap.add_argument("--dtype", default="float16", choices=list(ddp.DTYPES))
    ap.add_argument("--vllm", action="store_true", help="colocate vLLM for generation (single process, needs a >=24 GB GPU)")
    ap.add_argument("--max-steps", type=int); ap.add_argument("--limit-rows", type=int)
    ap.add_argument("--per-device-batch", type=int); ap.add_argument("--grad-accum", type=int); ap.add_argument("--grad-norm", type=float)
    ap.add_argument("--save-steps", type=int, help="checkpoint interval; a rerun of the same command resumes automatically")
    ap.add_argument("--num-generations", type=int, default=None,
                    help="GRPO group size G (k rollouts per prompt). Overrides config default 4; use distinct "
                         "out dirs per k (runs/grpo_<task>_k4, grpo_<task>_k8, grpo_<task>_k16) so checkpoints "
                         "never overwrite.")
    ap.add_argument("--max-tool-calling-iterations", type=int, default=None,
                    help=f"tool-calling turns per rollout (default {DEFAULT_MAX_TOOL_CALLING_ITERATIONS})")
    ap.add_argument("--max-completion-length", type=int, default=None,
                    help=f"token budget for the WHOLE multi-turn completion (default {DEFAULT_MAX_COMPLETION_LENGTH})")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--hf-sync", choices=("auto", "on", "off"), default=os.environ.get("THERMO_HF_SYNC", "auto"))
    ap.add_argument("--hf-repo", default=None, help="must be <token owner>/thermo-agent-checkpoints")
    ap.add_argument("--rollout-log", default=None, help="JSONL path for scored rollouts (audit and paper figures)")
    a = ap.parse_args(); g = GrpoSettings()
    if a.per_device_batch: g.per_device_batch = a.per_device_batch
    if a.grad_accum: g.grad_accum = a.grad_accum
    if a.grad_norm is not None: g.max_grad_norm = a.grad_norm
    if a.save_steps: g.save_steps = a.save_steps
    if a.num_generations is not None: g.num_generations = a.num_generations
    tool_iterations = (a.max_tool_calling_iterations if a.max_tool_calling_iterations is not None
                       else DEFAULT_MAX_TOOL_CALLING_ITERATIONS)
    completion_length = (a.max_completion_length if a.max_completion_length is not None
                         else DEFAULT_MAX_COMPLETION_LENGTH)
    world = ddp.world_size_from_env()
    if a.vllm and world > 1:
        raise SystemExit("--vllm colocate mode is single-process; under DDP let each rank use HF generate")
    if a.dtype != "float16":
        raise SystemExit("only fp16 LoRA DDP is supported on the dual-T4 target; pass --dtype float16")
    _report_topology(a.dtype, g.num_generations, g)

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    init = ddp.model_init_kwargs(a.dtype)
    rows = load_rows(a.data, task=a.task)[: a.limit_rows]
    # The stage includes k so the three group sizes get three distinct remote prefixes and local dirs.
    stage = stage_name(a.task, g.num_generations)
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
    lora = LoraConfig(r=g.lora.r, lora_alpha=g.lora.alpha, lora_dropout=g.lora.dropout,
                      target_modules=g.lora.target_modules, task_type="CAUSAL_LM")
    mt = multiturn_config(tool_iterations, completion_length)
    cfg = GRPOConfig(**supported(GRPOConfig, output_dir=a.out, learning_rate=g.lr, num_generations=g.num_generations,
                     temperature=g.temperature,
                     per_device_train_batch_size=g.per_device_batch, gradient_accumulation_steps=g.grad_accum,
                     max_steps=a.max_steps or g.max_steps, loss_type=g.loss_type, beta=g.beta, scale_rewards=g.scale_rewards,
                     reward_weights=TASKS[a.task][1],
                     remove_unused_columns=False, use_vllm=a.vllm, vllm_mode="colocate", log_completions=True,
                     gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                     max_grad_norm=g.max_grad_norm, dataloader_num_workers=a.num_workers,
                     gradient_as_bucket_view=True, ddp_find_unused_parameters=False, ddp_bucket_cap_mb=64,
                     optim="adamw_torch", save_total_limit=2, seed=0,
                     logging_steps=5, save_steps=g.save_steps, model_init_kwargs=init, report_to="none",
                     fp16=(a.dtype == "float16"), bf16=(a.dtype == "bfloat16"), **mt))
    # Refuse any accidental 4-bit/FSDP/ZeRO configuration: the supported recipe is fp16 LoRA DDP.
    ddp.assert_fp16_lora_ddp(a.dtype, init, lora,
                             deepspeed=getattr(cfg, "deepspeed", None), fsdp=getattr(cfg, "fsdp", None))
    resume = latest(a.out)
    if resume:
        print(f"resuming GRPO from {resume}", flush=True)
    funcs = rollout_log.wrap_rewards(TASKS[a.task][0], a.rollout_log, stage=stage)
    diag = RewardDiagnostics(TASKS[a.task][1], g.num_generations)
    funcs = diag.wrap(funcs)
    GRPOTrainer(model=init_model, reward_funcs=funcs, args=cfg, train_dataset=Dataset.from_list(rows),
                peft_config=lora,
                # No `tools=`: TRL extracts the wrapper's five public methods as tools at init and
                # again per rollout (grpo_trainer.py:750-763, :2527-2531). Passing `tools=` here would
                # append a duplicate schema for every tool (`:761`).
                environment_factory=make_env_factory(a.task, max_turns=tool_iterations),
                callbacks=([callback] if callback else [])).train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    main()
