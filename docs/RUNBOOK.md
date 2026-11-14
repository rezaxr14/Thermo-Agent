# Runbook

Goal: generate data, check it, measure the base model, run SFT then GRPO on Qwen3.5-4B, and report back a small bundle of files.
Everything under "Generate data" and every test runs on CPU. Everything that loads the model (evaluation, SFT, GRPO, merge) has never been run by the author (no GPU was available), so Section 10 lists what is most likely to need a fix. Run the steps in order and stop at the first failure; send back the failing command and its output.

## 0. Hardware
| GPU | Use |
|---|---|
| RTX PRO 6000 (48 GB) or RTX 4090 (24 GB) | bf16 LoRA, default settings. Add `--vllm` to GRPO if vLLM is installed. |
| Kaggle T4 (16 GB), one GPU | `--dtype float16 --qlora`. T4 has no bf16. |
| Kaggle TPU v5e-8 | not supported by these scripts (PyTorch/TRL on GPU only). |
| RTX 4050 (6 GB) | data generation, tests and non-model baselines only; 4B training with prompts of about 2k tokens is unlikely to fit. |

## 1. Install
```
unzip thermo-agent.zip && cd thermo-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt && pip install -e ".[dev,train]"
python -m thermo_agent.agent.build_harness --check          # must print nothing and exit 0
pytest -q -x                                                # about 5 minutes on CPU (longer on a slow machine); all tests must pass
```

## 2. Generate data (CPU, no model)
Measured on 4 workers in the authoring sandbox: Audit about 0.5 s per plant wall-clock; Design about 4 s per site wall-clock at `--max-evals 15` (expect two to three times more at 60, not measured); Repair about 0.5 s per record single-threaded.
```
mkdir -p reports data/processed
python -m thermo_agent.synth.build_repair_data data/processed -n 4000 --seed 0 --workers 8
python -m thermo_agent.audit.data              data/processed -n 800  --seed 0 --workers 8
python -m thermo_agent.design.data             data/processed -n 1500 --seed 0 --workers 8 --max-evals 60
python -m thermo_agent.ledger.data             data/processed -n 3000 --seed 0 --workers 8     # about 0.1 s per row wall-clock on 4 workers
python -m thermo_agent.eval.data_report data/processed > reports/data_report.json
```
`problems` in the report must be empty. If it names a missing split, generate more rows of that task. Reference designs must be built with `--max-evals 60` or more (a 15-evaluation polish was measured to be 1.6 to 3.1 percent worse).

## 3. Non-model baselines (the numbers a model has to beat)
```
python -m thermo_agent.eval.run_eval          data/processed --split test_ood_fault     --limit 40 --by-fault
python -m thermo_agent.eval.run_eval          data/processed --split test_ood_topology  --limit 40 --by-fault
python -m thermo_agent.design.run_eval        data/processed --split test_ood_climate   --limit 10
python -m thermo_agent.audit.run_eval         data/processed --split test_ood_topology  --limit 40
python -m thermo_agent.ledger.run_eval        data/processed --split test_ood_fault     --limit 200      # symptom coverage; a rule solves it by construction
```
Reference values measured on small samples: Repair, verifier-guided search repairs 92 to 100 percent with 8 to 45 solver calls (numeric faults) and about 220 to 280 (swapped ports, wrong fluid). Design, enumerating the defaults takes 43 to 48 evaluations at 0.7 to 1.2 percent from the best known design; random search with 12 evaluations is 7 to 10 percent worse. Ledger, the table of second-law symptom coverage (closure residual, negative destruction) by fault type; about 86 percent of rows show at least one symptom. Audit, first-law ranking hits the true top-1 (prime mover excluded) in 0 to 19 percent of plants; the fixed rule "raise turbine inlet temperature 10 percent" recovers 94 percent of the best gain on turbine plants and 33 percent on held-out engine plants.

## 3b. Will GRPO get a learning signal? (CPU, before any GPU time)
```
python -m thermo_agent.train.reward_diagnostics data/processed --groups 12 --out reports/reward_diagnostics.json
```
It simulates policies that answer correctly with probability 1 minus the error rate and draw groups of 8, then reports the reward spread inside groups and the share of groups whose rewards are all equal (those groups teach nothing, because GRPO normalises by the group's spread). Measured in the authoring sandbox on small samples: with a 5 percent error rate about 70 to 90 percent of groups are flat, at 10 percent about 30 to 50 percent, at 30 percent about 20 percent, at 60 percent none. Practical reading: once an SFT model answers a task correctly more than about 90 percent of the time, GRPO on that distribution has little to learn from; train GRPO on the tasks and splits where SFT is clearly below that, and watch the share of zero-variance groups in the training log (stop or change the prompt mix when it stays above about 50 percent). Fresh reward evaluation costs about 1 ms (ledger), 8 ms (repair), 23 ms (audit) and 35 ms (design) per completion, so rewards will not be the bottleneck next to generation.

## 4. Base model, no training
```
python -m thermo_agent.eval.model_eval data/processed --model Qwen/Qwen3.5-4B --limit 30 --harness none          --out reports/base_none.json
python -m thermo_agent.eval.model_eval data/processed --model Qwen/Qwen3.5-4B --limit 30 --harness agents+skills --out reports/base_harness.json
```
Add `--dtype float16 --load-4bit` on a T4. Check `reports/base_none.completions.jsonl`: are answers in the requested format? If almost none parse, the model is not following the format (expected for the base model); if the text starts with a thinking block, tell me.

## 5. Check the chat template before training
```
python -m thermo_agent.train.sft data/processed runs/tmp --tasks repair design audit ledger --print-sample
```
Send back the output. `PROMPT TAIL` must end where the assistant answer starts (for Qwen3-style templates, with an empty thinking block). `PREFIX CHECK` may say `prefix_matches: false`; that is why SFT uses the plain-text format of `train/render.py`, which is what GRPO and evaluation render.

## 6. SFT
Smoke test, then the full run (one adapter for all three tasks):
```
python -m thermo_agent.train.sft data/processed runs/sft_smoke --tasks repair design audit ledger --max-steps 20 --limit-rows 64
python -m thermo_agent.train.sft data/processed runs/sft       --tasks repair design audit ledger     # add --dtype float16 --qlora on a T4
```
Note the seconds per step from the smoke log and loss going down. Then evaluate (use the last `checkpoint-*` in `runs/sft`):
```
python -m thermo_agent.eval.model_eval data/processed --model Qwen/Qwen3.5-4B --adapter runs/sft/checkpoint-N --limit 30 --out reports/sft_none.json
python -m thermo_agent.eval.model_eval data/processed --model Qwen/Qwen3.5-4B --adapter runs/sft/checkpoint-N --limit 30 --harness agents+skills --out reports/sft_harness.json
```
Expect Repair and Audit format and Design validity to be high after SFT. If the format is still poor, the problem is the template or the data, not GRPO; stop and report.

## 7. GRPO (one task at a time, starting from the merged SFT model)
```
python -m thermo_agent.train.merge runs/sft/checkpoint-N runs/sft_merged
python -m thermo_agent.train.grpo data/processed runs/grpo_smoke --task repair --init runs/sft_merged --max-steps 5 --limit-rows 16
python -m thermo_agent.train.grpo data/processed runs/grpo_repair --task repair --init runs/sft_merged      # then --task ledger, --task audit, --task design
```
Look in the log for the mean of each reward function and the reward standard deviation. If the standard deviation is zero for most steps, every sample in a group scores the same and nothing is learned: tell me which task and the reward means. GRPO samples 8 completions per prompt with plain HF generation, which is slow; `--vllm` helps if it works. Ledger is a cheap, exact reward (pure arithmetic), so start GRPO there if time is short. Evaluate each result:
```
python -m thermo_agent.eval.model_eval data/processed --model runs/sft_merged --adapter runs/grpo_repair/checkpoint-N --tasks repair --limit 30 --out reports/grpo_repair.json
```

## 8. Ablations and forgetting
```
# harness on/off for base, SFT, GRPO: repeat the model_eval commands with --harness none and --harness agents+skills
# the held-out new component (gas engine) is in test_ood_topology for Repair and Audit
python -m thermo_agent.eval.thermoqa describe                                  # send back the columns, then:
python -m thermo_agent.eval.thermoqa run --question-col <Q> --answer-col <A> --limit 100 --out reports/tqa_base.json
python -m thermo_agent.eval.thermoqa run --question-col <Q> --answer-col <A> --limit 100 --adapter runs/sft/checkpoint-N --out reports/tqa_sft.json
```

Once the Frontier file is downloaded (see `thermo_agent.datasets.frontier.describe`), rank the catalog on the real trace; `--step-hours` is the sampling interval of the file and has no default on purpose. The result is a lower bound (ideal part-load, plant re-sized per load bin):
```
python -m thermo_agent.design.trace_eval path/to/frontier.xlsx --columns it_power=<col> t_supply=<col> t_return=<col> --power-unit MW --temp-unit C --chilled-fraction 0.5 --step-hours <hours> --top 8
```

To see what a modular fleet costs on the same trace, add `--spec-json '{...}' --scan-units 8` (one design, 1 to 8 identical units; the penalty is fuel above the ideal part-load bound):
```
python -m thermo_agent.design.trace_eval path/to/frontier.xlsx --columns it_power=<col> t_supply=<col> t_return=<col> --step-hours <hours> \
    --spec-json '{"prime_mover":"engine","cooling":"absorption","rejection":"wet","bottoming":"none","dT_rej":8.0,"T_hw":368.15,"dT_hw":10.0}' --scan-units 8
```

Design sample efficiency (the central claim): run the anytime protocol, in which the model is told the fuel of each valid design and may improve it, then plot best-result-after-b-evaluations against enumeration, random search and the enumerate-then-polish reference:
```
python -m thermo_agent.eval.model_eval data/processed --model runs/sft_merged --adapter runs/grpo_design/checkpoint-N --tasks design --refine-turns 6 --limit 30 --out reports/grpo_design_refine.json
python -m thermo_agent.design.curves data/processed --split test_id --limit 20 --report reports/grpo_design_refine.json --out paper/generated/design_anytime.png
```
Paired comparisons for the ablations (same examples, exact McNemar test for binary metrics, permutation test otherwise, Holm correction across everything listed):
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
5. Template: `enable_thinking=False` is passed to `apply_chat_template`; if the template ignores it, SFT, GRPO and evaluation still agree with each other, but a thinking block may appear in answers (the evaluator strips `<think>...</think>`).
6. Time per GRPO step on a T4 is unknown; use the smoke run to extrapolate before committing to 300 steps.
