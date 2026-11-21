# thermo-agent

Second-Law-Verified AI Agents for On-Site CHP and Heat Recovery in AI Data Centers.
All results are simulation-driven (real load traces into a CoolProp / NASA-7 plant model); no measured CHP data centre is involved.

```
thermo_agent/
  core/      schema.py  props.py  gasprops.py  fluids.py  exergy.py  components.py  solve.py  cgam_plant.py
             machines: compressor, turbine, combustor, recuperator, HRSG, ORC evaporator/turbine, pump, condenser,
                       gas_engine, dc_load, absorption_chiller, electric_chiller, heat_pump, heat_rejector
  verify/    verifiers.py (V1-V6, V8 site requirements)  grounding.py (V7)
  agent/     dsl.py (plant text, edit language)  loop.py (model-in-the-loop repair)
  synth/     topologies.py (cgam_orc, dc_chp)  faults.py  repair_dataset.py  sft_data.py  build_repair_data.py
  train/     prompts.py  rewards.py (GRPO rewards)  reward_diagnostics.py  perturb.py  render.py  sft.py  merge.py  grpo.py  config.py
  eval/      compare.py (paired tests)  repair.py  baselines.py  run_eval.py  model_eval.py (any model, all tasks)  summarize.py  paper_tables.py  data_report.py  thermoqa.py
  design/    spec.py (catalog, parser)  build.py (site+design -> plant, evaluate)  search.py (enumerate, polish, random)
             prompts.py  rewards.py  data.py  loop.py  run_eval.py
  ledger/    expr.py (safe arithmetic)  program.py  faults.py  prompts.py  rewards.py  data.py  run_eval.py
  audit/     changes.py (allowed edits, resize)  truth.py  spec.py  prompts.py  rewards.py  data.py  run_eval.py
  agent/     also AGENTS.md, skills/*/SKILL.md (generated sections), harness.py (ablation modes), build_harness.py
  design/    also trace_eval.py (trace-weighted fuel, ideal part-load)  modular.py (N identical units at rated load)  curves.py (anytime curves)
  datasets/  frontier.py (explicit column mapping -> Site and loop settings)
data/  requirements-lock.txt  .github/workflows/tests.yml (not run by the author)  tests/  docs/ (RUNBOOK.md, PROVENANCE.md)  paper/ (outline.md, methods_draft.md, refs.bib, method_justification.md)
```

    pip install -e ".[dev]" && pytest
    python -m thermo_agent.synth.repair_dataset data/processed/repair_v0.jsonl -n 300 --seed 0

Everything needed to run on a GPU, with expected outputs and the baselines to compare against, is in docs/RUNBOOK.md.

Ledger pipeline:

    python -m thermo_agent.ledger.data data/processed -n 3000 --seed 0 --workers 4
    python -m thermo_agent.ledger.run_eval data/processed --split test_ood_fault
    python -m thermo_agent.train.grpo data/grpo runs/grpo_ledger --task ledger --init runs/sft_merged

Audit pipeline:

    python -m thermo_agent.audit.data data/processed -n 400 --seed 0 --workers 4
    python -m thermo_agent.audit.run_eval data/processed --split test_ood_topology
    python -m thermo_agent.train.grpo data/grpo runs/grpo_audit --task audit --init runs/sft_merged

Design pipeline:

    python -m thermo_agent.design.data data/processed -n 600 --seed 0 --workers 4 --max-evals 60
    python -m thermo_agent.design.run_eval data/processed --split test_ood_climate
    python -m thermo_agent.train.sft data/sft runs/sft --tasks design
    python -m thermo_agent.train.grpo data/grpo runs/grpo_design --task design --init runs/sft_merged

Repair pipeline:

    python -m thermo_agent.synth.build_repair_data data/processed -n 3000 --seed 0
    python -m thermo_agent.eval.run_eval data/processed data/processed_arch --split test_ood_fault
    python -m thermo_agent.train.sft data/sft runs/sft --tasks repair   # LoRA SFT on Qwen3.5-4B
    python -m thermo_agent.train.grpo data/grpo runs/grpo_repair --task repair --init runs/sft_merged

Resumable training: with `HF_TOKEN` in `.env` (or a Kaggle Secret), each Trainer save pushes the full resume state
(adapter, optimizer, scheduler, scaler, RNG, trainer state) to the single private repo
`<token owner>/thermo-agent-checkpoints`; a restarted session pulls the newest checkpoint automatically.
`python -m thermo_agent.train.hf_sync init` creates the repo; `status`, `push`, `pull`, `push-merged` and
`pull-merged` inspect or move the state. Nothing is written to any other Hugging Face repo. The Kaggle training
notebook is `kaggle/train.ipynb` (generated from `train_runner.py`); see `docs/RUNBOOK.md`.
