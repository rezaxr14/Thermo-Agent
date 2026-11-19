# What makes this paper competitive, and what is still missing

This is an internal checklist, not part of the compiled paper. It separates what is already in the manuscript from
what can only be written after the Kaggle teacher pass and the GPU runs.

## In place now

1. **A new task, not a new optimizer.** The contribution is a setting in which physics verifiers are the reward: mass, energy, second law and site checks decide whether an answer exists. Four task families (Repair, Design, Audit, Ledger) share one simulator and one reward interface.
2. **Structural out-of-distribution splits.** Whole diagram combinations are held out (`engine-*-ext*`, `cgam_steam_orc`), while ORC, engines, turbines and district heat remain in training. The MIX leak (engine plants with an ideal boundary were silently in train) was found and fixed; `test_ood_topology` now contains 837 Repair, 135 Audit and 640 Ledger rows from `data/processed` alone, plus the architecture catalog.
3. **A frozen protocol.** `sec:protocol` fixes metrics, budgets, decoding, splits, Wilson intervals, paired McNemar/bootstrap tests and Holm correction before any model result is seen. Eight explicit RQs map to result slots.
4. **Comparisons that isolate the harness.** The evaluator has arms for our base/SFT/GRPO model with and without inference text, frontier zero-shot, frontier with our inference text, frontier with physics tools, and other models. `model_eval --backend replay` scores saved Kaggle answers on the same rows with the same verifiers; tool calls, reasoning traces and Kaggle token costs are recorded, not guessed.
5. **Cost accounting.** Solver calls/design evaluations, GPU hours and provider-reported nanodollars are separate columns. `teacher.progress.usage_totals` and `teacher.transfer.ingest` carry them into the ingest report.
6. **Reproducibility.** Seeds, data reports, corpus reports, token audits, corpus verifier, generated value macros with source/command provenance, four generated Kaggle notebooks, a resumable private HF checkpoint repo, and a single private teacher-progress file. Every table cell is generated from a report JSON; none is transcribed.
7. **Qualitative evidence without cherry-picking.** `collect_examples.py` selects traces deterministically from six stages and writes `examples_schema.json`; `figures_from_examples.py` draws them as a vector figure. The selection rule is code, so the figure reproduces.
8. **Honest status.** `sec:status` states exactly which claims are measured and which runs can fill the pending tables. Limitations are grouped as construct/internal/external/conclusion threats with mitigations.

## Needs the remaining runs (do not write these claims yet)

1. Actual numbers in `tab:results`, `tab:comparison`, `tab:ablation`: trained 4B vs baselines vs frontier arms, with CIs and Holm-corrected paired tests.
2. Teacher acceptance rate and cost per accepted row for the chosen model, including the model-selection probe (`docs/TEACHER_MODEL_CHOICES.md`).
3. GRPO-vs-SFT delta on each reward channel, and zero-variance group share measured with the *trained* SFT policy rather than the simulated one.
4. Tool-usage statistics: calls per answer, calls vs success, and reasoning-trace quality, from `reports/tool_usage.json`.
5. Failure-memory ablation (none / retrieved / random) with repeated-failure counts from `data/memory/failure_reasons.json`.
6. GPU-hour and dollar totals for training and inference, filled from the Kaggle run logs.

## New objective: runs to add before camera-ready

All of these are scored on the single frozen held-out set `data/heldout` (fresh ground-truth seed **987654**,
excluded from every training and prompt corpus), so each score is directly comparable with the others.

1. **Frontier zero-shot baseline on `data/heldout`.** The legacy zero-shot arm (`kaggle/teacher_benchmark.py`,
   reasoning on, no tools) scored on all four tasks and every split; this is the baseline every other arm is
   compared against.
2. **Frontier + harness + tools + skills + memory on `data/heldout`.** Trace-level results for the LangGraph arm
   with real physics tools, AGENTS.md/skills and fresh failure memory. Report the **accepted tool-CoT trace yield**
   (accepted tool-using trajectories / attempts, by model and task) and tool-usage statistics (calls per answer,
   calls vs success, revision rate) from `reports/tool_usage.json` and the trace ledger. Accepted trajectories
   become SFT completions (`source=frontier_tools`).
3. **SFT vs SFT+GRPO k=4 / 8 / 16 comparison.** Score our SFT checkpoint and three separately saved GRPO checkpoints
   (group size k = 4, 8, 16) on the same held-out set, with per-task metrics, Wilson intervals and Holm-corrected
   paired tests. Each checkpoint must live at a distinct path (`runs/sft`, `runs/grpo_k4`, `runs/grpo_k8`,
   `runs/grpo_k16`) so runs never overwrite one another.
4. **Other thermodynamics LLMs** on the same held-out set (domain-model arm), recorded once a real thermo-model id is
   supplied rather than the current GPT-nano stand-in.
5. **Model-choice note.** The chosen budget generators are `gpt-6-luna` then `claude-haiku-5-5` (reasoning on);
   the expensive tier (`gpt-6.2-sol`, `claude-sonnet-5-5`, `claude-opus-5-5`, `gemini-4-argon`) comes after that, and
   `claude-sonnet-5-5` needs the user's explicit go-ahead. `gemini-3.5-flash-lite` is excluded: the user judged it the
   weakest of the three budget models, so it must not consume quota.

## Highest-value additions if time permits

- **A pairwise "same plant, three policies" figure** (frontier zero-shot, frontier with tools, our trained 4B) on one held-out engine/steam-ORC case, with the verifier verdict beside each answer. The schema in `paper/assets/examples_schema.json` already accommodates it; it needs the three answers from the eval notebook.
- **A significance figure**: effect size with Holm-corrected intervals for RQ1--RQ8, instead of a table of stars. The data are already in the per-example outputs of `model_eval`.
- **A cost--quality frontier plot**: success or sample efficiency against recorded cost per solved row for every arm. This is the clearest Q1-style systems contribution and uses data we already record.
- **A reward-ablation curve** over GRPO steps (with/without site, minimality and grounding channels), using the per-rollout log. This turns RQ5 from a table into a training-dynamics result.
- **An explicit acceptance-rate curve** for teacher answers against model cost, showing that verifier gating, not model identity, determines corpus quality.
- **A held-out-diagram failure taxonomy** (which component or boundary the model misses on `engine-*-ext*` and `cgam_steam_orc`), counted from eval examples; this supports RQ3 qualitatively.

## Paper mechanics already handled

- Generated values carry source/command/seed and are checked by `verify_paper.py`; no undefined `\val` keys.
- Figure assets and their data sources are declared in `paper/figures/figure_manifest.json`; `fig_trace_walkthrough` is regenerated from examples rather than drawn by hand.
- The abstract, introduction, experiments, limitations and conclusion have been rewritten to state contributions, preregistered questions, status and threats to validity without claiming unrun results.
