# Method justification (draft text for Methods and Related Work)

**Scope statement.** All results are simulation-driven: real load traces drive a CoolProp/NASA-7 plant model. No measured CHP data centre is used.

## 1. Why a trained small model with a physics verifier
- Frontier LLMs are strong on property lookups but degrade on multi-step cycle analysis (ThermoQA, Düzkar 2026; single-author preprint). Prompted tool-use agents on gas-turbine problems needed large models; smaller ones struggled with tool use and parameter extraction (Song et al. 2024).
- Prompted multi-agent flowsheet generation exists (Tian et al., AAAI 2026) but uses a frozen model. RL for cycle structure exists (Li et al. 2026) but uses a learned property surrogate and no language model.
- Closest methodological precedent: SFT followed by GRPO against a solver-grounded reward for automated modelling of nuclear thermal-hydraulic systems (Luo et al. 2026: solver-convergence rate 72.4 % after SFT, 88.8 % after GRPO on their 50-case set), and RL with solver verifiers for optimisation modelling (SIRL, NeurIPS 2025). Our contribution relative to these: second-law verifiers (exergy closure, non-negative destruction, temperature-cross) as the reward, a data-centre site layer, and a fault-injection Repair benchmark with out-of-distribution splits.

## 2. Role of the AI (what the model decides)
The language model formulates and diagnoses: it reads a site spec or a failing plant plus verifier output and emits structured edits (Repair), a plant layout and parameters (Design), or an irreversibility ranking with a proposed change and predicted effect (Audit). The solver and verifiers are the environment and the reward; a local optimiser may polish continuous parameters after the model fixes structure. We do not claim to beat optimisation on the final optimum for a fixed superstructure; the claim is sample efficiency (solver calls per valid plant) and breadth (formulation, new topologies, repair).

## 3. Model and training recipe
- Base: Qwen3.5-4B, dense, Apache-2.0 (licence stated on derivative model cards; confirm on the upstream card before release).
- Stage 1, SFT on verified traces. Stage 2, GRPO (Shao et al. 2024) with verifiable rewards (Lambert et al. 2024). RLVR needs a policy that already succeeds sometimes, which is the reason for SFT first.
- LoRA on all linear layers; RL needs little adapter capacity and LoRA's best learning rate is about 10x the full-fine-tuning rate (Schulman et al. 2025). This keeps training on a 24 GB GPU plausible.
- Reward channels (Repair): format, validity (apply edits, solve, verify), site satisfaction, minimality, exact match when the fault is deducible, number grounding.

## 4. Encoder-decoder alternative
Adapting decoder-only models to encoder-decoder improves quality at similar inference cost, especially after instruction tuning (Zhang et al. 2025). Repair is single-turn structured-to-structured, which is the regime where this could help. The RL tooling we rely on (TRL GRPO, vLLM) targets decoder-only models, so this is an ablation, not the main system.

## 5. Data and grounding
- Frontier measurements (Sun et al. 2024): supply 18-25 C, return 25-40 C, waste heat 5-25 MW; the dataset authors propose heat pumps for upgrading it. Our cooling-loop defaults sit in this band.
- Absorption chiller: measured single-effect LiBr/H2O COP 0.55-0.70 (Ochoa Villa et al. 2019). Our chiller is a system-level model (reversible-limit fraction plus a first-law cap), not a LiBr property model.
- On-site CHP for data centres: Darrow and Hedman (ORNL 2009). Vendor figures are sanity targets only.

## 6. Known weak spots
1. The three current fault injectors are caught mostly by range checks; harder in-range faults are needed before Repair can carry a paper.
2. Chiller COP sits at its first-law cap at an 18 C chilled supply; the reversible-fraction parameters are defaults, not fitted.
3. Verifier gaming (reward hacking) is possible; edits are restricted to existing numeric fields and the site check is part of the reward.
4. Qwen3.5-4B tool-use ability is unmeasured here; no training has been run in this environment.
