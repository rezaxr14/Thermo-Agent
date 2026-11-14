# Paper outline

**Working title:** Second-Law-Verified Language-Model Agents for On-Site Power, Cooling and Heat Recovery in AI Data Centers
(alternatives: "Teaching a 4B Language Model Thermodynamics: Exergy-Verified Reinforcement Learning for Data-Centre Energy Systems"; "Exergy as Reward: ...")

**Scope statement (repeat in abstract, methods, limitations).** Simulation-driven: measured load and coolant traces set the operating point of a CoolProp / NASA-7 plant model. No measured on-site-generation data centre is used. Design-point steady state; traces enter through a trace-weighted fuel evaluation with ideal part-load (plant re-sized to each load bin), a lower bound; modular gensets (N identical units at rated load, dispatch by ceiling) bound the part-load loss from the other side; no real part-load curves, no start-up or ramp limits.

## 1. Problem
AI data centres are adding on-site gas generation, and the waste heat (low grade: coolant 18-25 C supply, 25-40 C return in the measured Frontier system) can drive absorption cooling, heat pumps or bottoming cycles. Choosing the layout, keeping the plant model physically consistent, and finding where exergy is lost are expert tasks. General LLMs are unreliable at multi-step thermodynamics, and a plant model can look plausible while violating the second law.

## 2. What is built
- Plant simulator: schema-typed graph of components (compressor, turbine, combustor, gas engine, recuperator, HRSG, ORC evaporator/turbine, pump, condenser, IT load, absorption chiller, electric chiller, heat pump, dry cooler / cooling tower). Ideal gases from NASA-7, water and R245fa from CoolProp, exergy against an explicit dead state (ambient air has zero exergy).
- Verifiers (also the reward): mass and energy balance, exergy closure, non-negative destruction per component, temperature cross / pinch, Carnot-type bounds, unit and range checks, site requirements, number grounding of reasoning text.
- Tasks: Repair (built), Ledger repair (built: fix equation-level faults in the exergy ledger, which range checks cannot see), Audit (built: rank the exergy destroyers, propose one allowed set-point change, predict its fuel effect at constant services), Design (built: prime mover, cooling, heat rejection, ORC bottoming with three working fluids, district heat from exhaust or heat pump, plus continuous parameters; fuel-minimising for fixed services). Model: Qwen3.5-4B, LoRA SFT then GRPO. Harness ablation: AGENTS.md and skills, including a held-out new component.

## 3. Contributions (claim only what the experiments support)
1. A second-law-verified environment for data-centre energy systems whose checks double as a dense, verifiable reward.
2. A Repair benchmark with five parameter-level fault types plus a ledger-repair benchmark with thirteen equation-level fault types (unit slip, sign flip, efficiency above 1, swapped ports, wrong fluid) and structural out-of-distribution splits: unseen fault types and unseen topologies / components (steam-ORC plant, gas engine).
3. A training recipe and evaluation in solver calls per valid plant against search baselines.
4. Design-space findings from the simulator (for example, exhaust-only recovery limits of gas engines vs turbines for absorption cooling).
Not claimed: beating optimisers on the final optimum for a fixed superstructure.

## 4. Experiments
| Question | Metric | Compared with |
|---|---|---|
| Does SFT+GRPO repair faults? | valid-plant rate, exact match (deducible faults) | no-op, prompted Qwen3.5-4B, verifier-guided search |
| Is it more sample-efficient? | Repair: solver calls per valid plant (model: 1-3; search measured: 8-45 numeric, 220-280 structural). Design: share within 2 % of the best known design after b design evaluations (anytime protocol, `design/curves.py`) | search / optimiser; enumeration, random search, enumerate-then-polish |
| Are the differences real? | paired exact McNemar (binary) or sign-flip permutation (other), Holm-corrected, same examples (`eval/compare.py`) | |
| Does it generalise? | success on test_ood_fault and test_ood_topology | in-distribution |
| Does the reward need all channels? | ablate validity / site / minimality / grounding | full reward |
| Does a harness help a held-out component? | success with vs without AGENTS.md and SKILL.md (the gas-engine skill is the only description of the held-out component; its ports and parameters are generated from the code) | `--harness none` vs `agents+skills` in `eval/model_eval.py` |
| Is SFT needed before GRPO? | reward curve, zero-variance group fraction | GRPO from base |
| Encoder-decoder alternative | same metrics | decoder-only (ablation only) |
Design baselines (small-sample measurement on held-out hot-climate and high-chilled-fraction sites; the catalog now has 32 configurations, 64 when the site has a heat demand, and 10.7 are valid per site on average): enumerate the defaults 43-48 evaluations, 0.7-1.2 % from the best known design; enumerate + Nelder-Mead polish about 125-133 evaluations (this is the reference); random search 12 evaluations, 7-10 % worse and sometimes invalid. A model must choose a valid, near-best configuration in 1-3 evaluations to be worth reporting. Reference designs for training must be built with a full polish budget (--max-evals 60 or more).
Audit baselines (small sample, 47 plants; ranking is scored on components other than the combustor / gas engine, which was the largest destroyer in every plant tested): first-law ranking (blame the largest heat dissipation) hits the true top-1 in 0 of 10 in-distribution plants and 19 % of held-out ones, against 29-31 % for random order, i.e. first-law analysis points at the heat rejector while the exergy loss is elsewhere. A fixed rule "raise turbine inlet temperature 10 %" recovers 94 % of the best allowed gain on turbine plants (valid in 6 of 10) but only 33 % on held-out engine plants (valid in 25 of 37). Exhaustive search over the allowed changes costs about 12 evaluations; a model needs 1. The prediction channel scores relative error against the actual effect (floor 0.2 % of fuel); trivial predictors (small sample, 37 plants): predicting no effect scores 0.00 to 0.12, predicting the average relative effect scores 0.45 to 0.52. The best allowed change saves a median 1.7 % of fuel (range 0.1 to 5.2 %).
Ledger baselines (400 generated rows): no-op solves none; a rule that rebuilds the ledger from the topology solves all by construction. Second-law symptom coverage of the injected faults: closure residual non-zero in 100 percent of component-line faults but only 10 to 31 percent of stream-line faults; negative ED in 28 to 89 percent; either symptom in 86 percent of rows overall; parameter-range checks see none. Metric for models: all-lines-correct rate with Wilson interval, by fault type, on test_ood_fault and test_ood_topology.
Also report ThermoQA before and after training to check for forgetting.

## 5. Datasets
| Dataset | Role | Status |
|---|---|---|
| Frontier energy dataset (Sun et al. 2024, figshare) | IT load level, coolant supply/return temperatures, heat-capture ratio -> Site specs and dc_chp settings | verified; loader written (explicit column mapping); file not yet downloaded |
| ThermoQA (Duzkar 2026, HF olivenet/thermoqa) | LLM thermodynamic-reasoning baseline, forgetting check | verified; not yet run |
| ClusterWise (Summit telemetry), SustainDC / DCRL-Green, DCcluster-Opt | load variability, weather and carbon traces for ambient design points | from the project brief; licences and schemas not yet checked |
| ASHRAE RP-1043 chiller data | calibrate the electric-chiller second-law fraction and approach | from the project brief; not yet checked |
| NREL/LBNL HVAC fault data | realism of the fault taxonomy | optional |
Current training data is synthetic (sampled plants plus injected faults); measured data enter through Site specs and calibration.

## 6. Limits and risks
- Absorption chiller is a system-level model (reversible-limit fraction with a first-law cap), calibrated only to the published COP range 0.55-0.70.
- Gas-engine jacket heat is rejected, not recovered; engine recovery limits are therefore conservative.
- Verifier gaming is possible; edits are restricted to existing fields and the site check is in the reward.
- Design outcomes depend on assumed technology constants (engine electrical efficiency 0.43, turbine defaults, chiller COP cap 0.8, heat-pump Carnot fraction 0.5, ORC fluid temperature limits 420/480/560 K), none fitted to data. Engines win most sites partly because of the assumed 0.43. Thermal storage and real part-load curves are not modelled: the trace evaluation brackets part-load between ideal part-load (lower bound) and modular gensets; the modularity penalty is nearly technology-independent in this model. A single-stage ORC cannot use turbine exhaust at about 850 K (fluid limit vs stack floor); steam bottoming would be needed.
- Water use is an upper bound (all rejected heat evaporated); fan power fraction is a default, not fitted.
- Audit: the prime mover is always the top exergy destroyer here, so ranking is scored without it; the best change is mostly pressure ratio or turbine inlet temperature on turbine plants, which a rule nearly solves in-distribution. Component efficiencies are excluded from allowed changes on purpose. Change effects use re-sized generation for equal net electricity.
- Ledger: the correct form is derivable by a short rule, so a program solves it; the task is a skill and localisation test, and 14 percent of faults show no second-law symptom at all (T0 faults are the most silent). Exergy numbers are given at 4 significant figures in the prompt, so a model cannot check closure by hand.
- No training has been run yet; every learning claim is pending.
