#!/usr/bin/env bash
# Non-model baselines (runbook section 3) + GRPO reward diagnostics (3b).
# These are the numbers a trained model has to beat, and they need no GPU.
set -euo pipefail

cd /mnt/c/Users/rezax/Home/Code/thermo_agent/Repo
PY=.venv/Scripts/python.exe
mkdir -p reports

step() { echo "[$(date -Is)] $*"; }

step "repair baselines, in-distribution"
$PY -m thermo_agent.eval.run_eval data/processed --split test_id --limit 40 --by-fault \
    > reports/baseline_repair_test_id.txt

step "repair baselines, unseen fault types"
$PY -m thermo_agent.eval.run_eval data/processed --split test_ood_fault --limit 40 --by-fault \
    > reports/baseline_repair_test_ood_fault.txt

step "repair baselines, unseen topologies"
$PY -m thermo_agent.eval.run_eval data/processed --split test_ood_topology --limit 40 --by-fault \
    > reports/baseline_repair_test_ood_topology.txt

step "design baselines (enumeration / enumerate+polish / random)"
$PY -m thermo_agent.design.run_eval data/processed --split test_ood_climate --limit 10 \
    > reports/baseline_design_test_ood_climate.txt

step "audit baselines"
$PY -m thermo_agent.audit.run_eval data/processed --split test_ood_topology --limit 40 \
    > reports/baseline_audit_test_ood_topology.txt

step "ledger baselines (symptom coverage)"
$PY -m thermo_agent.ledger.run_eval data/processed --split test_ood_fault --limit 200 \
    > reports/baseline_ledger_test_ood_fault.txt

step "GRPO reward diagnostics"
$PY -m thermo_agent.train.reward_diagnostics data/processed --groups 12 \
    --out reports/reward_diagnostics.json > reports/reward_diagnostics.txt

step "done"
