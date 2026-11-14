#!/usr/bin/env bash
# Generate the full CPU dataset set for thermo-agent (runbook section 2).
# Safe to re-run: generators are seeded and write per-task files.
set -euo pipefail

cd /mnt/c/Users/rezax/Home/Code/thermo_agent/Repo
PY=.venv/Scripts/python.exe
mkdir -p reports data/processed

stamp() { date -Is; }

echo "[$(stamp)] 1/5 repair dataset (n=4000)"
$PY -m thermo_agent.synth.build_repair_data data/processed -n 4000 --seed 0 --workers 8

echo "[$(stamp)] 2/5 audit dataset (n=800)"
$PY -m thermo_agent.audit.data data/processed -n 800 --seed 0 --workers 8

echo "[$(stamp)] 3/5 design dataset (n=1500, max-evals 60)"
$PY -m thermo_agent.design.data data/processed -n 1500 --seed 0 --workers 8 --max-evals 60

echo "[$(stamp)] 4/5 ledger dataset (n=3000)"
$PY -m thermo_agent.ledger.data data/processed -n 3000 --seed 0 --workers 8

echo "[$(stamp)] 5/5 data report"
$PY -m thermo_agent.eval.data_report data/processed > reports/data_report.json
cat reports/data_report.json

echo "[$(stamp)] done"
