# Non-model baselines (runbook 3) and GRPO reward diagnostics (3b). No GPU needed.
# Called directly, with output tee'd to reports/, so each run leaves a record.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path 'reports' | Out-Null

function Step([string]$msg) { Write-Output ("[{0}] {1}" -f (Get-Date -Format o), $msg) }

Step 'repair baselines: test_id'
& $py -m thermo_agent.eval.run_eval data/processed --split test_id --limit 40 --by-fault 2>&1 |
    Tee-Object -FilePath 'reports/baseline_repair_test_id.txt'
if ($LASTEXITCODE -ne 0) { throw "repair test_id failed ($LASTEXITCODE)" }

Step 'repair baselines: test_ood_fault'
& $py -m thermo_agent.eval.run_eval data/processed --split test_ood_fault --limit 40 --by-fault 2>&1 |
    Tee-Object -FilePath 'reports/baseline_repair_test_ood_fault.txt'
if ($LASTEXITCODE -ne 0) { throw "repair test_ood_fault failed ($LASTEXITCODE)" }

Step 'repair baselines: test_ood_topology'
& $py -m thermo_agent.eval.run_eval data/processed --split test_ood_topology --limit 40 --by-fault 2>&1 |
    Tee-Object -FilePath 'reports/baseline_repair_test_ood_topology.txt'
if ($LASTEXITCODE -ne 0) { throw "repair test_ood_topology failed ($LASTEXITCODE)" }

Step 'design baselines: test_ood_climate'
& $py -m thermo_agent.design.run_eval data/processed --split test_ood_climate --limit 10 2>&1 |
    Tee-Object -FilePath 'reports/baseline_design_test_ood_climate.txt'
if ($LASTEXITCODE -ne 0) { throw "design baseline failed ($LASTEXITCODE)" }

Step 'audit baselines: test_ood_topology'
& $py -m thermo_agent.audit.run_eval data/processed --split test_ood_topology --limit 40 2>&1 |
    Tee-Object -FilePath 'reports/baseline_audit_test_ood_topology.txt'
if ($LASTEXITCODE -ne 0) { throw "audit baseline failed ($LASTEXITCODE)" }

Step 'ledger baselines: test_ood_fault'
& $py -m thermo_agent.ledger.run_eval data/processed --split test_ood_fault --limit 200 2>&1 |
    Tee-Object -FilePath 'reports/baseline_ledger_test_ood_fault.txt'
if ($LASTEXITCODE -ne 0) { throw "ledger baseline failed ($LASTEXITCODE)" }

Step 'reward diagnostics'
& $py -m thermo_agent.train.reward_diagnostics data/processed --groups 12 --out reports/reward_diagnostics.json 2>&1 |
    Tee-Object -FilePath 'reports/reward_diagnostics.txt'
if ($LASTEXITCODE -ne 0) { throw "reward diagnostics failed ($LASTEXITCODE)" }

Step 'done'
