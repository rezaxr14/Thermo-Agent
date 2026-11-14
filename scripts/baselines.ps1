# Non-model baselines (runbook 3) and GRPO reward diagnostics (3b). No GPU needed.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path 'reports' | Out-Null

function Step([string]$msg) { Write-Output ("[{0}] {1}" -f (Get-Date -Format o), $msg) }

function Capture([string]$label, [string]$outFile, [string[]]$Args) {
    Step $label
    & $py @Args 2>&1 | Tee-Object -FilePath $outFile
    if ($LASTEXITCODE -ne 0) { throw "$label failed with exit code $LASTEXITCODE" }
}

Capture 'repair baselines: test_id' 'reports/baseline_repair_test_id.txt' `
    @('-m', 'thermo_agent.eval.run_eval', 'data/processed', '--split', 'test_id', '--limit', '40', '--by-fault')
Capture 'repair baselines: test_ood_fault' 'reports/baseline_repair_test_ood_fault.txt' `
    @('-m', 'thermo_agent.eval.run_eval', 'data/processed', '--split', 'test_ood_fault', '--limit', '40', '--by-fault')
Capture 'repair baselines: test_ood_topology' 'reports/baseline_repair_test_ood_topology.txt' `
    @('-m', 'thermo_agent.eval.run_eval', 'data/processed', '--split', 'test_ood_topology', '--limit', '40', '--by-fault')

Capture 'design baselines: test_ood_climate' 'reports/baseline_design_test_ood_climate.txt' `
    @('-m', 'thermo_agent.design.run_eval', 'data/processed', '--split', 'test_ood_climate', '--limit', '10')

Capture 'audit baselines: test_ood_topology' 'reports/baseline_audit_test_ood_topology.txt' `
    @('-m', 'thermo_agent.audit.run_eval', 'data/processed', '--split', 'test_ood_topology', '--limit', '40')

Capture 'ledger baselines: test_ood_fault' 'reports/baseline_ledger_test_ood_fault.txt' `
    @('-m', 'thermo_agent.ledger.run_eval', 'data/processed', '--split', 'test_ood_fault', '--limit', '200')

Capture 'reward diagnostics' 'reports/reward_diagnostics.txt' `
    @('-m', 'thermo_agent.train.reward_diagnostics', 'data/processed', '--groups', '12', '--out', 'reports/reward_diagnostics.json')

Step 'done'
