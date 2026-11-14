# Non-model baselines (runbook 3) and GRPO reward diagnostics (3b). No GPU needed.
# Limits are environment-driven so the same script covers a smoke run and the paper numbers:
#   THERMO_EVAL_REPAIR (200)  THERMO_EVAL_DESIGN (30)  THERMO_EVAL_AUDIT (100)  THERMO_EVAL_LEDGER (400)
#   THERMO_DIAG_GROUPS (20)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path 'reports' | Out-Null

function Step([string]$msg) { Write-Output ("[{0}] {1}" -f (Get-Date -Format o), $msg) }
function EnvInt([string]$name, [int]$default) {
    $raw = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrWhiteSpace($raw)) { return $default }
    return [int]$raw
}
$nRepair = EnvInt 'THERMO_EVAL_REPAIR' 200
$nDesign = EnvInt 'THERMO_EVAL_DESIGN' 30
$nAudit  = EnvInt 'THERMO_EVAL_AUDIT' 100
$nLedger = EnvInt 'THERMO_EVAL_LEDGER' 400
$nGroups = EnvInt 'THERMO_DIAG_GROUPS' 20

function Run([string]$name, [string]$module, [string[]]$args) {
    Step $name
    & $py -m $module @args 2>&1 | Tee-Object -FilePath "reports/$name.txt"
    if ($LASTEXITCODE -ne 0) { throw "$name failed ($LASTEXITCODE)" }
}

foreach ($split in @('test_id', 'test_ood_fault', 'test_ood_topology')) {
    Run "baseline_repair_$split" 'thermo_agent.eval.run_eval' @('data/processed', '--split', $split, '--limit', "$nRepair", '--by-fault')
}
foreach ($split in @('test_ood_climate', 'test_ood_load', 'test_ood_scale')) {
    Run "baseline_design_$split" 'thermo_agent.design.run_eval' @('data/processed', '--split', $split, '--limit', "$nDesign")
}
foreach ($split in @('test_ood_topology', 'test_id')) {
    Run "baseline_audit_$split" 'thermo_agent.audit.run_eval' @('data/processed', '--split', $split, '--limit', "$nAudit")
}
foreach ($split in @('test_ood_fault', 'test_ood_topology')) {
    Run "baseline_ledger_$split" 'thermo_agent.ledger.run_eval' @('data/processed', '--split', $split, '--limit', "$nLedger")
}

Step 'reward diagnostics'
& $py -m thermo_agent.train.reward_diagnostics data/processed --groups $nGroups --out reports/reward_diagnostics.json 2>&1 |
    Tee-Object -FilePath 'reports/reward_diagnostics.txt'
if ($LASTEXITCODE -ne 0) { throw "reward diagnostics failed ($LASTEXITCODE)" }

Step 'done'
