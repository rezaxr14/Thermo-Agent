# Generate the CPU datasets for thermo-agent (runbook section 2). Resumable: a step whose output exists is skipped.
# Design uses 14 workers on this 16-core machine; max-evals 60 is kept because the runbook requires reference
# designs built with 60 or more evaluations.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path 'reports', 'data/processed' | Out-Null

function Step([string]$msg) { Write-Output ("[{0}] {1}" -f (Get-Date -Format o), $msg) }

$repairDone = Test-Path 'data/processed/repair_train.jsonl'
$auditDone = Test-Path 'data/processed/audit_train.jsonl'
$ledgerDone = Test-Path 'data/processed/ledger_train.jsonl'
$designDone = Test-Path 'data/processed/design_train.jsonl'

if ($repairDone) { Step '1/5 repair dataset: exists, skip' } else {
    Step '1/5 repair dataset (n=4000)'
    & $py -m thermo_agent.synth.build_repair_data data/processed -n 4000 --seed 0 --workers 8
    if ($LASTEXITCODE -ne 0) { throw "repair dataset failed with exit code $LASTEXITCODE" }
}

if ($auditDone) { Step '2/5 audit dataset: exists, skip' } else {
    Step '2/5 audit dataset (n=800)'
    & $py -m thermo_agent.audit.data data/processed -n 800 --seed 0 --workers 8
    if ($LASTEXITCODE -ne 0) { throw "audit dataset failed with exit code $LASTEXITCODE" }
}

if ($ledgerDone) { Step '3/5 ledger dataset: exists, skip' } else {
    Step '3/5 ledger dataset (n=3000)'
    & $py -m thermo_agent.ledger.data data/processed -n 3000 --seed 0 --workers 8
    if ($LASTEXITCODE -ne 0) { throw "ledger dataset failed with exit code $LASTEXITCODE" }
}

if ($designDone) { Step '4/5 design dataset: exists, skip' } else {
    Step '4/5 design dataset (n=1500, max-evals 60, workers 14)'
    & $py -m thermo_agent.design.data data/processed -n 1500 --seed 0 --workers 14 --max-evals 60
    if ($LASTEXITCODE -ne 0) { throw "design dataset failed with exit code $LASTEXITCODE" }
}

Step '5/5 data report'
& $py -m thermo_agent.eval.data_report data/processed | Tee-Object -FilePath 'reports/data_report.json'
if ($LASTEXITCODE -ne 0) { throw "data report failed with exit code $LASTEXITCODE" }

Step 'done'
