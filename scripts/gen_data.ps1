# Generate the CPU datasets for thermo-agent (runbook section 2). Seeded, safe to re-run.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path 'reports', 'data/processed' | Out-Null

function Step([string]$msg) { Write-Output ("[{0}] {1}" -f (Get-Date -Format o), $msg) }
function Run([string]$label, [string[]]$Args) {
    Step $label
    & $py @Args
    if ($LASTEXITCODE -ne 0) { throw "$label failed with exit code $LASTEXITCODE" }
}

Run '1/5 repair dataset (n=4000)' @('-m', 'thermo_agent.synth.build_repair_data', 'data/processed', '-n', '4000', '--seed', '0', '--workers', '8')
Run '2/5 audit dataset (n=800)' @('-m', 'thermo_agent.audit.data', 'data/processed', '-n', '800', '--seed', '0', '--workers', '8')
Run '3/5 design dataset (n=1500, max-evals 60)' @('-m', 'thermo_agent.design.data', 'data/processed', '-n', '1500', '--seed', '0', '--workers', '8', '--max-evals', '60')
Run '4/5 ledger dataset (n=3000)' @('-m', 'thermo_agent.ledger.data', 'data/processed', '-n', '3000', '--seed', '0', '--workers', '8')

Step '5/5 data report'
& $py -m thermo_agent.eval.data_report data/processed | Tee-Object -FilePath 'reports/data_report.json'
if ($LASTEXITCODE -ne 0) { throw "data report failed with exit code $LASTEXITCODE" }

Step 'done'
