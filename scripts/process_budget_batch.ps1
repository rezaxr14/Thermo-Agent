# Gate, ingest and audit one downloaded luna/haiku hosted batch. No Kaggle calls.
param(
  [Parameter(Mandatory=$true)][string]$Model,
  [Parameter(Mandatory=$true)][string]$Journal,
  [Parameter(Mandatory=$true)][string]$RunId,
  [string]$OutDir = ""
)
if ($Model -notin @('gpt-6-luna','claude-haiku-5-5')) {
  throw "budget-only batch: model must be gpt-6-luna or claude-haiku-5-5"
}
if (-not $OutDir) { $OutDir = "data/frontier_tools_cohort/$Model" }
$journalPath = (Resolve-Path $Journal).Path
$dir = Split-Path -Parent $journalPath
$gated = Join-Path $dir "gated.jsonl"
$tokemanifest = Join-Path $dir "tokengate_manifest.json"
$env:PYTHONIOENCODING = 'utf-8'
& .venv/Scripts/python.exe scripts/frontier_cohort_tokengate.py journal $Model $journalPath $gated $tokemanifest
if ($LASTEXITCODE -ne 0) { throw "token gate failed for $Model" }
& .venv/Scripts/python.exe -m thermo_agent.teacher.transfer.ingest data/processed $gated $OutDir `
  --model $Model --run-id $RunId --memory-dir data/memory `
  --extra-export-dir data/processed_arch --extra-export-dir data/frontier_prov --extra-export-dir data/a_slice
if ($LASTEXITCODE -ne 0) { throw "ingest failed for $Model" }
& .venv/Scripts/python.exe scripts/check_datagen_complete.py
