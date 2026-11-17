# One command for the CPU part of the final hand-off: optional teacher ingest -> corpus -> paper assets -> PDF.
#
#   pwsh scripts/finalize.ps1                                   # no teacher yet: rebuild corpus and paper from ground truth
#   pwsh scripts/finalize.ps1 -Completions completions.jsonl -Model anthropic/claude-sonnet-5-5@default
param(
    [string]$Completions = '',
    [string]$Model = '',
    [int]$Variants = 20
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = if (Test-Path '.venv/Scripts/python.exe') { '.venv/Scripts/python.exe' } else { 'python' }
function Step([string]$msg) { "[{0:o}] {1}" -f (Get-Date), $msg }

if ($Completions) {
    if (-not $Model) { throw '-Model is required with -Completions' }
    Step "ingest teacher completions from $Completions"
    & $py -m thermo_agent.teacher.transfer ingest data/frontier_export $Completions data/frontier `
        --memory-dir data/memory --model $Model
    if ($LASTEXITCODE -ne 0) { throw 'teacher ingest failed' }
}

Step "build both corpora (variants=$Variants)"
& ./scripts/build_corpus.ps1 -Variants $Variants
if ($LASTEXITCODE -ne 0) { throw 'corpus build failed' }

Step 'regenerate paper assets'
& $py -m pytest -q 2>&1 | Tee-Object -FilePath logs/pytest_final.log | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'tests failed' }
& $py paper/tools/generate_paper_assets.py --tests-log logs/pytest_final.log --force
if ($LASTEXITCODE -ne 0) { throw 'paper asset generation failed' }

Step 'build main.pdf'
Push-Location paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex > $null 2>&1
if ($LASTEXITCODE -ne 0) { throw 'pdflatex 1 failed' }
bibtex main > $null 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex > $null 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex > $null 2>&1
if ($LASTEXITCODE -ne 0) { throw 'pdflatex final failed' }
Pop-Location
Step 'done: review paper/main.pdf and git status'
