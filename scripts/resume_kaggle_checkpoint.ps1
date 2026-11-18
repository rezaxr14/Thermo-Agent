# Relay the newest checkpoint from a Kaggle kernel output into the private resume dataset.
#
#   pwsh scripts/resume_kaggle_checkpoint.ps1 -Stage sft -Kernel rezanadimi/thermo-train-run
#
# Run this after a training session finishes and BEFORE the next session starts. It handles
# only LoRA weights + optimizer/RNG state: no credentials, no corpus, no logs.
param(
    [string]$Stage = 'sft',
    [string]$Kernel = 'rezanadimi/thermo-train-run'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:PYTHONIOENCODING = 'utf-8'
$env:HTTP_PROXY = 'socks5h://localhost:10808'
$env:HTTPS_PROXY = 'socks5h://localhost:10808'
$env:KAGGLE_API_TOKEN = (& .venv/Scripts/python.exe -c "from pathlib import Path; print([l.split('=',1)[1].strip() for l in Path('.env').read_text().splitlines() if l.startswith('KAGGLE_API_TOKEN=')][0])")

$meta = 'data/kaggle_datasets/thermo-agent-resume'
New-Item -ItemType Directory -Force $meta | Out-Null
$metadata = Join-Path $meta 'dataset-metadata.json'
if (-not (Test-Path $metadata)) {
    Set-Content $metadata -Encoding utf8 -Value '{
 "title": "thermo-agent resume checkpoints",
 "id": "rezanadimi/thermo-agent-resume",
 "licenses": [{"name": "MIT"}]
}'
}

$out = "runs/kaggle_outputs/$Stage"
New-Item -ItemType Directory -Force $out | Out-Null
$pattern = "thermo-agent/runs/$Stage/checkpoint-[0-9]+/"
& .venv/Scripts/kaggle.exe kernels output $Kernel -p (Resolve-Path $out).Path -o -q --file-pattern $pattern 2>&1 |
    Tee-Object -FilePath "logs/resume_download_$Stage.log"
if ($LASTEXITCODE -ne 0) { throw "kernel output download failed ($LASTEXITCODE)" }

$src = Join-Path $out "thermo-agent/runs/$Stage"
$ckpt = @(Get-ChildItem -Path $src -Directory -Filter 'checkpoint-*' -ErrorAction SilentlyContinue |
          Where-Object { $_.Name.Substring(11) -match '^\d+$' } |
          Sort-Object { [int]$_.Name.Substring(11) } -Descending | Select-Object -First 1)
if ($ckpt.Count -eq 0) { throw "no checkpoint for $Stage in $src" }
$ckpt = $ckpt[0]
$stateFile = Join-Path $ckpt.FullName 'trainer_state.json'
if (-not (Test-Path $stateFile)) { throw "incomplete checkpoint: no trainer_state.json" }
$null = Get-Content $stateFile -Raw -Encoding utf8 | ConvertFrom-Json
if (-not ((Test-Path (Join-Path $ckpt.FullName 'optimizer.pt')) -and
          ((Test-Path (Join-Path $ckpt.FullName 'adapter_model.safetensors')) -or
           (Test-Path (Join-Path $ckpt.FullName 'adapter_model.bin'))))) {
    throw "incomplete checkpoint: missing optimizer or adapter"
}

$step = [int]$ckpt.Name.Substring(11)
$zip = Join-Path $meta "$Stage.zip"
Remove-Item -Force $zip -ErrorAction SilentlyContinue
Compress-Archive -Path $ckpt.FullName -DestinationPath $zip -CompressionLevel Optimal
$exists = & .venv/Scripts/kaggle.exe datasets list -s thermo-agent-resume 2>&1 | Select-String 'thermo-agent-resume'
if ($exists) {
    & .venv/Scripts/kaggle.exe datasets version -p (Resolve-Path $meta).Path -m "$Stage step $step" -t -r skip
} else {
    & .venv/Scripts/kaggle.exe datasets create -p (Resolve-Path $meta).Path -t -r skip
}
if ($LASTEXITCODE -ne 0) { throw "dataset publish failed ($LASTEXITCODE)" }
Write-Output "published $Stage $step to rezanadimi/thermo-agent-resume"
