# Package the code for the Kaggle training notebook as a private dataset (`thermo-agent-code`).
#
#   pwsh scripts/make_kaggle_bundle.ps1
#   # upload the resulting kaggle_bundle.zip at https://www.kaggle.com/datasets -> New dataset (private)
#
# The corpus is a separate Kaggle dataset (`thermo-agent-corpus`) holding data/sft and data/grpo. Neither bundle
# contains the .env file, the HF token, ground-truth data or checkpoints.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

$zip = 'kaggle_bundle.zip'
Remove-Item -Force $zip -ErrorAction SilentlyContinue
$parts = @('thermo_agent', 'kaggle', 'scripts', 'pyproject.toml', 'requirements-lock.txt', 'README.md', 'AGENTS.md')
$parts = $parts | Where-Object { Test-Path $_ }
Compress-Archive -Path $parts -DestinationPath $zip -CompressionLevel Optimal
Get-Item $zip | Select-Object Name, Length, LastWriteTime
