# Package the code for the Kaggle training notebook as a private dataset (`thermo-agent-code`).
#
#   pwsh scripts/make_kaggle_bundle.ps1
#   # upload the resulting kaggle_bundle.zip at https://www.kaggle.com/datasets -> New dataset (private)
#
# The corpus is a separate Kaggle dataset (`thermo-agent-corpus`) holding data/sft and data/grpo. Neither bundle
# contains the .env file, the HF token, ground-truth data or checkpoints.
#
# data/memory carries the success/failure notes (D10/D11) so a HOSTED run serves the same retrieval memory as
# the local driver. It holds notes only: no ground-truth answers, no tokens, and the compact renderer strips
# model ids/timestamps/cost before anything reaches a model.
#
# Build in a staging directory: `Compress-Archive -Path data/memory` would strip the `data` parent and land the
# notes at `memory/`, where the hosted task does not look. Staging preserves the real relative paths.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

$zip = 'kaggle_bundle.zip'
Remove-Item -Force $zip -ErrorAction SilentlyContinue

$stage = Join-Path ([System.IO.Path]::GetTempPath()) ("thermo-bundle-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $stage | Out-Null
try {
    # Whole trees, copied as-is.
    foreach ($part in @('thermo_agent', 'kaggle', 'scripts')) {
        if (Test-Path $part) { Copy-Item -Recurse -Force $part -Destination $stage }
    }
    # Single files.
    foreach ($part in @('pyproject.toml', 'requirements-lock.txt', 'README.md', 'AGENTS.md')) {
        if (Test-Path $part) { Copy-Item -Force $part -Destination $stage }
    }
    # Retrieval memory must keep its `data/` prefix, or the hosted task cannot find it.
    if (Test-Path 'data/memory') {
        New-Item -ItemType Directory -Path (Join-Path $stage 'data') -Force | Out-Null
        Copy-Item -Recurse -Force 'data/memory' -Destination (Join-Path $stage 'data')
    }

    # Byte-compiled caches are host-specific, stale and pure bloat.
    Get-ChildItem -Path $stage -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force
    Get-ChildItem -Path $stage -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in @('.pyc', '.pyo') } |
        Remove-Item -Force

    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal
}
finally {
    Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
}

Get-Item $zip | Select-Object Name, Length, LastWriteTime
