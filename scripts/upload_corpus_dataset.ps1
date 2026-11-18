$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $MyInvocation.MyCommand.Path -Parent)
$dir = (Resolve-Path 'data/kaggle_datasets/thermo-agent-corpus' -ErrorAction SilentlyContinue)
if (-not $dir) { $dir = New-Item -ItemType Directory -Force -Path 'data/kaggle_datasets/thermo-agent-corpus' }
$dir = $dir.FullName
foreach ($name in @('sft','grpo')) {
    $zip = Join-Path $dir "$name.zip"
    if (-not (Test-Path $zip)) {
        Compress-Archive -Path "data/$name/*" -DestinationPath $zip -CompressionLevel Optimal
    }
    Get-Item $zip | Select-Object Name, Length
}
Set-Content (Join-Path $dir 'dataset-metadata.json') -Encoding utf8 -Value '{
 "title": "thermo-agent training corpora",
 "id": "rezanadimi/thermo-agent-corpus",
 "licenses": [{"name": "MIT"}],
 "files": [{"name": "sft.zip", "description": "SFT corpus with 20 paraphrases per ground-truth row"}, {"name": "grpo.zip", "description": "GRPO prompts, one per plant"}]
}'
$env:KAGGLE_API_TOKEN = (& .venv/Scripts/python.exe -c "from pathlib import Path; print([l.split('=',1)[1].strip() for l in Path('.env').read_text().splitlines() if l.startswith('KAGGLE_API_TOKEN=')][0])")
& .venv/Scripts/kaggle.exe datasets create -p "$dir" -t -r skip
