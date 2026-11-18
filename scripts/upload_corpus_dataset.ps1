$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
Set-Location $root
$dir = (Resolve-Path 'data/kaggle_datasets/thermo-agent-corpus' -ErrorAction SilentlyContinue)
if (-not $dir) { $dir = New-Item -ItemType Directory -Force -Path 'data/kaggle_datasets/thermo-agent-corpus' }
$dir = [string]$dir
foreach ($name in @('sft','grpo','sft_all','grpo_all','heldout')) {
    $zip = Join-Path $dir "$name.zip"
    # Always rebuild heldout.zip so a regenerated fresh-seed test set is never stale.
    if (-not (Test-Path $zip) -or $name -eq 'heldout') {
        if (Test-Path "data/$name") {
            Remove-Item -Force $zip -ErrorAction SilentlyContinue
            Compress-Archive -Path "data/$name/*" -DestinationPath $zip -CompressionLevel Optimal
        }
    }
    Get-Item $zip -ErrorAction SilentlyContinue | Select-Object Name, Length
}
Set-Content (Join-Path $dir 'dataset-metadata.json') -Encoding utf8 -Value '{
 "title": "thermo-agent training corpora",
 "id": "rezanadimi/thermo-agent-corpus",
 "licenses": [{"name": "MIT"}],
 "files": [{"name": "sft.zip", "description": "original SFT corpus"}, {"name": "grpo.zip", "description": "original GRPO prompts"}, {"name": "sft_all.zip", "description": "SFT corpus plus expert-scale cycle rows"}, {"name": "grpo_all.zip", "description": "GRPO prompts plus expert-scale cycle rows"}, {"name": "heldout.zip", "description": "fresh-seed (987654) held-out test set no cohort has seen; all arms scored here"}]
}'
$env:PYTHONIOENCODING = 'utf-8'
$env:HTTP_PROXY = 'socks5h://localhost:10808'
$env:HTTPS_PROXY = 'socks5h://localhost:10808'
$env:KAGGLE_API_TOKEN = (& .venv/Scripts/python.exe -c "from pathlib import Path; print([l.split('=',1)[1].strip() for l in Path('.env').read_text().splitlines() if l.startswith('KAGGLE_API_TOKEN=')][0])")
$exists = & .venv/Scripts/kaggle.exe datasets list -s thermo-agent-corpus 2>&1 | Select-String 'thermo-agent-corpus'
if ($exists) {
    & .venv/Scripts/kaggle.exe datasets version -p "$dir" -m "add merged corpora with expert-scale cycles" -t -r skip
} else {
    & .venv/Scripts/kaggle.exe datasets create -p "$dir" -t -r skip
}
