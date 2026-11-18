# Hosted teacher pass, round 2 (user-selected cheap models, reasoning ON).
# Models: gemini-3.8-flash (v3 default, reasoning ON), claude-haiku-5-5,
# gemini-3.5-flash-lite, gpt-6-luna. One additive journal per model.
# A Kaggle task runs ONE model at a time, so we wait for completion between runs.
# Claude Sonnet 5.5 is deliberately deferred (user: "do it later" -- remind the user).
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:PYTHONIOENCODING = 'utf-8'
$env:HTTP_PROXY = 'socks5h://localhost:10808'
$env:HTTPS_PROXY = 'socks5h://localhost:10808'
$env:KAGGLE_API_TOKEN = (& .venv/Scripts/python.exe -c "from pathlib import Path; print([l.split('=',1)[1].strip() for l in Path('.env').read_text().splitlines() if l.startswith('KAGGLE_API_TOKEN=')][0])")
$k = '.venv/Scripts/kaggle.exe'
$py = '.venv/Scripts/python.exe'
$task = 'thermo-teacher-big-s00'

function Wait-Task([int]$minutes) {
    $deadline = (Get-Date).AddMinutes($minutes)
    while ((Get-Date) -lt $deadline) {
        $out = & $k benchmarks tasks status $task 2>&1 | Out-String
        if ($out -match 'Status:\s+Completed') { Write-Output 'task completed'; return }
        if ($out -match 'Status:\s+Failed|Status:\s+Errored') { throw "task failed: $out" }
        Start-Sleep -Seconds 45
    }
    throw "task did not complete within $minutes minutes"
}

function Save-Ingest([string]$model, [string]$tag) {
    $dest = (Resolve-Path 'data/teacher_runs').Path
    $dir = Join-Path $dest 'kaggle_hosted'
    & $k benchmarks tasks download $task -m $model -o $dir -f 2>&1 | Out-String -Width 200 | Write-Output
    $journal = Get-ChildItem $dir -Recurse -Filter 'completions.jsonl' |
        Where-Object { $_.FullName -like "*$model*" } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $journal) { Write-Output "no journal for $model; skipping"; return }
    $target = Join-Path $dest "completions_big_${tag}_$model.jsonl"
    Copy-Item -Force $journal.FullName $target
    $slug = (& $py -c "import json,sys; r=json.loads(open(sys.argv[1],encoding='utf-8').readline()); print(r.get('model',''))" $target).Trim()
    if (-not $slug) { $slug = $model }
    Write-Output "journal $target model=$slug"
    & $py -m thermo_agent.teacher.transfer ingest data/frontier_export $target data/frontier --memory-dir data/memory --model $slug 2>&1 |
        Tee-Object -FilePath "logs/ingest_big_${tag}_$model.json" | Out-String -Width 200 | Write-Output
    & $k benchmarks quota --format json 2>&1 | Out-String -Width 200 | Write-Output
}

# The push auto-started gemini-3.8-flash on v3 (reasoning ON). Capture it first.
Write-Output '=== waiting for the v3 default run (gemini-3.8-flash, reasoning ON) ==='
Wait-Task 150
Save-Ingest 'gemini-3.8-flash' 'v3'

foreach ($model in @('claude-haiku-5-5', 'gemini-3.5-flash-lite', 'gpt-6-luna')) {
    Write-Output "=== run $model ==="
    & $k benchmarks tasks run $task -m $model --wait 5400 2>&1 | Out-String -Width 200 | Write-Output
    Save-Ingest $model 'v2'
}
Write-Output 'big teacher pass round 2 done'
