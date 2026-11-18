# Run the hosted big-model teacher pass additively, then ingest each model's answers.
# The pushed task is already running on the account default; this waits for it, then runs
# the three chosen big models one at a time and keeps one journal per model.
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
        if ($out -match 'Status:\s+Failed') { throw "task failed: $out" }
        Start-Sleep -Seconds 60
    }
    throw "task did not complete within $minutes minutes"
}

Write-Output 'waiting for the default-model run started by push'
Wait-Task 120

$models = @('claude-sonnet-5-5-default', 'gemini-4-argon', 'gpt-6-sol')
foreach ($model in $models) {
    Write-Output "=== run $model ==="
    & $k benchmarks tasks run $task -m $model --wait 5400 2>&1 | Out-String -Width 200 | Write-Output
    $dest = (Resolve-Path 'data/teacher_runs').Path
    $dir = Join-Path $dest 'kaggle_hosted'
    & $k benchmarks tasks download $task -m $model -o $dir -f 2>&1 | Out-String -Width 200 | Write-Output
    $journal = Get-ChildItem $dir -Recurse -Filter 'completions.jsonl' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $journal) { Write-Output "no journal for $model; skipping"; continue }
    $target = Join-Path $dest "completions_big_$model.jsonl"
    Copy-Item -Force $journal.FullName $target
    $slug = (& $py -c "import json,sys; r=json.loads(open(sys.argv[1],encoding='utf-8').readline()); print(r.get('model',''))" $target).Trim()
    if (-not $slug) { $slug = $model }
    Write-Output "journal $target model=$slug"
    & $py -m thermo_agent.teacher.transfer ingest data/frontier_export $target data/frontier --memory-dir data/memory --model $slug 2>&1 |
        Tee-Object -FilePath "logs/ingest_big_$model.json" | Out-String -Width 200 | Write-Output
    & $k benchmarks quota --format json 2>&1 | Out-String -Width 200 | Write-Output
}
Write-Output 'big teacher pass done'
