# Hosted LangGraph + real-tools teacher pass, one Kaggle Benchmarks task, one model at a time.
#
#   pwsh -File scripts/run_teacher_hosted.ps1 -DryRun                 # safe default: print the full plan
#   pwsh -File scripts/run_teacher_hosted.ps1 -Execute -Limit 200 -Rounds 3
#
# Dry-run is the default unless -Execute is passed. No push, no run, no download, no ingest
# and no API call happens in a dry run. Tokens are read from the environment or .env and
# are never printed. Each model's journal is APPENDED to its own shard, so no model can
# overwrite another model's evidence.
<#
.SYNOPSIS
    Push thermo-teacher-tools-s00, run it against each hosted model, download and ingest
    that model's journal, then print campaign stats.
.DESCRIPTION
    Kaggle Benchmarks exposes the big slugs only to hosted tasks (-m <slug>), never to the
    local SDK. This driver uses the exact `tasks push/run/download` flags from --help.
#>
param(
    [string[]]$Models = @('gpt-6-luna', 'claude-haiku-5-5'),
    [string]$Task = 'thermo-teacher-tools-s00',
    [int]$Limit = $(if ($env:HARNESS_LIMIT) { [int]$env:HARNESS_LIMIT } else { 0 }),
    # 0 = let the hosted task use agent_graph.DEFAULT_MAX_ROUNDS (currently 4).
    [int]$Rounds = $(if ($env:HARNESS_ROUNDS) { [int]$env:HARNESS_ROUNDS } else { 0 }),
    [switch]$DryRun,
    [switch]$Execute,
    [int]$Wait = 5400
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$env:PYTHONIOENCODING = 'utf-8'
$env:HTTP_PROXY = 'socks5h://localhost:10808'
$env:HTTPS_PROXY = 'socks5h://localhost:10808'

$k = '.venv/Scripts/kaggle.exe'
$py = '.venv/Scripts/python.exe'
$taskFile = 'kaggle/teacher_harness_task_s00.py'
$dsCode = 'rezanadimi/thermo-agent-code'
$dsPrompts = 'rezanadimi/thermo-teacher-prompts-big'
$runsDir = 'data/teacher_runs'
$downloadRoot = Join-Path $runsDir 'kaggle_hosted'
$statsScript = 'scripts/teacher_campaign_stats.py'
# The hosted task writes /kaggle/working/harness_completions.jsonl; accept the older
# completions.jsonl name as well. Download preserves output filenames under -o <dir>.
$journalNames = @('harness_completions.jsonl', 'completions.jsonl')

function Get-Secret([string]$name) {
    $value = [Environment]::GetEnvironmentVariable($name, 'Process')
    if (-not [string]::IsNullOrWhiteSpace($value)) { return $value }
    $envFile = Join-Path $root '.env'
    if (Test-Path $envFile) {
        foreach ($line in [System.IO.File]::ReadLines($envFile)) {
            $t = $line.TrimStart()
            if ($t.StartsWith($name + '=')) {
                $v = $t.Substring($name.Length + 1).Trim().Trim([char]34, [char]39)
                if (-not [string]::IsNullOrWhiteSpace($v)) { return $v }
            }
        }
    }
    return $null
}

function Show-Cmd([string[]]$argv) {
    return (($argv | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }) -join ' ')
}

function ConvertTo-Slug([string]$value) {
    return (($value -replace '[^A-Za-z0-9._-]', '-').Trim('-'))
}

# PREFLIGHT: refuse before any command runs. The token value is never echoed.
$token = Get-Secret 'KAGGLE_API_TOKEN'
if ([string]::IsNullOrWhiteSpace($token)) {
    throw 'preflight failed: KAGGLE_API_TOKEN is missing (set it in the environment or .env; value not shown)'
}
$env:KAGGLE_API_TOKEN = $token
Remove-Variable token -ErrorAction SilentlyContinue
Write-Output '[preflight] KAGGLE_API_TOKEN present (value hidden); HTTP(S)_PROXY set to socks5h://localhost:10808'

if (-not (Test-Path $taskFile)) { throw "task file not found: $taskFile" }
if (-not (Test-Path $k)) { throw "kaggle CLI not found: $k" }
if (-not (Test-Path $py)) { throw "python not found: $py" }

# pwsh -File passes a comma/space separated -Models value as one string.
$selected = @($Models | ForEach-Object { $_ -split '[,;\s]+' } | Where-Object { $_ } | Select-Object -Unique)
if ($selected.Count -eq 0) { throw '-Models must name at least one model' }

$isDry = $DryRun -or (-not $Execute)
$state = if ($isDry) { 'DRY RUN' } else { 'EXECUTE' }
Write-Output "=== hosted teacher [$state] ==="
Write-Output ("[task] {0}  [file] {1}  [wait/model] {2}s  [limit] {3}  [rounds] {4}" -f `
    $Task, $taskFile, $Wait, $(if ($Limit -gt 0) { $Limit } else { 'task default (200)' }), `
    $(if ($Rounds -gt 0) { $Rounds } else { 'task default (4 = agent_graph.DEFAULT_MAX_ROUNDS)' }))
for ($i = 0; $i -lt $selected.Count; $i++) {
    Write-Output ("[model {0}/{1}] {2}" -f ($i + 1), $selected.Count, $selected[$i])
}
if ($Limit -gt 0) { Write-Output "[env] HARNESS_LIMIT=$Limit (set before each run)" }
if ($Rounds -gt 0) { Write-Output "[env] HARNESS_ROUNDS=$Rounds (set before each run)" }

# --------------------------------------------------------------------------- #
# push (exact flags from `kaggle benchmarks tasks push --help`)
# --------------------------------------------------------------------------- #
$pushArgs = @('benchmarks', 'tasks', 'push', $Task, '-f', $taskFile,
              '-d', $dsCode, '-d', $dsPrompts)
Write-Output "PUSH: $(Show-Cmd (@($k) + $pushArgs))"
if (-not $isDry) {
    & $k @pushArgs 2>&1 | Out-String -Width 200 | Write-Output
    if ($LASTEXITCODE -ne 0) { throw "push failed for task $Task" }
}

foreach ($model in $selected) {
    $slug = ConvertTo-Slug $model
    $dir = Join-Path $downloadRoot $slug
    $target = Join-Path $runsDir "hosted_${slug}.jsonl"
    $targetAbs = Join-Path $root $target

    $runArgs = @('benchmarks', 'tasks', 'run', $Task, '-m', $model, '--wait', "$Wait")
    $downloadArgs = @('benchmarks', 'tasks', 'download', $Task, '-m', $model, '-o', $dir)
    $ingestArgs = @('-m', 'thermo_agent.teacher.transfer', 'ingest',
                    'data/frontier_export', $target, 'data/frontier',
                    '--memory-dir', 'data/memory', '--model', $model)

    Write-Output "--- $model (journal: $target) ---"
    Write-Output "RUN: $(Show-Cmd (@($k) + $runArgs))"
    Write-Output "DOWNLOAD: $(Show-Cmd (@($k) + $downloadArgs))"
    Write-Output "JOURNAL: find $($journalNames -join '|') under $dir (fallback: scan $downloadRoot for raw '$model' or path slug '$slug')"
    Write-Output "APPEND: <downloaded journal for $model> -> $target (id-deduped, never overwrite another model)"
    Write-Output "INGEST: $(Show-Cmd (@($py) + $ingestArgs))"

    if ($isDry) { continue }

    if ($Limit -gt 0) { $env:HARNESS_LIMIT = "$Limit" }
    if ($Rounds -gt 0) { $env:HARNESS_ROUNDS = "$Rounds" }

    & $k @runArgs 2>&1 | Out-String -Width 200 | Write-Output
    if ($LASTEXITCODE -ne 0) { throw "run failed for $model" }

    New-Item -ItemType Directory -Force $dir | Out-Null
    & $k @downloadArgs 2>&1 | Out-String -Width 200 | Write-Output
    if ($LASTEXITCODE -ne 0) { throw "download failed for $model" }

    # `tasks download -m <model> -o <dir>`: -m selects the run, -o only names the output
    # directory; the journal filename itself is model-independent. Prefer the per-model
    # -o directory, then fall back to a path scan that tolerates a slug containing '/'.
    $journal = Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $journalNames -contains $_.Name } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $journal) {
        $rawModel = $model.ToLowerInvariant().Replace('\', '/').Trim('/')
        $slugKey = '/' + $slug.ToLowerInvariant() + '/'
        $journal = Get-ChildItem $downloadRoot -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object {
                ($journalNames -contains $_.Name) -and (
                    ($_.FullName.ToLowerInvariant().Replace('\', '/').Contains($slugKey)) -or
                    ($_.FullName.ToLowerInvariant().Replace('\', '/').Contains($rawModel))
                )
            } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
    }
    if (-not $journal) {
        Write-Output "no journal for $model under $dir (or $downloadRoot); skipping its append/ingest"
        continue
    }

    New-Item -ItemType Directory -Force (Split-Path $targetAbs -Parent) | Out-Null
    $existing = @{}
    if (Test-Path $targetAbs) {
        foreach ($line in [System.IO.File]::ReadLines($targetAbs)) {
            $m = [regex]::Match($line, '"id"\s*:\s*"([^"]+)"')
            if ($m.Success) { $existing[$m.Groups[1].Value] = $true }
        }
    }
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    $added = 0
    foreach ($line in [System.IO.File]::ReadLines($journal.FullName)) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $m = [regex]::Match($line, '"id"\s*:\s*"([^"]+)"')
        $rid = if ($m.Success) { $m.Groups[1].Value } else { '' }
        if ($rid -and $existing.ContainsKey($rid)) { continue }
        [System.IO.File]::AppendAllText($targetAbs, $line + "`n", $utf8)
        if ($rid) { $existing[$rid] = $true }
        $added++
    }
    Write-Output "appended $added new line(s) from $($journal.FullName) to $target"

    & $py @ingestArgs 2>&1 | Out-String -Width 200 | Write-Output
    if ($LASTEXITCODE -ne 0) { throw "ingest failed for $model (journal kept: $target)" }
}

if (Test-Path $statsScript) {
    Write-Output "STATS: $(Show-Cmd (@($py, $statsScript)))"
    if (-not $isDry) {
        & $py $statsScript 2>&1 | Out-String -Width 200 | Write-Output
        if ($LASTEXITCODE -ne 0) { Write-Output '[stats] stats script failed; continuing' }
    }
} else {
    Write-Output "[stats] $statsScript not found yet; continuing"
}

if ($isDry) {
    Write-Output "=== dry run complete: $($selected.Count) models planned, 0 network calls, 0 writes ==="
} else {
    Write-Output "=== execute complete: $($selected.Count) models; journals in $runsDir ==="
}
