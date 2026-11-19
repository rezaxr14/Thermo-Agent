# Orchestrate the tool-using CoT teacher campaign: run the BUDGET tier first, expensive tier opt-in.
#
#   pwsh scripts/run_teacher_campaign.ps1 -DryRun                       # safe default: print exact commands
#   pwsh scripts/run_teacher_campaign.ps1 -Execute -Rows 200 -Resume
#   pwsh scripts/run_teacher_campaign.ps1 -Tier expensive -AllowExpensive -Execute -Rows 200
#
# Each model writes its own append-only shard under data/teacher_runs/, so no model overwrites another.
# Dry-run is the default: pass -Execute to make any API call or write. Tokens are never printed.
<#
.SYNOPSIS
    Runs scripts/kaggle_harness_cli.py once per model, then ingests each shard with transfer.ingest.
.DESCRIPTION
    Default model order is the budget trio (gpt-6-luna, claude-haiku-5-5, gemini-3.5-flash-lite).
    The expensive tier is selected only with -Tier expensive, and execution additionally requires
    -AllowExpensive so budget quota must be consciously signed off before it is spent.
#>
param(
    [string[]]$Models = @(),
    [ValidateSet('budget', 'expensive')][string]$Tier = 'budget',
    [int]$Rows = 200,
    [string]$Task = 'all',
    [switch]$DryRun,
    [switch]$Execute,
    [switch]$Resume,
    [switch]$AllowExpensive,
    [string]$Split = 'train',
    [string]$ExportDir = 'data/frontier_export',
    [string]$OutDir = 'data/teacher_runs'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$py = if (Test-Path '.venv/Scripts/python.exe') { '.venv/Scripts/python.exe' } else { 'python' }

# model order is deliberate: cheapest first, so quota goes furthest before any expensive call.
# These are the exact keys the Kaggle Benchmarks SDK exposes to this account
# (`kbench.llms`; check with `python -c "import kaggle_benchmarks as k; print(sorted(k.llms))"`).
# The hosted-task catalog is larger (luna/haiku-5-5/gemini-3.5-flash-lite), but the SDK does NOT
# expose those slugs: `kbench.llms.get` falls back to the account default, so the driver now
# refuses a miss instead of mislabelling provenance.
$budget = @('google/gemini-3.1-flash-lite-preview', 'openai/gpt-5.4-nano-2026-03-17')
$expensive = @('google/gemini-3-flash-preview', 'anthropic/claude-sonnet-5@default',
               'qwen/qwen3-next-80b-a3b-instruct')
$expensiveSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$expensive)
$tasks = @('repair', 'ledger', 'audit', 'design')

# Explicit -Models wins; otherwise -Tier picks the default list.
if ($PSBoundParameters.ContainsKey('Models') -and $Models.Count -gt 0) {
    # pwsh -File passes a comma/space separated list as one string; split it into model ids.
    $selected = @($Models | ForEach-Object { $_ -split '[,;\s]+' } | Where-Object { $_ } | Select-Object -Unique)
} elseif ($Tier -eq 'expensive') {
    $selected = @($expensive)
} else {
    $selected = @($budget)
}

if ($Rows -lt 1) {
    throw '-Rows must be >= 1 (kaggle_harness_cli.py slices [:limit], so 0 generates no rows)'
}
if ($Task -ne 'all' -and $tasks -notcontains $Task) {
    throw "-Task must be 'all' or one of: $($tasks -join ', ')"
}

# Budget models always run before expensive ones, even in an explicit mixed -Models list.
$budgetFirst = @($selected | Where-Object { $budget -contains $_ })
$others = @($selected | Where-Object { $budget -notcontains $_ })
if ($others.Count -gt 0 -and $budgetFirst.Count -gt 0) {
    Write-Output '[order] budget models moved ahead of expensive models'
}
$selected = @($budgetFirst) + @($others)

$isDry = $DryRun -or (-not $Execute)

function Test-Secret([string]$name) {
    $v = [Environment]::GetEnvironmentVariable($name, 'Process')
    if (-not [string]::IsNullOrWhiteSpace($v)) { return $true }
    $envFile = Join-Path $root '.env'
    if (Test-Path $envFile) {
        foreach ($line in [System.IO.File]::ReadLines($envFile)) {
            $t = $line.TrimStart()
            if ($t.StartsWith($name + '=')) {
                $val = $t.Substring($name.Length + 1).Trim().Trim([char]34, [char]39)
                if (-not [string]::IsNullOrWhiteSpace($val)) { return $true }
            }
        }
    }
    return $false
}

# PREFLIGHT: refuse before any command runs, and never echo a token value.
foreach ($name in @('KAGGLE_API_TOKEN', 'HF_TOKEN')) {
    if (-not (Test-Secret $name)) {
        throw "preflight failed: $name is missing (set it in the environment or .env; value not shown)"
    }
    Write-Output "[preflight] $name present (value hidden)"
}

$usesExpensive = @($selected | Where-Object { $expensiveSet.Contains($_) }).Count -gt 0
if ($usesExpensive -and -not $AllowExpensive) {
    if ($isDry) {
        Write-Output '[guard] WARNING: expensive tier selected; -AllowExpensive is required to execute'
    } else {
        throw 'expensive tier requires -AllowExpensive: exhaust/confirm the budget tier first'
    }
}

function ConvertTo-ShardSlug([string]$value) {
    return (($value -replace '[^A-Za-z0-9._-]', '-').Trim('-'))
}
function Show-Cmd([string[]]$argv) {
    return (($argv | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }) -join ' ')
}

$state = if ($isDry) { 'DRY RUN' } else { 'EXECUTE' }
Write-Output "=== teacher campaign [$state] ==="
Write-Output ("[tier] {0}  [rows/model] {1}  [task] {2}  [split] {3}  [resume] {4}" -f `
    $Tier, $Rows, $Task, $Split, $(if ($Resume) { 'on' } else { 'off' }))
for ($i = 0; $i -lt $selected.Count; $i++) {
    Write-Output ("[model {0}/{1}] {2}" -f ($i + 1), $selected.Count, $selected[$i])
}

# -Resume narrows the driver's export to ids NOT already accepted into data/frontier.
# The snapshot is built once, before the model loop, so every model sees the same pending rows.
$driverExport = $ExportDir
$filterCode = @'
import json,pathlib,sys; exp=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); split=sys.argv[3]; done={str(json.loads(l).get('record_id') or json.loads(l).get('id')).split('#',1)[0] for p in pathlib.Path('data/frontier').glob('frontier_*_'+split+'.jsonl') for l in p.open(encoding='utf-8') if l.strip()}; out.mkdir(parents=True,exist_ok=True); rows={p.name:[json.loads(l) for l in p.open(encoding='utf-8') if l.strip()] for p in exp.glob('*'+split+'.jsonl')}; kept={name:[r for r in rs if str(r.get('id')).split('#',1)[0] not in done] for name,rs in rows.items()}; [(out/name).write_text(''.join(json.dumps(r)+'\n' for r in rs),encoding='utf-8') for name,rs in kept.items()]; print('pending rows',sum(map(len,kept.values())),'already-ingested ids',len(done))
'@
if ($Resume) {
    $driverExport = Join-Path $OutDir '_campaign_pending_export'
    $filterArgs = @($py, '-c', ($filterCode -replace '\r?\n$', ''), $ExportDir, $driverExport, $Split)
    Write-Output "[resume] export filter: $(Show-Cmd $filterArgs)"
    if ($isDry) {
        Write-Output "[resume] DRY RUN: no pending export written; driver would use $driverExport"
    } else {
        & $py -c $filterCode $ExportDir $driverExport $Split
        if ($LASTEXITCODE -ne 0) { throw 'failed to build the pending export (already-ingested ids)' }
    }
}

$taskSlug = if ($Task -eq 'all') { 'all' } else { ConvertTo-ShardSlug $Task }

foreach ($model in $selected) {
    $slug = ConvertTo-ShardSlug $model
    $shard = Join-Path $OutDir "harness_${slug}_${taskSlug}.jsonl"
    Write-Output "--- $model (shard: $shard) ---"

    # The driver's quota guard (daily reserve) and per-row try/except live inside kaggle_harness_cli.py;
    # it appends to this shard and skips ids already present with a usable completion.
    $driverArgs = @('scripts/kaggle_harness_cli.py',
        '--out', $shard,
        '--export-dir', $driverExport,
        '--split', $Split,
        '--limit', "$Rows",
        '--model', $model)
    Write-Output "DRIVER: $(Show-Cmd (@($py) + $driverArgs))"

    # transfer.ingest has no --source flag: validate.teacher_row sets source='frontier_tools'
    # automatically when a shard record carries trajectory + sft_completion (the driver writes both).
    # --model is the supported provenance flag: it tags each accepted row id and the ingest report.
    $ingestArgs = @('-m', 'thermo_agent.teacher.transfer', 'ingest',
        $ExportDir, $shard, 'data/frontier',
        '--memory-dir', 'data/memory',
        '--model', $model)
    if ($Task -ne 'all') { $ingestArgs += @('--tasks', $Task) }
    Write-Output "INGEST: $(Show-Cmd (@($py) + $ingestArgs))"
    Write-Output "SOURCE: source=frontier_tools is derived from the shard trajectory (no --source flag); --model preserves provenance"

    if (-not $isDry) {
        & $py @driverArgs 2>&1 | Out-String -Width 200 | Write-Output
        if ($LASTEXITCODE -ne 0) { throw "driver failed for $model; shard left intact for resume: $shard" }
        New-Item -ItemType Directory -Force 'logs' | Out-Null
        $report = Join-Path 'logs' "teacher_campaign_ingest_${slug}_${taskSlug}.json"
        & $py @ingestArgs 2>&1 |
            Tee-Object -FilePath $report | Out-String -Width 200 | Write-Output
        if ($LASTEXITCODE -ne 0) { throw "ingest failed for $model (report: $report)" }
    }
}

if ($isDry) {
    Write-Output "=== dry run complete: $($selected.Count) models, 0 rows generated, 0 API calls, 0 writes ==="
} else {
    Write-Output "=== execute complete: $($selected.Count) models; shards in $OutDir ==="
    Write-Output "[verify] & $py -c `"import json,pathlib,collections; print(collections.Counter(json.loads(l).get('model','?') for p in pathlib.Path('data/teacher_runs').glob('harness_*.jsonl') for l in p.open(encoding='utf-8') if l.strip()))`""
}
