# Point the teacher sampler at a Kaggle-hosted endpoint (see kaggle/README.md).
#   pwsh scripts/kaggle_teacher.ps1 -Url https://<random>.trycloudflare.com -Samples 2 -Limit 50
# Sampling is resumable: accepted ids already on disk are skipped, so a dropped tunnel costs only in-flight rows.
param(
    [string]$Url = $env:TEACHER_BASE_URL,
    [string]$Model = $(if ($env:TEACHER_MODEL) { $env:TEACHER_MODEL } else { 'teacher' }),
    [int]$Samples = 2,
    [int]$Limit = 0,
    [string]$Tasks = 'repair ledger audit design',
    [string]$Mode = 'agents+skills'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path 'data/frontier', 'data/memory', 'reports' | Out-Null

function Step([string]$msg) { Write-Output ("[{0}] {1}" -f (Get-Date -Format o), $msg) }

if ([string]::IsNullOrWhiteSpace($Url)) { throw 'pass -Url or set TEACHER_BASE_URL' }

Step "probing $Url"
& $py -m thermo_agent.teacher.kaggle --probe $Url
if ($LASTEXITCODE -ne 0) { throw "teacher endpoint not reachable: $Url" }

$extra = @()
if ($Limit -gt 0) { $extra += @('--limit', "$Limit") }
$extra += @('--extra-data', 'data/processed_arch')

foreach ($task in $Tasks.Split(' ') | Where-Object { $_ }) {
    Step "teacher sample: $task (samples=$Samples limit=$Limit mode=$Mode)"
    & $py -m thermo_agent.teacher.sample data/processed data/frontier `
        --task $task --samples $Samples --base-url $Url --model $Model --mode $Mode `
        --memory data/memory --k-memory 3 @extra |
        Tee-Object -FilePath "reports/teacher_$task.txt"
    if ($LASTEXITCODE -ne 0) { throw "teacher sampling failed for $task ($LASTEXITCODE)" }
}

Step 'done'
