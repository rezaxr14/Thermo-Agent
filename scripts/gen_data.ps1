# Generate every CPU dataset for thermo-agent, then mine failure memory and combine the SFT corpus.
#
# Everything is driven by the environment, so the same script scales up or down without editing:
#   THERMO_N_REPAIR   default 12000      THERMO_N_AUDIT   default 3000
#   THERMO_N_LEDGER   default 9000       THERMO_N_DESIGN  default 4000
#   THERMO_N_ARCH     default 3000 (requested rows; about 70% survive detection)
#   THERMO_MAX_EVALS  default 60         THERMO_SEED      default 0
#   THERMO_WORKERS    default: auto      THERMO_N_MEMORY  default 1000
#   THERMO_VARIANTS   default 20         THERMO_SKIP      comma list of steps to skip
#
# Auto worker count leaves two cores for the OS and the SSH session, and never asks for more RAM than is free
# (about 300 MB per worker). Resumable: a step is skipped when its meta_*.json already records n >= target.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path 'reports', 'data/processed', 'data/processed_arch', 'data/memory' | Out-Null

function Step([string]$msg) { Write-Output ("[{0}] {1}" -f (Get-Date -Format o), $msg) }
function EnvInt([string]$name, [int]$default) {
    $raw = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrWhiteSpace($raw)) { return $default }
    return [int]$raw
}
function Get-Workers([int]$memPerWorkerMB = 300) {
    $override = [Environment]::GetEnvironmentVariable('THERMO_WORKERS')
    if (-not [string]::IsNullOrWhiteSpace($override)) { return [Math]::Max(1, [int]$override) }
    $cpu = [Environment]::ProcessorCount
    $freeMB = [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
    $byCpu = [Math]::Max(1, $cpu - 2)
    $byMem = [Math]::Max(1, [int]($freeMB / $memPerWorkerMB))
    return [Math]::Min($byCpu, $byMem)
}
function MetaOf([string]$dir, [string]$task) {
    $p = Join-Path $dir "meta_$task.json"
    if (-not (Test-Path $p)) { return $null }
    return Get-Content $p -Raw | ConvertFrom-Json
}
function HasN([string]$dir, [string]$task, [int]$n) {
    $m = MetaOf $dir $task
    if ($null -eq $m) { return $false }
    return ([int]$m.params.n -ge $n)
}
function HasDesign([int]$n, [int]$evals) {
    $m = MetaOf 'data/processed' 'design'
    if ($null -eq $m) { return $false }
    return ([int]$m.params.n -ge $n) -and ([int]$m.params.max_evals -ge $evals)
}

$nRepair = EnvInt 'THERMO_N_REPAIR' 12000
$nAudit  = EnvInt 'THERMO_N_AUDIT' 3000
$nLedger = EnvInt 'THERMO_N_LEDGER' 9000
$nDesign = EnvInt 'THERMO_N_DESIGN' 4000
$nArch   = EnvInt 'THERMO_N_ARCH' 3000
$nArchAudit  = EnvInt 'THERMO_N_ARCH_AUDIT' 2000
$nArchLedger = EnvInt 'THERMO_N_ARCH_LEDGER' 3000
$nMemory = EnvInt 'THERMO_N_MEMORY' 1000
$maxEvals = EnvInt 'THERMO_MAX_EVALS' 60
$seed = EnvInt 'THERMO_SEED' 0
$variants = EnvInt 'THERMO_VARIANTS' 20
$workers = Get-Workers
$skip = @()
if ($env:THERMO_SKIP) { $skip = $env:THERMO_SKIP.Split(',') | ForEach-Object { $_.Trim() } }

Step ("workers={0} (cpu={1}) repair={2} audit={3} ledger={4} design={5} arch={6}/{7}/{8} max_evals={9}" -f `
    $workers, [Environment]::ProcessorCount, $nRepair, $nAudit, $nLedger, $nDesign, $nArch, $nArchAudit, $nArchLedger, $maxEvals)

if ('repair' -notin $skip) {
    if (HasN 'data/processed' 'repair' $nRepair) { Step "repair: meta already has n>=$nRepair, skip" } else {
        Step "repair dataset (n=$nRepair, workers=$workers)"
        & $py -m thermo_agent.synth.build_repair_data data/processed -n $nRepair --seed $seed --workers $workers
        if ($LASTEXITCODE -ne 0) { throw "repair dataset failed ($LASTEXITCODE)" }
    }
}
if ('audit' -notin $skip) {
    if (HasN 'data/processed' 'audit' $nAudit) { Step "audit: meta already has n>=$nAudit, skip" } else {
        Step "audit dataset (n=$nAudit, workers=$workers)"
        & $py -m thermo_agent.audit.data data/processed -n $nAudit --seed $seed --workers $workers
        if ($LASTEXITCODE -ne 0) { throw "audit dataset failed ($LASTEXITCODE)" }
    }
}
if ('ledger' -notin $skip) {
    if (HasN 'data/processed' 'ledger' $nLedger) { Step "ledger: meta already has n>=$nLedger, skip" } else {
        Step "ledger dataset (n=$nLedger, workers=$workers)"
        & $py -m thermo_agent.ledger.data data/processed -n $nLedger --seed $seed --workers $workers
        if ($LASTEXITCODE -ne 0) { throw "ledger dataset failed ($LASTEXITCODE)" }
    }
}
if ('design' -notin $skip) {
    if (HasDesign $nDesign $maxEvals) { Step "design: meta already has n>=$nDesign and max_evals>=$maxEvals, skip" } else {
        Step "design dataset (n=$nDesign, max-evals=$maxEvals, workers=$workers)"
        & $py -m thermo_agent.design.data data/processed -n $nDesign --seed $seed --workers $workers --max-evals $maxEvals
        if ($LASTEXITCODE -ne 0) { throw "design dataset failed ($LASTEXITCODE)" }
    }
}
if ('arch' -notin $skip) {
    if (HasN 'data/processed_arch' 'repair_arch' $nArch) { Step "architectures: repair meta already has n>=$nArch, skip" } else {
        Step "architecture repair dataset (n=$nArch over 65 diagrams, workers=$workers)"
        & $py -m thermo_agent.synth.architectures data/processed_arch -n $nArch --seed $seed --workers $workers
        if ($LASTEXITCODE -ne 0) { throw "architecture dataset failed ($LASTEXITCODE)" }
    }
    if (HasN 'data/processed_arch' 'audit_arch' $nArchAudit) { Step "architectures: audit meta already has n>=$nArchAudit, skip" } else {
        Step "architecture audit dataset (n=$nArchAudit over 65 diagrams, workers=$workers)"
        & $py -m thermo_agent.audit.data data/processed_arch -n $nArchAudit --seed $seed --workers $workers --architectures
        if ($LASTEXITCODE -ne 0) { throw "architecture audit dataset failed ($LASTEXITCODE)" }
    }
    if (HasN 'data/processed_arch' 'ledger_arch' $nArchLedger) { Step "architectures: ledger meta already has n>=$nArchLedger, skip" } else {
        Step "architecture ledger dataset (n=$nArchLedger over 65 diagrams, workers=$workers)"
        & $py -m thermo_agent.ledger.data data/processed_arch -n $nArchLedger --seed $seed --workers $workers --architectures
        if ($LASTEXITCODE -ne 0) { throw "architecture ledger dataset failed ($LASTEXITCODE)" }
    }
}
if ('memory' -notin $skip) {
    Step "hindsight memory (mine $nMemory repair rows)"
    & $py -m thermo_agent.teacher.validate --mine data/processed --out data/memory --limit $nMemory
    if ($LASTEXITCODE -ne 0) { throw "memory mining failed ($LASTEXITCODE)" }
}
if ('report' -notin $skip) {
    Step 'data report'
    & $py -m thermo_agent.eval.data_report data/processed | Tee-Object -FilePath 'reports/data_report.json'
    if ($LASTEXITCODE -ne 0) { throw "data report failed ($LASTEXITCODE)" }
}
if ('combine' -notin $skip) {
    Step "build both corpora, token audit and swap (scripts/build_corpus.ps1 -Variants $variants)"
    & ./scripts/build_corpus.ps1 -Variants $variants
    if ($LASTEXITCODE -ne 0) { throw "build_corpus failed ($LASTEXITCODE)" }
}

Step 'done'
