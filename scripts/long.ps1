#Requires -Version 5.1
<#
.SYNOPSIS
  Start and inspect long-running jobs. Jobs are created through WMI (Win32_Process.Create), so their parent is WmiPrvSE,
  not the SSH session: they keep running after the connection closes, which Start-Process children do not.

.EXAMPLE
  pwsh scripts/long.ps1 start gen-data "scripts/gen_data.ps1"

.EXAMPLE
  pwsh scripts/long.ps1 status gen-data -Lines 40

.EXAMPLE
  pwsh scripts/long.ps1 list
#>
param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateSet('start', 'list', 'tail', 'kill', 'logs', 'status')]
    [string]$Action,

    [Parameter(Position = 1)]
    [string]$Name = 'work',

    [Parameter(Position = 2)]
    [string]$Command,

    [int]$Lines = 60
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Get-JobInfo([string]$jobName) {
    $pidFile = Join-Path $logs "$jobName.pid"
    if (-not (Test-Path $pidFile)) { return $null }
    $id = (Get-Content $pidFile -Raw).Trim()
    $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
    [pscustomobject]@{
        Name  = $jobName
        Pid   = [int]$id
        Alive = [bool]$proc
        Cpu   = if ($proc) { [math]::Round($proc.CPU, 1) } else { $null }
        Log   = (Join-Path $logs "$jobName.log")
    }
}

switch ($Action) {
    'start' {
        if (-not $Command) { throw 'Provide a script to run, e.g. start gen-data "scripts/gen_data.ps1"' }
        $jobScript = Join-Path $logs "$Name.ps1"
        $logPath = Join-Path $logs "$Name.log"
        $body = @"
`$ErrorActionPreference = 'Continue'
Set-Location '$root'
`$env:Path = "`$env:USERPROFILE\.local\bin;`$env:USERPROFILE\tools\bin;`$env:Path"
Start-Transcript -Path '$logPath' -Append -Force | Out-Null
Write-Output "=== START `$(Get-Date -Format o) [$Name] ==="
try {
$Command
} catch {
    Write-Output "JOB ERROR: `$_"
}
Write-Output "=== EXIT=`$LASTEXITCODE `$(Get-Date -Format o) [$Name] ==="
Stop-Transcript | Out-Null
"@
        Set-Content -Path $jobScript -Value $body -Encoding utf8

        $cmdLine = 'pwsh -NoProfile -ExecutionPolicy Bypass -File "{0}"' -f $jobScript
        $res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmdLine } -ErrorAction Stop
        if ($res.ReturnValue -ne 0) { throw "Win32_Process.Create failed: return=$($res.ReturnValue)" }
        Set-Content -Path (Join-Path $logs "$Name.pid") -Value $res.ProcessId
        Write-Output "started '$Name' pid=$($res.ProcessId) (detached)"
        Write-Output "log: $logPath"
    }

    'status' {
        $info = Get-JobInfo $Name
        if (-not $info) { Write-Output "no job '$Name'"; return }
        if ($info.Alive) { Write-Output "'$Name' RUNNING pid=$($info.Pid) cpu_s=$($info.Cpu)" }
        else { Write-Output "'$Name' FINISHED (or died) pid=$($info.Pid)" }
        if (Test-Path $info.Log) { Write-Output '--- last lines ---'; Get-Content $info.Log -Tail $Lines }
    }

    'list' {
        $rows = Get-ChildItem $logs -Filter *.pid -ErrorAction SilentlyContinue | ForEach-Object { Get-JobInfo $_.BaseName }
        if (-not $rows) { Write-Output 'no jobs recorded' } else { $rows | Format-Table -AutoSize | Out-String -Width 120 }
    }

    'tail' {
        $log = Join-Path $logs "$Name.log"
        if (Test-Path $log) { Get-Content $log -Tail $Lines } else { Write-Output "no log: $log" }
    }

    'logs' {
        Get-ChildItem $logs -Filter *.log -ErrorAction SilentlyContinue |
            Select-Object Name, @{n = 'KB'; e = { [math]::Round($_.Length / 1KB, 1) } }, LastWriteTime |
            Format-Table -AutoSize | Out-String -Width 120
    }

    'kill' {
        $info = Get-JobInfo $Name
        if ($info -and $info.Alive) { Stop-Process -Id $info.Pid -Force; Write-Output "killed '$Name' pid=$($info.Pid)" }
        else { Write-Output "'$Name' is not running" }
    }
}
