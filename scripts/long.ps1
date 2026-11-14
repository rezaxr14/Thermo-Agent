#Requires -Version 5.1
<#
.SYNOPSIS
  Run long jobs in a detached tmux session inside WSL, so they survive SSH disconnects.

.EXAMPLE
  pwsh scripts/long.ps1 start pytest ".venv/Scripts/python.exe -m pytest -q"

.EXAMPLE
  pwsh scripts/long.ps1 tail pytest -Lines 80
#>
param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateSet('start', 'list', 'tail', 'kill', 'logs')]
    [string]$Action,

    [Parameter(Position = 1)]
    [string]$Name = 'work',

    [Parameter(Position = 2)]
    [string]$Command,

    [int]$Lines = 60
)

$ErrorActionPreference = 'Stop'
$distro   = 'Ubuntu-24.04'
$repoWin  = Split-Path -Parent $PSScriptRoot
$drive    = $repoWin.Substring(0, 1).ToLower()
$rest     = $repoWin.Substring(2) -replace '\\', '/'
$repoWsl  = "/mnt/$drive$rest"
$logsWin  = Join-Path $repoWin 'logs'

switch ($Action) {
    'start' {
        if (-not $Command) { throw 'Provide -Command, e.g. start pytest ".venv/Scripts/python.exe -m pytest -q"' }
        New-Item -ItemType Directory -Force -Path $logsWin | Out-Null

        $body = @"
#!/usr/bin/env bash
cd "$repoWsl" || { echo "cannot cd to $repoWsl"; exit 127; }
mkdir -p logs
exec > >(tee -a "logs/$Name.log") 2>&1
echo "=== START `$(date -Is) [$Name] ==="
$Command
rc=`$?
echo "=== EXIT=`$rc `$(date -Is) [$Name] ==="
exit `$rc
"@
        $body = $body -replace "`r`n", "`n"
        $shPath = Join-Path $logsWin "$Name.sh"
        [System.IO.File]::WriteAllText($shPath, $body, (New-Object System.Text.UTF8Encoding($false)))

        & wsl -d $distro -- tmux kill-session -t $Name 2>&1 | Out-Null
        & wsl -d $distro -- tmux new-session -d -s $Name bash "logs/$Name.sh"
        if ($LASTEXITCODE -ne 0) { throw "tmux failed to start session '$Name'" }
        Write-Output "started tmux session '$Name'"
        Write-Output "repo (wsl): $repoWsl"
        Write-Output "log: $logsWin\$Name.log"
        Write-Output "attach: wsl -d $distro -- tmux attach -t $Name"
    }

    'list' {
        $out = & wsl -d $distro -- tmux ls 2>&1
        if ($LASTEXITCODE -ne 0) { Write-Output 'no sessions' } else { $out }
    }

    'tail' {
        $pane = & wsl -d $distro -- tmux capture-pane -p -t $Name 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Output "(no live session '$Name'; showing log tail)"
        }
        $log = Join-Path $logsWin "$Name.log"
        if (Test-Path $log) {
            Get-Content $log -Tail $Lines
        } else {
            Write-Output "no log yet: $log"
        }
    }

    'logs' {
        Get-ChildItem -Path $logsWin -Filter *.log -ErrorAction SilentlyContinue |
            Select-Object Name, @{n = 'KB'; e = { [math]::Round($_.Length / 1KB, 1) } }, LastWriteTime |
            Format-Table -AutoSize
    }

    'kill' {
        & wsl -d $distro -- tmux kill-session -t $Name 2>&1
        Write-Output "killed '$Name'"
    }
}
