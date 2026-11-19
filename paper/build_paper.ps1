# Build the paper IN PLACE from the paper/ directory using MiKTeX.
# Produces paper/main.pdf directly (no -output-directory, no copying from a build dir).
#
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File paper\build_paper.ps1
# Or from paper/:     .\build_paper.ps1

$ErrorActionPreference = 'Stop'

# Always operate on the directory that contains this script (i.e. paper/).
$paperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location -LiteralPath $paperDir

try {
    # Prefer the MiKTeX x64 binaries; fall back to whatever pdflatex/bibtex is on PATH.
    $miktexBin = Join-Path $env:LOCALAPPDATA 'Programs\MiKTeX\miktex\bin\x64'
    $pdflatex  = Join-Path $miktexBin 'pdflatex.exe'
    $bibtex    = Join-Path $miktexBin 'bibtex.exe'
    if (-not (Test-Path -LiteralPath $pdflatex)) { $pdflatex = (Get-Command pdflatex -ErrorAction Stop).Source }
    if (-not (Test-Path -LiteralPath $bibtex))   { $bibtex   = (Get-Command bibtex   -ErrorAction Stop).Source }

    Write-Host "Working directory : $(Get-Location)"
    Write-Host "pdflatex          : $pdflatex"
    Write-Host "bibtex            : $bibtex"

    function Invoke-BuildStep([string]$label, [string]$exe, [string[]]$cmdArgs) {
        Write-Host ""
        Write-Host "=== $label ==="
        # Native tools write progress/diagnostics to stderr; do not let PowerShell turn
        # that into a terminating error. Success is decided by the exit code below.
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $output = (& $exe @cmdArgs 2>&1 | Out-String)
        }
        finally {
            $ErrorActionPreference = $prev
        }
        Write-Host $output
        if ($LASTEXITCODE -ne 0) {
            throw "$label failed with exit code $LASTEXITCODE"
        }
    }

    # LaTeX -> BibTeX -> LaTeX -> LaTeX (resolves citations, then cross-references).
    Invoke-BuildStep 'pdflatex pass 1' $pdflatex @('-interaction=nonstopmode', '-file-line-error', 'main.tex')
    Invoke-BuildStep 'bibtex'          $bibtex   @('main')
    Invoke-BuildStep 'pdflatex pass 2' $pdflatex @('-interaction=nonstopmode', '-file-line-error', 'main.tex')
    Invoke-BuildStep 'pdflatex pass 3' $pdflatex @('-interaction=nonstopmode', '-file-line-error', 'main.tex')

    $pdf = Join-Path $paperDir 'main.pdf'
    if (-not (Test-Path -LiteralPath $pdf)) { throw 'main.pdf was not produced in the paper/ root' }

    Write-Host ""
    Write-Host "=== RESULT ==="
    $fi = Get-Item -LiteralPath $pdf
    Write-Host "PDF: $($fi.FullName)  ($($fi.Length) bytes, modified $($fi.LastWriteTime.ToString('o')))"

    $pdfinfo = Join-Path $miktexBin 'pdfinfo.exe'
    if (Test-Path -LiteralPath $pdfinfo) {
        Write-Host (( & $pdfinfo $pdf ) 2>&1 | Select-String '^Pages:' | Out-String)
    }
}
finally {
    Pop-Location
}
