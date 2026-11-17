# Rebuild the Kaggle teacher hand-off from the current SFT corpus, so the zip can never lag the corpus.
#
#   pwsh scripts/make_teacher_export.ps1
#   # upload data/frontier_export.zip as a private Kaggle dataset named thermo-teacher-prompts
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = if (Test-Path '.venv/Scripts/python.exe') { '.venv/Scripts/python.exe' } else { 'python' }

& $py -m thermo_agent.teacher.transfer export data/sft data/frontier_export --per-task 400
if ($LASTEXITCODE -ne 0) { throw 'teacher export failed' }
Compress-Archive -Path data/frontier_export -DestinationPath data/frontier_export.zip -Force
Get-Item data/frontier_export.zip | Select-Object Name, Length, LastWriteTime
& $py -c "import json;d=json.load(open('data/frontier_export/export_manifest.json'));print(json.dumps(d,indent=1))"
