# Score the saved Kaggle answers on the frozen evaluation splits with the same evaluator as our model.
# Requires: data/teacher_runs/eval_<split>.jsonl and harness_completions.jsonl.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = if (Test-Path '.venv/Scripts/python.exe') { '.venv/Scripts/python.exe' } else { 'python' }
New-Item -ItemType Directory -Force -Path 'reports' | Out-Null

$merged = 'data/teacher_runs/eval_all.jsonl'
Get-ChildItem 'data/teacher_runs' -Filter 'eval_test_*.jsonl' | Where-Object Name -notlike 'eval_gptnano_*' |
    ForEach-Object { Get-Content $_.FullName } | Set-Content $merged -Encoding utf8
"merged frontier eval journal: $merged"

$arms = @(
    @{ name='frontier_zero'; replay='data/teacher_runs/eval_all.jsonl'; harness='none'; family='frontier'; tools='none' },
    @{ name='frontier_tools'; replay='data/teacher_runs/harness_eval_completions.jsonl'; harness='tools'; family='frontier'; tools='tools' },
    @{ name='other_model'; replay='data/teacher_runs/eval_gptnano_all.jsonl'; harness='none'; family='domain-model'; tools='none' }
)
foreach ($a in $arms) {
    if (-not (Test-Path $a.replay)) { "skip $($a.name): missing $($a.replay)"; continue }
    $splitArgs = if ($a.name -eq 'frontier_tools') { @('--splits','test_id') } else { @() }
    & $py -m thermo_agent.eval.model_eval data/processed data/processed_arch `
        --backend replay --replay $a.replay --tasks repair design audit ledger @splitArgs `
        --limit 30 --harness $a.harness --model-family $a.family --tool-access $a.tools `
        --zero-shot --display-name $a.name --out "reports/eval_$($a.name).json"
    if ($LASTEXITCODE -ne 0) { throw "eval failed: $($a.name)" }
}

# Build the comparison manifest from the reports that actually exist, then the table input.
$manifest = @{ arms = @{} }
foreach ($name in @('frontier_zero','frontier_tools','other_model','ours_base','ours_harness')) {
    $p = "reports/eval_$name.json"
    if (Test-Path $p) {
        $spec = $arms | Where-Object name -eq $name | Select-Object -First 1
        $manifest.arms[$name] = @{
            report = (Resolve-Path $p).Path
            family = if ($spec) { $spec.family } else { 'ours' }
            harness = if ($spec) { $spec.harness } else { 'none' }
            tool_access = if ($spec) { $spec.tools } else { 'none' }
            zero_shot = $true
            display_name = $name
        }
    }
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content 'paper/assets/model_results_manifest.json' -Encoding utf8
& $py -m thermo_agent.eval.comparison paper/assets/model_results_manifest.json --out paper/assets/model_results.json
& $py -c "import json;from thermo_agent.eval import tool_usage;from pathlib import Path;s=tool_usage.summarize(tool_usage.load('data/teacher_runs/harness_completions.jsonl'));Path('reports/tool_usage.json').write_text(json.dumps(s,indent=1))"
'run_replay_evals done'
