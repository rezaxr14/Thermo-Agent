# Overnight budget frontier loop: 100-row luna/haiku batches, process each, repeat.
# No new Kaggle object, no gemini, no premium, no local data upload to Kaggle.
param(
  [int]$MaxBatches = 100,
  [double]$MinDailyQuota = 0.50,
  [int]$RebuildEvery = 1
)

$env:PYTHONIOENCODING = 'utf-8'
Remove-Item Env:HTTP_PROXY, Env:HTTPS_PROXY -ErrorAction SilentlyContinue
$TASK = 'thermo-teacher-cotproof-s00'
$MODELS = @('gpt-6-luna', 'claude-haiku-5-5')

function Log([string]$msg) { Write-Host "[$(Get-Date -Format o)] $msg" }
function Fail([string]$msg) { Log "FATAL: $msg"; throw $msg }

function Invoke-K([string[]]$KArgs) {
  # Kaggle's gateway returns transient 429/5xx and empty bodies. Retry with backoff
  # instead of killing a multi-hour loop on a single blip.
  for ($i = 1; $i -le 10; $i++) {
    $out = & powershell -NoProfile -File scripts/kaggle_project.ps1 @KArgs 2>&1
    $s = ($out | Out-String)
    $code = $LASTEXITCODE
    if ($code -eq 0 -and $s -notmatch 'aborted|Forbidden|ConnectionAborted') { return $out }
    if ($s -match 'Pushed new version|Done: [0-9]+ run|All runs completed') { return $out }
    Log "kaggle retry $i for $($KArgs -join ' '): " + (($s -split "`n" | Select-Object -First 1) -replace '\s+', ' ')
    Start-Sleep -Seconds ([Math]::Min(60, 5 * $i))
  }
  Fail "kaggle command failed: $($KArgs -join ' ')"
}

function Get-TaskStatus([string]$name = $TASK) {
  # Best effort: a transient failure here must not abort the batch.
  try {
    $out = & powershell -NoProfile -File scripts/kaggle_project.ps1 benchmarks tasks status $name 2>&1
    if ($LASTEXITCODE -eq 0) { return ($out | Out-String) }
  } catch { }
  return ''
}

function Get-DailyQuota {
  $raw = (Invoke-K @('benchmarks', 'quota', '--format', 'json') | Out-String)
  $m = [regex]::Match($raw, '"period":\s*"Daily".*?"remaining":\s*"\$([0-9.]+)"',
                      [System.Text.RegularExpressions.RegexOptions]::Singleline)
  if (-not $m.Success) { Fail "could not parse daily quota" }
  return [double]$m.Groups[1].Value
}

function Wait-Task([string]$Version = '') {
  for ($i = 1; $i -le 240; $i++) {
    $s = Get-TaskStatus
    $rightVersion = (-not $Version) -or ($s -match "Version:\s+$Version\b")
    if ($rightVersion -and $s -match 'Status:\s+Completed') { return }
    if ($rightVersion -and $s -match 'Status:\s+Errored') { Fail "task errored: $s" }
    if (-not $s) { Log "task status unavailable (attempt $i); still polling" }
    Start-Sleep -Seconds 20
  }
  Fail "task did not complete in time"
}

function Wait-Kernel {
  for ($i = 1; $i -le 180; $i++) {
    $s = (Invoke-K @('kernels', 'status', 'rezanadimi/thermo-agent-hf-to-kaggle-dataset-bridge') | Out-String)
    if ($s -match 'COMPLETE') { return }
    if ($s -match 'ERROR') { Fail "bridge kernel errored: $s" }
    Start-Sleep -Seconds 20
  }
  Fail "bridge kernel did not complete in time"
}

function Commit-Work([string]$message) {
  git add data/frontier_tools_cohort data/memory docs/BUDGET_BATCH_REPORT.md data/frontier_tools_accepted data/sft_mix_v2 data/grpo_mix_v2 reports/DATAGEN_PROGRESS.md reports/datagen_progress.json
  $diff = git diff --cached --stat
  if ($diff) {
    git commit -m $message | Out-Null
    git push | Out-Null
    Log "committed: $message"
  }
}

function Update-TrainingData {
  Log "rebuilding mix with all accepted frontier traces"
  & .venv/Scripts/python.exe scripts/build_accepted_frontier.py
  if ($LASTEXITCODE -ne 0) { Fail "accepted frontier flatten failed" }
  & .venv/Scripts/python.exe scripts/mix_tool_corpora.py `
    --deterministic-sft data/sft_tools_v2_h --deterministic-grpo data/grpo_tools_v2_h `
    --frontier data/frontier_tools_accepted --out-sft data/sft_mix_v2 --out-grpo data/grpo_mix_v2 `
    --manifest data/sft_mix_v2/MANIFEST.json --harness-prompt --a-slice data/a_slice --seed 0 --force
  if ($LASTEXITCODE -ne 0) { Fail "mix rebuild failed" }
  & .venv/Scripts/python.exe -m pytest -q tests/test_final_mix_tiers_provenance.py tests/test_check_datagen_complete.py
  if ($LASTEXITCODE -ne 0) { Fail "post-mix tests failed" }
  & .venv/Scripts/python.exe scripts/sync_corpora_hf.py --dirs data/frontier_tools_cohort data/frontier_tools_accepted data/memory data/sft_mix_v2 data/grpo_mix_v2 `
    --message "budget frontier traces incorporated into training mix" --tag "budget-mix-$(Get-Date -Format yyyyMMddHHmmss)"
  if ($LASTEXITCODE -ne 0) { Fail "HF sync failed" }
  # Refresh the human progress view every batch so cell coverage is never a guess.
  & .venv/Scripts/python.exe scripts/report_datagen_progress.py --json reports/datagen_progress.json --md reports/DATAGEN_PROGRESS.md | Out-Null
  Commit-Work "budget: refresh training mix with accepted frontier traces"
}

Log "overnight loop start; max=$MaxBatches min_daily=$MinDailyQuota rebuild_every=$RebuildEvery"
for ($b = 1; $b -le $MaxBatches; $b++) {
  $quota = Get-DailyQuota
  Log "batch $b : daily quota remaining `$ $quota"
  if ($quota -lt $MinDailyQuota) { Log "daily quota low; stopping"; break }

  Log "batch $b : building shards"
  & .venv/Scripts/python.exe scripts/build_next_budget_batch.py
  if ($LASTEXITCODE -ne 0) { Fail "batch shard build failed" }

  Log "batch $b : syncing prompts/config to HF"
  & .venv/Scripts/python.exe scripts/sync_corpora_hf.py --dirs data/frontier_tools_cohort_prompts `
    --message "budget batch $b shards" --tag "budget-batch-$b"
  if ($LASTEXITCODE -ne 0) { Fail "HF prompt sync failed" }

  Log "batch $b : refreshing Kaggle dataset from HF"
  Invoke-K @('kernels', 'push', '--path', 'kaggle/hf_to_kaggle_kernel') | Out-Null
  Wait-Kernel

  Log "batch $b : pushing generic runner and waiting for auto-run skip"
  $pushOut = (Invoke-K @('benchmarks', 'tasks', 'push', $TASK, '--file', 'notebooks/kaggle_model_runner.py',
             '--kaggle-dataset', 'rezanadimi/thermo-agent-data') | Out-String)
  # `tasks push` auto-runs the session default model, which the runner guard skips
  # before any model call. Wait for THAT version, then launch luna+haiku explicitly.
  $version = [regex]::Match($pushOut, '[Vv]ersion\s+(\d+)').Groups[1].Value
  Log "batch $b : new task version $(if ($version) { $version } else { 'unknown' })"
  Wait-Task $version

  Log "batch $b : running luna + haiku"
  Invoke-K @('benchmarks', 'tasks', 'run', $TASK, '--model', 'gpt-6-luna',
             '--model', 'claude-haiku-5-5', '--wait', '1800', '-v') | Out-Null

  # Name the batch after the journal in harness_config so the shard, journal and
  # provenance run id always agree (no b+1 drift).
  $journalName = (& .venv/Scripts/python.exe -c "import json;print(json.load(open('data/frontier_tools_cohort_prompts/harness_config.json'))['journal'])" | Out-String).Trim()
  $batchTag = ($journalName -replace '_completions\.jsonl$', '')
  if (-not $batchTag) { $batchTag = "batch$b" }
  $download = "data/teacher_runs/kaggle_hosted/budget_batches/$batchTag"
  Log "batch $b : downloading journals to $download"
  Invoke-K @('benchmarks', 'tasks', 'download', $TASK, '--model', 'gpt-6-luna',
             '--model', 'claude-haiku-5-5', '--output', $download) | Out-Null

  foreach ($model in $MODELS) {
    $journal = (Get-ChildItem $download -Recurse -Filter '*_completions.jsonl' |
                Where-Object { $_.FullName -replace '\\', '/' -match "/$([regex]::Escape($model))/" } |
                Select-Object -First 1).FullName
    if (-not $journal) { Fail "no journal for $model in $download" }
    $runid = "budget-$batchTag-$model"
    Log "batch $b : gating/ingesting $model"
    powershell -NoProfile -File scripts/process_budget_batch.ps1 -Model $model -Journal $journal -RunId $runid |
      Select-String -Pattern 'accepted_tool_rows|survived|dropped_over_gate|accepted frontier tool traces' |
      ForEach-Object { Log "  $model $($_.Line.Trim())" }
    if ($LASTEXITCODE -ne 0) { Fail "gate/ingest failed for $model ($journal); see empty/error-journal guard" }
  }

  & .venv/Scripts/python.exe scripts/build_accepted_frontier.py | ForEach-Object { Log "  $_" }
  Add-Content -Path docs/BUDGET_BATCH_REPORT.md -Value "`n## $batchTag (overnight)`n- Ran luna+haiku, gated and ingested accepted traces; memory updated by transfer.ingest.`n"
  Commit-Work "budget: process $batchTag"

  if ($b % $RebuildEvery -eq 0) { Update-TrainingData }
}
Update-TrainingData
Log "overnight loop done"
