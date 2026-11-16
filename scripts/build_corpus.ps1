# Build both training corpora from the processed datasets, then prove every row fits the training sequence length.
#
#   pwsh scripts/build_corpus.ps1                 # variants 20 -> data/sft and data/grpo
#   pwsh scripts/build_corpus.ps1 -Variants 20    # 20 paraphrases per ground-truth repair row
#
# The SFT corpus carries the harness on 35 percent of rows and hindsight memory where it retrieves; the GRPO
# corpus has neither, because GRPO samples G completions per prompt and long prompts multiply the KV cache.
# Paraphrases rewrite only the reference diagnosis, and GRPO reads prompt + reward columns only, so the GRPO
# corpus keeps one prompt per plant (`-GrpoVariants 1`): 20 identical prompts would waste rollouts.
# Both are staged, audited with the real Qwen3.5 tokenizer, and only then swapped into place, so a failed build
# never leaves a half-written corpus behind.
param(
    [int]$Variants = 20,
    [int]$GrpoVariants = 1,
    [int]$MaxLength = 4096,
    [string]$Sft = 'data/sft',
    [string]$Grpo = 'data/grpo'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = if (Test-Path '.venv/Scripts/python.exe') { '.venv/Scripts/python.exe' } else { 'python' }
New-Item -ItemType Directory -Force -Path 'reports' | Out-Null

function Step($message) { "[{0:o}] {1}" -f (Get-Date), $message }

function Build-Corpus([string]$out, [string]$harnessFraction, [string]$kMemory, [int]$nVariants) {
    $stage = "$out.new"
    Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
    & $py -m thermo_agent.synth.combine data/processed data/frontier data/memory $stage `
        --extra-processed data/processed_arch --variants $nVariants `
        --harness-fraction $harnessFraction --k-memory $kMemory --check-limit 25 |
        Tee-Object -FilePath "reports/combine_$(Split-Path $out -Leaf).json" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "combine failed for $out" }
    return $stage
}

Step "combine SFT corpus (variants=$Variants, harness=0.35, memory=3) -> $Sft"
$sftStage = Build-Corpus $Sft '0.35' '3' $Variants
Step "combine GRPO prompts (variants=$GrpoVariants, no harness, no memory) -> $Grpo"
$grpoStage = Build-Corpus $Grpo '0' '0' $GrpoVariants

Step "token audit with the Qwen3.5 tokenizer (max_length=$MaxLength)"
& $py -m thermo_agent.train.token_audit $sftStage --max-length $MaxLength --out reports/token_audit_sft.json
if ($LASTEXITCODE -ne 0) { throw "SFT corpus does not fit max_length=$MaxLength; left in $sftStage" }
& $py -m thermo_agent.train.token_audit $grpoStage --max-length $MaxLength --out reports/token_audit_grpo.json
if ($LASTEXITCODE -ne 0) { throw "GRPO corpus does not fit max_length=$MaxLength; left in $grpoStage" }

function Swap-In([string]$stage, [string]$final) {
    Remove-Item -Recurse -Force $final -ErrorAction SilentlyContinue
    Rename-Item -Path $stage -NewName (Split-Path $final -Leaf)
    Step "swapped $stage -> $final"
}
Swap-In $sftStage $Sft
Swap-In $grpoStage $Grpo
Step 'done'
