# Build both training corpora from the processed datasets, then prove every row fits the training sequence length.
#
#   pwsh scripts/build_corpus.ps1                 # variants 20 -> data/sft and data/grpo
#   pwsh scripts/build_corpus.ps1 -Variants 20    # 20 paraphrases per ground-truth repair row
#
# Both corpora carry the harness on 35 percent of rows and the hindsight memory where it retrieves: GRPO must see
# the same prompts the SFT model learned to answer, otherwise the two stages disagree about the input format. GRPO
# samples G completions per prompt, so its corpus keeps one prompt per plant (`-GrpoVariants 1`); 20 identical
# prompts would waste rollouts. Paraphrases rewrite only the reference diagnosis.
#
# Both corpora are staged, audited with the real Qwen3.5 tokenizer, verified against their report, and only then
# swapped into place, so a failed build never leaves a half-written corpus behind.
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
Step "combine GRPO prompts (variants=$GrpoVariants, harness=0.35, memory=3) -> $Grpo"
$grpoStage = Build-Corpus $Grpo '0.35' '3' $GrpoVariants

Step "token audit with the Qwen3.5 tokenizer (max_length=$MaxLength)"
& $py -m thermo_agent.train.token_audit $sftStage --max-length $MaxLength --out reports/token_audit_sft.json
if ($LASTEXITCODE -ne 0) { throw "SFT corpus does not fit max_length=$MaxLength; left in $sftStage" }
& $py -m thermo_agent.train.token_audit $grpoStage --max-length $MaxLength --out reports/token_audit_grpo.json
if ($LASTEXITCODE -ne 0) { throw "GRPO corpus does not fit max_length=$MaxLength; left in $grpoStage" }

Step 'verify the staged corpora against their reports'
& $py -m thermo_agent.synth.verify $sftStage --max-tokens 4000 |
    Tee-Object -FilePath reports/verify_sft.json | Out-Null
if ($LASTEXITCODE -ne 0) { throw "staged SFT corpus failed verification; left in $sftStage" }
& $py -m thermo_agent.synth.verify $grpoStage --max-tokens 4000 --allow-missing-completion |
    Tee-Object -FilePath reports/verify_grpo.json | Out-Null
if ($LASTEXITCODE -ne 0) { throw "staged GRPO corpus failed verification; left in $grpoStage" }

function Swap-In([string]$stage, [string]$final) {
    if (-not (Test-Path $stage)) { throw "stage directory missing: $stage" }
    New-Item -ItemType Directory -Force -Path (Split-Path $final -Parent) | Out-Null
    if (Test-Path $final) {
        # Rename-Item -NewName silently moves the stage *inside* an existing target directory
        # (data/sft/sft.new), which is how one repair_train.jsonl went missing once. Move the old corpus aside
        # first and delete it after the swap, so the stage always ends up at exactly $final.
        $old = "$final.old"
        Remove-Item -Recurse -Force $old -ErrorAction SilentlyContinue
        Move-Item -Force $final $old
        Remove-Item -Recurse -Force $old -ErrorAction SilentlyContinue
    }
    Move-Item -Force $stage $final
    if (-not (Test-Path $final)) { throw "swap failed: $stage -> $final" }
    Step "swapped $stage -> $final"
}

Swap-In $sftStage $Sft
Swap-In $grpoStage $Grpo

Step 'verify the swapped corpora in place'
& $py -m thermo_agent.synth.verify $Sft --max-tokens 4000 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "swapped SFT corpus failed verification: $Sft" }
& $py -m thermo_agent.synth.verify $Grpo --max-tokens 4000 --allow-missing-completion | Out-Null
if ($LASTEXITCODE -ne 0) { throw "swapped GRPO corpus failed verification: $Grpo" }
Step 'done'
