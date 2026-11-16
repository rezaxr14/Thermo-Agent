#!/usr/bin/env bash
# Dual-T4 launcher: two DDP ranks, full fp16 LoRA (no 4-bit), T4-safe dtype and overflow handling.
# Usage:
#   scripts/train_dual_t4.sh sft
#   scripts/train_dual_t4.sh grpo repair
#   scripts/train_dual_t4.sh all
#   scripts/train_dual_t4.sh merge runs/sft/checkpoint-N runs/sft_merged
#
# Environment:
#   DATA=data/sft        combined corpus written by thermo_agent.synth.combine
#   GRPO_DATA=data/grpo  same prompts with no harness and no memory (short prompts for G rollouts)
#   OUT=runs             adapter / checkpoint root
#   MODEL=Qwen/Qwen3.5-4B
#   TASKS="repair design audit ledger"
#   INIT=runs/sft_merged merged SFT model that GRPO starts from
#   NGPU / WORKERS       auto-detected; set to override
#   STEPS / ROWS         optional smoke limits (--max-steps / --limit-rows)
set -euo pipefail

cd "$(dirname "$0")/.."

DATA=${DATA:-data/sft}          # combined SFT corpus (harness on 35% of rows)
GRPO_DATA=${GRPO_DATA:-data/grpo}  # same prompts without harness/memory: GRPO samples G=4 per prompt
OUT=${OUT:-runs}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
TASKS=${TASKS:-"repair design audit ledger"}
INIT=${INIT:-runs/sft_merged}
PY=${PY:-python}
[ -x .venv/bin/python ] && PY=.venv/bin/python

# Detect the GPUs and dataloader workers instead of assuming the Kaggle 2xT4 shape.
if [ -z "${NGPU:-}" ]; then
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  [ "${NGPU:-0}" -ge 1 ] 2>/dev/null || NGPU=1
fi
if [ -z "${WORKERS:-}" ]; then
  CPUS=$(nproc 2>/dev/null || echo 4)
  WORKERS=$(( CPUS / 2 )); [ "$WORKERS" -lt 1 ] && WORKERS=1
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( WORKERS > 4 ? 4 : WORKERS ))}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=1           # no InfiniBand on Kaggle; avoids a probe that can stall
# export NCCL_P2P_DISABLE=1        # only if NCCL hangs or reports an unsupported P2P setup

LIMITS=()
[ -n "${STEPS:-}" ] && LIMITS+=(--max-steps "$STEPS")
[ -n "${ROWS:-}" ] && LIMITS+=(--limit-rows "$ROWS")

preflight() {
  "$PY" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible")
n = torch.cuda.device_count()
print(f"visible GPUs: {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    cap = torch.cuda.get_device_capability(i)
    print(f"  {i}: {p.name} cc={cap[0]}.{cap[1]} mem={p.total_memory / 1024**3:.1f} GB")
if n < 2:
    sys.exit("expected 2 GPUs; set NGPU if the runtime differs")
PY
}

run_sft() {
  preflight
  torchrun --standalone --nproc_per_node="$NGPU" --module thermo_agent.train.sft \
    "$DATA" "$OUT/sft" --model "$MODEL" --dtype float16 --tasks $TASKS \
    --per-device-batch 2 --grad-accum 8 --grad-norm 1.0 --num-workers "$WORKERS" "${LIMITS[@]}"
}

run_grpo() {
  local task=${1:?task required: repair|design|audit|ledger}
  preflight
  torchrun --standalone --nproc_per_node="$NGPU" --module thermo_agent.train.grpo \
    "$GRPO_DATA" "$OUT/grpo_$task" --task "$task" --init "$INIT" --dtype float16 \
    --per-device-batch 4 --grad-accum 2 --grad-norm 1.0 --num-workers "$WORKERS" "${LIMITS[@]}"
}

run_merge() {
  "$PY" -m thermo_agent.train.merge "${1:?checkpoint dir}" "${2:?output dir}"
}

case "${1:-all}" in
  sft)   run_sft ;;
  grpo)  run_grpo "${2:-}" ;;
  merge) run_merge "${2:-}" "${3:-}" ;;
  all)
    run_sft
    latest=$(find "$OUT/sft" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
    [ -n "$latest" ] || { echo "no checkpoint under $OUT/sft" >&2; exit 1; }
    run_merge "$latest" "$INIT"
    for t in $TASKS; do run_grpo "$t"; done
    ;;
  *) echo "unknown stage: $1 (use sft|grpo|merge|all)" >&2; exit 2 ;;
esac
