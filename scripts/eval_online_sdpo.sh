#!/usr/bin/env bash
set -euo pipefail

# Run online SDPO evaluation: interleaved training + evaluation loop.
#
# Usage:
#   ./scripts/eval_online_sdpo.sh [--dry-run]
#
# Common overrides:
#   MODEL="Qwen/Qwen3-8B" STYLE="no_emojis" ./scripts/eval_online_sdpo.sh
#   LR=1e-5 TRAIN_N=30 EVAL_N=50 ./scripts/eval_online_sdpo.sh
#   BASELINE_MODEL=/path/to/baseline ./scripts/eval_online_sdpo.sh

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
  echo "Dry run mode enabled. Commands will be printed but not executed."
fi

run() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "$*"
  else
    eval "$*"
  fi
}

# =============================================================================
# Paths
# =============================================================================
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Load API keys from .env if present
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

# =============================================================================
# Configuration
# =============================================================================
MODEL="${MODEL:-Qwen/Qwen3-8B}"
BASELINE_MODEL="${BASELINE_MODEL:-}"  # leave empty to use same model as baseline before any training
USER_MODEL="${USER_MODEL:-Qwen/Qwen3-32B}"
STYLE="${STYLE:-no_emojis}"
EVAL_STYLES="${EVAL_STYLES:-}"  # optionally add additional eval user profiles
LR="${LR:-5e-6}"
LOSS_MODE="${LOSS_MODE:-full_distillation}"
TRAIN_N="${TRAIN_N:-15}"
EVAL_N="${EVAL_N:-100}"
EVAL_EVERY="${EVAL_EVERY:-3}"
TRAIN_STEPS_PER_EXAMPLE="${TRAIN_STEPS_PER_EXAMPLE:-1}"
SEED="${SEED:-1234}"
RUN_NAME="${RUN_NAME:-eval_online_sdpo}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/helpsteer_prompts}"

# =============================================================================
# Output + caches (portable)
# =============================================================================
BASE_WORK="${BASE_WORK:-${SCRATCH:-${TMPDIR:-/tmp}}}"
RUN_ID="${RUN_ID:-eval-online-sdpo-$(date +%Y%m%d-%H%M%S)}"

export OUTPUT_DIR="${OUTPUT_DIR:-$BASE_WORK/eval-online-sdpo/$RUN_ID}"
CACHE_DIR="${CACHE_DIR:-$BASE_WORK/hf-cache}"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"/{hf,datasets,hub}

export HF_HOME="$CACHE_DIR/hf"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/hub"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

unset SSL_CERT_FILE SSL_CERT_DIR || true

# =============================================================================
# Summary
# =============================================================================
echo "=== EVAL ONLINE SDPO ==="
echo "GPUs:       $(nvidia-smi -L 2>/dev/null | wc -l || echo 'N/A')"
echo "Date:       $(date)"
echo "Baseline:   ${BASELINE_MODEL:-<same as model>}"
echo "Model:      $MODEL"
echo "User model: $USER_MODEL"
echo "Style:      $STYLE"
echo "LR=$LR LOSS=$LOSS_MODE TRAIN_N=$TRAIN_N EVAL_N=$EVAL_N"
echo "DATA_DIR:   $DATA_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# =============================================================================
# Run
# =============================================================================
cd "$REPO_ROOT"

run "python eval_online_sdpo.py \
  --model \"$MODEL\" \
  --user_model_name_or_path \"$USER_MODEL\" \
  --judge_local \
  --user_vllm \
  --lr \"$LR\" \
  --loss_mode \"$LOSS_MODE\" \
  --style \"$STYLE\" \
  --use_lora \
  --use_vllm \
  --train_jsonl \"$DATA_DIR/train.jsonl\" \
  --val_jsonl \"$DATA_DIR/validation.jsonl\" \
  --train_n \"$TRAIN_N\" \
  --eval_n \"$EVAL_N\" \
  --eval_every \"$EVAL_EVERY\" \
  --train_steps_per_example \"$TRAIN_STEPS_PER_EXAMPLE\" \
  --out_dir \"$OUTPUT_DIR\" \
  --run_name \"$RUN_NAME\" \
  --seed \"$SEED\" \
  --eval_styles $EVAL_STYLES \
  --debug_first_example \
  --save_checkpoints \
  ${BASELINE_MODEL:+--baseline_model \"$BASELINE_MODEL\"} \
  \"\$@\""
