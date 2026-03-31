#!/usr/bin/env bash
set -euo pipefail

# Offline SDPO training on pre-collected interaction data (e.g. WildFeedback).
#
# Usage:
#   TRAIN_JSONL=/path/to/interactions.jsonl ./scripts/train_offline_sdpo.sh [--dry-run]
#
# Common overrides:
#   BASE_MODEL="Qwen/Qwen3-4B" LR=2e-6 BS=4 GA=8 ./scripts/train_offline_sdpo.sh
#   WORLD_SIZE=4 ./scripts/train_offline_sdpo.sh
#   ACCELERATE_CONFIG=./my_config.yaml ./scripts/train_offline_sdpo.sh

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
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/offline_sdpo/main_offline_sdpo.py}"

# Load API keys from .env if present
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

# Optional accelerate config. If unset, we do `accelerate launch --num_processes $WORLD_SIZE`
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$REPO_ROOT/multigpu_accelerate_config.yaml}"
WORLD_SIZE="${WORLD_SIZE:-4}"

# =============================================================================
# Configuration
# =============================================================================
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
LR="${LR:-2e-6}"
BS="${BS:-4}"
GA="${GA:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"

TRAIN_JSONL="${TRAIN_JSONL:-}"  # REQUIRED

if [[ -z "$TRAIN_JSONL" ]]; then
  echo "ERROR: TRAIN_JSONL is required (path to interaction data JSONL)"
  echo "  Example: TRAIN_JSONL=/path/to/wildfeedback_interactions.jsonl $0"
  exit 1
fi

# Tracking
WANDB_PROJECT="${WANDB_PROJECT:-offline-sdpo}"
WANDB_NAME="${WANDB_NAME:-sdpo-${BASE_MODEL//\//-}-lr${LR}-bs${BS}-ga${GA}}"

# =============================================================================
# Output + caches (portable)
# =============================================================================
BASE_WORK="${BASE_WORK:-${SCRATCH:-${TMPDIR:-/tmp}}}"
RUN_ID="${RUN_ID:-offline-sdpo-$(date +%Y%m%d-%H%M%S)}"

export OUTPUT_DIR="${OUTPUT_DIR:-$BASE_WORK/offline-sdpo/$RUN_ID}"
CACHE_DIR="${CACHE_DIR:-$BASE_WORK/hf-cache}"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"/{hf,datasets,hub}

export HF_HOME="$CACHE_DIR/hf"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/hub"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export WANDB_PROJECT
export WANDB_NAME

unset SSL_CERT_FILE SSL_CERT_DIR || true

# =============================================================================
# Summary
# =============================================================================
echo "=== OFFLINE SDPO TRAINING ==="
echo "GPUs:        $(nvidia-smi -L 2>/dev/null | wc -l || echo 'N/A')"
echo "Date:        $(date)"
echo "Base model:  $BASE_MODEL"
echo "LR=$LR BS=$BS GA=$GA EPOCHS=$NUM_EPOCHS"
echo "TRAIN_JSONL: $TRAIN_JSONL"
echo "OUTPUT_DIR:  $OUTPUT_DIR"
echo "WANDB:       $WANDB_PROJECT / $WANDB_NAME"
echo ""

# =============================================================================
# Run
# =============================================================================
cd "$REPO_ROOT"

SCRIPT_ARGS="\"$TRAIN_SCRIPT\" \
  --learning_rate $LR \
  --batch_size $BS \
  --grad_accum $GA \
  --base_model $BASE_MODEL \
  --train_jsonl $TRAIN_JSONL \
  --num_epochs $NUM_EPOCHS"

if [[ "${WORLD_SIZE}" -le 1 ]]; then
  run "python $SCRIPT_ARGS \"\$@\""
else
  if [[ -n "$ACCELERATE_CONFIG" && -f "$ACCELERATE_CONFIG" ]]; then
    run "accelerate launch --config_file \"$ACCELERATE_CONFIG\" $SCRIPT_ARGS \"\$@\""
  else
    run "accelerate launch --num_processes \"$WORLD_SIZE\" $SCRIPT_ARGS \"\$@\""
  fi
fi
