#!/usr/bin/env bash
set -euo pipefail

# Evaluate an in-context oracle baseline: the model receives the style
# instruction directly in the prompt rather than learning from interactions.
#
# Usage:
#   ./scripts/eval_incontext_oracle.sh [--dry-run]
#
# Common overrides:
#   MODEL="Qwen/Qwen3-8B" ./scripts/eval_incontext_oracle.sh
#   EVAL_N=50 SEED=42 ./scripts/eval_incontext_oracle.sh
#   DATA_DIR=/path/to/data ./scripts/eval_incontext_oracle.sh

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
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-32B}"
EVAL_N="${EVAL_N:-100}"
SEED="${SEED:-1234}"
# Note: data/ is not included in the repo — prepare it first (see README)
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/tldr_prompts_unique}"

# =============================================================================
# Output + caches (portable)
# =============================================================================
BASE_WORK="${BASE_WORK:-${SCRATCH:-${TMPDIR:-/tmp}}}"
CACHE_DIR="${CACHE_DIR:-$BASE_WORK/hf-cache}"

mkdir -p "$CACHE_DIR"/{hf,datasets,hub}

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
echo "=== IN-CONTEXT ORACLE ==="
echo "GPUs:        $(nvidia-smi -L 2>/dev/null | wc -l || echo 'N/A')"
echo "Date:        $(date)"
echo "Model:       $MODEL"
echo "Judge:       $JUDGE_MODEL"
echo "EVAL_N=$EVAL_N SEED=$SEED"
echo "DATA_DIR:    $DATA_DIR"
echo ""

# =============================================================================
# Run
# =============================================================================
cd "$REPO_ROOT"

run "python auxiliary/eval_incontext_oracle.py \
  --model \"$MODEL\" \
  --judge_model \"$JUDGE_MODEL\" \
  --val_jsonl \"$DATA_DIR/validation.jsonl\" \
  --eval_n \"$EVAL_N\" \
  --seed \"$SEED\" \
  \"\$@\""
