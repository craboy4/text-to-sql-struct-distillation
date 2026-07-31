#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_NAME [LORA_ADAPTER_PATH]" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/text2sql_qwen3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
EVALUATOR_DIR="${EVALUATOR_DIR:-$PROJECT_ROOT/evaluation_upstream_minidev}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3-8B}"
RUN_DIR="$PROJECT_ROOT/outputs/$1"
PROMPTS="$PROJECT_ROOT/eval/minidev_author_prompt_messages.jsonl"
RESPONSES="$RUN_DIR/responses.jsonl"
PREDICTIONS="$RUN_DIR/predictions.jsonl"
AUTHOR_PREDICTIONS="$RUN_DIR/author_predictions.json"
EX_LOG="$RUN_DIR/author_ex.txt"

if [[ -e "$PREDICTIONS" || -e "$AUTHOR_PREDICTIONS" || -e "$EX_LOG" ]]; then
  echo "refusing to overwrite completed artifacts in $RUN_DIR" >&2
  exit 2
fi

if [[ ! -f "$PROMPTS" ]]; then
  "$PYTHON_BIN" "$EVALUATOR_DIR/prepare_author_minidev_prompts.py" \
    --source "$EVALUATOR_DIR/mini_dev_prompt.jsonl" \
    --output "$PROMPTS"
fi

mkdir -p "$RUN_DIR"
INFERENCE=(
  "$PYTHON_BIN" "$EVALUATOR_DIR/run_minidev_local.py"
  --model "$MODEL_PATH"
  --prompts "$PROMPTS"
  --output "$RESPONSES"
  --batch-size "${BATCH_SIZE:-8}"
  --max-input-tokens "${MAX_INPUT_TOKENS:-30720}"
  --max-new-tokens "${MAX_NEW_TOKENS:-1024}"
)
if [[ $# -eq 2 ]]; then
  INFERENCE+=(--adapter "$2")
fi
"${INFERENCE[@]}"

"$PYTHON_BIN" "$EVALUATOR_DIR/extract_author_minidev_responses.py" \
  --source-prompts "$EVALUATOR_DIR/mini_dev_prompt.jsonl" \
  --responses "$RESPONSES" \
  --predictions "$PREDICTIONS" \
  --author-predictions "$AUTHOR_PREDICTIONS"

PYTHONUTF8=1 "$PYTHON_BIN" "$EVALUATOR_DIR/evaluation_ex.py" \
  --db_root_path "$PROJECT_ROOT/eval/MINIDEV/dev_databases/" \
  --predicted_sql_path "$AUTHOR_PREDICTIONS" \
  --ground_truth_path "$PROJECT_ROOT/eval/MINIDEV/mini_dev_sqlite_gold.sql" \
  --diff_json_path "$EVALUATOR_DIR/mini_dev_prompt.jsonl" \
  --num_cpus "${NUM_CPUS:-16}" \
  --meta_time_out "${SQL_TIMEOUT_SECONDS:-30}" \
  --sql_dialect SQLite \
  --output_log_path "$EX_LOG"
