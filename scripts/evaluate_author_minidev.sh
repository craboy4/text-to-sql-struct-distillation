#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PREDICTIONS_JSONL OUTPUT_DIR" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/text2sql_qwen3}"
PREDICTIONS="$1"
OUTPUT_DIR="$2"
EVALUATOR_DIR="${EVALUATOR_DIR:-$PROJECT_ROOT/evaluation_upstream_minidev}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

mkdir -p "$OUTPUT_DIR"
AUTHOR_PREDICTIONS="$OUTPUT_DIR/author_predictions.json"
AUTHOR_LOG="$OUTPUT_DIR/author_ex.txt"

test ! -e "$AUTHOR_PREDICTIONS"
test ! -e "$AUTHOR_LOG"

"$PYTHON_BIN" "$EVALUATOR_DIR/prepare_author_ex_predictions.py" \
  --upstream-prompts "$EVALUATOR_DIR/mini_dev_prompt.jsonl" \
  --predictions "$PREDICTIONS" \
  --output "$AUTHOR_PREDICTIONS"

PYTHONUTF8=1 "$PYTHON_BIN" "$EVALUATOR_DIR/evaluation_ex.py" \
  --db_root_path "$PROJECT_ROOT/eval/MINIDEV/dev_databases/" \
  --predicted_sql_path "$AUTHOR_PREDICTIONS" \
  --ground_truth_path "$PROJECT_ROOT/eval/MINIDEV/mini_dev_sqlite_gold.sql" \
  --diff_json_path "$EVALUATOR_DIR/mini_dev_prompt.jsonl" \
  --num_cpus "${NUM_CPUS:-16}" \
  --meta_time_out "${SQL_TIMEOUT_SECONDS:-30}" \
  --sql_dialect SQLite \
  --output_log_path "$AUTHOR_LOG"
