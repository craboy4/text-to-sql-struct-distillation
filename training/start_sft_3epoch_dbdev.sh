#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"

DATASET="$PROJECT_ROOT/data/qwen_sft_messages_train_dbdev4.jsonl"
MODEL_DIR="$PROJECT_ROOT/models/Qwen3-8B"
RUN_ID="${RUN_ID:-qwen3_8b_bird_lora_3epoch_dbdev4_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="$PROJECT_ROOT/outputs/experiments/$RUN_ID"
LOG_FILE="$OUTPUT_DIR/train.log"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to reuse existing run directory: $OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

python "$SCRIPT_DIR/preflight.py" \
  --project-root "$PROJECT_ROOT" \
  --dataset "$DATASET" \
  --expected-records 5160 | tee "$OUTPUT_DIR/preflight.json"
date +%s > "$OUTPUT_DIR/run_started_at.txt"

exec > >(tee -a "$LOG_FILE") 2>&1
printf '%s\n' "$RUN_ID" > "$OUTPUT_DIR/run_id.txt"
echo "SFT log: $LOG_FILE"
echo "Use: python $SCRIPT_DIR/monitor_training.py --output-dir $OUTPUT_DIR --follow"

CUDA_VISIBLE_DEVICES=0 swift sft \
  --model "$MODEL_DIR" \
  --dataset "$DATASET" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --max_length 32768 \
  --truncation_strategy delete \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.03 \
  --seed 42 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_checkpointing true \
  --attn_impl sdpa \
  --dataset_num_proc 12 \
  --dataloader_num_workers 4 \
  --load_from_cache_file true \
  --save_strategy steps \
  --save_steps 50 \
  --save_total_limit 25 \
  --create_checkpoint_symlink true \
  --save_only_model false \
  --logging_steps 1 \
  --report_to tensorboard \
  --output_dir "$OUTPUT_DIR" \
  --add_version false \
  --split_dataset_ratio 0
