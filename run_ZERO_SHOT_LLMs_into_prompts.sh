#!/bin/bash
# ==========================================================
# Run ZERO-SHOT LLMs on ONLY RoBERTa-LARGE prompt JSON files
# ==========================================================

PYTHON=python
SCRIPT=Inference_on_LLMs.py

LOG_DIR=logs
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG="$LOG_DIR/run_all_zero_shot_llms_$TIMESTAMP.log"

# ----------------------------------------------------------
# ZERO-SHOT MODELS
# ----------------------------------------------------------
ZERO_SHOT_MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
)

echo "Starting ZERO-SHOT inference at $TIMESTAMP" | tee "$GLOBAL_LOG"
echo "==================================================" | tee -a "$GLOBAL_LOG"

# ----------------------------------------------------------
# Loop over ONLY RoBERTa-LARGE prompt files
# ----------------------------------------------------------
for PROMPT in Data/2_Prompts/*/PLM_ROBERTA-LARGE_*_prompts.json; do

  DATASET=$(basename "$(dirname "$PROMPT")")        # TACRED / TACREV / RETACRED
  BASE=$(basename "$PROMPT" _prompts.json)          # PLM_ROBERTA-LARGE_TACRED_TEST etc.
  PROCESSED_CSV="Data/1_Processed/${DATASET}/${BASE}.csv"

  if [ ! -f "$PROCESSED_CSV" ]; then
    echo "Missing CSV: $PROCESSED_CSV -> skipping" | tee -a "$GLOBAL_LOG"
    continue
  fi

  # ========================================================
  # ZERO-SHOT RUNS
  # ========================================================
  for MODEL in "${ZERO_SHOT_MODELS[@]}"; do

    MODEL_TAG=$(basename "$MODEL")
    MODEL_TAG="ZEROSHOT_${MODEL_TAG}"

    MODEL_LOG="$LOG_DIR/${BASE}_${MODEL_TAG}_$TIMESTAMP.log"
    OUTPUT_CSV="Data/3_Final_predictions/${DATASET}/${BASE}_${MODEL_TAG}_LLM.csv"

    mkdir -p "$(dirname "$OUTPUT_CSV")"

    echo "" | tee -a "$GLOBAL_LOG"
    echo "ZERO-SHOT | DATASET=$DATASET | MODEL=$MODEL" | tee -a "$GLOBAL_LOG"

    $PYTHON $SCRIPT \
      --prompts_json "$PROMPT" \
      --processed_csv "$PROCESSED_CSV" \
      --output_csv "$OUTPUT_CSV" \
      --model_path "$MODEL" \
      2>&1 | tee "$MODEL_LOG" | tee -a "$GLOBAL_LOG"

  done

done

echo "" | tee -a "$GLOBAL_LOG"
echo "ALL ZERO-SHOT LLMs finished successfully at $(date)" | tee -a "$GLOBAL_LOG"