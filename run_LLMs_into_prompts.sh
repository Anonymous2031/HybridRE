#!/bin/bash
# ==========================================================
# Run ONLY LoRA LLMs on ONLY RoBERTa-LARGE prompt JSON files
# ==========================================================

PYTHON=python
SCRIPT=Inference_on_LLMs.py

LOG_DIR=logs
mkdir -p $LOG_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG="$LOG_DIR/run_all_llms_$TIMESTAMP.log"

# ----------------------------------------------------------
# LoRA-ADAPTED LOCAL MODELS
# ----------------------------------------------------------
BASE_DIR=$(pwd)

LORA_MODELS=(
    "$BASE_DIR/LLM/QWEEN_RETACRED/checkpoint-1600-merged"
    "$BASE_DIR/LLM/QWEEN_TACRED/checkpoint-950-merged"
)

echo "Starting LoRA-only inference at $TIMESTAMP" | tee $GLOBAL_LOG
echo "==================================================" | tee -a $GLOBAL_LOG

# ----------------------------------------------------------
# Loop over ONLY RoBERTa-LARGE prompt files
# ----------------------------------------------------------
for PROMPT in Data/2_Prompts/*/PLM_ROBERTA-LARGE_*_prompts.json; do

  DATASET=$(basename "$(dirname "$PROMPT")")      # TACRED / TACREV / RETACRED
  BASE=$(basename "$PROMPT" _prompts.json)       # PLM_ROBERTA-LARGE_TACRED_TEST etc.
  SPLIT=$(echo "$BASE" | awk -F_ '{print tolower($NF)}')
  PROCESSED_CSV="Data/1_Processed/${DATASET}/${BASE}.csv"

  if [ ! -f "$PROCESSED_CSV" ]; then
    echo "Missing CSV: $PROCESSED_CSV -> skipping" | tee -a $GLOBAL_LOG
    continue
  fi

  # ========================================================
  # LoRA-ADAPTED RUNS (dataset-aware)
  # ========================================================
  for MODEL in "${LORA_MODELS[@]}"; do

    # --- Enforce dataset ↔ LoRA compatibility ---
    # Mistral_TACRED → TACRED + TACREV
    if [[ "$MODEL" == *"TACRED"* && "$MODEL" != *"RETACRED"* ]]; then
      if [[ "$DATASET" == "RETACRED" ]]; then
        continue
      fi
    fi

    # Mistral_ReTACRED → ReTACRED only
    if [[ "$MODEL" == *"RETACRED"* && "$DATASET" != "RETACRED" ]]; then
      continue
    fi
    # -------------------------------------------

    MODEL_NAME=$(basename "$(dirname "$MODEL")")
    CKPT=$(basename "$MODEL")
    MODEL_TAG="LORA_${MODEL_NAME}_${CKPT}"

    MODEL_LOG="$LOG_DIR/${BASE}_${MODEL_TAG}_$TIMESTAMP.log"
    OUTPUT_CSV="Data/3_Final_predictions/$DATASET/${BASE}_${MODEL_TAG}_LLM.csv"

    echo "" | tee -a $GLOBAL_LOG
    echo "LORA | DATASET=$DATASET | MODEL=$MODEL_TAG" | tee -a $GLOBAL_LOG

    $PYTHON $SCRIPT \
      --prompts_json "$PROMPT" \
      --processed_csv "$PROCESSED_CSV" \
      --output_csv "$OUTPUT_CSV" \
      --model_path "$MODEL" \
      2>&1 | tee "$MODEL_LOG" | tee -a "$GLOBAL_LOG"
  done

done

echo "" | tee -a $GLOBAL_LOG
echo "ALL LoRA LLMs finished successfully at $(date)" | tee -a "$GLOBAL_LOG"
