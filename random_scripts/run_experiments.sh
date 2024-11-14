#!/bin/bash

# Base directory for datasets
DATASET_DIR="datasets/"

# Output directory for fine-tuned models
OUTPUT_DIR="finetuning/"

# Array of learning rates
LEARNING_RATES=(1e-3 5e-4 1e-4)

# Array of batch sizes
BATCH_SIZES=(32 64)

# Array of model sizes
MODEL_SIZES=("Small" "Base" "Large")

# Array for scheduler use
# USE_SCHEDULER=(1 0)
# USE_SCHEDULER=("false" "true")

# Fixed parameters
EPOCHS=15
WARMUP_STEPS=500

# Create a new experiment number
EXP_NUM=$(ls -1d logs/exp_* 2>/dev/null | wc -l)
EXP_NUM=$((EXP_NUM + 1))
EXP_DIR="logs/exp_$EXP_NUM"
mkdir -p "$EXP_DIR"

echo "Starting experiment $EXP_NUM"

# Loop through all combinations
for lr in "${LEARNING_RATES[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        for model_size in "${MODEL_SIZES[@]}"; do
            # for use_scheduler in "${USE_SCHEDULER[@]}"; do
            echo "Running experiment with lr=${lr}, batch_size=${batch_size}, model_size=${model_size}, use_scheduler=${use_scheduler}"
            
            python train.py \
                -i "$DATASET_DIR" \
                -o "$OUTPUT_DIR" \
                -e "$EPOCHS" \
                -b "$batch_size" \
                -m "$model_size" \
                -l "$lr" \
                -s "$use_scheduler" \
                -w "$WARMUP_STEPS" \
                -ld "$EXP_DIR"
            
            echo "Experiment completed"
            echo "----------------------------------------"
            # done
        done
    done
done

echo "All experiments completed"