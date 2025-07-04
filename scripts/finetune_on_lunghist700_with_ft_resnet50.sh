#!/bin/bash

# filepath: /home/valentin/workspaces/histobench/scripts/fine_tune_lunghist700.sh

# Fine-tuning task parameters
MODEL="resnet50"
WEIGHTS_PATH=""
INPUT_DIR="data/LungHist700/LungHist700_10x"
CSV_METADATA="data/LungHist700/metadata.csv"
GPU_ID=0
BATCH_SIZE=32
NUM_WORKERS=4
LR=0.001
WEIGHT_DECAY=0.00001
MAX_EPOCHS=500
CHECKPOINT_DIR="checkpoints/resnet50/"
CSV_FOLD_PATH="/home/valentin/workspaces/histobench/data/cv_splits_lunghist700_superclass_n_fold_4.csv"

# Run the fine-tuning script
python /home/valentin/workspaces/histobench/histobench/finetuning/finetune_lunghist700_avg_pool_with_finetuner.py \
    --model "$MODEL" \
    --model-weights-path "$WEIGHTS_PATH" \
    --input-dir "$INPUT_DIR" \
    --csv-metadata "$CSV_METADATA" \
    --gpu-id "$GPU_ID" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --max-epochs "$MAX_EPOCHS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --fold-csv-path "$CSV_FOLD_PATH" 