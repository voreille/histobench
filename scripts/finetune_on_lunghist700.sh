#!/bin/bash

# filepath: /home/valentin/workspaces/histobench/scripts/fine_tune_lunghist700.sh

# Fine-tuning task parameters
MODEL="resnet50"
INPUT_DIR="data/LungHist700/LungHist700_10x"
CSV_METADATA="data/LungHist700/metadata.csv"
GPU_ID=0
BATCH_SIZE=32
NUM_WORKERS=4
LR=0.001
WEIGHT_DECAY=0.00001
MAX_EPOCHS=20
CHECKPOINT_DIR="checkpoints"

# Run the fine-tuning script
python /home/valentin/workspaces/histobench/histobench/finetuning/finetune_lunghist700_no_avg_pool.py \
    --model "$MODEL" \
    --input-dir "$INPUT_DIR" \
    --csv-metadata "$CSV_METADATA" \
    --gpu-id "$GPU_ID" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --max-epochs "$MAX_EPOCHS" \
    --checkpoint-dir "$CHECKPOINT_DIR"