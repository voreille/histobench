#!/bin/bash

# filepath: /home/valentin/workspaces/histobench/scripts/evaluate_finetuned_model.sh

# Evaluation task parameters
CHECKPOINT_PATH="/home/valentin/workspaces/histobench/checkpoints/moco_superpixel_cluster_bioptimus/epoch=289-step=6090.ckpt"
# CHECKPOINT_PATH="/home/valentin/workspaces/histobench/checkpoints/moco_superpixel_cluster_bioptimus/epoch=252-step=5313.ckpt"
INPUT_DIR="data/LungHist700/LungHist700_10x"
CSV_METADATA="data/LungHist700/metadata.csv"
BATCH_SIZE=32
NUM_WORKERS=4
GPU_ID=0
LABEL_COLUMN="superclass"

# Run the evaluation script
python /home/valentin/workspaces/histobench/histobench/finetuning/evaluate_finetuned_models_lunghist700.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --input-dir "$INPUT_DIR" \
    --csv-metadata "$CSV_METADATA" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --gpu-id "$GPU_ID" \
    --label-column "$LABEL_COLUMN"