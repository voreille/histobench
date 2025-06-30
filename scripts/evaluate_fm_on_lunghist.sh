#!/bin/bash

# Variables for compute_embeddings_lunghist700.py
# MODEL="H-optimus-0"  # Change to your desired model
MODEL="moco_superpixel_cluster_bioptimus"  # Change to your desired model
# MODEL="resnet50"  # Change to your desired model
# MODEL="moco_superpixel_cluster_bioptimus"
# MODEL="moco_v2"
LABEL="superpixel_cluster"  # Change to your desired label
# LABEL="resnet50"  # Change to your desired label
WEIGHTS_PATH="/mnt/nas7/data/Personal/Darya/Checkpoints/superpixel_cluster_clean/cluster_bioptimus_queue_wo_fn_queue20480_alpha0.01_beta0.005_epochs60_moco_superpixel_cluster_bioptimus/best_model_epoch46.pth"  # Leave empty if not needed
# WEIGHTS_PATH=""  # Leave empty if not needed

INPUT_DIR="data/LungHist700/LungHist700_10x"
INPUT_BASENAME=$(basename "$INPUT_DIR")
GPU_ID=1
AGGREGATION="whole_roi"  # Options: whole_roi, tile_no_overlap, tile_with_overlap
# AGGREGATION="whole_roi"  # Options: whole_roi, tile_no_overlap, tile_with_overlap
TILE_SIZE=224
BATCH_SIZE=32
NUM_WORKERS=4
EMBEDDINGS_PATH="data/embeddings/lunghist700/${LABEL}_${INPUT_BASENAME}_${AGGREGATION}.h5"

# Variables for evaluate_lunghist700.py
CSV_METADATA="data/LungHist700/metadata.csv"
KNN_N_NEIGHBORS=5
REPORT_PATH="reports/lunghist700/${LABEL}_${INPUT_BASENAME}_${AGGREGATION}_KNNn_${KNN_N_NEIGHBORS}_cv_report.csv"
N_SPLITS=5 # Number of splits for cross-validation

# Check if embeddings file exists
if [ -f "$EMBEDDINGS_PATH" ]; then
    echo "Embeddings file $EMBEDDINGS_PATH already exists. Skipping embedding computation."
else
    # Run compute_embeddings_lunghist700.py
    python histobench/evaluation/compute_embeddings_lunghist700.py \
        --model "$MODEL" \
        --model-weights-path "$WEIGHTS_PATH" \
        --input-dir "$INPUT_DIR" \
        --gpu-id "$GPU_ID" \
        --aggregation "$AGGREGATION" \
        --tile-size "$TILE_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --embeddings-path "$EMBEDDINGS_PATH"
fi

# Run evaluate_lunghist700.py
python histobench/evaluation/evaluate_lunghist700.py \
    --csv-metadata "$CSV_METADATA" \
    --embeddings-path "$EMBEDDINGS_PATH" \
    --knn-n-neighbors "$KNN_N_NEIGHBORS" \
    --report-path "$REPORT_PATH" \
    --n-splits "$N_SPLITS"