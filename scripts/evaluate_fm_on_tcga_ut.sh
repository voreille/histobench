#!/bin/bash

# Variables for compute_embeddings_tcga_ut.py
# MODEL="moco_superpixel_cluster_bioptimus"
MODEL="resnet50"
WEIGHTS_PATH="/mnt/nas7/data/Personal/Darya/Checkpoints/superpixel_cluster_clean/cluster_bioptimus_bs256_lr0.003_epochs100_clusters50_moco_superpixel_cluster_bioptimus/best_model.pth"
MAGNIFICATION_KEY=5
TCGA_UT_DIR="data/tcga-ut"
GPU_ID=1
BATCH_SIZE=256
NUM_WORKERS=24
EMBEDDINGS_PATH="data/embeddings/tcga_ut/${MODEL}__mag_key_${MAGNIFICATION_KEY}.h5"

# Variables for evaluate_tcag_ut.py
CSV_SPLIT_PATH="data/tcga-ut/train_val_test_split.csv"
KNN_N_NEIGHBORS=5
REPORT_PATH="reports/tcga-ut/${MODEL}__mag_key_5__internal_external_report.csv"

# Check if embeddings file exists
if [ -f "$EMBEDDINGS_PATH" ]; then
    echo "Embeddings file $EMBEDDINGS_PATH already exists. Skipping embedding computation."
else
    # Run compute_embeddings_tcga_ut.py
    python histobench/evaluation/compute_embeddings_tcga_ut.py \
        --model "$MODEL" \
        --model-weights-path "$WEIGHTS_PATH" \
        --magnification-key "$MAGNIFICATION_KEY" \
        --tcga-ut-dir "$TCGA_UT_DIR" \
        --gpu-id "$GPU_ID" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --embeddings-path "$EMBEDDINGS_PATH"
fi

# Run evaluate_tcag_ut.py
python histobench/evaluation/evaluate_tcag_ut.py \
    --csv-split-path "$CSV_SPLIT_PATH" \
    --embeddings-path "$EMBEDDINGS_PATH" \
    --knn-n-neighbors "$KNN_N_NEIGHBORS" \
    --report-path "$REPORT_PATH"