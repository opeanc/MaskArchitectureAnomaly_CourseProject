#!/bin/bash
set -e
BASE_DIR="/Users/francescodedominicis/Desktop/POLITO/Advanced_ML/Project/Validation_Dataset"

DATASET="FS_LostFound_full"

echo "=== TEMPERATURE SCALING EXPERIMENT ON $DATASET ==="

# Testiamo 3 temperature comuni
for TEMP in 0.5 0.75 1 1.1; do
    echo "[1/5] Processing FS_LostFound_full..."
    echo ">> Running with T = $TEMP"
    python evalAnomaly_TS.py --num-workers 0 --subset FS_LostFound_full \
        --input "$BASE_DIR/FS_LostFound_full/images/*.png" \
        --temperature $TEMP
done

DATASET="fs_static"

echo "=== TEMPERATURE SCALING EXPERIMENT ON $DATASET ==="

# Testiamo 3 temperature comuni
for TEMP in 0.5 0.75 1 1.1; do
    echo "[2/5] Processing fs_static..."
    echo ">> Running with T = $TEMP"
    python evalAnomaly_TS.py --num-workers 0 --subset fs_static \
        --input "$BASE_DIR/fs_static/images/*.jpg" \
        --temperature $TEMP
done

DATASET="RoadAnomaly"

echo "=== TEMPERATURE SCALING EXPERIMENT ON $DATASET ==="

# Testiamo 3 temperature comuni
for TEMP in 0.5 0.75 1 1.1; do
    echo "[3/5] Processing RoadAnomaly..."
    echo ">> Running with T = $TEMP"
    python evalAnomaly_TS.py --num-workers 0 --subset RoadAnomaly \
        --input "$BASE_DIR/RoadAnomaly/images/*.jpg" \
        --temperature $TEMP
done

DATASET="RoadAnomaly21"

echo "=== TEMPERATURE SCALING EXPERIMENT ON $DATASET ==="

# Testiamo 3 temperature comuni
for TEMP in 0.5 0.75 1 1.1; do
    echo "[4/5] Processing RoadAnomaly21..."
    echo ">> Running with T = $TEMP"
    python evalAnomaly_TS.py --num-workers 0 --subset RoadAnomaly21 \
        --input "$BASE_DIR/RoadAnomaly21/images/*.png" \
        --temperature $TEMP
done

DATASET="RoadObsticle21"

echo "=== TEMPERATURE SCALING EXPERIMENT ON $DATASET ==="

# Testiamo 3 temperature comuni
for TEMP in 0.5 0.75 1 1.1; do
    echo "[5/5] Processing RoadObsticle21..."
    echo ">> Running with T = $TEMP"
    python evalAnomaly_TS.py --num-workers 0 --subset RoadObsticle21 \
        --input "$BASE_DIR/RoadObsticle21/images/*.webp" \
        --temperature $TEMP
done