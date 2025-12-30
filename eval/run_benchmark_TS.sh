#!/bin/bash
set -e
BASE_DIR="/Users/francescodedominicis/Desktop/POLITO/Advanced_ML/Project/Validation_Dataset"

# Definisci il dataset di test (es. RoadAnomaly21)
DATASET="RoadAnomaly21"

echo "=== TEMPERATURE SCALING EXPERIMENT ON $DATASET ==="

# Testiamo 3 temperature comuni
for TEMP in 0.5 0.75 1.1; do
    echo "[1/5] Processing RoadAnomaly21..."
    echo ">> Running with T = $TEMP"
    python evalAnomaly_TS.py --num-workers 0 --subset RoadAnomaly21 \
        --input "$BASE_DIR/RoadAnomaly21/images/*.png" \
        --temperature $TEMP
done