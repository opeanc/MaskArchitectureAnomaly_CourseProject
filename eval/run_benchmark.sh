#!/bin/bash

# Attiva l'uscita immediata in caso di errore
set -e

# Percorso base dei dataset
BASE_DIR="/Users/francescodedominicis/Desktop/POLITO/Advanced_ML/Project/Validation_Dataset"

echo "========================================================"
echo "   STARTING ERFNet ANOMALY BENCHMARK (STEP 4)   "
echo "========================================================"

# 1. FS LostFound Full
echo ""
echo "[1/5] Processing FS_LostFound_full..."
python evalAnomaly.py --num-workers 0 --subset FS_LostFound_full \
    --input "$BASE_DIR/FS_LostFound_full/images/*.png"

# 2. FS Static
echo ""
echo "[2/5] Processing fs_static..."
python evalAnomaly.py --num-workers 0 --subset fs_static \
    --input "$BASE_DIR/fs_static/images/*.jpg"

# 3. RoadAnomaly
echo ""
echo "[3/5] Processing RoadAnomaly..."
python evalAnomaly.py --num-workers 0 --subset RoadAnomaly \
    --input "$BASE_DIR/RoadAnomaly/images/*.jpg"

# 4. RoadAnomaly21
echo ""
echo "[4/5] Processing RoadAnomaly21..."
python evalAnomaly.py --num-workers 0 --subset RoadAnomaly21 \
    --input "$BASE_DIR/RoadAnomaly21/images/*.png"

# 5. RoadObsticle21
echo ""
echo "[5/5] Processing RoadObsticle21..."
python evalAnomaly.py --num-workers 0 --subset RoadObsticle21 \
    --input "$BASE_DIR/RoadObsticle21/images/*.webp"

echo ""
echo "========================================================"
echo "   BENCHMARK COMPLETED SUCCESSFULLY! 🚀"
echo "   Check results in: eval/results.txt"
echo "========================================================"