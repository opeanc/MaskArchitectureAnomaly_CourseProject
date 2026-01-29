#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "Configurazione Percorsi Rilevata:"
echo "Script Dir:   $SCRIPT_DIR"
echo "Project Root: $PROJECT_ROOT"

# Path definition
CITYSCAPES_DIR="$PROJECT_ROOT/datasets/cityscapes"  
OBJ_DIR="$PROJECT_ROOT/datasets/final_ds"           
SAVE_DIR="$PROJECT_ROOT/trained_models"
PRETRAINED_WEIGHTS="$PROJECT_ROOT/trained_models/eomt_cityscapes.bin"
CONFIG_YAML="$SCRIPT_DIR/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"

echo "Weights:      $PRETRAINED_WEIGHTS"
echo "Output Dir:   $SAVE_DIR"
echo "=================================================="

# create save directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# python execution
python "$SCRIPT_DIR/finetune_eomt.py" \
    --cityscapes_dir "$CITYSCAPES_DIR" \
    --obj_dir "$OBJ_DIR" \
    --save_dir "$SAVE_DIR" \
    --pretrained_weights "$PRETRAINED_WEIGHTS" \
    --config_path "$CONFIG_YAML" \
    --batch-size 1 \
    --epochs 4 \
    --lr 1e-4 \
    --subset "train"
