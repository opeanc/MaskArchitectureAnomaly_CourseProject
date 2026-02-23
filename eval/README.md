# Anomaly Segmentation Evaluation & Post-Hoc Analysis

This module contains the comprehensive evaluation suite used to benchmark our models. We significantly extended the base evaluation scripts to support high-resolution inputs via a custom sliding window, Temperature Scaling grid search, and custom ODIN perturbation strategies for query-based architectures.

## 🏆 Key Engineering Contributions in this Module

1. **High-Resolution "Sliding Window" Inference:** Mask-based Transformers like EoMT are extremely memory-intensive. We implemented a robust `window_inference` logic across our scripts that cleanly slices $1024 \times 2048$ inputs, processes them, and stitches the output logits (or mask/class queries) back together without artifacts.
2. **Automated Temperature Scaling (TS) Search:** Implemented an automated grid search pipeline (`evalAnomaly_TS_eomt.py`) that efficiently extracts valid pixel logits, unloads the model from VRAM to prevent OOM errors, and computes the optimal Temperature $T$ to maximize AUPRC.
3. **ODIN on Mask Transformers:** Designed and implemented two novel ways to adapt ODIN (Out-of-DIstribution classifier for Neural networks) for the EoMT architecture, requiring custom forward-backward passes during inference to apply input perturbations.

## ⚙️ Requirements & Data Setup

Make sure you are running these scripts from within the project's Python environment.

⚠️ **Dataset Configuration:**
To run the evaluation scripts correctly, you must place the benchmark datasets inside the `Validation_Dataset` folder located at the root of the repository. The folder structure must exactly match the following:

\`\`\`text
MaskArchitectureAnomaly_CourseProject/
├── Validation_Dataset/
│   ├── FS_LostFound_full/
│   ├── fs_static/
│   ├── RoadAnomaly/
│   ├── RoadAnomaly21/
│   └── RoadObsticle21/
├── eval/
│   └── ...
\`\`\`

*(Note: Ensure the dataset directories contain the respective `images` and `labels_masks` subfolders as expected by the dataloaders).*

You can pass the correct path to the scripts using the `--input` argument. For example, to evaluate on RoadObsticle21 from within the `eval` directory:
\`\`\`bash
python evalAnomaly_eomt.py --input '../Validation_Dataset/RoadObsticle21/images/*.webp'
\`\`\`

## 🚀 Available Evaluation Scripts

We categorized our evaluation scripts based on the specific technique being benchmarked:

### 1. Standard Anomaly Evaluation
Computes baseline anomaly scores (MSP, MaxLogit, MaxEntropy, and our customized RbA score).
* **`evalAnomaly.py`**: Evaluates the baseline ERFNet model.
* **`evalAnomaly_eomt.py`**: Evaluates the fine-tuned EoMT model, including the Region-based Anomaly (RbA) calculation on object queries.

\`\`\`bash
# Example: Evaluate EoMT on RoadObstacle21
python evalAnomaly_eomt.py --input '../datasets/RoadObstacle21/images/*.webp' --loadWeights 'your_model.ckpt'
\`\`\`

### 2. Temperature Scaling (TS)
Enhances separation between In-Distribution and OOD samples by scaling logits.
* **`evalAnomaly_TS.py`**: TS for the ERFNet baseline.
* **`evalAnomaly_TS_eomt.py`**: TS for EoMT. Supports both fixed-temperature evaluation (`--temperature 1.5`) and an automated grid search for the optimal $T$ (`--best-temperature`).

\`\`\`bash
# Example: Find the optimal Temperature for EoMT
python evalAnomaly_TS_eomt.py --input '../datasets/RoadObstacle21/images/*.webp' --best-temperature
\`\`\`

### 3. ODIN (Input Perturbation)
*Note: As detailed in our paper, these scripts demonstrate that ODIN struggles with mask-based architectures due to the decoupling of spatial masks and semantic queries.*
* **`evalAnomaly_odin_pixel_based.py`**: Applies ODIN perturbation based on the fused pixel-wise probabilities.
* **`evalAnomaly_odin_query_level.py`**: Applies ODIN perturbation exclusively targeting the valid object queries' class logits.

### 4. Closed-Set Semantic Segmentation (mIoU)
Ensures that the anomaly fine-tuning does not degrade the model's standard segmentation capabilities on Cityscapes.
* **`eval_iou_eomt.py`**: Calculates the Mean Intersection over Union (mIoU) for the EoMT model.

\`\`\`bash
python eval_iou_eomt.py --datadir '../datasets/cityscapes/' --subset val
\`\`\`

---
*For visualization scripts (e.g., color map generation) and server evaluation tools inherited from the original repository, please refer to `eval_cityscapes_color.py` and `eval_cityscapes_server.py`.*