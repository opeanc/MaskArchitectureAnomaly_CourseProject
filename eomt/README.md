# EoMT Architecture & Fine-Tuning Pipeline

This directory contains the core implementation of the Encoder-only Mask Transformer (EoMT), extensively modified and extended by our team to support dynamic Outlier Exposure (OE) and custom loss functions for Anomaly Segmentation.

## 🏆 Key Contributions in this Module

Instead of relying solely on the base repository's training routines, we implemented a custom fine-tuning pipeline tailored for our research objectives:

1. **On-the-fly Anomaly Injection (`datasets/anomaly_cityscapes.py`)**: We designed a custom PyTorch `Dataset` class (`CityscapesAnomalyDataset`) that dynamically samples objects from our `final_ds` dataset and injects them into Cityscapes scenes during training. This includes bounding box constraints and on-the-fly harmonization.
2. **Custom Fine-Tuning Loop (`finetune_eomt.py`)**: We developed a dedicated training script that:
   - Freezes the backbone and isolates the `class_head` and `mask_head` for fine-tuning.
   - Dynamically applies either standard `MaskClassificationLoss` for clean images or our custom Out-of-Distribution losses (`RbALoss` or `KLLoss`) when anomalies are detected in the batch.
   - Implements **Gradient Accumulation** to simulate a batch size of 16 (required for stable training on high-resolution $1024 \times 2048$ images) while physically operating with a batch size of 1 to fit within GPU VRAM limits.
3. **Execution Script (`run_finetuning.sh`)**: A streamlined Bash script to manage paths and launch the fine-tuning process.

## 📂 Directory Structure & Datasets Warning

⚠️ **Important Note regarding `datasets` folders:** There are two distinct `datasets` directories in this project:
* `../datasets/` (Root level): **This is where your raw data goes.** You must place the `cityscapes/` directory (containing `leftImg8bit` and `gtFine`) and the `final_ds/` directory (our custom anomaly objects dataset) here. The link to download `final_ds` is available in our paper. Do not put `.zip` files here; extract them.
* `./datasets/` (Inside `eomt/`): **This contains the Python code** for the data loaders, including our custom `anomaly_cityscapes.py`.

## 🚀 Usage

### 1. Data Preparation
Ensure your root `datasets` directory is structured as follows:
\`\`\`text
MaskArchitectureAnomaly_CourseProject/
├── datasets/
│   ├── cityscapes/
│   │   ├── leftImg8bit/
│   │   └── gtFine/
│   └── final_ds/  # Download link in the paper
├── eomt/
│   └── ...
\`\`\`

### 2. Fine-Tuning the Model
We provide a shell script to easily launch our custom fine-tuning pipeline. Ensure you have the base pre-trained weights (`eomt_cityscapes.bin`) inside the `../trained_models/` directory.

\`\`\`bash
cd eomt
chmod +x run_finetuning.sh
./run_finetuning.sh
\`\`\`

By default, the script is configured to use a batch size of 1 and run for 4 epochs, simulating a batch size of 16 via gradient accumulation. To switch between the `RbALoss` and the `KLLoss`, modify the `useRbaLoss` boolean variable directly inside `finetune_eomt.py`.

### 3. Evaluation
For evaluating the generated models, please refer to the tools provided in the `../eval/` directory.