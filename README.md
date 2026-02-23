# 🚦 Mask Architecture for Road Anomaly Segmentation 

> **💡 Note on this Repository:** This project was developed as a team effort for the Advanced Machine Learning course at Politecnico di Torino. It adapts the Mask Classification paradigm (using EoMT) for Road Anomaly Detection.

## 🏆 My Key Contributions
While the project was collaborative, my specific engineering and research focus included:
1. **Dynamic Anomaly Dataset Generation:** Implemented an on-the-fly Outlier Exposure (OE) pipeline injecting synthetic anomalies into Cityscapes scenes, complete with harmonization techniques (Color Transfer, Depth-dependent Blur, Noise Injection).
2. **High-Resolution Inference (Sliding Window):** Engineered a custom sliding window mechanism allowing the base model (designed for 1024x1024 inputs) to effectively process high-resolution 1024x2048 urban imagery without memory bottlenecks.
3. **ODIN Implementation on EoMT:** Extended the post-hoc ODIN method to the query-based EoMT architecture, designing and evaluating both *Pixel-wise* and *Query-level* perturbation strategies.

## 📂 Repository Structure

* **`eomt/`**: Contains the core architecture, training scripts, and configuration files. Adapted from the original EoMT repository to support our dynamic anomaly dataset and ODIN implementations.
* **`eval/`**: Contains scripts and tools for evaluating the model's output, visualizing predictions, and performing anomaly segmentation inference.
* **`datasets/`**: Directory for the training datasets (e.g., Cityscapes and our custom `final_ds` anomaly objects).
* **`Validation_Dataset/`**: Directory dedicated to the 5 evaluation benchmarks (RoadAnomaly, RoadObsticle21, Fishyscapes, etc.) required for the inference scripts.
* **`trained_models/`**: Pre-trained weights and checkpoints.

## 📄 Full Paper
For a deep dive into the methodology, loss functions (RbA and KL Divergence), and comprehensive benchmark results on RoadAnomaly, RoadObstacle, and Fishyscapes, please refer to our full paper: [Read the PDF Paper here](./Mask_Architecture_Anomaly_Segmentation_for_Road_Scenes.pdf).

## 🚀 Getting Started
Please refer to the specific documentation inside each module:
- [EoMT Training & Setup](eomt/README.md)
- [Evaluation & Inference](eval/README.md)

