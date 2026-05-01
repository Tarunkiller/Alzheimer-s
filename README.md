# 🧠 Multi-Modal Alzheimer’s Disease Detection using Deep Learning (IEEE Published)

## 📌 Overview

This project presents a research-driven multi-modal deep learning system for Alzheimer’s disease detection by integrating heterogeneous medical data sources including MRI, CT, PET scans, and structured clinical data.

Unlike traditional single-modality approaches, this system leverages multiple data sources and multiple deep learning models to improve diagnostic accuracy and robustness. The work was independently developed and published in an IEEE international conference.

---

## 🚀 Key Contributions

* Designed a **multi-modal learning system** integrating MRI, CT, PET, and clinical data
* Built a **multi-model architecture** combining:

  * 27-layer Custom CNN (spatial feature extraction)
  * ResNet18 (deep residual learning)
  * Vision Transformer (global feature understanding)
* Trained on **~87,000+ medical data samples** across multiple modalities
* Implemented **multi-modal data fusion**, improving performance over single-input systems
* Achieved **~90%+ accuracy with fast convergence in just 10 epochs**
* Published in an **IEEE International Conference**, validating research contribution

---

## 🧠 Model Architecture

### 🔹 CNN (27 Layers)

Captures fine-grained spatial features from medical imaging data.

### 🔹 ResNet18

Uses residual connections to improve gradient flow and enable deeper feature learning.

### 🔹 Vision Transformer (ViT)

Captures global relationships and long-range dependencies in medical images.

### 🔹 Multi-Modal Fusion Layer

Combines outputs from all models along with clinical data to produce final predictions, improving robustness and generalization.

---

## ⚙️ System Workflow

1. Data Collection (~87K samples across MRI, CT, PET, clinical data)
2. Data Preprocessing (normalization, resizing, structuring)
3. Feature Extraction using CNN, ResNet18, and ViT
4. Multi-modal feature fusion
5. Classification into Alzheimer’s stages
6. Model evaluation and optimization

---

## 📊 Performance

* Dataset Size: ~87,000+ samples
* Accuracy: ~90%+
* Training Efficiency: Converged within 10 epochs
* Processing Speed: Optimized pipeline for faster training and inference

---

## 🔬 Research Contribution

* Demonstrates effectiveness of **multi-modal learning in healthcare AI**
* Improves diagnostic accuracy compared to single-model systems
* Provides a scalable architecture for real-world clinical applications

---

## 📄 Publication

IEEE International Conference — Alzheimer’s Detection using Multi-Modal Deep Learning

---

## 🛠️ Tech Stack

Python | TensorFlow / PyTorch | CNN | ResNet18 | Vision Transformer | NumPy | Pandas

---

## 👨‍💻 Author

Padamati Tarun Krishna
AI/ML Engineer (Aspiring)
