# 🧠 Multi-Modal Early-Stage Alzheimer’s Disease Detection using Deep Learning (IEEE Published)

## 📌 Overview

This project presents a research-driven deep learning system for **early-stage Alzheimer’s disease detection** using a multi-modal approach that integrates MRI, CT, PET scans, and clinical data.

Unlike traditional systems that rely on a single modality, this approach combines multiple data sources and multiple models to detect subtle early-stage patterns, enabling faster and more accurate diagnosis.

This work was independently developed and published in an IEEE international conference.

---

## 🚀 Key Contributions

* Designed a system focused on **early-stage Alzheimer’s detection**
* Integrated **multi-modal inputs**:

  * MRI
  * CT
  * PET
  * Clinical Data
* Built a **multi-model architecture**:

  * 27-layer Custom CNN
  * ResNet18
  * Vision Transformer (ViT)
* Trained on **~87,000+ medical data samples**
* Implemented **multi-modal feature fusion**
* Achieved **~90%+ accuracy with convergence in just 10 epochs**
* Published in an **IEEE International Conference**

---

## 🧠 Model Architecture

### 🔹 Custom CNN (27 Layers)

* Extracts detailed spatial features from medical images
* Effective for detecting subtle early-stage patterns

### 🔹 ResNet18

* Uses residual learning for deep feature extraction
* Improves gradient flow and training stability

### 🔹 Vision Transformer (ViT)

* Captures global dependencies in image data
* Enhances understanding of complex medical patterns

### 🔹 Multi-Modal Fusion

* Combines outputs from all models
* Integrates clinical data with imaging features
* Produces final prediction with improved accuracy

---

## ⚙️ Workflow

1. Data Collection (~87,000+ samples across modalities)
2. Preprocessing:

   * Image normalization and resizing
   * Clinical data structuring
3. Feature Extraction (CNN, ResNet18, ViT)
4. Multi-modal feature fusion
5. Classification into Alzheimer’s stages
6. Model evaluation and optimization

---

## 📊 Performance

* Dataset Size: ~87,000+ samples
* Accuracy: ~90%+
* Fast convergence within 10 epochs
* Optimized processing pipeline
* Strong performance in **early-stage detection**

---

## 🔬 Research Contribution

* Demonstrates effectiveness of **multi-modal learning in healthcare AI**
* Improves early detection compared to single-model systems
* Provides a scalable solution for real-world medical applications

---

## 📄 Publication

IEEE International Conference
Topic: Multi-Modal Deep Learning for Early Alzheimer’s Detection

---

## 🛠️ Tech Stack

Python | TensorFlow / PyTorch | CNN | ResNet18 | Vision Transformer | NumPy | Pandas

---

## 📂 Project Structure

```
data/
models/
notebooks/
train.py
evaluate.py
README.md
```

---

## 👨‍💻 Author

Padamati Tarun Krishna
AI/ML Engineer (Aspiring)

---

## 📌 Note

This project was independently developed as a final-year research work focusing on AI-based healthcare solutions.
