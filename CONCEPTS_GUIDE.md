# 🫁 Comprehensive Concepts & Architecture Guide
## Pneumonia Detection using TensorFlow, Keras & Computer Vision

---

## 📌 1. Medical Background & Problem Statement

### 🫁 What is Pneumonia?
Pneumonia is an inflammatory lung infection caused by bacteria, viruses, or fungi. It causes the pulmonary alveoli (air sacs) in one or both lungs to fill with fluid or pus (known as **infiltrate** or **consolidation**).

### 🩺 Radiographic Indicators on Chest X-Rays (CXR):
- **Normal X-Ray**: Clean, dark lung fields (air appears black on radiographs because X-rays pass through freely).
- **Pneumonic X-Ray**: White/gray cloudy patches (**opacities**) indicating fluid-filled lung tissue, blurry lung margins, or lobar consolidation.

### 🎯 Objective of this System:
Automate binary classification of Chest X-Ray images into **`NORMAL`** or **`PNEUMONIA`** using TensorFlow & Keras Deep Learning models, providing radiologists with:
1. Probability confidence scores.
2. Visual spatial heatmaps (**Grad-CAM via `tf.GradientTape`**) identifying the exact lung regions driving the decision.
3. Persistent SQLite prediction logging for diagnostic audits.

---

## 🖼️ 2. Computer Vision & TensorFlow Data Pipeline

### 🔄 Image Preprocessing Steps:
1. **Resizing**: Standardized to **\(224 \times 224\)** pixels to match input dimensions required by deep neural networks.
2. **Preprocessing**: Model-specific scaling (Rescaling `1./255`, `tf.keras.applications.mobilenet_v2.preprocess_input`, `tf.keras.applications.resnet50.preprocess_input`).

### 🎲 Keras Data Augmentation Layers (Train Set):
- **`RandomFlip("horizontal")`**: Simulates minor patient positioning variations.
- **`RandomRotation(0.05)`**: Accounts for slight camera angle rotations.
- **`RandomBrightness(0.1)` & `RandomContrast(0.1)`**: Simulates varying X-ray exposure across different scanners.

---

## 🏗️ 3. Deep Learning Architectures Explained

This repository supports **3 distinct model architectures** built in TensorFlow / Keras:

### 🧠 A. Custom CNN (Keras Sequential/Functional)
Built from scratch to understand fundamental feature extraction layers:

```
Input Image (224, 224, 3)
   │
   ├── Conv2D(32, 3x3) ──► BatchNormalization ──► ReLU ──► MaxPool(2x2)
   ├── Conv2D(64, 3x3) ──► BatchNormalization ──► ReLU ──► MaxPool(2x2)
   ├── Conv2D(128, 3x3) ──► BatchNormalization ──► ReLU ──► MaxPool(2x2)
   │
   ├── GlobalAveragePooling2D
   ├── Dropout(0.4)
   ├── Dense(128, activation='relu')
   └── Dense(2, activation='softmax') ──► Output Probabilities
```

---

### 📱 B. MobileNetV2 Transfer Learning
- **Base Model**: `tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet')`
- **Feature Extraction**: Backbone weights are frozen (`trainable = False`), and custom classification head is trained.

---

### 🏛️ C. ResNet50 Transfer Learning
- **Base Model**: `tf.keras.applications.ResNet50(include_top=False, weights='imagenet')`
- **Residual Learning**: Uses residual skip connections to extract complex spatial features across 50 deep layers.

---

## 🎯 4. Explainable AI (XAI): TensorFlow Grad-CAM

### 📐 Mathematical Formulation with `tf.GradientTape()`:
1. **Feature Map Tracing**: Extract activations \(A^k\) from the last convolutional layer.
2. **Gradient Tape**: Compute gradients of class score \(y^c\) with respect to feature map activations \(A^k\):
   \[
   \frac{\partial y^c}{\partial A^k_{i, j}}
   \]
3. **Channel Importance Pooling**:
   \[
   \alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A^k_{i, j}}
   \]
4. **Heatmap Output**: Weighted combination followed by ReLU activation:
   \[
   L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)
   \]

---

## 📊 5. Evaluation Metrics & Clinical Significance

| Metric | Formula | Medical Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | \(\frac{TP + TN}{TP + TN + FP + FN}\) | Overall percentage of correct diagnostic predictions. |
| **Sensitivity (Recall)** | \(\frac{TP}{TP + FN}\) | **Critical Medical Metric**: Proportion of actual Pneumonia cases correctly identified. |
| **Specificity** | \(\frac{TN}{TN + FP}\) | Proportion of Healthy/Normal cases correctly identified. |
| **Precision** | \(\frac{TP}{TP + FP}\) | Reliability of a Pneumonia positive diagnosis. |
| **F1-Score** | \(2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\) | Harmonic mean balancing precision and recall. |
| **ROC-AUC** | Area under ROC Curve | Discriminative capability across decision thresholds. |

---

## ⚠️ Medical Disclaimer

> **This system is intended for educational and research purposes only. It is not a medical diagnostic tool and should not replace evaluation by a qualified healthcare professional.**
