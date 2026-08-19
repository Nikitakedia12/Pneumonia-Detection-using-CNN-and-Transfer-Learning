# 🫁 Pneumonia Detection using TensorFlow 2.x, Keras & Computer Vision

An end-to-end medical deep learning system for automated chest X-ray pneumonia detection featuring **TensorFlow 2.x + Keras**, Transfer Learning (Custom CNN, MobileNetV2, ResNet50), **Grad-CAM visual heatmaps via `tf.GradientTape`**, SQLite prediction database logging, and a Streamlit Web Dashboard.

---

## 📁 Repository Structure

```text
pneumonia-detection/
│
├── data/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── models/
│   ├── custom_cnn.keras
│   ├── mobilenetv2.keras
│   ├── resnet50.keras
│   └── final_model.keras
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── gradcam.py
│   └── preprocessing.py
│
├── database/
│   ├── predictions.db
│   └── database.py
│
├── app/
│   └── app.py
│
├── outputs/
│   ├── plots/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── gradcam/
│
├── config/
│   └── config.py
│
├── model_metrics.json
├── requirements.txt
├── CONCEPTS_GUIDE.md
└── README.md
```

---

## 🚀 Quick Start & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train TensorFlow Models
Train deep learning models (Custom CNN, MobileNetV2, ResNet50) and select the best model:
```bash
python src/train.py
```
This automatically evaluates on the test dataset, generates `outputs/confusion_matrix.png`, `outputs/roc_curve.png`, updates `model_metrics.json`, and saves the best model to `models/final_model.keras`.

### 3. Launch Streamlit Web Application
```bash
py -m streamlit run app/app.py
```
Open **`http://localhost:8501`** in your web browser.

---

## ⚙️ Key Features

1. **TensorFlow + Keras Pipeline**: Fully built using `tf.data` and `tf.keras`.
2. **Multi-Model Support**: Custom CNN architecture, MobileNetV2 (Transfer Learning), and ResNet50.
3. **Grad-CAM Heatmaps**: Explainable AI visualization using `tf.GradientTape()` gradient tracing.
4. **SQLite Database Audit**: Inferences are automatically logged into `database/predictions.db`.
5. **Interactive Dashboard**: Streamlit interface with upload, sample gallery, confidence scores, and analytics.

---

## ⚠️ Medical Disclaimer

> **This system is intended for educational and research purposes only. It is not a medical diagnostic tool and should not replace evaluation by a qualified healthcare professional.**
