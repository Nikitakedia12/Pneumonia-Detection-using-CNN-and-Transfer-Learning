import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def evaluate_model(model, test_loader):
    """
    Evaluates a PyTorch model on a test DataLoader.
    Returns: metrics dict, y_true numpy array, y_pred_probs numpy array.
    """
    model.eval()
    y_true_list = []
    y_pred_probs_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            y_true_list.append(labels.numpy())
            y_pred_probs_list.append(probs)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred_classes)
    prec = precision_score(y_true, y_pred_classes, zero_division=0)
    rec = recall_score(y_true, y_pred_classes, zero_division=0)
    f1 = f1_score(y_true, y_pred_classes, zero_division=0)
    
    # Specificity Calculation
    cm = confusion_matrix(y_true, y_pred_classes)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    pneu_probs = y_pred_probs[:, 1] if y_pred_probs.shape[1] > 1 else y_pred_probs[:, 0]
    try:
        auc = roc_auc_score(y_true, pneu_probs)
    except Exception:
        auc = 0.5

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(auc)
    }

    return metrics, y_true, y_pred_probs

def generate_evaluation_plots(y_true, y_pred_probs):
    """
    Generates and saves Confusion Matrix and ROC Curve plots.
    """
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    pneu_probs = y_pred_probs[:, 1] if y_pred_probs.shape[1] > 1 else y_pred_probs[:, 0]

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASSES,
                yticklabels=config.CLASSES)
    plt.title('Pneumonia Detection Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, pneu_probs)
    auc_val = roc_auc_score(y_true, pneu_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall / Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(config.ROC_CURVE_PATH, dpi=300)
    plt.close()

    print(f"📊 Saved Confusion Matrix plot to: {config.CONFUSION_MATRIX_PATH}")
    print(f"📈 Saved ROC Curve plot to: {config.ROC_CURVE_PATH}")
