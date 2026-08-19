import os
import torch

# Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
PLOTS_DIR = os.path.join(OUTPUTS_DIR, 'plots')
GRADCAM_DIR = os.path.join(OUTPUTS_DIR, 'gradcam')
DATABASE_DIR = os.path.join(BASE_DIR, 'database')
DB_PATH = os.path.join(DATABASE_DIR, 'predictions.db')

CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, 'confusion_matrix.png')
ROC_CURVE_PATH = os.path.join(OUTPUTS_DIR, 'roc_curve.png')
METRICS_JSON_PATH = os.path.join(BASE_DIR, 'model_metrics.json')

# Hyperparameters & Settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 5

CLASSES = ['NORMAL', 'PNEUMONIA']
CLASS_TO_IDX = {'NORMAL': 0, 'PNEUMONIA': 1}
IDX_TO_CLASS = {0: 'NORMAL', 1: 'PNEUMONIA'}

# Compute Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Paths
CUSTOM_CNN_PATH = os.path.join(MODELS_DIR, 'custom_cnn.pth')
MOBILENET_PATH = os.path.join(MODELS_DIR, 'mobilenetv2.pth')
RESNET_PATH = os.path.join(MODELS_DIR, 'resnet50.pth')
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, 'final_model.pth')

# Image Normalization Statistics
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
