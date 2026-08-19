import sys
import os
import time
import json
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from src.preprocessing import get_dataloaders
from src.evaluate import evaluate_model, generate_evaluation_plots

class LoggerTee:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            encoding = getattr(self.terminal, 'encoding', 'utf-8') or 'utf-8'
            self.terminal.write(message.encode(encoding, errors='replace').decode(encoding, errors='replace'))
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def build_custom_cnn(num_classes=2):
    return CustomCNN(num_classes=num_classes)

def build_mobilenetv2(num_classes=2):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def build_resnet50(num_classes=2):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def train_model(model_name, build_fn, save_path, train_loader, val_loader, epochs=config.EPOCHS):
    print(f"\n==========================================")
    print(f"🚀 Training Architecture: {model_name}")
    print(f"==========================================")

    model = build_fn().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = float(correct) / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)

        val_acc = float(val_correct) / val_total
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model, save_path)

    duration = time.time() - start_time
    print(f"✅ Training completed in {duration:.2f}s | Saved to {save_path}")
    return model

def main():
    log_path = os.path.join(config.OUTPUTS_DIR, "training.log")
    sys.stdout = LoggerTee(log_path)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device Engine: {config.DEVICE}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    architectures = [
        ("custom_cnn", build_custom_cnn, config.CUSTOM_CNN_PATH),
        ("mobilenetv2", build_mobilenetv2, config.MOBILENET_PATH),
        ("resnet50", build_resnet50, config.RESNET_PATH)
    ]

    for name, build_fn, path in architectures:
        train_model(name, build_fn, path, train_loader, val_loader)

if __name__ == "__main__":
    main()
