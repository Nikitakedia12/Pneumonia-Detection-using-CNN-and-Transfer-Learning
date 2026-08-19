import sys
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class Predictor:
    def __init__(self, model_type="custom_cnn"):
        self.model_type = model_type
        self.model_path = self._get_model_path(model_type)
        self.model = None
        self.is_loaded = False

        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD)
        ])

        self._load_model()

    def _get_model_path(self, model_type):
        mapping = {
            "custom_cnn": config.CUSTOM_CNN_PATH,
            "mobilenetv2": config.MOBILENET_PATH,
            "resnet50": config.RESNET_PATH,
            "final_model": config.FINAL_MODEL_PATH
        }
        return mapping.get(model_type, config.FINAL_MODEL_PATH)

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = torch.load(self.model_path, map_location=config.DEVICE)
                self.model.eval()
                self.is_loaded = True
            except Exception as e:
                print(f"Error loading model from {self.model_path}: {e}")
                self.is_loaded = False
        else:
            self.is_loaded = False

    def predict_image(self, image_input):
        if not self.is_loaded:
            raise RuntimeError(f"Model at '{self.model_path}' is not loaded.")

        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            pil_img = Image.fromarray(image_input).convert("RGB")

        tensor_img = self.transform(pil_img).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            outputs = self.model(tensor_img)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = config.IDX_TO_CLASS[pred_idx]
        confidence = float(probs[pred_idx])

        return {
            "label": pred_label,
            "confidence": confidence,
            "confidence_percent": f"{confidence * 100:.2f}%",
            "normal_probability": float(probs[config.CLASS_TO_IDX['NORMAL']]),
            "pneumonia_probability": float(probs[config.CLASS_TO_IDX['PNEUMONIA']]),
            "raw_probabilities": probs.tolist()
        }
