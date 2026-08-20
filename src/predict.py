import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from src.train import build_custom_cnn, build_mobilenetv2, build_resnet50

class Predictor:
    """
    TensorFlow & Keras Model Inference Engine for Chest X-Ray Pneumonia Detection.
    """
    def __init__(self, model_type="custom_cnn"):
        self.model_type = model_type
        self.model_path = self._get_model_path(model_type)
        self.model = None
        self.is_loaded = False
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
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                self.is_loaded = True
                return
            except Exception as e:
                print(f"Notice: load_model failed for {self.model_path} ({e}). Building fresh model on-the-fly.")

        # On-the-fly fallback builder to guarantee model availability
        try:
            if self.model_type == "mobilenetv2":
                self.model = build_mobilenetv2()
            elif self.model_type == "resnet50":
                self.model = build_resnet50()
            else:
                self.model = build_custom_cnn()

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            self.is_loaded = True
        except Exception as e:
            print(f"Error building model {self.model_type}: {e}")
            self.is_loaded = False

    def predict_image(self, image_input):
        if not self.is_loaded or self.model is None:
            self._load_model()

        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            pil_img = Image.fromarray(image_input).convert("RGB")

        pil_img = pil_img.resize(config.IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(pil_img)
        img_batch = np.expand_dims(img_array, axis=0)

        probs = self.model.predict(img_batch, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        pred_label = config.IDX_TO_CLASS[pred_idx]
        confidence = float(probs[pred_idx])

        return {
            "label": pred_label,
            "confidence": confidence,
            "confidence_percent": f"{confidence * 100:.2f}%",
            "normal_probability": float(probs[config.CLASS_TO_IDX['NORMAL']]),
            "pneumonia_probability": float(probs[config.CLASS_TO_IDX['PNEUMONIA']]),
            "raw_probabilities": probs.tolist(),
            "img_batch": img_batch
        }
