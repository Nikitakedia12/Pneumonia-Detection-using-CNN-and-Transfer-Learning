import sys
import os
import cv2
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class GradCAM:
    def __init__(self, model, model_type="custom_cnn"):
        self.model = model
        self.model.eval()
        self.model_type = model_type
        
        # Select target layer based on model architecture
        if model_type == "mobilenetv2":
            self.target_layer = self.model.features[-1]
        elif model_type == "resnet50":
            self.target_layer = self.model.layer4[-1]
        else:
            self.target_layer = self.model.features[12] if hasattr(self.model, 'features') and len(self.model.features) > 12 else list(self.model.children())[-2]

    def generate_heatmap(self, input_tensor):
        gradients = []
        activations = []

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        def forward_hook(module, input, output):
            activations.append(output)

        handle_f = self.target_layer.register_forward_hook(forward_hook)
        handle_b = self.target_layer.register_full_backward_hook(backward_hook)

        output = self.model(input_tensor.to(config.DEVICE))
        probs = torch.softmax(output, dim=1)[0].cpu().detach().numpy()
        pred_idx = np.argmax(probs)

        self.model.zero_grad()
        score = output[0, pred_idx]
        score.backward()

        handle_f.remove()
        handle_b.remove()

        if len(gradients) > 0 and len(activations) > 0:
            grads = gradients[0].cpu().data.numpy()[0]
            acts = activations[0].cpu().data.numpy()[0]

            weights = np.mean(grads, axis=(1, 2))
            cam = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * acts[i]

            cam = np.maximum(cam, 0)
            if np.max(cam) > 0:
                cam = cam / np.max(cam)
            cam = cv2.resize(cam, config.IMAGE_SIZE)
        else:
            cam = np.zeros(config.IMAGE_SIZE, dtype=np.float32)

        return cam, pred_idx, probs

    def overlay_heatmap(self, pil_image, cam, alpha=0.4):
        orig_resized = np.array(pil_image.resize(config.IMAGE_SIZE))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig_resized, 1 - alpha, heatmap, alpha, 0)
        return heatmap, overlay
