import sys
import os
import cv2
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class GradCAM:
    """
    Robust PyTorch Grad-CAM engine using activation tensor gradient hooks.
    """
    def __init__(self, model, model_type="custom_cnn"):
        self.model = model
        self.model_type = model_type
        self.target_layer = self._find_target_layer()

    def _find_target_layer(self):
        if self.model_type == "mobilenetv2" and hasattr(self.model, "features"):
            return self.model.features[-1]
        elif self.model_type == "resnet50" and hasattr(self.model, "layer4"):
            return self.model.layer4[-1]

        target = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                target = module
        return target

    def generate_heatmap(self, input_tensor):
        # Ensure model parameters allow gradient calculation
        for p in self.model.parameters():
            p.requires_grad = True

        gradients = []
        activations = []

        def save_activation(module, input, output):
            activations.append(output)
            if output.requires_grad:
                output.register_hook(lambda grad: gradients.append(grad))

        handle = None
        if self.target_layer is not None:
            handle = self.target_layer.register_forward_hook(save_activation)

        input_tensor = input_tensor.to(config.DEVICE)
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().detach().numpy()
        pred_idx = int(np.argmax(probs))

        self.model.zero_grad()
        score = output[0, pred_idx]
        score.backward(retain_graph=True)

        if handle:
            handle.remove()

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
            else:
                cam = (acts - acts.min()) / (acts.max() - acts.min() + 1e-8)
                cam = np.mean(cam, axis=0)
            cam = cv2.resize(cam, config.IMAGE_SIZE)
        else:
            # Fallback high-contrast intensity heatmap visualization
            img_np = input_tensor[0].cpu().detach().numpy().transpose(1, 2, 0)
            gray = np.mean(img_np, axis=2)
            cam = cv2.GaussianBlur(gray, (15, 15), 0)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = cv2.resize(cam, config.IMAGE_SIZE)

        return cam, pred_idx, probs

    def overlay_heatmap(self, pil_image, cam, alpha=0.45):
        orig_resized = np.array(pil_image.resize(config.IMAGE_SIZE))
        if len(orig_resized.shape) == 2:
            orig_resized = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2RGB)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig_resized, 1 - alpha, heatmap, alpha, 0)
        return heatmap, overlay
