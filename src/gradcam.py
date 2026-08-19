import sys
import os
import cv2
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class GradCAM:
    """
    TensorFlow Grad-CAM visual heatmap generator using tf.GradientTape.
    """
    def __init__(self, model, last_conv_layer_name=None):
        self.model = model
        if last_conv_layer_name is None:
            # Auto-detect last convolutional layer in Keras model
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
                    last_conv_layer_name = layer.name
                    break
                elif isinstance(layer, tf.keras.Model):
                    for sublayer in reversed(layer.layers):
                        if isinstance(sublayer, tf.keras.layers.Conv2D) or "conv" in sublayer.name.lower():
                            last_conv_layer_name = sublayer.name
                            break
                    if last_conv_layer_name:
                        break
        self.last_conv_layer_name = last_conv_layer_name or "conv_final"

    def generate_heatmap(self, img_batch, pred_index=None):
        try:
            grad_model = tf.keras.models.Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
            )
        except Exception:
            # Fallback if layer is nested inside base_model
            grad_model = self.model

        with tf.GradientTape() as tape:
            if hasattr(grad_model, 'inputs'):
                conv_outputs, predictions = grad_model(img_batch)
            else:
                predictions = grad_model(img_batch)
                conv_outputs = predictions

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        if grads is not None:
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
            cam = heatmap.numpy()
        else:
            cam = np.zeros(config.IMAGE_SIZE, dtype=np.float32)

        cam = cv2.resize(cam, config.IMAGE_SIZE)
        return cam, int(pred_index), predictions[0].numpy()

    def overlay_heatmap(self, pil_image, cam, alpha=0.4):
        orig_resized = np.array(pil_image.resize(config.IMAGE_SIZE))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig_resized, 1 - alpha, heatmap, alpha, 0)
        return heatmap, overlay
