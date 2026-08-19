import sys
import os
import time
import json
import shutil
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from src.preprocessing import get_datasets, get_data_augmentation
from src.evaluate import evaluate_model, generate_evaluation_plots

# Logger Tee to output logs both to terminal and training.log
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

def build_custom_cnn(input_shape=(224, 224, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    x = get_data_augmentation()(inputs)
    x = tf.keras.layers.Rescaling(1./255)(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', name='conv_final')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name="CustomCNN")
    return model

def build_mobilenetv2(input_shape=(224, 224, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    x = get_data_augmentation()(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    except Exception as e:
        print(f"Notice: MobileNetV2 ImageNet weights download failed ({e}). Building without pretrained weights.", flush=True)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None
        )

    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name="MobileNetV2")
    return model

def build_resnet50(input_shape=(224, 224, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    x = get_data_augmentation()(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    
    try:
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    except Exception as e:
        print(f"Notice: ResNet50 ImageNet weights download failed ({e}). Building without pretrained weights.", flush=True)
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights=None
        )

    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name="ResNet50")
    return model

def train_model(model_name, build_fn, save_path, train_ds, val_ds, epochs=config.EPOCHS):
    print(f"\n==========================================")
    print(f"🚀 Training Architecture: {model_name}")
    print(f"==========================================")

    model = build_fn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_accuracy', save_best_only=True)
    ]

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    duration = time.time() - start_time
    print(f"✅ Training completed in {duration:.2f}s")

    # Ensure best model is saved
    model.save(save_path)
    return model, history

def main():
    log_path = os.path.join(config.OUTPUTS_DIR, "training.log")
    sys.stdout = LoggerTee(log_path)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    train_ds, val_ds, test_ds, class_names = get_datasets()

    architectures = [
        ("custom_cnn", build_custom_cnn, config.CUSTOM_CNN_PATH),
        ("mobilenetv2", build_mobilenetv2, config.MOBILENET_PATH),
        ("resnet50", build_resnet50, config.RESNET_PATH)
    ]

    results = {}
    best_model_name = None
    best_f1_score = -1.0
    best_model_path = None

    for name, build_fn, path in architectures:
        model, history = train_model(name, build_fn, path, train_ds, val_ds)
        
        # Evaluate model on test set
        metrics, y_true, y_pred_probs = evaluate_model(model, test_ds)
        metrics['history'] = {
            'train_loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'train_acc': [float(x) for x in history.history['accuracy']],
            'val_acc': [float(x) for x in history.history['val_accuracy']]
        }
        results[name] = metrics

        print(f"\n📊 {name.upper()} Test Metrics:")
        print(f"   Accuracy: {metrics['accuracy']*100:.2f}% | Precision: {metrics['precision']*100:.2f}% | Recall: {metrics['recall']*100:.2f}% | F1-Score: {metrics['f1_score']*100:.2f}% | ROC-AUC: {metrics['roc_auc']:.4f}")

        if metrics['f1_score'] > best_f1_score:
            best_f1_score = metrics['f1_score']
            best_model_name = name
            best_model_path = path

    print(f"\n🏆 Best Selected Model: {best_model_name.upper()} (F1-Score: {best_f1_score*100:.2f}%)")
    shutil.copy(best_model_path, config.FINAL_MODEL_PATH)
    print(f"💾 Saved final model to: {config.FINAL_MODEL_PATH}")

    # Save metrics JSON
    with open(config.METRICS_JSON_PATH, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "all_models": results
        }, f, indent=4)

    # Generate plots for best model
    best_model = tf.keras.models.load_model(config.FINAL_MODEL_PATH)
    _, y_true, y_pred_probs = evaluate_model(best_model, test_ds)
    generate_evaluation_plots(y_true, y_pred_probs)

if __name__ == "__main__":
    main()
