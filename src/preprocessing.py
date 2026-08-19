import sys
import os
import tensorflow as tf

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def get_data_augmentation():
    """
    Keras Sequential data augmentation pipeline.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")
    return data_augmentation

def get_datasets(batch_size=config.BATCH_SIZE, image_size=config.IMAGE_SIZE):
    """
    Loads train, validation, and test datasets using tf.keras.utils.image_dataset_from_directory.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )

    class_names = train_ds.class_names

    # Performance optimization with prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
