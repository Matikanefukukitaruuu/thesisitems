"""
Model architectures used in the study:

1. build_baseline_model() -- a shallow CNN trained from scratch, used as the
   comparative reference point against the ResNet50 transfer-learning models.

2. build_resnet_model()   -- a ResNet50 backbone pretrained on ImageNet, used as
   a frozen feature extractor with a custom classification head. Pass
   preprocess=True to enable the data-augmentation + ResNet-specific
   normalization pipeline (the "preprocessed" configuration), or
   preprocess=False to train on resized-only images (the "non-preprocessed"
   configuration).
"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_baseline_model(num_classes: int = config.NUM_CLASSES,
                          img_size: tuple = config.IMG_SIZE,
                          learning_rate: float = 1e-4) -> keras.Model:
    """Shallow CNN baseline: 3 conv blocks -> dense -> softmax, trained from scratch."""
    inputs = keras.Input(shape=(*img_size, 3))
    x = inputs

    x = layers.Rescaling(1.0 / 255)(x)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="tilapia_baseline_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_resnet_model(preprocess: bool = True,
                        num_classes: int = config.NUM_CLASSES,
                        img_size: tuple = config.IMG_SIZE,
                        learning_rate: float = config.LEARNING_RATE) -> keras.Model:
    """
    ResNet50 transfer-learning model.

    preprocess=True  -> applies random flip/rotation/zoom augmentation and
                         ResNet-specific pixel normalization before the frozen
                         ResNet50 backbone ("preprocessed" configuration).
    preprocess=False -> feeds resized-only images directly to the backbone
                         ("non-preprocessed" configuration).
    """
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*img_size, 3),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*img_size, 3))
    x = inputs

    if preprocess:
        x = data_augmentation(x)
        x = keras.applications.resnet.preprocess_input(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model_name = "tilapia_resnet50_preprocessed" if preprocess else "tilapia_resnet50_raw"
    model = keras.Model(inputs, outputs, name=model_name)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
