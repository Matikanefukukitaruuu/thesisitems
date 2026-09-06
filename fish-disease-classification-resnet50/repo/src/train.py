"""
Training entry point for the three model configurations evaluated in the study:

    baseline            - shallow CNN trained from scratch
    resnet_raw          - ResNet50 transfer learning, resized-only images (no augmentation/normalization)
    resnet_preprocessed - ResNet50 transfer learning, with augmentation + ResNet-specific normalization

Usage:
    python src/train.py --model resnet_preprocessed
    python src/train.py --model resnet_raw --epochs 50
    python src/train.py --model baseline

Trained models are written to config.MODEL_DIR as `<model_name>.keras`.
"""

import argparse
import os
import sys

from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import load_datasets
from src.models import build_baseline_model, build_resnet_model


def build_model(model_choice: str, learning_rate: float):
    if model_choice == "baseline":
        return build_baseline_model(learning_rate=learning_rate)
    if model_choice == "resnet_raw":
        return build_resnet_model(preprocess=False, learning_rate=learning_rate)
    if model_choice == "resnet_preprocessed":
        return build_resnet_model(preprocess=True, learning_rate=learning_rate)
    raise ValueError(f"Unknown --model '{model_choice}'")


def main():
    parser = argparse.ArgumentParser(description="Train a tilapia fish disease classification model.")
    parser.add_argument(
        "--model",
        choices=["baseline", "resnet_raw", "resnet_preprocessed"],
        required=True,
        help="Which model configuration to train.",
    )
    parser.add_argument("--data-dir", default=config.TRAIN_DIR, help="Path to the training data directory.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE, help="Adam learning rate.")
    parser.add_argument(
        "--patience",
        type=int,
        default=config.EARLY_STOPPING_PATIENCE,
        help="Early-stopping patience (epochs with no val_loss improvement before stopping).",
    )
    parser.add_argument("--output-dir", default=config.MODEL_DIR, help="Directory to save the trained model.")
    args = parser.parse_args()

    train_ds, val_ds, class_names = load_datasets(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    print(f"Detected classes: {class_names}")

    model = build_model(args.model, learning_rate=args.learning_rate)
    model.summary()

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f"{model.name}.keras")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(checkpoint_path)
    print(f"Saved trained model to {checkpoint_path}")

    return history


if __name__ == "__main__":
    main()
