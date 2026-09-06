"""
Evaluate a trained model on the independent, held-out test set.

Computes overall accuracy, a full classification report (precision/recall/F1
per class), and plots a confusion matrix.

Usage:
    python src/evaluate.py --model-path saved_models/tilapia_resnet50_preprocessed.keras
    python src/evaluate.py --model-path saved_models/tilapia_baseline_cnn.keras --test-dir data/test --save-fig confusion_matrix.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import load_test_dataset


def evaluate_labeled_test(model, dataset, class_names):
    """Run predictions on `dataset`, print accuracy + classification report, and return the confusion matrix."""
    y_true = np.concatenate([y.numpy() for _, y in dataset], axis=0)
    y_true_labels = np.argmax(y_true, axis=1)

    preds = model.predict(dataset)
    y_pred_labels = np.argmax(preds, axis=1)

    accuracy = np.mean(y_true_labels == y_pred_labels)

    print("\n==============================")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("==============================")

    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=class_names))

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    return cm, accuracy


def plot_confusion_matrix(cm, class_names, save_path: str = None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix figure to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the held-out test set.")
    parser.add_argument("--model-path", required=True, help="Path to the saved .keras model file.")
    parser.add_argument("--test-dir", default=config.TEST_DIR, help="Path to the test data directory.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size.")
    parser.add_argument("--save-fig", default=None, help="If set, save the confusion matrix figure to this path instead of displaying it.")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)

    test_ds = load_test_dataset(test_dir=args.test_dir, batch_size=args.batch_size)
    class_names = test_ds.class_names

    cm, _ = evaluate_labeled_test(model, test_ds, class_names)
    plot_confusion_matrix(cm, class_names, save_path=args.save_fig)


if __name__ == "__main__":
    main()
