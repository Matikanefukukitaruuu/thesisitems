"""
Dataset loading utilities.

Expected directory layout (class-per-folder), e.g.:

    data/train/Col/...
    data/train/MAS/...
    data/train/NN/...
    data/train/STr/...
    data/train/TiLV/...

    data/test/Col/...
    data/test/MAS/...
    ...

`load_datasets()` splits data/train into training/validation subsets (80/20).
`load_test_dataset()` loads the fully held-out test set, never seen during
training or validation.
"""

import os
import sys

import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_datasets(data_dir: str = config.TRAIN_DIR,
                   img_size: tuple = config.IMG_SIZE,
                   batch_size: int = config.BATCH_SIZE,
                   validation_split: float = config.VALIDATION_SPLIT,
                   seed: int = config.SEED):
    """Load the training and validation datasets from a class-per-folder directory."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds, class_names


def load_test_dataset(test_dir: str = config.TEST_DIR,
                       img_size: tuple = config.IMG_SIZE,
                       batch_size: int = config.BATCH_SIZE):
    """Load the independent, held-out test set. shuffle=False keeps labels aligned with predictions."""
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    return test_ds
