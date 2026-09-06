# Fish Disease Classification using Transfer Learning with ResNet50

Image-based classification of Nile tilapia (*Oreochromis niloticus*) health
conditions using a ResNet50 transfer-learning model, benchmarked against a
shallow CNN baseline. This repository contains the model, training, and
evaluation code used in the accompanying thesis / journal manuscript.

The system classifies fish images into five categories:

| Code   | Class                                |
|--------|---------------------------------------|
| `Col`  | Columnaris Disease                    |
| `MAS`  | Motile Aeromonad Septicemia           |
| `NN`   | Healthy                               |
| `STr`  | Streptococcosis                       |
| `TiLV` | Tilapia Lake Virus                    |

## Results

Three configurations were trained and evaluated on an independent, held-out
test set of 274 images:

| Model                          | Test Accuracy | Macro F1 | Epochs to Converge |
|---------------------------------|:-------------:|:--------:|:-------------------:|
| Baseline CNN (from scratch)     | 54.38%        | 0.48     | 12                  |
| ResNet50 (non-preprocessed)     | 84.67%        | 0.84     | 34                  |
| ResNet50 (preprocessed)         | 87.96%        | 0.87     | 11                  |

The **preprocessed** configuration applies data augmentation (random flip,
rotation, zoom) and ResNet-specific pixel normalization before the frozen
ResNet50 backbone; the **non-preprocessed** configuration only resizes
images. See the paper for full class-level metrics, confusion matrices, and
a heatmap-based interpretability analysis.

## Project structure

```
.
├── config.py            # shared constants (paths, image size, batch size, etc.)
├── requirements.txt
├── src/
│   ├── models.py         # build_baseline_model(), build_resnet_model()
│   ├── dataset.py        # dataset loading (train/val split + held-out test set)
│   ├── train.py           # CLI training entry point
│   └── evaluate.py        # CLI evaluation entry point (metrics + confusion matrix)
└── saved_models/          # trained models are written here (gitignored)
```

## Setup

```bash
git clone <this-repo-url>
cd <this-repo>
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Dataset layout

This repository does not include the image dataset. Organize your images in
a class-per-folder layout, with a separate, disjoint folder for the held-out
test set:

```
data/
├── train/
│   ├── Col/
│   ├── MAS/
│   ├── NN/
│   ├── STr/
│   └── TiLV/
└── test/
    ├── Col/
    ├── MAS/
    ├── NN/
    ├── STr/
    └── TiLV/
```

`data/train` is automatically split 80/20 into training and validation
subsets at load time (see `src/dataset.py`); `data/test` is loaded separately
and is never seen during training or validation. Update the paths in
`config.py` if your data lives elsewhere.

## Training

```bash
# Baseline CNN (trained from scratch, no pretrained weights)
python src/train.py --model baseline

# ResNet50, resized-only inputs (no augmentation/normalization)
python src/train.py --model resnet_raw

# ResNet50, full preprocessing pipeline (augmentation + ResNet normalization)
python src/train.py --model resnet_preprocessed
```

Common flags: `--epochs`, `--batch-size`, `--learning-rate`, `--patience`
(early-stopping patience on validation loss), `--output-dir`. Run
`python src/train.py --help` for the full list. Trained models are saved to
`saved_models/<model_name>.keras`.

## Evaluation

```bash
python src/evaluate.py --model-path saved_models/tilapia_resnet50_preprocessed.keras
```

Prints overall accuracy and a full per-class precision/recall/F1
classification report, and displays (or saves, with `--save-fig`) a
confusion matrix heatmap.

## Model architecture

- **Baseline CNN**: 3 convolutional blocks (32 → 64 → 128 filters, 3×3,
  ReLU + max-pooling) → flatten → dense(128, ReLU) → dropout(0.5) →
  softmax(5). Trained from scratch with Adam.
- **ResNet50 transfer learning**: ResNet50 pretrained on ImageNet
  (`include_top=False`), convolutional base frozen → GlobalAveragePooling2D
  → dropout(0.4) → dense(256, ReLU) → dropout(0.3) → softmax(5). Optionally
  preceded by a `RandomFlip`/`RandomRotation`/`RandomZoom` augmentation
  stack and ResNet-specific `preprocess_input` normalization.

Final training configuration (selected via hyperparameter search over batch
size, learning rate, and optimizer — see the paper): Adam optimizer,
learning rate `1e-3`, batch size `32`, categorical cross-entropy loss,
input size `256×256×3`, max 50 epochs with early stopping on validation
loss (best weights restored).

## Citation

If you use this code, please cite the accompanying paper:

> M. A. Olalo Jr., A. K. M. Canlas, and J. G. D. Villegas, "Deep Transfer
> Learning for Fish Disease Classification in Nile Tilapia (*Oreochromis
> niloticus*) Using ResNet50: A Comparative Study on Image Preprocessing
> Effects," 2026.

## License

MIT — see [LICENSE](LICENSE).
