"""
Shared configuration constants for the Tilapia Fish Disease Classification project.

Edit these values (or override via CLI flags in train.py / evaluate.py) to match
your local dataset layout.
"""

# ---- Data ----
TRAIN_DIR = "data/train"        # expects one sub-folder per class, e.g. data/train/Col, data/train/MAS, ...
TEST_DIR = "data/test"          # same class-folder layout, held out and never used for training/validation
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 123

# Disease classes used in the study:
#   Col  - Columnaris Disease
#   MAS  - Motile Aeromonad Septicemia
#   NN   - Healthy (no disease)
#   STr  - Streptococcosis
#   TiLV - Tilapia Lake Virus
CLASS_NAMES = ["Col", "MAS", "NN", "STr", "TiLV"]
NUM_CLASSES = len(CLASS_NAMES)

# ---- Training ----
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5

# ---- Output ----
MODEL_DIR = "saved_models"
