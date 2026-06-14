"""
Project configuration for the Non-Invasive Neonatal Jaundice Detection system.

This file keeps medically meaningful constants, data paths, and training
hyperparameters in one place so that experiments, reports, and the Flask app
use the same thresholds and labels.

Important clinical note:
The thresholds below are useful for project staging and triage-oriented model
outputs. A real clinical deployment must be validated against local protocols,
gestational age, birth weight, age in hours, and serum bilirubin confirmation.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
SYNTHETIC_DIR = ROOT_DIR / "synthetic"
APP_DIR = ROOT_DIR / "app"
REPORTS_DIR = ROOT_DIR / "reports"

NORMAL_DIR = DATA_DIR / "normal"
JAUNDICED_DIR = DATA_DIR / "jaundiced"
MILD_DIR = JAUNDICED_DIR / "mild"
MODERATE_DIR = JAUNDICED_DIR / "moderate"
SEVERE_DIR = JAUNDICED_DIR / "severe"


# ---------------------------------------------------------------------------
# Labels and bilirubin thresholds
# ---------------------------------------------------------------------------

NUM_CLASSES = 4

CLASS_NAMES = ["Normal", "Mild", "Moderate", "Severe"]

CLASS_TO_IDX = {
    "normal": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}

IDX_TO_CLASS = {value: key for key, value in CLASS_TO_IDX.items()}

# Thresholds are expressed in mg/dL.
BILIRUBIN_THRESHOLDS = {
    "Normal": {"min": 0.0, "max": 5.0, "description": "< 5 mg/dL"},
    "Mild": {"min": 5.0, "max": 12.0, "description": "5-12 mg/dL"},
    "Moderate": {"min": 12.0, "max": 20.0, "description": "12-20 mg/dL"},
    "Severe": {"min": 20.0, "max": float("inf"), "description": "> 20 mg/dL"},
}


# ---------------------------------------------------------------------------
# Input image settings
# ---------------------------------------------------------------------------

IMAGE_SIZE = 224
IMAGE_CHANNELS = 3
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# EfficientNet-B0 uses ImageNet normalization when initialized with pretrained
# weights. Keep this separate from skin-tone normalization, which is performed
# in image/color space before tensor normalization.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Train/validation/test split
# ---------------------------------------------------------------------------

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

MODEL_NAME = "efficientnet_b0"
PRETRAINED = True
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.40
NUM_WORKERS = 2
PIN_MEMORY = True
EARLY_STOPPING_PATIENCE = 8
GRADIENT_CLIP_NORM = 1.0
CLASSIFICATION_LOSS_WEIGHT = 0.70
REGRESSION_LOSS_WEIGHT = 0.30
BILIRUBIN_TARGETS = {
    0: 2.5,
    1: 8.5,
    2: 16.0,
    3: 24.0,
}

# Class imbalance control. The dataset module can build a
# WeightedRandomSampler from the training subset when this flag is enabled.
USE_WEIGHTED_SAMPLER = True


# ---------------------------------------------------------------------------
# Augmentation hyperparameters
# ---------------------------------------------------------------------------

HORIZONTAL_FLIP_PROB = 0.50
ROTATION_DEGREES = 12
BRIGHTNESS_JITTER = 0.20
CONTRAST_JITTER = 0.15
SATURATION_JITTER = 0.15
HUE_JITTER = 0.03
GAUSSIAN_NOISE_STD = 0.025


# ---------------------------------------------------------------------------
# Skin-tone normalization settings
# ---------------------------------------------------------------------------

# Reference chroma values are intentionally conservative. They normalize broad
# illumination/color-cast differences while avoiding aggressive recoloring that
# could erase jaundice signals.
SKIN_NORMALIZATION_ENABLED = True
SKIN_MASK_MIN_AREA_RATIO = 0.03
REFERENCE_LAB_A = 11.0
REFERENCE_LAB_B = 18.0
CHROMA_BLEND_STRENGTH = 0.35
LUMINANCE_CLIP_LIMIT = 2.0


# ---------------------------------------------------------------------------
# YCbCr feature extraction settings
# ---------------------------------------------------------------------------

# Jaundice-related yellowing is often reflected by blue-yellow opponent
# behavior. In YCbCr, Cb is especially useful because yellow regions tend to
# reduce blue-difference chroma. We keep several robust summary statistics.
YCBCR_FEATURE_NAMES = [
    "y_mean",
    "y_std",
    "cb_mean",
    "cb_std",
    "cb_p10",
    "cb_p50",
    "cb_p90",
    "cr_mean",
    "cr_std",
    "yellow_blue_ratio",
]

YCBCR_FEATURE_DIM = len(YCBCR_FEATURE_NAMES)
