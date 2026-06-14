"""
Dataset and data-loader utilities for neonatal jaundice classification.

The project uses four classes:
- data/normal
- data/jaundiced/mild
- data/jaundiced/moderate
- data/jaundiced/severe

Each sample can be a skin image, a sclera image, or a pair. Pairing is supported
through filename conventions:
- baby001_skin.jpg and baby001_sclera.jpg
- baby001-skin.jpg and baby001-sclera.jpg
- baby001.jpg as a single unpaired image

The dataset returns:
{
    "image": Tensor[C, H, W],
    "features": Tensor[YCBCR_FEATURE_DIM],
    "label": LongTensor scalar,
    "path": original image path,
    "modality": "skin" | "sclera" | "combined"
}
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF

import config
from skin_tone_normalizer import SkinToneNormalizer


@dataclass(frozen=True)
class ImageSample:
    """Metadata for one training sample."""

    path: Path
    label: int
    class_name: str
    modality: str
    paired_path: Optional[Path] = None


class AddGaussianNoise:
    """Add mild sensor-like noise after tensor conversion."""

    def __init__(self, std: float = config.GAUSSIAN_NOISE_STD) -> None:
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class NeonatalJaundiceDataset(Dataset):
    """
    PyTorch dataset for skin/sclera jaundice images.

    Args:
        samples: List of ImageSample objects.
        train: Enables stochastic augmentation when True.
        image_size: Final square input size for EfficientNet-B0.
        normalize_skin_tone: Apply fairness-oriented skin-tone normalization.
    """

    def __init__(
        self,
        samples: Sequence[ImageSample],
        train: bool = False,
        image_size: int = config.IMAGE_SIZE,
        normalize_skin_tone: bool = config.SKIN_NORMALIZATION_ENABLED,
    ) -> None:
        self.samples = list(samples)
        self.train = train
        self.image_size = image_size
        self.normalize_skin_tone = normalize_skin_tone
        self.skin_normalizer = SkinToneNormalizer()
        self.transform = self._build_transform(train=train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]

        image = self._load_rgb(sample.path)
        if sample.paired_path is not None:
            paired = self._load_rgb(sample.paired_path)
            image = self._combine_skin_and_sclera(image, paired)

        if self.normalize_skin_tone:
            image = self.skin_normalizer.normalize(image)

        ycbcr_features = extract_ycbcr_features(image)
        pil_image = Image.fromarray(image)
        image_tensor = self.transform(pil_image)

        return {
            "image": image_tensor,
            "features": torch.tensor(ycbcr_features, dtype=torch.float32),
            "label": torch.tensor(sample.label, dtype=torch.long),
            "path": str(sample.path),
            "modality": sample.modality,
        }

    def _build_transform(self, train: bool) -> transforms.Compose:
        """Create training or evaluation transforms."""

        if train:
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
                    transforms.RandomRotation(config.ROTATION_DEGREES),
                    transforms.ColorJitter(
                        brightness=config.BRIGHTNESS_JITTER,
                        contrast=config.CONTRAST_JITTER,
                        saturation=config.SATURATION_JITTER,
                        hue=config.HUE_JITTER,
                    ),
                    transforms.ToTensor(),
                    AddGaussianNoise(config.GAUSSIAN_NOISE_STD),
                    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
                ]
            )

        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
            ]
        )

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        """Load an image from disk as RGB uint8."""

        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _combine_skin_and_sclera(self, image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
        """
        Combine paired skin and sclera images into one model input.

        The left half contains the first image and the right half contains the
        paired image. This keeps the EfficientNet input three-channel and avoids
        introducing custom model code at the dataset stage.
        """

        size = (self.image_size, self.image_size)
        left = cv2.resize(image_a, size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(image_b, size, interpolation=cv2.INTER_AREA)
        combined = np.concatenate([left[:, : self.image_size // 2], right[:, self.image_size // 2 :]], axis=1)
        return combined


def discover_samples(data_dir: Path = config.DATA_DIR) -> List[ImageSample]:
    """
    Discover labeled images from the project folder structure.

    Returns a flat list of ImageSample records. Sclera/skin files are paired
    when matching filename stems are found; unpaired files remain valid samples.
    """

    class_dirs = {
        "normal": config.NORMAL_DIR,
        "mild": config.MILD_DIR,
        "moderate": config.MODERATE_DIR,
        "severe": config.SEVERE_DIR,
    }

    samples: List[ImageSample] = []
    for class_name, folder in class_dirs.items():
        if not folder.exists():
            continue

        files = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS
        )
        samples.extend(_build_class_samples(files, class_name))

    return samples


def _build_class_samples(files: Sequence[Path], class_name: str) -> List[ImageSample]:
    """Pair skin/sclera files where possible and create ImageSample objects."""

    label = config.CLASS_TO_IDX[class_name]
    grouped: Dict[str, Dict[str, Path]] = {}
    untagged: List[Path] = []

    for path in files:
        base_stem, modality = _parse_modality(path.stem)
        if modality in {"skin", "sclera"}:
            grouped.setdefault(base_stem, {})[modality] = path
        else:
            untagged.append(path)

    samples: List[ImageSample] = []
    used_paths = set()

    for group in grouped.values():
        skin = group.get("skin")
        sclera = group.get("sclera")
        if skin is not None and sclera is not None:
            samples.append(
                ImageSample(
                    path=skin,
                    paired_path=sclera,
                    label=label,
                    class_name=class_name,
                    modality="combined",
                )
            )
            used_paths.update({skin, sclera})

    for group in grouped.values():
        for modality, path in group.items():
            if path not in used_paths:
                samples.append(
                    ImageSample(
                        path=path,
                        label=label,
                        class_name=class_name,
                        modality=modality,
                    )
                )

    for path in untagged:
        samples.append(
            ImageSample(
                path=path,
                label=label,
                class_name=class_name,
                modality="skin",
            )
        )

    return samples


def _parse_modality(stem: str) -> Tuple[str, Optional[str]]:
    """Infer modality from common filename suffixes."""

    lowered = stem.lower()
    for token in ("_skin", "-skin", " skin"):
        if lowered.endswith(token):
            return stem[: -len(token)], "skin"
    for token in ("_sclera", "-sclera", " sclera", "_eye", "-eye"):
        if lowered.endswith(token):
            return stem[: -len(token)], "sclera"
    return stem, None


def extract_ycbcr_features(image_rgb: np.ndarray) -> np.ndarray:
    """
    Extract robust YCbCr color statistics from an RGB image.

    Cb is explicitly included because yellowing lowers blue-difference chroma.
    The returned values are scaled to roughly human-readable [0, 1] ranges.
    """

    image_uint8 = _to_uint8_rgb(image_rgb)
    ycbcr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2YCrCb).astype(np.float32)

    # OpenCV's YCrCb channel order is Y, Cr, Cb. We rename explicitly to avoid
    # the common Cb/Cr swap mistake.
    y = ycbcr[:, :, 0] / 255.0
    cr = ycbcr[:, :, 1] / 255.0
    cb = ycbcr[:, :, 2] / 255.0

    yellow_blue_ratio = float(np.mean((1.0 - cb) / (y + 1e-6)))

    return np.array(
        [
            float(np.mean(y)),
            float(np.std(y)),
            float(np.mean(cb)),
            float(np.std(cb)),
            float(np.percentile(cb, 10)),
            float(np.percentile(cb, 50)),
            float(np.percentile(cb, 90)),
            float(np.mean(cr)),
            float(np.std(cr)),
            yellow_blue_ratio,
        ],
        dtype=np.float32,
    )


def create_train_val_test_split(
    samples: Sequence[ImageSample],
    train_ratio: float = config.TRAIN_SPLIT,
    val_ratio: float = config.VAL_SPLIT,
    test_ratio: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> Tuple[List[ImageSample], List[ImageSample], List[ImageSample]]:
    """
    Stratified 70/15/15 split by class.

    Splitting within each class helps preserve minority classes in validation
    and test sets when the dataset is imbalanced.
    """

    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    by_label: Dict[int, List[ImageSample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)

    rng = random.Random(seed)
    train_samples: List[ImageSample] = []
    val_samples: List[ImageSample] = []
    test_samples: List[ImageSample] = []

    for label_samples in by_label.values():
        label_samples = list(label_samples)
        rng.shuffle(label_samples)

        n = len(label_samples)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        # Keep at least one test item when possible after train/val allocation.
        if n >= 3 and n_train + n_val >= n:
            n_train = max(1, n_train - 1)

        train_samples.extend(label_samples[:n_train])
        val_samples.extend(label_samples[n_train : n_train + n_val])
        test_samples.extend(label_samples[n_train + n_val :])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    return train_samples, val_samples, test_samples


def build_weighted_sampler(samples: Sequence[ImageSample]) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to reduce class imbalance during training.

    Classes with fewer images receive higher sampling probability. This is
    preferred over naive oversampling on disk because it avoids duplicate files.
    """

    if not samples:
        raise ValueError("Cannot build a sampler from an empty sample list")

    labels = torch.tensor([sample.label for sample in samples], dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=config.NUM_CLASSES).float()
    class_counts = torch.clamp(class_counts, min=1.0)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(
    data_dir: Path = config.DATA_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    use_weighted_sampler: bool = config.USE_WEIGHTED_SAMPLER,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Discover images, split them 70/15/15, and return train/val/test loaders.
    """

    samples = discover_samples(data_dir)
    if not samples:
        raise ValueError(
            "No images found. Add images under data/normal and data/jaundiced/{mild,moderate,severe}."
        )

    train_samples, val_samples, test_samples = create_train_val_test_split(samples)

    train_dataset = NeonatalJaundiceDataset(train_samples, train=True)
    val_dataset = NeonatalJaundiceDataset(val_samples, train=False)
    test_dataset = NeonatalJaundiceDataset(test_samples, train=False)

    sampler = build_weighted_sampler(train_samples) if use_weighted_sampler else None
    shuffle = sampler is None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert numpy image arrays to RGB uint8."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected RGB image with shape H x W x 3")
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    if array.max(initial=0) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    all_samples = discover_samples()
    train, val, test = create_train_val_test_split(all_samples)
    print(f"Discovered {len(all_samples)} samples")
    print(f"Train/Val/Test: {len(train)}/{len(val)}/{len(test)}")
