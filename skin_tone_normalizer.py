"""
Skin-tone normalization utilities for neonatal jaundice images.

The goal is fairness and robustness across Indian newborns with different
complexions and across rural-hospital lighting conditions. This module reduces
lighting/color-cast variation without forcing every baby into one artificial
skin tone. That distinction matters: jaundice color is part of the diagnostic
signal, so normalization must be gentle and mask-aware.

Expected input/output convention:
- Public methods accept RGB images as numpy arrays with shape H x W x 3.
- Pixel values may be uint8 in [0, 255] or float in [0, 1].
- Returned images are uint8 RGB arrays in [0, 255].
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import config


@dataclass(frozen=True)
class NormalizationStats:
    """Small audit record that can be logged during experiments."""

    skin_area_ratio: float
    lab_a_shift: float
    lab_b_shift: float
    used_skin_mask: bool


class SkinToneNormalizer:
    """
    Normalize image color while preserving jaundice-relevant chroma cues.

    The pipeline is deliberately conservative:
    1. Apply gray-world white balance to reduce camera/lighting color casts.
    2. Detect probable skin pixels using YCrCb and HSV rules.
    3. Equalize luminance only on the L channel in LAB space.
    4. Gently blend skin chroma toward a reference range instead of replacing it.

    This is suitable for preprocessing and training consistency. It is not a
    substitute for collecting a representative, clinically labeled dataset.
    """

    def __init__(
        self,
        min_skin_area_ratio: float = config.SKIN_MASK_MIN_AREA_RATIO,
        reference_lab_a: float = config.REFERENCE_LAB_A,
        reference_lab_b: float = config.REFERENCE_LAB_B,
        chroma_blend_strength: float = config.CHROMA_BLEND_STRENGTH,
        luminance_clip_limit: float = config.LUMINANCE_CLIP_LIMIT,
    ) -> None:
        self.min_skin_area_ratio = min_skin_area_ratio
        self.reference_lab_a = reference_lab_a
        self.reference_lab_b = reference_lab_b
        self.chroma_blend_strength = float(np.clip(chroma_blend_strength, 0.0, 1.0))
        self.luminance_clip_limit = luminance_clip_limit

    def normalize(self, image_rgb: np.ndarray, return_stats: bool = False):
        """
        Normalize an RGB image.

        Args:
            image_rgb: RGB image as uint8 [0, 255] or float [0, 1].
            return_stats: When True, return (image, NormalizationStats).

        Returns:
            Normalized RGB uint8 image, optionally with stats.
        """

        image = self._to_uint8_rgb(image_rgb)
        balanced = self._gray_world_white_balance(image)
        skin_mask = self.create_skin_mask(balanced)

        skin_area_ratio = float(np.mean(skin_mask > 0))
        used_skin_mask = skin_area_ratio >= self.min_skin_area_ratio

        lab = cv2.cvtColor(balanced, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 0] = self._normalize_luminance(lab[:, :, 0])

        if used_skin_mask:
            lab, a_shift, b_shift = self._normalize_skin_chroma(lab, skin_mask)
        else:
            # When the mask is too small, applying chroma correction would be
            # unstable. Luminance normalization still helps low-light images.
            a_shift = 0.0
            b_shift = 0.0

        normalized = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

        if not return_stats:
            return normalized

        stats = NormalizationStats(
            skin_area_ratio=skin_area_ratio,
            lab_a_shift=float(a_shift),
            lab_b_shift=float(b_shift),
            used_skin_mask=used_skin_mask,
        )
        return normalized, stats

    def create_skin_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Build a probable skin mask using complementary color-space thresholds.

        YCrCb is useful for skin chroma clustering; HSV helps reject saturated
        backgrounds and dark shadows. The thresholds are intentionally broad
        because newborn skin appearance varies substantially.
        """

        image = self._to_uint8_rgb(image_rgb)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Broad YCrCb skin range used as a starting point for many skin-detection
        # pipelines. It is not ethnicity-specific and should be validated on the
        # project dataset.
        ycrcb_mask = cv2.inRange(
            ycrcb,
            np.array([35, 115, 70], dtype=np.uint8),
            np.array([255, 180, 150], dtype=np.uint8),
        )

        # HSV range removes extremely dark regions and high-saturation objects.
        hsv_mask = cv2.inRange(
            hsv,
            np.array([0, 10, 35], dtype=np.uint8),
            np.array([50, 210, 255], dtype=np.uint8),
        )

        mask = cv2.bitwise_and(ycrcb_mask, hsv_mask)

        # Morphological cleanup makes summary statistics less sensitive to
        # individual pixels, compression artifacts, or cloth/background edges.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return mask

    def _normalize_luminance(self, l_channel: np.ndarray) -> np.ndarray:
        """Apply CLAHE on LAB luminance to reduce uneven lighting."""

        l_uint8 = np.clip(l_channel, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=self.luminance_clip_limit,
            tileGridSize=(8, 8),
        )
        return clahe.apply(l_uint8).astype(np.float32)

    def _normalize_skin_chroma(
        self,
        lab: np.ndarray,
        skin_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Gently shift LAB a/b channels on probable skin pixels.

        We use median chroma instead of mean chroma because median is more
        robust to blankets, sensor noise, and specular highlights.
        """

        mask = skin_mask > 0
        if not np.any(mask):
            return lab, 0.0, 0.0

        a_median = float(np.median(lab[:, :, 1][mask]))
        b_median = float(np.median(lab[:, :, 2][mask]))

        a_target = 128.0 + self.reference_lab_a
        b_target = 128.0 + self.reference_lab_b
        a_shift = (a_target - a_median) * self.chroma_blend_strength
        b_shift = (b_target - b_median) * self.chroma_blend_strength

        # Soft mask prevents sharp color transitions around face/skin borders.
        alpha = (skin_mask.astype(np.float32) / 255.0) * self.chroma_blend_strength
        lab[:, :, 1] = lab[:, :, 1] + alpha * a_shift
        lab[:, :, 2] = lab[:, :, 2] + alpha * b_shift

        return lab, a_shift, b_shift

    def _gray_world_white_balance(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Correct global color cast using gray-world white balance.

        Rural clinical images may be captured under tube lights, daylight, or
        phone flash. This simple correction is transparent and fast enough for
        mobile/edge preprocessing.
        """

        image = image_rgb.astype(np.float32)
        channel_means = image.reshape(-1, 3).mean(axis=0)
        gray_mean = float(channel_means.mean())
        scale = gray_mean / np.maximum(channel_means, 1e-6)
        balanced = image * scale.reshape(1, 1, 3)
        return np.clip(balanced, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
        """Validate and convert common image array formats to uint8 RGB."""

        if image is None:
            raise ValueError("image_rgb cannot be None")

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape H x W x 3")

        if array.dtype == np.uint8:
            return array.copy()

        array = array.astype(np.float32)
        if array.max(initial=0) <= 1.0:
            array = array * 255.0

        return np.clip(array, 0, 255).astype(np.uint8)


def normalize_skin_tone(image_rgb: np.ndarray) -> np.ndarray:
    """Convenience function for one-off normalization calls."""

    return SkinToneNormalizer().normalize(image_rgb)
