"""
Offline OpenCV region detection for neonatal jaundice screening.

This module does not require internet access. It uses OpenCV Haar cascades,
skin-color masks, and image-quality checks to:
- Detect the baby face.
- Extract forehead/chest skin regions.
- Extract likely sclera regions from eye photos.
- Return retake guidance when regions are unclear.

The detector is intentionally conservative. If the app cannot confidently find
usable skin/sclera regions, it should guide the user to retake the image instead
of silently producing a low-quality screening result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from skin_tone_normalizer import SkinToneNormalizer


@dataclass
class RegionResult:
    """Detection output for one image."""

    region_rgb: Optional[np.ndarray]
    overlay_rgb: np.ndarray
    mask: np.ndarray
    confidence: float
    clear: bool
    message: str
    box: Optional[Tuple[int, int, int, int]] = None


class OfflineRegionDetector:
    """OpenCV-only detector for skin and sclera regions."""

    def __init__(self) -> None:
        self.normalizer = SkinToneNormalizer()
        haar_dir = Path(cv2.data.haarcascades)
        self.face_cascade = cv2.CascadeClassifier(str(haar_dir / "haarcascade_frontalface_default.xml"))
        self.eye_cascade = cv2.CascadeClassifier(str(haar_dir / "haarcascade_eye.xml"))

    def analyze(self, skin_photo_rgb: np.ndarray, eye_photo_rgb: np.ndarray) -> Dict[str, RegionResult]:
        """Run face/skin and sclera detection together."""

        skin = self.extract_skin_regions(skin_photo_rgb)
        sclera = self.extract_sclera_region(eye_photo_rgb)
        return {"skin": skin, "sclera": sclera}

    def extract_skin_regions(self, image_rgb: np.ndarray) -> RegionResult:
        """
        Extract forehead/chest skin area.

        If a face is detected, forehead is preferred. Otherwise, the detector
        falls back to the largest skin-colored region, often chest/limb.
        """

        image = ensure_uint8_rgb(image_rgb)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        skin_mask = self.normalizer.create_skin_mask(image)

        overlay = image.copy()
        selected_box: Optional[Tuple[int, int, int, int]] = None
        region = None
        confidence = 0.0

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            forehead_y1 = max(y + int(0.08 * h), 0)
            forehead_y2 = min(y + int(0.34 * h), image.shape[0])
            forehead_x1 = max(x + int(0.22 * w), 0)
            forehead_x2 = min(x + int(0.78 * w), image.shape[1])
            selected_box = (forehead_x1, forehead_y1, forehead_x2 - forehead_x1, forehead_y2 - forehead_y1)
            region = image[forehead_y1:forehead_y2, forehead_x1:forehead_x2]
            mask_crop = skin_mask[forehead_y1:forehead_y2, forehead_x1:forehead_x2]
            confidence = 0.60 + 0.35 * float(np.mean(mask_crop > 0))
        else:
            selected_box = largest_mask_box(skin_mask)
            if selected_box is not None:
                x, y, w, h = selected_box
                region = image[y : y + h, x : x + w]
                confidence = 0.40 + 0.45 * float(np.mean(skin_mask[y : y + h, x : x + w] > 0))

        if selected_box is not None:
            x, y, w, h = selected_box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 180, 90), 3)
            overlay[skin_mask > 0] = (0.65 * overlay[skin_mask > 0] + 0.35 * np.array([46, 204, 113])).astype(np.uint8)

        quality_ok, quality_message = image_quality_message(image)
        area_ok = selected_box is not None and selected_box[2] * selected_box[3] >= image.shape[0] * image.shape[1] * 0.01
        clear = bool(region is not None and confidence >= 0.55 and quality_ok and area_ok)
        message = "Skin region detected." if clear else retake_message("skin", quality_message, area_ok, confidence)

        return RegionResult(region, overlay, skin_mask, float(min(confidence, 1.0)), clear, message, selected_box)

    def extract_sclera_region(self, image_rgb: np.ndarray) -> RegionResult:
        """Extract likely sclera pixels from an eye/face photo."""

        image = ensure_uint8_rgb(image_rgb)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(24, 18))
        sclera_mask = create_sclera_mask(image)
        overlay = image.copy()
        selected_box: Optional[Tuple[int, int, int, int]] = None

        if len(eyes) > 0:
            x, y, w, h = max(eyes, key=lambda box: box[2] * box[3])
            selected_box = (x, y, w, h)
        else:
            selected_box = largest_mask_box(sclera_mask)

        region = None
        confidence = 0.0
        area_ok = False
        if selected_box is not None:
            x, y, w, h = selected_box
            region = image[y : y + h, x : x + w]
            crop_mask = sclera_mask[y : y + h, x : x + w]
            area_ratio = float(np.mean(crop_mask > 0))
            area_ok = crop_mask.sum() > 0 and area_ratio >= 0.015
            confidence = 0.35 + 0.60 * min(area_ratio * 5.0, 1.0)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 110, 220), 3)
            overlay[sclera_mask > 0] = (0.62 * overlay[sclera_mask > 0] + 0.38 * np.array([52, 152, 219])).astype(np.uint8)

        quality_ok, quality_message = image_quality_message(image)
        clear = bool(region is not None and confidence >= 0.50 and quality_ok and area_ok)
        message = "Sclera region detected." if clear else retake_message("sclera", quality_message, area_ok, confidence)

        return RegionResult(region, overlay, sclera_mask, float(min(confidence, 1.0)), clear, message, selected_box)


def create_sclera_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Detect bright low-saturation white/yellowish eye regions."""

    image = ensure_uint8_rgb(image_rgb)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    hsv_mask = cv2.inRange(hsv, np.array([0, 0, 70], np.uint8), np.array([70, 115, 255], np.uint8))
    chroma_mask = cv2.inRange(ycrcb, np.array([80, 100, 70], np.uint8), np.array([255, 175, 155], np.uint8))
    mask = cv2.bitwise_and(hsv_mask, chroma_mask)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def largest_mask_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return bounding box for the largest connected component."""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:
        return None
    return tuple(int(value) for value in cv2.boundingRect(contour))


def image_quality_message(image_rgb: np.ndarray) -> Tuple[bool, str]:
    """Check blur and exposure before inference."""

    gray = cv2.cvtColor(ensure_uint8_rgb(image_rgb), cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_value = float(gray.mean())
    if blur_score < 35:
        return False, "Image is blurry."
    if mean_value < 45:
        return False, "Image is too dark."
    if mean_value > 235:
        return False, "Image is overexposed."
    return True, "Image quality is acceptable."


def retake_message(region_name: str, quality_message: str, area_ok: bool, confidence: float) -> str:
    """Build user-facing retake guidance."""

    if quality_message != "Image quality is acceptable.":
        return f"Retake {region_name} photo: {quality_message} Use steady focus and even light."
    if not area_ok:
        return f"Retake {region_name} photo: region is too small or partly hidden."
    if confidence < 0.55:
        return f"Retake {region_name} photo: detected region is unclear."
    return f"Retake {region_name} photo with better framing."


def ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Validate and convert image arrays to RGB uint8."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected RGB image with shape H x W x 3")
    if array.dtype == np.uint8:
        return array.copy()
    array = array.astype(np.float32)
    if array.max(initial=0) <= 1.0:
        array *= 255.0
    return np.clip(array, 0, 255).astype(np.uint8)
