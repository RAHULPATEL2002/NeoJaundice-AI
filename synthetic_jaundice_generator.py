"""
Synthetic jaundice generation, skin-tone adaptation, and ablation tooling.

Research contribution:
This module creates bilirubin-like yellowing on normal neonatal skin images in
YCbCr/YCrCb space while preserving high-frequency skin texture. It also includes
skin-tone-aware prediction calibration and reporting across six Fitzpatrick
levels, with special attention to Indian neonatal skin-tone diversity.

The synthetic images are intended for research augmentation and ablation
studies. They must not be treated as a replacement for clinically measured,
ethics-approved neonatal datasets.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

import config
from skin_tone_normalizer import SkinToneNormalizer


FITZPATRICK_NAMES = {
    1: "I Very light",
    2: "II Light",
    3: "III Medium",
    4: "IV Brown",
    5: "V Dark brown",
    6: "VI Deeply pigmented",
}

# Indian neonatal datasets usually contain more tones III-V, but the module
# keeps all six Fitzpatrick groups so cross-site results can be reported.
INDIAN_TONE_PRIOR = {
    1: 0.03,
    2: 0.08,
    3: 0.27,
    4: 0.34,
    5: 0.22,
    6: 0.06,
}


@dataclass(frozen=True)
class SeverityProfile:
    """YCbCr/YCrCb channel shifts for one jaundice severity."""

    name: str
    cb_shift: float
    cr_shift: float
    luminance_shift: float
    bilirubin_target: float


SEVERITY_PROFILES = {
    "mild": SeverityProfile("mild", cb_shift=-8.0, cr_shift=3.0, luminance_shift=1.0, bilirubin_target=8.5),
    "moderate": SeverityProfile("moderate", cb_shift=-17.0, cr_shift=7.0, luminance_shift=2.0, bilirubin_target=16.0),
    "severe": SeverityProfile("severe", cb_shift=-28.0, cr_shift=12.0, luminance_shift=3.0, bilirubin_target=24.0),
}


class SyntheticJaundiceGenerator:
    """
    Generate synthetic jaundice progression from normal baby skin photos.

    OpenCV names the color space YCrCb, with channel order Y, Cr, Cb. The code
    explicitly assigns channel names to avoid the common Cb/Cr swap.
    """

    def __init__(
        self,
        output_dir: Path = config.SYNTHETIC_DIR,
        seed: int = config.RANDOM_SEED,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.rng = np.random.default_rng(seed)
        self.normalizer = SkinToneNormalizer()

    def generate_from_directory(
        self,
        input_dir: Path = config.NORMAL_DIR,
        severities: Sequence[str] = ("mild", "moderate", "severe"),
        variants_per_image: int = 2,
    ) -> pd.DataFrame:
        """
        Generate synthetic images for all normal images in input_dir.

        Returns:
            A manifest DataFrame with original path, synthetic path, severity,
            bilirubin target, and estimated Fitzpatrick tone.
        """

        input_paths = sorted(
            path
            for path in Path(input_dir).rglob("*")
            if path.is_file() and path.suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS
        )
        if not input_paths:
            raise FileNotFoundError(f"No normal images found in {input_dir}")

        records: List[Dict[str, object]] = []
        for image_path in input_paths:
            image_rgb = read_rgb(image_path)
            tone = SkinToneAdapter.estimate_fitzpatrick_tone(image_rgb)
            for severity in severities:
                profile = SEVERITY_PROFILES[severity]
                severity_dir = self.output_dir / severity
                severity_dir.mkdir(parents=True, exist_ok=True)
                for variant_index in range(variants_per_image):
                    synthetic = self.simulate_progression(image_rgb, profile, variant_index)
                    output_path = severity_dir / f"{image_path.stem}_{severity}_v{variant_index + 1}.jpg"
                    write_rgb(output_path, synthetic)
                    records.append(
                        {
                            "original_path": str(image_path),
                            "synthetic_path": str(output_path),
                            "severity": severity,
                            "class_index": config.CLASS_TO_IDX[severity],
                            "bilirubin_target": profile.bilirubin_target,
                            "fitzpatrick": tone,
                            "fitzpatrick_name": FITZPATRICK_NAMES[tone],
                            "method": "YCbCr spatial chroma shift",
                        }
                    )

        manifest = pd.DataFrame(records)
        manifest_path = self.output_dir / "synthetic_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        return manifest

    def simulate_progression(
        self,
        image_rgb: np.ndarray,
        profile: SeverityProfile,
        variant_index: int = 0,
    ) -> np.ndarray:
        """
        Simulate jaundice by spatially shifting Cb/Cr channels.

        The yellowing map is stronger on the upper-central face area and weaker
        near image borders/limbs. Texture is preserved because shifts are applied
        mostly to low-frequency chroma, leaving luminance and local detail intact.
        """

        image = ensure_uint8_rgb(image_rgb)
        normalized = self.normalizer.normalize(image)
        ycrcb = cv2.cvtColor(normalized, cv2.COLOR_RGB2YCrCb).astype(np.float32)

        y_channel = ycrcb[:, :, 0]
        cr_channel = ycrcb[:, :, 1]
        cb_channel = ycrcb[:, :, 2]

        spatial_map = self.create_spatial_yellowing_map(image.shape[:2], variant_index)
        texture_guard = self.create_texture_guard(y_channel)
        strength = spatial_map * texture_guard

        # Add tiny variant noise to prevent duplicate augmented samples while
        # keeping the jaundice pattern physiologically plausible.
        noise = self.rng.normal(0.0, 0.035, size=strength.shape).astype(np.float32)
        strength = np.clip(strength + noise, 0.0, 1.0)

        cb_channel = cb_channel + profile.cb_shift * strength
        cr_channel = cr_channel + profile.cr_shift * strength
        y_channel = y_channel + profile.luminance_shift * strength

        shifted = np.stack([y_channel, cr_channel, cb_channel], axis=2)
        shifted = np.clip(shifted, 0, 255).astype(np.uint8)
        synthetic = cv2.cvtColor(shifted, cv2.COLOR_YCrCb2RGB)

        # Preserve realistic skin texture by mixing back a small amount of the
        # original high-frequency component.
        return preserve_texture(normalized, synthetic)

    def create_spatial_yellowing_map(self, shape: Tuple[int, int], variant_index: int) -> np.ndarray:
        """Create a face-biased yellowing map with smooth regional variation."""

        height, width = shape
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        x = xx / max(width - 1, 1)
        y = yy / max(height - 1, 1)

        face_center_x = 0.50 + 0.04 * np.sin(variant_index + 1)
        face_center_y = 0.36 + 0.03 * np.cos(variant_index + 1)
        face = np.exp(-(((x - face_center_x) ** 2) / 0.11 + ((y - face_center_y) ** 2) / 0.08))
        torso = 0.55 * np.exp(-(((x - 0.50) ** 2) / 0.22 + ((y - 0.68) ** 2) / 0.18))
        limb_falloff = 0.65 + 0.35 * np.exp(-((x - 0.50) ** 2) / 0.22)
        vignette = 0.75 + 0.25 * np.exp(-(((x - 0.50) ** 2) + ((y - 0.50) ** 2)) / 0.35)

        yellowing = (face + torso) * limb_falloff * vignette
        yellowing = cv2.GaussianBlur(yellowing, (0, 0), sigmaX=max(width, height) / 35)
        yellowing = normalize_zero_one(yellowing)
        return yellowing.astype(np.float32)

    @staticmethod
    def create_texture_guard(luminance: np.ndarray) -> np.ndarray:
        """
        Reduce color shift on strong edges/highlights to preserve texture.

        This prevents the synthetic method from painting flat yellow regions
        over creases, hair, cloth edges, and camera highlights.
        """

        luminance_uint8 = np.clip(luminance, 0, 255).astype(np.uint8)
        edges = cv2.Laplacian(luminance_uint8, cv2.CV_32F)
        edge_strength = normalize_zero_one(np.abs(edges))
        guard = 1.0 - (0.35 * edge_strength)
        return np.clip(guard, 0.65, 1.0).astype(np.float32)


class SkinToneAdapter:
    """
    Skin-tone estimation and prediction calibration across Fitzpatrick I-VI.

    The tone estimator uses masked skin pixels and an ITA-like luminance/chroma
    calculation. For Indian neonatal images, the thresholds are tuned to avoid
    collapsing medium-brown and dark-brown tones into one group.
    """

    normalizer = SkinToneNormalizer()

    @classmethod
    def estimate_fitzpatrick_tone(cls, image_rgb: np.ndarray) -> int:
        """Estimate Fitzpatrick category from probable skin pixels."""

        image = ensure_uint8_rgb(image_rgb)
        mask = cls.normalizer.create_skin_mask(image) > 0
        if np.mean(mask) < 0.02:
            mask = np.ones(image.shape[:2], dtype=bool)

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_star = lab[:, :, 0][mask] * (100.0 / 255.0)
        b_star = lab[:, :, 2][mask] - 128.0
        ita = np.degrees(np.arctan2(np.median(l_star) - 50.0, np.median(b_star) + 1e-6))

        # Adapted ITA thresholds with slightly wider III-V buckets for Indian
        # infants, whose skin tones are frequently underrepresented in datasets.
        if ita > 48:
            return 1
        if ita > 35:
            return 2
        if ita > 20:
            return 3
        if ita > 5:
            return 4
        if ita > -18:
            return 5
        return 6

    @staticmethod
    def calibrate_probabilities(probabilities: np.ndarray, fitzpatrick_tone: int) -> np.ndarray:
        """
        Normalize predictions across skin tones with tone-aware calibration.

        Darker tones can reduce visible yellow contrast in RGB images. This
        calibration mildly increases attention to jaundiced classes for tones
        IV-VI and uses temperature scaling to avoid overconfident predictions.
        """

        probs = np.asarray(probabilities, dtype=np.float32)
        probs = np.clip(probs, 1e-6, 1.0)
        logits = np.log(probs)

        temperature_by_tone = {1: 0.98, 2: 1.00, 3: 1.03, 4: 1.07, 5: 1.10, 6: 1.14}
        jaundice_bias_by_tone = {1: -0.03, 2: -0.01, 3: 0.00, 4: 0.04, 5: 0.07, 6: 0.10}

        temperature = temperature_by_tone.get(fitzpatrick_tone, 1.0)
        bias = jaundice_bias_by_tone.get(fitzpatrick_tone, 0.0)
        logits = logits / temperature
        logits[1:] += bias
        calibrated = np.exp(logits - np.max(logits))
        calibrated = calibrated / calibrated.sum()
        return calibrated.astype(np.float32)

    @classmethod
    def accuracy_by_skin_tone(
        cls,
        predictions_csv: Path,
        output_csv: Path = config.REPORTS_DIR / "research" / "per_skin_tone_accuracy.csv",
    ) -> pd.DataFrame:
        """
        Compute accuracy by Fitzpatrick category from prediction records.

        Expected columns:
            image_path, y_true, y_pred
        Optional:
            fitzpatrick
        """

        df = pd.read_csv(predictions_csv)
        if "fitzpatrick" not in df.columns:
            df["fitzpatrick"] = [
                cls.estimate_fitzpatrick_tone(read_rgb(Path(path))) for path in df["image_path"]
            ]

        rows = []
        for tone in range(1, 7):
            subset = df[df["fitzpatrick"] == tone]
            accuracy = accuracy_score(subset["y_true"], subset["y_pred"]) if len(subset) else np.nan
            rows.append(
                {
                    "fitzpatrick": tone,
                    "skin_tone": FITZPATRICK_NAMES[tone],
                    "n": int(len(subset)),
                    "accuracy": accuracy,
                }
            )

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result = pd.DataFrame(rows)
        result.to_csv(output_csv, index=False)
        return result


class SyntheticAblationRunner:
    """
    Prepare and summarize ablation studies for the synthetic-data method.

    This class supports two research workflows:
    1. Prepare synthetic data and manifests for training externally.
    2. Compare completed experiment prediction CSVs for paper metrics.
    """

    def __init__(self, research_dir: Path = config.REPORTS_DIR / "research") -> None:
        self.research_dir = Path(research_dir)
        self.research_dir.mkdir(parents=True, exist_ok=True)

    def prepare_ablation_datasets(self, synthetic_ratio: float = 1.0) -> Dict[str, Path]:
        """
        Create manifest files for baseline and synthetic-training experiments.

        The baseline manifest points to real data only. The synthetic manifest
        includes generated images up to the requested ratio.
        """

        generator = SyntheticJaundiceGenerator()
        synthetic_manifest = generator.generate_from_directory()
        if synthetic_ratio < 1.0:
            synthetic_manifest = (
                synthetic_manifest.groupby("severity", group_keys=False)
                .sample(frac=synthetic_ratio, random_state=config.RANDOM_SEED)
                .reset_index(drop=True)
            )

        baseline_manifest = self.create_real_data_manifest()
        baseline_path = self.research_dir / "baseline_real_manifest.csv"
        synthetic_path = self.research_dir / f"with_synthetic_ratio_{synthetic_ratio:.2f}_manifest.csv"
        baseline_manifest.to_csv(baseline_path, index=False)
        synthetic_manifest.to_csv(synthetic_path, index=False)
        return {"baseline_manifest": baseline_path, "synthetic_manifest": synthetic_path}

    def create_real_data_manifest(self) -> pd.DataFrame:
        """Create a manifest for existing real images in data/."""

        rows = []
        class_dirs = {
            "normal": config.NORMAL_DIR,
            "mild": config.MILD_DIR,
            "moderate": config.MODERATE_DIR,
            "severe": config.SEVERE_DIR,
        }
        for class_name, folder in class_dirs.items():
            for path in sorted(folder.rglob("*")) if folder.exists() else []:
                if path.is_file() and path.suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS:
                    rows.append(
                        {
                            "image_path": str(path),
                            "severity": class_name,
                            "class_index": config.CLASS_TO_IDX[class_name],
                            "source": "real",
                            "fitzpatrick": SkinToneAdapter.estimate_fitzpatrick_tone(read_rgb(path)),
                        }
                    )
        return pd.DataFrame(rows)

    def compare_prediction_files(
        self,
        baseline_predictions: Path,
        synthetic_predictions: Path,
        output_csv: Path = config.REPORTS_DIR / "research" / "ablation_results.csv",
    ) -> pd.DataFrame:
        """
        Compare baseline vs synthetic-trained predictions.

        Expected columns:
            y_true, y_pred, fitzpatrick
        Optional:
            severity
        """

        rows = [
            self.compute_ablation_metrics("Without synthetic data", pd.read_csv(baseline_predictions)),
            self.compute_ablation_metrics("With synthetic data", pd.read_csv(synthetic_predictions)),
        ]
        result = pd.DataFrame(rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        with open(output_csv.with_suffix(".json"), "w", encoding="utf-8") as file:
            json.dump(result.to_dict(orient="records"), file, indent=2)
        return result

    @staticmethod
    def compute_ablation_metrics(method: str, df: pd.DataFrame) -> Dict[str, float]:
        """Compute overall, dark-tone, and rare-severe accuracy."""

        if "fitzpatrick" not in df.columns:
            raise ValueError("Prediction CSV must contain a fitzpatrick column.")

        overall = accuracy_score(df["y_true"], df["y_pred"]) if len(df) else np.nan
        dark_subset = df[df["fitzpatrick"].astype(int) >= 5]
        severe_subset = df[df["y_true"].astype(int) == config.CLASS_TO_IDX["severe"]]

        dark_accuracy = accuracy_score(dark_subset["y_true"], dark_subset["y_pred"]) if len(dark_subset) else np.nan
        severe_accuracy = accuracy_score(severe_subset["y_true"], severe_subset["y_pred"]) if len(severe_subset) else np.nan

        return {
            "method": method,
            "overall_accuracy": float(overall),
            "dark_skin_accuracy": float(dark_accuracy) if not np.isnan(dark_accuracy) else np.nan,
            "rare_severe_accuracy": float(severe_accuracy) if not np.isnan(severe_accuracy) else np.nan,
            "n_samples": int(len(df)),
            "n_dark_skin": int(len(dark_subset)),
            "n_severe": int(len(severe_subset)),
        }

    def create_demo_ablation_results(self) -> pd.DataFrame:
        """
        Create clearly labeled demo ablation numbers for figure plumbing.

        Use only for manuscript-layout dry runs before real experiments finish.
        """

        rows = [
            {
                "method": "Without synthetic data",
                "overall_accuracy": 0.812,
                "dark_skin_accuracy": 0.716,
                "rare_severe_accuracy": 0.641,
                "n_samples": 0,
                "n_dark_skin": 0,
                "n_severe": 0,
                "status": "demo_placeholder",
            },
            {
                "method": "With synthetic data",
                "overall_accuracy": 0.884,
                "dark_skin_accuracy": 0.832,
                "rare_severe_accuracy": 0.781,
                "n_samples": 0,
                "n_dark_skin": 0,
                "n_severe": 0,
                "status": "demo_placeholder",
            },
        ]
        result = pd.DataFrame(rows)
        result.to_csv(self.research_dir / "ablation_results.csv", index=False)
        return result


def read_rgb(path: Path) -> np.ndarray:
    """Read image from disk as RGB uint8."""

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    """Write RGB image to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(ensure_uint8_rgb(image_rgb), cv2.COLOR_RGB2BGR))


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


def normalize_zero_one(values: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1] with numerical safety."""

    values = values.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def preserve_texture(original_rgb: np.ndarray, synthetic_rgb: np.ndarray) -> np.ndarray:
    """Blend original high-frequency texture into the synthetic image."""

    original = ensure_uint8_rgb(original_rgb).astype(np.float32)
    synthetic = ensure_uint8_rgb(synthetic_rgb).astype(np.float32)
    original_blur = cv2.GaussianBlur(original, (0, 0), sigmaX=2.5)
    high_frequency = original - original_blur
    restored = synthetic + 0.35 * high_frequency
    return np.clip(restored, 0, 255).astype(np.uint8)


def create_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description="Synthetic jaundice research tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate synthetic jaundice images")
    generate.add_argument("--input-dir", type=Path, default=config.NORMAL_DIR)
    generate.add_argument("--output-dir", type=Path, default=config.SYNTHETIC_DIR)
    generate.add_argument("--variants", type=int, default=2)

    tone_report = subparsers.add_parser("skin-tone-report", help="Compute accuracy by skin tone")
    tone_report.add_argument("--predictions", type=Path, required=True)
    tone_report.add_argument("--output", type=Path, default=config.REPORTS_DIR / "research" / "per_skin_tone_accuracy.csv")

    ablation = subparsers.add_parser("ablation", help="Compare baseline and synthetic prediction CSVs")
    ablation.add_argument("--baseline-predictions", type=Path)
    ablation.add_argument("--synthetic-predictions", type=Path)
    ablation.add_argument("--demo", action="store_true", help="Create demo placeholder results")

    prepare = subparsers.add_parser("prepare-ablation", help="Generate synthetic data and experiment manifests")
    prepare.add_argument("--synthetic-ratio", type=float, default=1.0)

    return parser


def main() -> None:
    """CLI entry point."""

    parser = create_parser()
    args = parser.parse_args()

    if args.command == "generate":
        generator = SyntheticJaundiceGenerator(output_dir=args.output_dir)
        manifest = generator.generate_from_directory(args.input_dir, variants_per_image=args.variants)
        print(f"Generated {len(manifest)} synthetic images")
        print(f"Manifest: {args.output_dir / 'synthetic_manifest.csv'}")

    elif args.command == "skin-tone-report":
        report = SkinToneAdapter.accuracy_by_skin_tone(args.predictions, args.output)
        print(report.to_string(index=False))

    elif args.command == "ablation":
        runner = SyntheticAblationRunner()
        if args.demo:
            report = runner.create_demo_ablation_results()
        else:
            if not args.baseline_predictions or not args.synthetic_predictions:
                raise ValueError("Provide --baseline-predictions and --synthetic-predictions, or use --demo.")
            report = runner.compare_prediction_files(args.baseline_predictions, args.synthetic_predictions)
        print(report.to_string(index=False))

    elif args.command == "prepare-ablation":
        runner = SyntheticAblationRunner()
        paths = runner.prepare_ablation_datasets(args.synthetic_ratio)
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
