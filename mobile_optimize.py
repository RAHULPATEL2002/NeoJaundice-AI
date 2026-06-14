"""
Mobile optimization utilities for the neonatal jaundice model.

Pipeline:
1. Export PyTorch dual-input model to ONNX.
2. Apply dynamic INT8 quantization with ONNX Runtime.
3. Report model size and CPU inference latency.

Targets:
- Quantized model under 20 MB.
- CPU inference under 3 seconds per sample for Android-class devices.

The script measures the current host CPU. Always benchmark on the target Android
phone before claiming production latency.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

import config
from model import DualInputJaundiceNet


class ONNXExportWrapper(nn.Module):
    """Convert model dict output into tuple output for ONNX export."""

    def __init__(self, model: DualInputJaundiceNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, skin_image: torch.Tensor, sclera_image: torch.Tensor, ycbcr_features: torch.Tensor):
        outputs = self.model(skin_image, sclera_image, ycbcr_features)
        return outputs["logits"], outputs["bilirubin"]


def load_model_for_export(checkpoint_path: Optional[Path]) -> DualInputJaundiceNet:
    """Load trained weights if available; otherwise export initialized model."""

    model = DualInputJaundiceNet(pretrained=False)
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def export_to_onnx(
    checkpoint_path: Path = config.MODEL_DIR / "best_model.pth",
    output_path: Path = config.MODEL_DIR / "jaundice_model.onnx",
) -> Path:
    """Export PyTorch model to ONNX."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = ONNXExportWrapper(load_model_for_export(checkpoint_path))
    skin = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    sclera = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    features = torch.randn(1, config.YCBCR_FEATURE_DIM)

    torch.onnx.export(
        model,
        (skin, sclera, features),
        output_path,
        input_names=["skin_image", "sclera_image", "ycbcr_features"],
        output_names=["logits", "bilirubin"],
        dynamic_axes={
            "skin_image": {0: "batch"},
            "sclera_image": {0: "batch"},
            "ycbcr_features": {0: "batch"},
            "logits": {0: "batch"},
            "bilirubin": {0: "batch"},
        },
        opset_version=17,
    )
    return output_path


def quantize_onnx_int8(
    onnx_path: Path = config.MODEL_DIR / "jaundice_model.onnx",
    output_path: Path = config.MODEL_DIR / "jaundice_model_int8.onnx",
) -> Path:
    """Apply ONNX Runtime dynamic INT8 quantization."""

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise RuntimeError("Install onnxruntime to quantize: pip install onnxruntime") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    return output_path


def benchmark_onnx(model_path: Path, warmup: int = 3, runs: int = 10) -> Dict[str, float]:
    """Benchmark ONNX model latency on CPU."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install onnxruntime to benchmark: pip install onnxruntime") from exc

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = {
        "skin_image": np.random.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).astype(np.float32),
        "sclera_image": np.random.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).astype(np.float32),
        "ycbcr_features": np.random.randn(1, config.YCBCR_FEATURE_DIM).astype(np.float32),
    }

    for _ in range(warmup):
        session.run(None, inputs)

    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, inputs)
        latencies.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": float(np.mean(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "target_under_3000ms": bool(np.percentile(latencies, 95) < 3000.0),
    }


def model_size_mb(path: Path) -> float:
    """Return model file size in megabytes."""

    return path.stat().st_size / (1024.0 * 1024.0)


def optimize_mobile(checkpoint_path: Path, output_dir: Path = config.MODEL_DIR) -> Dict[str, object]:
    """Run export, quantization, size check, and CPU benchmark."""

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_to_onnx(checkpoint_path, output_dir / "jaundice_model.onnx")
    quantized_path = quantize_onnx_int8(onnx_path, output_dir / "jaundice_model_int8.onnx")
    benchmark = benchmark_onnx(quantized_path)

    report = {
        "onnx_path": str(onnx_path),
        "quantized_path": str(quantized_path),
        "onnx_size_mb": round(model_size_mb(onnx_path), 2),
        "quantized_size_mb": round(model_size_mb(quantized_path), 2),
        "size_target_under_20mb": model_size_mb(quantized_path) < 20.0,
        "cpu_benchmark": benchmark,
    }
    with open(output_dir / "mobile_optimization_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return report


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Export and quantize model for mobile CPU inference")
    parser.add_argument("--checkpoint", type=Path, default=config.MODEL_DIR / "best_model.pth")
    parser.add_argument("--output-dir", type=Path, default=config.MODEL_DIR)
    args = parser.parse_args()

    report = optimize_mobile(args.checkpoint, args.output_dir)
    print(json.dumps(report, indent=2))
    if not report["size_target_under_20mb"]:
        print("Warning: quantized model is above 20 MB. Consider sharing one backbone or pruning.")
    if not report["cpu_benchmark"]["target_under_3000ms"]:
        print("Warning: CPU p95 latency is above 3 seconds on this machine.")


if __name__ == "__main__":
    main()
