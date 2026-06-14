"""
Evaluation utilities for medical reporting.

This module reports:
- Full confusion matrix.
- Overall and per-class classification accuracy.
- MAE for bilirubin regression in mg/dL.
- Sensitivity and specificity per severity class.
- Multiclass AUC-ROC when the test set contains enough classes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, confusion_matrix, mean_absolute_error, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch import nn

import config
from dataset import build_dataloaders
from model import build_model


def compute_medical_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    bilirubin_targets: np.ndarray,
    bilirubin_predictions: np.ndarray,
) -> Dict[str, object]:
    """Compute classification, regression, sensitivity, specificity, and AUC metrics."""

    cm = confusion_matrix(labels, predictions, labels=list(range(config.NUM_CLASSES)))
    total = int(cm.sum())
    overall_accuracy = float(np.trace(cm) / total) if total else 0.0

    per_class = {}
    for class_index, class_name in enumerate(config.CLASS_NAMES):
        tp = int(cm[class_index, class_index])
        fn = int(cm[class_index, :].sum() - tp)
        fp = int(cm[:, class_index].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)

        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        accuracy = tp / cm[class_index, :].sum() if cm[class_index, :].sum() else 0.0

        per_class[class_name] = {
            "accuracy": float(accuracy),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "support": int(cm[class_index, :].sum()),
        }

    mae = float(mean_absolute_error(bilirubin_targets, bilirubin_predictions)) if len(labels) else 0.0
    auc_roc = compute_auc_roc(labels, probabilities)

    return {
        "overall_accuracy": overall_accuracy,
        "bilirubin_mae_mg_dl": mae,
        "per_severity": per_class,
        "auc_roc": auc_roc,
        "confusion_matrix": cm.tolist(),
    }


def compute_auc_roc(labels: np.ndarray, probabilities: np.ndarray) -> Optional[float]:
    """Compute macro one-vs-rest AUC-ROC when at least two classes are present."""

    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None

    y_true = label_binarize(labels, classes=list(range(config.NUM_CLASSES)))
    try:
        return float(roc_auc_score(y_true, probabilities, average="macro", multi_class="ovr"))
    except ValueError:
        return None


def plot_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, output_path: Path) -> None:
    """Save a labeled confusion matrix plot."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(labels, predictions, labels=list(range(config.NUM_CLASSES)))

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(config.NUM_CLASSES)
    plt.xticks(tick_marks, config.CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, config.CLASS_NAMES)
    plt.xlabel("Predicted severity")
    plt.ylabel("True severity")

    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > threshold else "black"
            plt.text(col, row, str(cm[row, col]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_multiclass_roc(labels: np.ndarray, probabilities: np.ndarray, output_path: Path) -> None:
    """Save one-vs-rest ROC curves for each class."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if labels.size == 0 or len(np.unique(labels)) < 2:
        return

    y_true = label_binarize(labels, classes=list(range(config.NUM_CLASSES)))

    plt.figure(figsize=(7, 6))
    for class_index, class_name in enumerate(config.CLASS_NAMES):
        if y_true[:, class_index].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, class_index], probabilities[:, class_index])
        class_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} AUC={class_auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Sensitivity")
    plt.title("One-vs-Rest AUC-ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: Path = config.MODEL_DIR / "best_model.pth") -> Dict[str, object]:
    """Load a saved model and evaluate it on the test split."""

    # Local import avoids a circular import at module load time.
    from train import validate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = build_dataloaders()

    model = build_model(device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()
    stats = validate(model, test_loader, cls_loss_fn, reg_loss_fn, device)

    plot_confusion_matrix(
        stats["labels"],
        stats["predictions"],
        config.REPORTS_DIR / "confusion_matrix.png",
    )
    plot_multiclass_roc(
        stats["labels"],
        stats["probabilities"],
        config.REPORTS_DIR / "auc_roc_curve.png",
    )

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "evaluation_report.json", "w", encoding="utf-8") as file:
        json.dump(stats["metrics"], file, indent=2)

    return stats["metrics"]


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Evaluate the best jaundice model checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.MODEL_DIR / "best_model.pth",
        help="Path to a saved model checkpoint.",
    )
    args = parser.parse_args()

    metrics = evaluate_checkpoint(args.checkpoint)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
