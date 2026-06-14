"""
Training script for the dual-input multi-task jaundice model.

Loss:
total_loss = 0.7 * CrossEntropy(severity) + 0.3 * MSE(bilirubin_mg_dl)

Because the Stage 1 dataset stores class labels but not measured bilirubin
values, this script derives regression targets from severity band midpoints in
config.BILIRUBIN_TARGETS. When measured serum bilirubin labels are available,
extend the dataset to return "bilirubin" and this script will use them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import config
from dataset import build_dataloaders
from evaluate import compute_medical_metrics, plot_confusion_matrix, plot_multiclass_roc
from model import build_model


def get_device() -> torch.device:
    """Use CUDA when available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, ...]:
    """
    Move a batch to device and provide dual images.

    Future datasets may return separate "skin_image" and "sclera_image" tensors.
    The Stage 1 dataset returns a single "image"; in that case both branches get
    the available tensor so training remains runnable.
    """

    if "skin_image" in batch and "sclera_image" in batch:
        skin_image = batch["skin_image"].to(device, non_blocking=True)
        sclera_image = batch["sclera_image"].to(device, non_blocking=True)
    else:
        skin_image = batch["image"].to(device, non_blocking=True)
        sclera_image = batch["image"].to(device, non_blocking=True)

    features = batch.get("features")
    if features is not None:
        features = features.to(device, non_blocking=True)

    labels = batch["label"].to(device, non_blocking=True)

    if "bilirubin" in batch:
        bilirubin = batch["bilirubin"].float().to(device, non_blocking=True)
    else:
        bilirubin = labels_to_bilirubin_targets(labels).to(device)

    return skin_image, sclera_image, features, labels, bilirubin


def labels_to_bilirubin_targets(labels: torch.Tensor) -> torch.Tensor:
    """Map severity labels to bilirubin band midpoint targets in mg/dL."""

    target_values = torch.tensor(
        [config.BILIRUBIN_TARGETS[index] for index in range(config.NUM_CLASSES)],
        dtype=torch.float32,
        device=labels.device,
    )
    return target_values[labels.long()]


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    cls_loss_fn: nn.Module,
    reg_loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch and return aggregate losses."""

    model.train()
    running_total = 0.0
    running_cls = 0.0
    running_reg = 0.0
    sample_count = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        skin_image, sclera_image, features, labels, bilirubin = prepare_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(skin_image, sclera_image, features)

        cls_loss = cls_loss_fn(outputs["logits"], labels)
        reg_loss = reg_loss_fn(outputs["bilirubin"], bilirubin)
        total_loss = (
            config.CLASSIFICATION_LOSS_WEIGHT * cls_loss
            + config.REGRESSION_LOSS_WEIGHT * reg_loss
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)
        optimizer.step()

        batch_size = labels.size(0)
        running_total += float(total_loss.item()) * batch_size
        running_cls += float(cls_loss.item()) * batch_size
        running_reg += float(reg_loss.item()) * batch_size
        sample_count += batch_size

    return {
        "loss": running_total / max(sample_count, 1),
        "classification_loss": running_cls / max(sample_count, 1),
        "regression_loss": running_reg / max(sample_count, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    cls_loss_fn: nn.Module,
    reg_loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    """Evaluate a loader and return losses, predictions, probabilities, and metrics."""

    model.eval()
    running_total = 0.0
    running_cls = 0.0
    running_reg = 0.0
    sample_count = 0

    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_probabilities: List[np.ndarray] = []
    all_bilirubin_targets: List[float] = []
    all_bilirubin_predictions: List[float] = []

    for batch in tqdm(loader, desc="Validation", leave=False):
        skin_image, sclera_image, features, labels, bilirubin = prepare_batch(batch, device)
        outputs = model(skin_image, sclera_image, features)

        cls_loss = cls_loss_fn(outputs["logits"], labels)
        reg_loss = reg_loss_fn(outputs["bilirubin"], bilirubin)
        total_loss = (
            config.CLASSIFICATION_LOSS_WEIGHT * cls_loss
            + config.REGRESSION_LOSS_WEIGHT * reg_loss
        )

        probabilities = torch.softmax(outputs["logits"], dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        batch_size = labels.size(0)
        running_total += float(total_loss.item()) * batch_size
        running_cls += float(cls_loss.item()) * batch_size
        running_reg += float(reg_loss.item()) * batch_size
        sample_count += batch_size

        all_labels.extend(labels.cpu().numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_probabilities.extend(probabilities.cpu().numpy())
        all_bilirubin_targets.extend(bilirubin.cpu().numpy().tolist())
        all_bilirubin_predictions.extend(outputs["bilirubin"].cpu().numpy().tolist())

    labels_np = np.array(all_labels, dtype=np.int64)
    predictions_np = np.array(all_predictions, dtype=np.int64)
    probabilities_np = np.array(all_probabilities, dtype=np.float32)
    bilirubin_targets_np = np.array(all_bilirubin_targets, dtype=np.float32)
    bilirubin_predictions_np = np.array(all_bilirubin_predictions, dtype=np.float32)

    metrics = compute_medical_metrics(
        labels_np,
        predictions_np,
        probabilities_np,
        bilirubin_targets_np,
        bilirubin_predictions_np,
    )

    return {
        "loss": running_total / max(sample_count, 1),
        "classification_loss": running_cls / max(sample_count, 1),
        "regression_loss": running_reg / max(sample_count, 1),
        "labels": labels_np,
        "predictions": predictions_np,
        "probabilities": probabilities_np,
        "bilirubin_targets": bilirubin_targets_np,
        "bilirubin_predictions": bilirubin_predictions_np,
        "metrics": metrics,
    }


def plot_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    """Save loss and accuracy curves for reports."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Multi-task Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_accuracy"], label="Val accuracy")
    plt.plot(epochs, history["val_mae"], label="Val MAE mg/dL")
    plt.xlabel("Epoch")
    plt.title("Validation Metrics")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, object]) -> None:
    """Save the best model checkpoint."""

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "class_names": config.CLASS_NAMES,
        "bilirubin_targets": config.BILIRUBIN_TARGETS,
    }
    torch.save(checkpoint, config.MODEL_DIR / "best_model.pth")


def main() -> None:
    """Train the model and write reports/plots."""

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders()
    model = build_model(device=device)

    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_mae": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        train_stats = train_one_epoch(model, train_loader, optimizer, cls_loss_fn, reg_loss_fn, device)
        val_stats = validate(model, val_loader, cls_loss_fn, reg_loss_fn, device)
        val_metrics = val_stats["metrics"]

        scheduler.step(float(val_stats["loss"]))

        history["train_loss"].append(float(train_stats["loss"]))
        history["val_loss"].append(float(val_stats["loss"]))
        history["val_accuracy"].append(float(val_metrics["overall_accuracy"]))
        history["val_mae"].append(float(val_metrics["bilirubin_mae_mg_dl"]))

        print(
            "Train loss: "
            f"{train_stats['loss']:.4f} | Val loss: {val_stats['loss']:.4f} | "
            f"Val acc: {val_metrics['overall_accuracy']:.4f} | "
            f"Val MAE: {val_metrics['bilirubin_mae_mg_dl']:.2f} mg/dL"
        )

        if float(val_stats["loss"]) < best_val_loss:
            best_val_loss = float(val_stats["loss"])
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, val_metrics)
            print("Saved new best model to models/best_model.pth")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    plot_training_curves(history, config.REPORTS_DIR / "training_curves.png")

    # Reload the best checkpoint before final test reporting.
    checkpoint = torch.load(config.MODEL_DIR / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_stats = validate(model, test_loader, cls_loss_fn, reg_loss_fn, device)

    plot_confusion_matrix(
        test_stats["labels"],
        test_stats["predictions"],
        config.REPORTS_DIR / "confusion_matrix.png",
    )
    plot_multiclass_roc(
        test_stats["labels"],
        test_stats["probabilities"],
        config.REPORTS_DIR / "auc_roc_curve.png",
    )

    report = {
        "history": history,
        "test_metrics": test_stats["metrics"],
    }
    with open(config.REPORTS_DIR / "training_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nFinal test metrics:")
    print(json.dumps(test_stats["metrics"], indent=2))


if __name__ == "__main__":
    main()
