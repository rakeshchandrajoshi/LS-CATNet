"""Plotting helpers for LS-CATNet results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import roc_curve_data


def plot_history(history: Dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [a * 100 for a in history["train_acc"]], label="Train")
    axes[0].plot(epochs, [a * 100 for a in history["val_acc"]], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("LS-CATNet Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.4)

    axes[1].plot(epochs, history["train_loss"], label="Train")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("LS-CATNet Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("LS-CATNet Confusion Matrix")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = roc_curve_data(y_true, y_prob, class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in class_names:
        ax.plot(data[name]["fpr"], data[name]["tpr"], lw=1.8, label=f"{name} (AUC={data[name]['auc']:.4f})")
    ax.plot(data["macro"]["fpr"], data["macro"]["tpr"], lw=2.5, label=f"Macro (AUC={data['macro']['auc']:.4f})")
    ax.plot([0, 1], [0, 1], linestyle=":", lw=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("LS-CATNet ROC Curves")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_per_class_csv(metrics: Dict, out_path: str | Path) -> None:
    df = pd.DataFrame(metrics["per_class"])
    df.to_csv(out_path, index=False)
