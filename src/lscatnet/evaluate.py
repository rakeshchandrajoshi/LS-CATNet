"""Evaluate a trained LS-CATNet checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import make_loaders
from .metrics import bootstrap_confidence_intervals, compute_multiclass_metrics
from .model import LSCATNet, count_parameters
from .plots import plot_confusion_matrix, plot_roc_curves, save_per_class_csv
from .utils import ensure_dir, get_device, save_json, set_seed


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for images, labels in tqdm(loader, desc="test", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(probs.argmax(axis=1))
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LS-CATNet")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", default="outputs_lscatnet_eval")
    parser.add_argument("--class-names", nargs="+", default=["fognonsmoke", "fogsmoke", "nonsmoke", "smoke"])
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap samples for CIs; 0 disables")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)
    device = get_device(args.device)

    _, _, test_loader, _, _, test_ds = make_loaders(args.data_root, args.class_names, args.img_size, args.batch_size, args.num_workers)
    model = LSCATNet(num_classes=args.num_classes, pretrained_backbone=not args.no_pretrained).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    params = count_parameters(model)

    y_true, y_pred, y_prob = predict_all(model, test_loader, device)
    metrics = compute_multiclass_metrics(y_true, y_pred, y_prob, args.class_names)
    metrics["params"] = params
    if args.bootstrap > 0:
        metrics["bootstrap_ci_95"] = bootstrap_confidence_intervals(
            y_true, y_pred, y_prob, args.class_names, n_boot=args.bootstrap, seed=args.seed
        )

    save_json(metrics, outdir / "metrics.json")
    print(metrics["classification_report"])
    print("Overall:")
    for k, v in metrics["overall"].items():
        print(f"  {k}: {v}")
    if "bootstrap_ci_95" in metrics:
        print("95% bootstrap confidence intervals:")
        for k, v in metrics["bootstrap_ci_95"].items():
            print(f"  {k}: {v['lower']:.4f}–{v['upper']:.4f}")

    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, args.class_names, outdir / "confusion_matrix.png")
    plot_roc_curves(y_true, y_prob, args.class_names, outdir / "roc_curves.png")
    save_per_class_csv(metrics, outdir / "per_class_metrics.csv")
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred, **{f"prob_{c}": y_prob[:, i] for i, c in enumerate(args.class_names)}}).to_csv(
        outdir / "predictions.csv", index=False
    )
    print(f"Saved evaluation outputs to {outdir}")


if __name__ == "__main__":
    main()
