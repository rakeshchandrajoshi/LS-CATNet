"""Train LS-CATNet on split image folders."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .data import class_counts, make_loaders
from .model import LSCATNet, count_parameters
from .plots import plot_history
from .utils import ensure_dir, get_device, save_json, set_seed


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, save_path: Path):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_acc = 0.0
        self.best_epoch = 0
        self.counter = 0

    def step(self, val_acc: float, epoch: int, model: torch.nn.Module) -> bool:
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.counter = 0
            torch.save({"model_state": model.state_dict(), "best_acc": val_acc, "epoch": epoch}, self.save_path)
            print(f"  saved best checkpoint: val_acc={val_acc * 100:.2f}%")
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f"  early stopping at epoch {epoch}; best epoch={self.best_epoch}")
            return True
        return False


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=f"{total_loss / total:.4f}", acc=f"{correct / total * 100:.2f}%")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train LS-CATNet")
    parser.add_argument("--data-root", required=True, help="Path with train/val/test subfolders")
    parser.add_argument("--outdir", default="outputs_lscatnet", help="Output directory")
    parser.add_argument("--class-names", nargs="+", default=["fognonsmoke", "fogsmoke", "nonsmoke", "smoke"])
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained EfficientNet-B0 backbone")
    parser.add_argument("--weighted-loss", action="store_true", help="Use inverse-frequency class weights")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable CosineAnnealingLR")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_classes != len(args.class_names):
        raise ValueError("--num-classes must match the number of --class-names")

    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)
    device = get_device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, _, train_ds, val_ds, test_ds = make_loaders(
        args.data_root, args.class_names, args.img_size, args.batch_size, args.num_workers
    )
    print(f"Train={len(train_ds):,}, Val={len(val_ds):,}, Test={len(test_ds):,}")
    print(f"Class mapping: {train_ds.class_to_idx}")

    model = LSCATNet(num_classes=args.num_classes, pretrained_backbone=not args.no_pretrained).to(device)
    params = count_parameters(model)
    print("Parameters:", params, {k + "_M": v / 1e6 for k, v in params.items()})

    if args.weighted_loss:
        counts = class_counts(train_ds, args.num_classes).float()
        weights = counts.sum() / (args.num_classes * torch.clamp(counts, min=1))
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        loss_type = "weighted CrossEntropyLoss"
    else:
        criterion = nn.CrossEntropyLoss()
        loss_type = "standard CrossEntropyLoss"

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None if args.no_scheduler else optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    checkpoint_path = outdir / "LSCATNet_best.pth"
    stopper = EarlyStopping(args.patience, args.min_delta, checkpoint_path)

    config = vars(args).copy()
    config.update({"loss": loss_type, "optimizer": "Adam", "scheduler": None if scheduler is None else "CosineAnnealingLR", "params": params})
    save_json(config, outdir / "training_config.json")

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_acc={train_acc*100:.2f}% loss={train_loss:.4f} | "
            f"val_acc={val_acc*100:.2f}% loss={val_loss:.4f} | lr={lr_now:.2e}"
        )
        if stopper.step(val_acc, epoch, model):
            break

    elapsed = time.time() - start
    print(f"Training complete in {elapsed / 60:.2f} min. Best val acc={stopper.best_acc*100:.2f}%")
    save_json(history, outdir / "history.json")
    with (outdir / "history.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))
    plot_history(history, outdir / "training_curves.png")
    save_json({"best_val_acc": stopper.best_acc, "best_epoch": stopper.best_epoch, "elapsed_sec": elapsed}, outdir / "training_summary.json")


if __name__ == "__main__":
    main()
