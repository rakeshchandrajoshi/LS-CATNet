"""Dataset and transform utilities for LS-CATNet."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def pil_loader_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


class SmokeImageFolder(Dataset):
    """Image-folder dataset with an explicit class order.

    Expected directory structure:
        root/class_0/*.jpg
        root/class_1/*.jpg
        ...
    """

    def __init__(self, root: str | Path, class_names: Sequence[str], transform=None):
        self.root = Path(root)
        self.class_names = list(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")

        for class_name in self.class_names:
            class_dir = self.root / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Missing class folder: {class_dir}. Expected classes: {self.class_names}"
                )
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                    self.samples.append((str(path), self.class_to_idx[class_name]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No image files found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = pil_loader_rgb(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(img_size: int = 224):
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def make_datasets(data_root: str | Path, class_names: Sequence[str], img_size: int = 224):
    data_root = Path(data_root)
    train_transform, eval_transform = build_transforms(img_size)
    train_ds = SmokeImageFolder(data_root / "train", class_names, transform=train_transform)
    val_ds = SmokeImageFolder(data_root / "val", class_names, transform=eval_transform)
    test_ds = SmokeImageFolder(data_root / "test", class_names, transform=eval_transform)
    return train_ds, val_ds, test_ds


def make_loaders(
    data_root: str | Path,
    class_names: Sequence[str],
    img_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
):
    train_ds, val_ds, test_ds = make_datasets(data_root, class_names, img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def class_counts(dataset: SmokeImageFolder, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, label in dataset.samples:
        counts[label] += 1
    return counts
