#!/usr/bin/env python
"""Leakage-safe train/validation/test split preparation.

This script must be run on the raw, unaugmented image folders. It first creates
train/val/test splits at the original-image level, then optionally applies
offline augmentation only to the training subset.

Expected input structure:
    raw_root/class_name/*.jpg

Output structure:
    output_root/train/class_name/*.jpg
    output_root/val/class_name/*.jpg
    output_root/test/class_name/*.jpg
    output_root/split_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image, ImageEnhance, ImageOps

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(raw_root: Path, class_names: Sequence[str]) -> Dict[str, List[Path]]:
    files = {}
    for cls in class_names:
        cls_dir = raw_root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {cls_dir}")
        imgs = [p for p in sorted(cls_dir.rglob("*")) if p.is_file() and p.suffix.lower() in VALID_EXT]
        if not imgs:
            raise RuntimeError(f"No images found in: {cls_dir}")
        files[cls] = imgs
    return files


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def original_id(path: Path, class_name: str) -> str:
    # Stable original-image ID. This is saved to the manifest and can be used
    # to prove that no original image appears in more than one split.
    return f"{class_name}_{sha1_text(str(path.resolve()))}_{path.stem}"


def split_list(items: List[Path], train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def augment_image(img: Image.Image, rng: random.Random) -> Image.Image:
    """Random shift/flip/rotation/zoom/color augmentation for training images only."""
    img = img.convert("RGB")
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    if rng.random() < 0.3:
        img = ImageOps.flip(img)

    angle = rng.uniform(-15, 15)
    translate_x = rng.uniform(-0.08, 0.08) * img.size[0]
    translate_y = rng.uniform(-0.08, 0.08) * img.size[1]
    scale = rng.uniform(0.9, 1.1)

    # PIL affine matrix maps output coordinates to input coordinates.
    w, h = img.size
    inv_scale = 1.0 / scale
    matrix = (inv_scale, 0, -translate_x, 0, inv_scale, -translate_y)
    img = img.transform((w, h), Image.Transform.AFFINE, matrix, resample=Image.Resampling.BILINEAR)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))

    # Mild color jitter similar to the training transform.
    if rng.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.9, 1.1))
    return img


def save_augmented(src: Path, dst: Path, rng: random.Random) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        aug = augment_image(img, rng)
        aug.save(dst, quality=95)


def main():
    parser = argparse.ArgumentParser(description="Create leakage-safe split folders for LS-CATNet")
    parser.add_argument("--raw-root", required=True, help="Raw, unaugmented class-folder dataset")
    parser.add_argument("--output-root", required=True, help="Output split dataset root")
    parser.add_argument("--class-names", nargs="+", default=["fognonsmoke", "fogsmoke", "nonsmoke", "smoke"])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-originals", action="store_true", help="Copy original files instead of symlinking")
    parser.add_argument("--augment-train", action="store_true", help="Create offline augmented variants only for train split")
    parser.add_argument("--augment-factor", type=int, default=4, help="Number of augmented variants per original train image")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = list_images(raw_root, args.class_names)
    rng = random.Random(args.seed)
    rows = []

    for cls in args.class_names:
        train, val, test = split_list(files[cls], args.train_ratio, args.val_ratio, args.seed)
        split_map = {"train": train, "val": val, "test": test}
        for split, paths in split_map.items():
            for src in paths:
                orig_id = original_id(src, cls)
                ext = src.suffix.lower()
                dst = output_root / split / cls / f"{orig_id}{ext}"
                if args.copy_originals:
                    safe_copy(src, dst)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    dst.symlink_to(src.resolve())
                rows.append(
                    {
                        "orig_id": orig_id,
                        "class": cls,
                        "split": split,
                        "source_path": str(src),
                        "saved_path": str(dst),
                        "augmented": "False",
                    }
                )

                if split == "train" and args.augment_train:
                    for aug_idx in range(1, args.augment_factor + 1):
                        aug_dst = output_root / "train" / cls / f"{orig_id}_aug{aug_idx:02d}.jpg"
                        save_augmented(src, aug_dst, rng)
                        rows.append(
                            {
                                "orig_id": orig_id,
                                "class": cls,
                                "split": "train",
                                "source_path": str(src),
                                "saved_path": str(aug_dst),
                                "augmented": "True",
                            }
                        )

    manifest_path = output_root / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["orig_id", "class", "split", "source_path", "saved_path", "augmented"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Prepared split dataset at: {output_root}")
    print(f"Manifest saved to: {manifest_path}")
    print("Important: splitting was performed before augmentation; augmentation was applied only to train when requested.")


if __name__ == "__main__":
    main()
