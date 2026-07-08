#!/usr/bin/env python
"""Audit train/validation/test leakage using the split manifest and file hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def audit_manifest(manifest: Path) -> int:
    by_orig = defaultdict(set)
    with manifest.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_orig[row["orig_id"]].add(row["split"])
    leaks = {k: v for k, v in by_orig.items() if len(v) > 1}
    if leaks:
        print("Original-image-level leakage detected:")
        for k, v in list(leaks.items())[:20]:
            print(f"  {k}: {sorted(v)}")
        print(f"Total leaking original IDs: {len(leaks)}")
        return 1
    print(f"Manifest audit passed: {len(by_orig):,} original images are confined to one split each.")
    return 0


def audit_hashes(data_root: Path) -> int:
    by_hash = defaultdict(list)
    for split in ["train", "val", "test"]:
        split_root = data_root / split
        if not split_root.exists():
            continue
        for path in split_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VALID_EXT:
                by_hash[sha256_file(path)].append((split, path))
    leaks = {}
    for h, items in by_hash.items():
        splits = {s for s, _ in items}
        if len(splits) > 1:
            leaks[h] = items
    if leaks:
        print("Exact file duplicates detected across splits:")
        for h, items in list(leaks.items())[:20]:
            print(f"  hash={h[:12]} splits={sorted({s for s, _ in items})}")
            for split, path in items[:5]:
                print(f"    {split}: {path}")
        print(f"Total duplicated hashes across splits: {len(leaks)}")
        return 1
    print("Exact-hash audit passed: no identical image files occur across different splits.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Audit leakage across train/val/test splits")
    parser.add_argument("--data-root", required=True, help="Split dataset root")
    parser.add_argument("--manifest", default=None, help="Path to split_manifest.csv")
    parser.add_argument("--hash-audit", action="store_true", help="Also perform SHA256 duplicate audit across split folders")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    manifest = Path(args.manifest) if args.manifest else data_root / "split_manifest.csv"
    status = 0
    if manifest.exists():
        status |= audit_manifest(manifest)
    else:
        print(f"Manifest not found: {manifest}; skipping original-ID audit")
    if args.hash_audit:
        status |= audit_hashes(data_root)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
