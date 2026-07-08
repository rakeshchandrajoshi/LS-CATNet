"""Print LS-CATNet parameter counts."""

from __future__ import annotations

import argparse

from .model import LSCATNet, count_parameters, module_wise_trainable_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Count LS-CATNet parameters")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weight loading")
    return parser.parse_args()


def main():
    args = parse_args()
    model = LSCATNet(num_classes=args.num_classes, pretrained_backbone=not args.no_pretrained)
    counts = count_parameters(model)
    print("=" * 70)
    print("Model: LS-CATNet with adaptive branch-wise gated fusion")
    print("=" * 70)
    print(f"Total parameters      : {counts['total']:,} ({counts['total']/1e6:.6f} M)")
    print(f"Trainable parameters  : {counts['trainable']:,} ({counts['trainable']/1e6:.6f} M)")
    print(f"Frozen parameters     : {counts['frozen']:,} ({counts['frozen']/1e6:.6f} M)")
    print("-" * 70)
    print("Module-wise trainable parameters")
    for name, params in module_wise_trainable_parameters(model).items():
        print(f"{name:<25} {params:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
