"""Single-image inference with LS-CATNet."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .data import IMAGENET_MEAN, IMAGENET_STD
from .model import LSCATNet
from .utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a single image with LS-CATNet")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--class-names", nargs="+", default=["fognonsmoke", "fogsmoke", "nonsmoke", "smoke"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    image = Image.open(args.image).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    model = LSCATNet(num_classes=len(args.class_names), pretrained_backbone=not args.no_pretrained).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    pred_idx = int(probs.argmax().item())
    print(f"Predicted class: {args.class_names[pred_idx]}")
    print("Probabilities:")
    for name, prob in zip(args.class_names, probs.cpu().tolist()):
        print(f"  {name:<15} {prob:.6f}")


if __name__ == "__main__":
    main()
