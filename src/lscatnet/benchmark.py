"""Inference latency, FPS and FLOP benchmarking for LS-CATNet."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .model import LSCATNet, count_parameters
from .utils import get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LS-CATNet")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="benchmark.json")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    model = LSCATNet(num_classes=args.num_classes, pretrained_backbone=not args.no_pretrained).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()

    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    gmacs, gflops = None, None
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        gmacs = macs / 1e9
        gflops = 2 * macs / 1e9
    except Exception as exc:
        print(f"THOP unavailable or failed; skipping FLOPs. Reason: {exc}")

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(args.runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    result = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "input_shape": [1, 3, args.img_size, args.img_size],
        "mean_latency_ms": float(arr.mean()),
        "std_latency_ms": float(arr.std()),
        "median_latency_ms": float(np.median(arr)),
        "fps": float(1000.0 / arr.mean()),
        "gmacs": None if gmacs is None else float(gmacs),
        "gflops_2x_macs": None if gflops is None else float(gflops),
        "params": count_parameters(model),
    }
    save_json(result, args.out)
    print(result)


if __name__ == "__main__":
    main()
