"""Inference + Kaggle submission for SML Project 2.

Loads a checkpoint, runs the model over the test set, applies sigmoid +
threshold, and writes a Kaggle-formatted CSV via utils.save_predictions.

    python predict.py \\
        --config configs/unet_residual.yaml \\
        --checkpoint checkpoints/unet_residual_best.pt \\
        --output predictions/submission_unet_residual.csv
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from eth_mugs_dataset import ETHMugsDataset
from models import build_model
from utils import get_device, save_predictions, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Kaggle predictions.")
    p.add_argument("--config", type=str, required=True, help="YAML config used at training time.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    p.add_argument("--output", type=str, default="predictions/submission.csv")
    p.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold.")
    p.add_argument("--tta", action="store_true", help="Apply horizontal-flip TTA.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict_batch(model, images, tta: bool) -> torch.Tensor:
    """Return per-pixel probabilities for a batch."""
    probs = torch.sigmoid(model(images))
    if tta:
        flipped = torch.flip(images, dims=[3])
        probs_flipped = torch.sigmoid(model(flipped))
        probs_flipped = torch.flip(probs_flipped, dims=[3])
        probs = (probs + probs_flipped) / 2
    return probs


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device()
    print(f"[predict] device: {device}")

    # Test dataset.
    test_root = cfg["data"]["test_root"]
    img_size = tuple(cfg["data"].get("img_size", [252, 378]))
    test_ds = ETHMugsDataset(test_root, mode="test", img_size=img_size)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"].get("batch_size", 8),
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
    )
    print(f"[predict] test images: {len(test_ds)}")

    # Model.
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    best_iou = ckpt.get("best_val_iou", float("nan"))
    print(f"[predict] loaded {args.checkpoint} (best val IoU during training: {best_iou:.4f})")

    # Run inference.
    predictions: dict[str, np.ndarray] = {}
    for images, image_ids in tqdm(test_loader, desc="predict"):
        images = images.to(device, non_blocking=True)
        probs = predict_batch(model, images, tta=args.tta)
        masks = (probs > args.threshold).cpu().numpy().astype(np.uint8)
        # masks has shape (B, 1, H, W); collapse the channel dim.
        masks = masks.squeeze(1)
        for i, img_id in enumerate(image_ids):
            predictions[img_id] = masks[i]

    # Write CSV.
    output_path = Path(args.output)
    save_predictions(predictions, output_path)


if __name__ == "__main__":
    main()
