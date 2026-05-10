"""Utility functions for SML Project 2: ETH Mug Segmentation.

Includes:
- IoU metric (per-image and batched).
- Run-length encoding (RLE) and the Kaggle submission writer.
- Reproducibility helpers (set_seed).
- Mask <-> tensor conversions.
- Simple visualization helper for sanity checking.
"""
from __future__ import annotations

import os
import random
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Slight performance hit but better reproducibility:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Mean IoU over the batch.

    Args:
        pred:   {0,1} mask tensor of shape (B, 1, H, W) or (B, H, W).
        target: {0,1} mask tensor of the same shape.
        eps:    small constant to avoid 0/0 (e.g., empty image with empty prediction).

    Returns:
        Scalar tensor — mean IoU over the batch.
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def compute_iou_from_logits(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """IoU after applying sigmoid + threshold to model logits."""
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    return iou_score(pred, target)


# ---------------------------------------------------------------------------
# Run-length encoding for Kaggle submissions
# ---------------------------------------------------------------------------

def mask_to_rle(mask: np.ndarray) -> str:
    """Encode a binary mask in run-length format.

    Format follows the standard Kaggle convention used in segmentation
    challenges: pairs of (start_pixel, run_length), 1-indexed, in column-major
    (Fortran) order.

    Args:
        mask: 2-D array with values in {0, 1}.

    Returns:
        Space-separated RLE string. Empty string if the mask is empty.
    """
    pixels = mask.flatten(order="F")
    # Pad with zeros at start and end so we always detect leading/trailing runs.
    pixels = np.concatenate([[0], pixels, [0]])
    # Indices where 0 -> 1 or 1 -> 0.
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if len(runs) == 0:
        return ""
    runs[1::2] -= runs[::2]  # convert end-indices to lengths
    return " ".join(str(int(x)) for x in runs)


def rle_to_mask(rle: str, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of mask_to_rle (handy for debugging / sanity checks).

    Args:
        rle:   RLE-encoded string.
        shape: (H, W) of the target mask.
    """
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    if rle.strip() == "":
        return mask.reshape(shape, order="F")
    tokens = list(map(int, rle.split()))
    starts = np.array(tokens[0::2]) - 1  # back to 0-indexed
    lengths = np.array(tokens[1::2])
    for s, L in zip(starts, lengths):
        mask[s : s + L] = 1
    return mask.reshape(shape, order="F")


def save_predictions(
    predictions: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    """Write a Kaggle-formatted CSV of predictions.

    Args:
        predictions: mapping from ImageId (e.g. "0001") to a 2-D binary mask.
        output_path: path of the output CSV file.

    The CSV has two columns: ImageId, EncodedPixels.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId", "EncodedPixels"])
        for image_id in sorted(predictions.keys()):
            mask = predictions[image_id].astype(np.uint8)
            writer.writerow([image_id, mask_to_rle(mask)])

    print(f"[save_predictions] Wrote {len(predictions)} rows to {output_path}")


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA if available, else MPS (Apple Silicon), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def list_image_ids(folder: str | Path, suffix: str = "_rgb.jpg") -> list[str]:
    """Return sorted list of image IDs (4-digit prefix) found in ``folder``."""
    folder = Path(folder)
    ids = []
    for p in folder.iterdir():
        if p.name.endswith(suffix):
            ids.append(p.name[: -len(suffix)])
    return sorted(ids)
