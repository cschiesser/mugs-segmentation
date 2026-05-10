"""Training script for SML Project 2: ETH Mug segmentation.

Run a training experiment defined by a YAML config file:

    python train.py --config configs/unet_small.yaml

Saves two checkpoints per run under ``checkpoints/<run_name>_{best,last}.pt``.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from eth_mugs_dataset import make_train_val_split
from models import build_model, BCEDiceLoss
from utils import (
    compute_iou_from_logits,
    count_parameters,
    get_device,
    set_seed,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a segmentation model.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed-precision.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, criterion, device, scaler, use_amp: bool
):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            iou = compute_iou_from_logits(logits, masks)

        total_loss += loss.item()
        total_iou += iou.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou.item():.4f}")

    return total_loss / n_batches, total_iou / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0

    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        iou = compute_iou_from_logits(logits, masks)

        total_loss += loss.item()
        total_iou += iou.item()
        n_batches += 1

    return total_loss / n_batches, total_iou / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    run_name = cfg.get("run_name", Path(args.config).stem)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = get_device()
    print(f"[train] device: {device}")
    print(f"[train] run_name: {run_name}")

    # Datasets / loaders -----------------------------------------------------
    data_root = cfg["data"]["train_root"]
    img_size = tuple(cfg["data"].get("img_size", [252, 378]))
    val_fraction = cfg["data"].get("val_fraction", 0.15)

    train_ds, val_ds = make_train_val_split(
        root=data_root,
        val_fraction=val_fraction,
        seed=seed,
        img_size=img_size,
    )
    print(f"[train] train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model ------------------------------------------------------------------
    model = build_model(cfg).to(device)
    print(f"[train] model: {cfg['model']['name']} | params: {count_parameters(model):,}")

    # Loss / optimiser / scheduler ------------------------------------------
    loss_cfg = cfg.get("loss", {})
    criterion = BCEDiceLoss(
        bce_weight=loss_cfg.get("bce_weight", 0.5),
        dice_weight=loss_cfg.get("dice_weight", 0.5),
    )

    lr = cfg["training"]["lr"]
    weight_decay = cfg["training"].get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[train] AMP enabled: {use_amp}")

    # Resume -----------------------------------------------------------------
    start_epoch = 0
    best_val_iou = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_iou = ckpt.get("best_val_iou", -1.0)
        print(f"[train] resumed from {args.resume} (epoch {start_epoch}, best IoU {best_val_iou:.4f})")

    # Training loop ----------------------------------------------------------
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{run_name}_best.pt"
    last_path = ckpt_dir / f"{run_name}_last.pt"

    history = []
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step()

        dt = time.time() - t0
        log_line = (
            f"[epoch {epoch+1:3d}/{epochs}] "
            f"train_loss {train_loss:.4f} train_iou {train_iou:.4f} | "
            f"val_loss {val_loss:.4f} val_iou {val_iou:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s"
        )
        print(log_line)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou,
        })

        # Save last checkpoint every epoch.
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_iou": best_val_iou,
            "config": cfg,
        }, last_path)

        # Save best checkpoint.
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_iou": best_val_iou,
                "config": cfg,
            }, best_path)
            print(f"[train]   ↑ new best val IoU {best_val_iou:.4f} -> saved {best_path.name}")

    # Persist training history for the report.
    history_path = ckpt_dir / f"{run_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] saved history to {history_path}")
    print(f"[train] best val IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
