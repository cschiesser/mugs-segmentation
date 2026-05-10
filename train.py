"""ETH Mug segmentation — training and prediction in one script.

    python train.py                                         # train with config.yaml
    python train.py --config config.yaml                    # same
    python train.py --predict \\
        --checkpoint checkpoints/unet_small_best.pt \\
        --output predictions/sub.csv                        # inference
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from eth_mugs_dataset import ETHMugsDataset, make_train_val_split


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.op = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Up(nn.Module):
    """Bilinear upsample + 1x1 conv, concatenate skip, then DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        widths = [base_channels * (2 ** i) for i in range(depth + 1)]
        self.input_conv = DoubleConv(in_channels, widths[0])
        self.downs = nn.ModuleList([Down(widths[i], widths[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList([
            Up(in_ch=widths[i], skip_ch=widths[i - 1], out_ch=widths[i - 1])
            for i in range(depth, 0, -1)
        ])
        self.output_conv = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.input_conv(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = skips.pop()
        for up in self.ups:
            x = up(x, skips.pop())
        return self.output_conv(x)


# ---------------------------------------------------------------------------
# Residual U-Net
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Two 3x3 convs with a residual skip and a 1x1 projection if needed."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.proj = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class DownRes(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ResidualBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class UpRes(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
        )
        self.block = ResidualBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.block(torch.cat([skip, x], dim=1))


class ResUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        widths = [base_channels * (2 ** i) for i in range(depth + 1)]
        self.input_block = ResidualBlock(in_channels, widths[0])
        self.downs = nn.ModuleList([DownRes(widths[i], widths[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList([
            UpRes(in_ch=widths[i], skip_ch=widths[i - 1], out_ch=widths[i - 1])
            for i in range(depth, 0, -1)
        ])
        self.output_conv = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.input_block(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = skips.pop()
        for up in self.ups:
            x = up(x, skips.pop())
        return self.output_conv(x)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if probs.dim() == 4 and probs.size(1) == 1:
            probs = probs.squeeze(1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        intersection = (probs * target).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        return 1.0 - ((2 * intersection + self.eps) / (union + self.eps)).mean()


class BCEDiceLoss(nn.Module):
    """Weighted BCE-with-logits + Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, target) + self.dice_weight * self.dice(logits, target)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_iou_from_logits(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > threshold).float()
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection
    return ((intersection + 1e-7) / (union + 1e-7)).mean()


def mask_to_rle(mask: np.ndarray) -> str:
    """Encode a binary mask as a Kaggle-format run-length string (column-major)."""
    pixels = np.concatenate([[0], mask.flatten(order="F"), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if len(runs) == 0:
        return ""
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def save_predictions(predictions: dict[str, np.ndarray], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId", "EncodedPixels"])
        for image_id in sorted(predictions.keys()):
            writer.writerow([image_id, mask_to_rle(predictions[image_id].astype(np.uint8))])
    print(f"[predict] wrote {len(predictions)} rows to {output_path}")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> nn.Module:
    name = cfg.get("model_name", "unet")
    kwargs = dict(
        in_channels=3,
        out_channels=1,
        base_channels=cfg.get("base_channels", 32),
        depth=cfg.get("depth", 4),
    )
    if name == "unet":
        return UNet(**kwargs)
    if name == "resunet":
        return ResUNet(**kwargs)
    raise ValueError(f"Unknown model_name: {name!r}. Choose 'unet' or 'resunet'.")


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    model.train()
    total_loss = total_iou = 0.0
    n = 0
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
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou.item():.4f}")
    return total_loss / n, total_iou / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = total_iou = 0.0
    n = 0
    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, masks).item()
        total_iou += compute_iou_from_logits(logits, masks).item()
        n += 1
    return total_loss / n, total_iou / n


# ---------------------------------------------------------------------------
# Main: training
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train or predict ETH mug segmentation.")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config (default: config.yaml)")
    p.add_argument("--resume", default=None, help="Checkpoint path to resume training from")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed-precision")
    p.add_argument("--predict", action="store_true", help="Run inference instead of training")
    p.add_argument("--checkpoint", default=None, help="[predict] checkpoint to load")
    p.add_argument("--output", default="predictions/submission.csv", help="[predict] output CSV path")
    p.add_argument("--threshold", type=float, default=0.5, help="[predict] sigmoid threshold")
    p.add_argument("--tta", action="store_true", help="[predict] horizontal-flip TTA")
    args = p.parse_args()

    if args.predict:
        main_predict(args)
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_name = cfg.get("run_name", "run")
    set_seed(cfg.get("seed", 42))
    device = get_device()
    print(f"[train] device: {device}")
    print(f"[train] run_name: {run_name}")

    train_ds, val_ds = make_train_val_split(
        root=cfg["train_root"],
        val_fraction=cfg.get("val_fraction", 0.15),
        seed=cfg.get("seed", 42),
    )
    print(f"[train] train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    batch_size = cfg.get("batch_size", 8)
    num_workers = cfg.get("num_workers", 2)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)
    print(f"[train] model: {cfg.get('model_name', 'unet')} | params: {count_parameters(model):,}")

    criterion = BCEDiceLoss(
        bce_weight=cfg.get("bce_weight", 0.5),
        dice_weight=cfg.get("dice_weight", 0.5),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    epochs = cfg.get("epochs", 60)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[train] AMP enabled: {use_amp}")

    start_epoch = 0
    best_val_iou = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_iou = ckpt.get("best_val_iou", -1.0)
        print(f"[train] resumed from {args.resume} (epoch {start_epoch})")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
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

        print(
            f"[epoch {epoch+1:3d}/{epochs}] "
            f"train_loss {train_loss:.4f} train_iou {train_iou:.4f} | "
            f"val_loss {val_loss:.4f} val_iou {val_iou:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s"
        )
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_iou": train_iou,
            "val_loss": val_loss, "val_iou": val_iou,
        })

        ckpt_data = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_val_iou": best_val_iou, "config": cfg,
        }
        torch.save(ckpt_data, last_path)
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            ckpt_data["best_val_iou"] = best_val_iou
            torch.save(ckpt_data, best_path)
            print(f"[train]   ↑ new best val IoU {best_val_iou:.4f} -> saved {best_path.name}")

    history_path = ckpt_dir / f"{run_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] saved history to {history_path}")
    print(f"[train] best val IoU: {best_val_iou:.4f}")


# ---------------------------------------------------------------------------
# Main: prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def main_predict(args: argparse.Namespace) -> None:
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required for --predict")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = get_device()
    print(f"[predict] device: {device}")

    test_ds = ETHMugsDataset(cfg["test_root"], mode="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
    )
    print(f"[predict] test images: {len(test_ds)}")

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    best_iou = ckpt.get("best_val_iou", float("nan"))
    print(f"[predict] loaded {args.checkpoint} (best val IoU: {best_iou:.4f})")

    predictions: dict[str, np.ndarray] = {}
    for images, image_ids in tqdm(test_loader, desc="predict"):
        images = images.to(device, non_blocking=True)
        probs = torch.sigmoid(model(images))
        if args.tta:
            probs_flip = torch.sigmoid(model(torch.flip(images, dims=[3])))
            probs = (probs + torch.flip(probs_flip, dims=[3])) / 2
        masks = (probs > args.threshold).cpu().numpy().astype(np.uint8).squeeze(1)
        for i, img_id in enumerate(image_ids):
            predictions[img_id] = masks[i]

    save_predictions(predictions, args.output)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
