"""Dataset class for the ETH Mug segmentation task.

Loads paired (RGB image, binary mask) samples and applies augmentations
where geometric transforms are synchronised between image and mask, while
photometric transforms are applied only to the image.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T


# Native resolution of the project images.
IMG_HEIGHT = 252
IMG_WIDTH = 378


class ETHMugsDataset(Dataset):
    """Loads RGB images and (optionally) binary segmentation masks.

    Folder layout expected (matches the project specification):

        root/
            rgb/   0001_rgb.jpg, 0002_rgb.jpg, ...
            masks/ 0001_mask.png, 0002_mask.png, ...    # only for train mode

    Args:
        root:    path to a folder that contains ``rgb/`` (and ``masks/``).
        mode:    one of {"train", "val", "test"}. Controls augmentations and
                 whether masks are loaded.
        img_size: optional (H, W) to resize to. None keeps native resolution.
    """

    def __init__(
        self,
        root: str | Path,
        mode: Literal["train", "val", "test"] = "train",
        img_size: tuple[int, int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.mode = mode
        self.img_size = img_size if img_size is not None else (IMG_HEIGHT, IMG_WIDTH)

        rgb_dir = self.root / "rgb"
        if not rgb_dir.is_dir():
            raise FileNotFoundError(f"Expected rgb folder at {rgb_dir}")

        # Discover IDs from the RGB folder.
        self.image_ids = sorted(
            p.name[: -len("_rgb.jpg")]
            for p in rgb_dir.iterdir()
            if p.name.endswith("_rgb.jpg")
        )
        if len(self.image_ids) == 0:
            raise RuntimeError(f"No images found in {rgb_dir}")

        # Mask folder only required when we have labels.
        self.has_masks = mode != "test"
        if self.has_masks:
            mask_dir = self.root / "masks"
            if not mask_dir.is_dir():
                raise FileNotFoundError(
                    f"mode={mode!r} but no masks folder at {mask_dir}"
                )

        # ImageNet stats are a fine general-purpose normaliser even though we
        # are not using pretrained weights — they centre/spread RGB sensibly.
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # Photometric augmentation applied to the image only (training mode).
        self.color_jitter = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_path = self.root / "rgb" / f"{image_id}_rgb.jpg"
        image = Image.open(img_path).convert("RGB")

        if self.has_masks:
            mask_path = self.root / "masks" / f"{image_id}_mask.png"
            mask = Image.open(mask_path)
            # Some masks are saved as L, some as P/RGBA — collapse to L.
            if mask.mode != "L":
                mask = mask.convert("L")
        else:
            mask = None

        if self.mode == "train":
            image, mask = self._train_transform(image, mask)
        else:
            image, mask = self._eval_transform(image, mask)

        if mask is None:
            # Test mode: return image and the ID so we can write predictions.
            return image, image_id
        return image, mask

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def _eval_transform(self, image: Image.Image, mask: Image.Image | None):
        """Resize (if needed), to-tensor, normalise. No randomness."""
        H, W = self.img_size
        if image.size != (W, H):
            image = TF.resize(image, [H, W], interpolation=TF.InterpolationMode.BILINEAR)
            if mask is not None:
                mask = TF.resize(mask, [H, W], interpolation=TF.InterpolationMode.NEAREST)

        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t, self.mean, self.std)

        if mask is None:
            return image_t, None

        mask_arr = np.array(mask, dtype=np.uint8)
        # Binarise: original masks may be {0, 255} or {0, 1}.
        mask_arr = (mask_arr > 127).astype(np.float32) if mask_arr.max() > 1 else mask_arr.astype(np.float32)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)  # (1, H, W)
        return image_t, mask_t

    def _train_transform(self, image: Image.Image, mask: Image.Image | None):
        """Synced geometric augs on both, photometric augs only on the image."""
        H, W = self.img_size

        # Resize first to working size (in case of unexpected inputs).
        if image.size != (W, H):
            image = TF.resize(image, [H, W], interpolation=TF.InterpolationMode.BILINEAR)
            if mask is not None:
                mask = TF.resize(mask, [H, W], interpolation=TF.InterpolationMode.NEAREST)

        # 1) Random horizontal flip (synced).
        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)

        # 2) Random rotation in [-15, 15] degrees (synced).
        angle = float(torch.empty(1).uniform_(-15.0, 15.0).item())
        image = TF.rotate(
            image, angle,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=[0, 0, 0],
        )
        if mask is not None:
            mask = TF.rotate(
                mask, angle,
                interpolation=TF.InterpolationMode.NEAREST,
                fill=[0],
            )

        # 3) Random resized crop (synced) — zoom-in/out augmentation.
        if torch.rand(1).item() < 0.5:
            i, j, h, w = T.RandomResizedCrop.get_params(
                image, scale=(0.7, 1.0), ratio=(0.9, 1.1)
            )
            image = TF.resized_crop(
                image, i, j, h, w, [H, W],
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            if mask is not None:
                mask = TF.resized_crop(
                    mask, i, j, h, w, [H, W],
                    interpolation=TF.InterpolationMode.NEAREST,
                )

        # 4) Photometric jitter — image only, leaves the mask alone.
        image = self.color_jitter(image)

        # To tensors + normalise.
        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t, self.mean, self.std)

        if mask is None:
            return image_t, None
        mask_arr = np.array(mask, dtype=np.uint8)
        mask_arr = (mask_arr > 127).astype(np.float32) if mask_arr.max() > 1 else mask_arr.astype(np.float32)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)
        return image_t, mask_t


def make_train_val_split(
    root: str | Path,
    val_fraction: float = 0.15,
    seed: int = 42,
    img_size: tuple[int, int] | None = None,
) -> tuple[ETHMugsDataset, ETHMugsDataset]:
    """Deterministically split ``root`` into train + val datasets.

    The split is done by ID, so an image in val never leaks into train.
    Returns two ETHMugsDataset instances that share the same underlying files
    but use different image_id lists and different (train vs eval) transforms.
    """
    base = ETHMugsDataset(root, mode="train", img_size=img_size)
    ids = list(base.image_ids)

    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(val_fraction * len(ids))))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])

    train_ds = ETHMugsDataset(root, mode="train", img_size=img_size)
    train_ds.image_ids = train_ids

    val_ds = ETHMugsDataset(root, mode="val", img_size=img_size)
    val_ds.image_ids = val_ids

    return train_ds, val_ds
