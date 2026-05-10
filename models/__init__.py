"""Model factory for SML Project 2."""
from __future__ import annotations

from .unet import UNet
from .unet_residual import ResUNet
from .losses import BCEDiceLoss, DiceLoss


def build_model(cfg: dict):
    """Construct a model from a config dict.

    Expects ``cfg`` to contain a ``model`` sub-dict with at least ``name``.
    """
    mcfg = cfg["model"]
    name = mcfg["name"].lower()

    common = dict(
        in_channels=mcfg.get("in_channels", 3),
        out_channels=mcfg.get("out_channels", 1),
        base_channels=mcfg.get("base_channels", 32),
        depth=mcfg.get("depth", 4),
    )

    if name == "unet":
        return UNet(**common)
    if name in {"resunet", "unet_residual"}:
        return ResUNet(**common)

    raise ValueError(f"Unknown model name: {name!r}")


__all__ = ["UNet", "ResUNet", "BCEDiceLoss", "DiceLoss", "build_model"]
