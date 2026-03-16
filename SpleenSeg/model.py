"""Shared model architecture and checkpoint utilities for SpleenSeg.

Single source of truth for:
  - The 2D UNet used across training, ONNX export, and inference.
  - Checkpoint metadata helpers (reading stored training hyperparameters).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def build_unet_2d(num_slices: int) -> torch.nn.Module:
    """Build the 2D UNet used across training, export, and inference.

    Args:
        num_slices: Number of input channels (stacked axial slices for 2.5D context).

    Returns:
        MONAI UNet instance (un-initialised weights).
    """
    from monai.networks.layers import Norm
    from monai.networks.nets import UNet

    return UNet(
        spatial_dims=2,
        in_channels=int(num_slices),
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )


def read_ckpt_meta(ckpt: Path | dict[str, Any]) -> dict[str, Any]:
    """Extract stored training hyperparameters from a checkpoint.

    Accepts either a filesystem :class:`Path` (loaded internally on CPU) or an
    already-loaded checkpoint dict so callers that already hold the dict do not
    need to read the file twice.

    Returns:
        dict with keys:
          - ``num_slices``: ``int`` or ``None``
          - ``roi_size``: ``tuple[int, int, int]`` or ``None``  (X, Y, Z ordering)
    """
    if isinstance(ckpt, Path):
        ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)

    args = ckpt.get("args")
    if not isinstance(args, dict):
        return {"num_slices": None, "roi_size": None}

    num_slices: int | None = None
    roi_size: tuple[int, int, int] | None = None

    slice_cfg = args.get("slice")
    if isinstance(slice_cfg, dict):
        try:
            num_slices = int(slice_cfg["num_slices"])
        except (KeyError, TypeError, ValueError):
            pass

    pre_cfg = args.get("preprocess")
    if isinstance(pre_cfg, dict) and "roi_size" in pre_cfg:
        try:
            r = pre_cfg["roi_size"]
            roi_size = (int(r[0]), int(r[1]), int(r[2]))
        except (IndexError, TypeError, ValueError):
            pass

    return {"num_slices": num_slices, "roi_size": roi_size}
