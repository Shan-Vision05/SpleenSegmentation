from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
)


@dataclass(frozen=True)
class PreprocessConfig:
    hu_min: float = -200.0
    hu_max: float = 300.0
    target_spacing: tuple[float, float, float] = (1.5, 1.5, 1.5)
    roi_size: tuple[int, int, int] = (128, 128, 128)
    axcodes: str = "RAS"


def build_preprocessing_transforms(config: PreprocessConfig) -> Compose:
    """Build preprocessing transforms for Decathlon CT spleen segmentation.

    Output shapes:
      - image: [1, D, H, W] float32, scaled to [0, 1]
      - label: [1, D, H, W] uint8/bool-ish (0/1)

    Notes:
      - We use RAS orientation for consistency across cases.
      - We resample to uniform spacing with bilinear (image) / nearest (label).
      - We crop to the foreground region (from image) then pad/crop to roi_size.
    """

    keys = ["image", "label"]
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes=config.axcodes, labels=None),
            Spacingd(
                keys=keys,
                pixdim=config.target_spacing,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config.hu_min,
                a_max=config.hu_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=keys, source_key="image"),
            ResizeWithPadOrCropd(keys=keys, spatial_size=config.roi_size),
            EnsureTyped(keys=keys, dtype=(np.float32, np.uint8)),
        ]
    )


def build_preprocessing_transforms_inference(config: PreprocessConfig) -> Compose:
    """Inference-time transforms for images only (no label)."""
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes=config.axcodes, labels=None),
            Spacingd(keys=["image"], pixdim=config.target_spacing, mode=("bilinear",)),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config.hu_min,
                a_max=config.hu_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=config.roi_size),
            EnsureTyped(keys=["image"], dtype=(np.float32,)),
        ]
    )


def _as_tuple3(vals: Sequence[float] | Sequence[int]) -> tuple:
    if len(vals) != 3:
        raise ValueError(f"Expected 3 values, got {len(vals)}")
    return (vals[0], vals[1], vals[2])


def config_from_args(
    hu_min: float,
    hu_max: float,
    target_spacing: Sequence[float],
    roi_size: Sequence[int],
    axcodes: str,
) -> PreprocessConfig:
    return PreprocessConfig(
        hu_min=float(hu_min),
        hu_max=float(hu_max),
        target_spacing=_as_tuple3([float(x) for x in target_spacing]),
        roi_size=_as_tuple3([int(x) for x in roi_size]),
        axcodes=str(axcodes),
    )
