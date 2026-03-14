from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.networks.layers import Norm
from monai.networks.nets import UNet


def _infer_num_slices_and_hw_from_ckpt(ckpt: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    num_slices = None
    h = None
    w = None

    args = ckpt.get("args")
    if isinstance(args, dict):
        slice_cfg = args.get("slice")
        if isinstance(slice_cfg, dict) and "num_slices" in slice_cfg:
            try:
                num_slices = int(slice_cfg["num_slices"])
            except Exception:
                pass

        pre_cfg = args.get("preprocess")
        if isinstance(pre_cfg, dict) and "roi_size" in pre_cfg:
            roi = pre_cfg["roi_size"]
            # roi_size is (X, Y, Z); our 2D network consumes (X,Y) as (H,W)
            try:
                h = int(roi[0])
                w = int(roi[1])
            except Exception:
                pass

    return num_slices, h, w


def _build_unet(num_slices: int) -> torch.nn.Module:
    return UNet(
        spatial_dims=2,
        in_channels=int(num_slices),
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: Export 2.5D UNet checkpoint to ONNX.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("spleen_run_dump/checkpoints/checkpoints/best.pt"),
        help="Path to a .pt checkpoint produced by training.",
    )
    parser.add_argument(
        "--onnx-out",
        type=Path,
        default=Path("spleen_run_dump/onnx/unet25d.onnx"),
        help="Output ONNX path.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--dynamic-axes",
        action="store_true",
        help="Export with dynamic batch/height/width axes.",
    )

    # Optional overrides (otherwise inferred from checkpoint when possible)
    parser.add_argument("--num-slices", type=int, default=0, help="Override input channels. 0 = infer")
    parser.add_argument("--height", type=int, default=0, help="Override input height. 0 = infer")
    parser.add_argument("--width", type=int, default=0, help="Override input width. 0 = infer")

    args = parser.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    inferred_slices, inferred_h, inferred_w = _infer_num_slices_and_hw_from_ckpt(ckpt)

    num_slices = int(args.num_slices) if int(args.num_slices) > 0 else inferred_slices
    height = int(args.height) if int(args.height) > 0 else inferred_h
    width = int(args.width) if int(args.width) > 0 else inferred_w

    if num_slices is None:
        raise ValueError("Could not infer --num-slices from checkpoint; pass --num-slices")
    if height is None or width is None:
        raise ValueError("Could not infer --height/--width from checkpoint; pass --height and --width")

    model = _build_unet(num_slices=num_slices)
    state = ckpt.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint missing model_state")
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.from_numpy(np.random.randn(1, num_slices, height, width).astype(np.float32))

    args.onnx_out.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if bool(args.dynamic_axes):
        dynamic_axes = {
            "image": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        model,
        dummy,
        str(args.onnx_out),
        input_names=["image"],
        output_names=["logits"],
        opset_version=int(args.opset),
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    print("Exported ONNX model:")
    print(f"  ckpt: {args.ckpt.resolve()}")
    print(f"  onnx: {args.onnx_out.resolve()}")
    print(f"  input: [B, C, H, W] = [B, {num_slices}, {height}, {width}]")
    print(f"  opset: {int(args.opset)} | dynamic_axes={bool(args.dynamic_axes)}")


if __name__ == "__main__":
    main()
