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
    parser = argparse.ArgumentParser(description="Verify ONNXRuntime output matches PyTorch for 2.5D UNet.")
    parser.add_argument("--onnx", type=Path, default=Path("spleen_run_dump/onnx/unet25d.onnx"))
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("spleen_run_dump/checkpoints/checkpoints/best.pt"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", type=str, default="CPU", choices=["CPU", "CUDA"], help="ONNXRuntime execution provider")
    parser.add_argument("--num-slices", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    if not args.onnx.exists():
        raise FileNotFoundError(f"Missing ONNX: {args.onnx}")
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

    # PyTorch forward
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    model = _build_unet(num_slices=num_slices)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    x = np.random.randn(1, num_slices, height, width).astype(np.float32)
    with torch.no_grad():
        y_pt = model(torch.from_numpy(x)).cpu().numpy()

    # ONNXRuntime forward
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if str(args.provider).upper() == "CUDA":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(str(args.onnx), providers=providers)
    y_ort = sess.run(None, {"image": x})[0]

    max_abs = float(np.max(np.abs(y_pt - y_ort)))
    mean_abs = float(np.mean(np.abs(y_pt - y_ort)))

    ok = np.allclose(y_pt, y_ort, rtol=float(args.rtol), atol=float(args.atol))
    print("ONNX vs PyTorch")
    print(f"  ok: {ok}")
    print(f"  shape: {y_pt.shape}")
    print(f"  max_abs_err: {max_abs:.6g}")
    print(f"  mean_abs_err: {mean_abs:.6g}")

    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
