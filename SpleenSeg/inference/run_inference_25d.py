from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from SpleenSeg.model import build_unet_2d, read_ckpt_meta
from SpleenSeg.preprocessing.transforms import (
    PreprocessConfig,
    build_preprocessing_transforms,
    build_preprocessing_transforms_inference_fullres,
    build_preprocessing_transforms_inference_fullres_with_label,
)


def _stack_slices(vol_xyz: np.ndarray, z_index: int, num_slices: int) -> np.ndarray:
    """Stack k adjacent axial slices (along Z=last axis) as channels.

    vol_xyz: [X, Y, Z]
    returns: [C=num_slices, X, Y]
    """
    if num_slices % 2 != 1:
        raise ValueError("num_slices must be odd")
    half = num_slices // 2
    z = vol_xyz.shape[2]
    zs = [min(max(z_index + dz, 0), z - 1) for dz in range(-half, half + 1)]
    return np.stack([vol_xyz[:, :, zi] for zi in zs], axis=0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _tile_starts(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    last = size - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _extract_patch_chw(stack_chw: np.ndarray, x0: int, y0: int, tile: int) -> tuple[np.ndarray, slice, slice]:
    """Extract [C, tile, tile] patch with zero-padding if needed."""
    c, h, w = stack_chw.shape
    x1 = min(x0 + tile, h)
    y1 = min(y0 + tile, w)

    patch = stack_chw[:, x0:x1, y0:y1]
    pad_x = tile - patch.shape[1]
    pad_y = tile - patch.shape[2]
    if pad_x > 0 or pad_y > 0:
        patch = np.pad(patch, ((0, 0), (0, pad_x), (0, pad_y)), mode="constant", constant_values=0)
    return patch, slice(x0, x1), slice(y0, y1)



def _dice(pred: np.ndarray, true: np.ndarray) -> float:
    pred = (pred > 0).astype(np.uint8)
    true = (true > 0).astype(np.uint8)
    inter = int((pred & true).sum())
    denom = int(pred.sum() + true.sum())
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def _save_nifti(path: Path, vol_xyz: np.ndarray, affine: np.ndarray | None) -> None:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    img = nib.Nifti1Image(vol_xyz, affine=affine)
    nib.save(img, str(path))


def _save_qc_png(path: Path, image_xyz: np.ndarray, mask_xyz: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    z_mid = image_xyz.shape[2] // 2
    z_max = int(np.argmax(mask_xyz.sum(axis=(0, 1)))) if mask_xyz.sum() > 0 else z_mid

    def _plot(ax, z: int, name: str):
        img2d = image_xyz[:, :, z]
        m2d = mask_xyz[:, :, z]
        ax.imshow(img2d.T, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(m2d.T == 0, m2d.T), cmap="autumn", alpha=0.5, origin="lower")
        ax.set_title(f"{name} z={z}")
        ax.axis("off")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)
    _plot(axes[0], z_mid, "mid")
    _plot(axes[1], z_max, "max-mask")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 7: Run 2.5D inference (PyTorch or ONNX) on a CT volume.")

    parser.add_argument("--image", type=Path, required=True, help="Path to CT .nii/.nii.gz")
    parser.add_argument("--label", type=Path, default=None, help="Optional label for Dice sanity check")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/inference"))
    parser.add_argument("--name", type=str, default="case", help="Output filename stem")
    parser.add_argument(
        "--save-preproc-image",
        action="store_true",
        help="Also save the preprocessed image volume as NIfTI for overlay visualization.",
    )

    # Model inputs
    parser.add_argument("--num-slices", type=int, default=5, help="2.5D context slices (odd)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold on sigmoid(logits)")

    # Backend
    parser.add_argument("--ckpt", type=Path, default=None, help="PyTorch checkpoint (.pt)")
    parser.add_argument("--onnx", type=Path, default=None, help="ONNX model (.onnx)")
    parser.add_argument("--ort-provider", type=str, default="CPU", choices=["CPU", "CUDA"], help="ONNXRuntime provider")

    # Preprocess (must match training)
    parser.add_argument("--hu-min", type=float, default=-200.0)
    parser.add_argument("--hu-max", type=float, default=300.0)
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.5, 1.5, 1.5))
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        help="Training ROI size; used here as tile size (X,Y) for 2D inference.",
    )
    parser.add_argument("--axcodes", type=str, default="RAS")

    # Tiling
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=32,
        help="Overlap (pixels) for 2D tiling. Effective stride = tile - overlap.",
    )

    # Runtime
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch device (cuda/cpu)")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bar")

    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "roi", "fullres"],
        help="auto: ROI if --label else fullres. roi: label-cropped fixed ROI (matches training). fullres: tiled full volume.",
    )

    args = parser.parse_args()

    if args.ckpt is None and args.onnx is None:
        raise ValueError("Pass --ckpt (PyTorch) or --onnx (ONNXRuntime)")
    if args.ckpt is not None and args.onnx is not None:
        raise ValueError("Pass only one of --ckpt or --onnx")

    cfg = PreprocessConfig(
        hu_min=float(args.hu_min),
        hu_max=float(args.hu_max),
        target_spacing=(float(args.spacing[0]), float(args.spacing[1]), float(args.spacing[2])),
        roi_size=(int(args.roi_size[0]), int(args.roi_size[1]), int(args.roi_size[2])),
        axcodes=str(args.axcodes),
    )

    # If using ckpt, try to infer num_slices/roi_size unless user explicitly set them.
    if args.ckpt is not None and args.ckpt.exists():
        meta = read_ckpt_meta(args.ckpt)
        if meta["num_slices"] is not None and int(args.num_slices) == 5:
            args.num_slices = int(meta["num_slices"])
        if meta["roi_size"] is not None and tuple(cfg.roi_size) == (128, 128, 128):
            cfg = PreprocessConfig(
                hu_min=cfg.hu_min,
                hu_max=cfg.hu_max,
                target_spacing=cfg.target_spacing,
                roi_size=meta["roi_size"],
                axcodes=cfg.axcodes,
            )

    if int(args.num_slices) % 2 != 1:
        raise ValueError("--num-slices must be odd")

    if not args.image.exists():
        raise FileNotFoundError(f"Missing image: {args.image}")

    mode = str(args.mode)
    if mode == "auto":
        mode = "roi" if args.label is not None else "fullres"

    if mode == "roi":
        if args.label is None:
            raise ValueError("--mode roi requires --label")
        if not args.label.exists():
            raise FileNotFoundError(f"Missing label: {args.label}")
        tfm = build_preprocessing_transforms(cfg)
        out = tfm({"image": str(args.image), "label": str(args.label)})
    elif mode == "fullres":
        if args.label is not None:
            if not args.label.exists():
                raise FileNotFoundError(f"Missing label: {args.label}")
            tfm = build_preprocessing_transforms_inference_fullres_with_label(cfg)
            out = tfm({"image": str(args.image), "label": str(args.label)})
        else:
            tfm = build_preprocessing_transforms_inference_fullres(cfg)
            out = tfm({"image": str(args.image)})
    else:
        raise ValueError(f"Unknown mode: {mode}")

    image = out["image"]
    if hasattr(image, "detach"):
        image_np = image.detach().cpu().numpy()
    else:
        image_np = np.asarray(image)

    # image_np: [1, X, Y, Z]
    if image_np.ndim != 4 or image_np.shape[0] != 1:
        raise ValueError(f"Unexpected image tensor shape: {image_np.shape}")

    vol = image_np[0].astype(np.float32, copy=False)

    true_mask = None
    if args.label is not None and "label" in out:
        lab = out["label"]
        if hasattr(lab, "detach"):
            lab_np = lab.detach().cpu().numpy()
        else:
            lab_np = np.asarray(lab)
        if lab_np.ndim != 4 or lab_np.shape[0] != 1:
            raise ValueError(f"Unexpected label tensor shape: {lab_np.shape}")
        true_mask = (lab_np[0] > 0).astype(np.uint8)

    meta = out.get("image_meta_dict", {})
    affine = None
    if isinstance(meta, dict) and "affine" in meta:
        try:
            affine = np.asarray(meta["affine"], dtype=np.float32)
        except Exception:
            affine = None

    x, y, z = vol.shape

    tile = int(cfg.roi_size[0])
    if int(cfg.roi_size[1]) != tile:
        raise ValueError(f"Expected roi_size[0]==roi_size[1] for square tiles, got {cfg.roi_size}")
    overlap = int(args.tile_overlap)
    if overlap < 0 or overlap >= tile:
        raise ValueError("--tile-overlap must satisfy 0 <= overlap < tile")
    stride = tile - overlap

    iterator = range(z)
    if bool(args.progress):
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(iterator, desc="infer", total=z)
        except Exception:
            iterator = range(z)

    logits_vol = np.zeros((x, y, z), dtype=np.float32)

    if args.ckpt is not None:
        if not args.ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model = build_unet_2d(num_slices=int(args.num_slices)).to(device)

        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model_state")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint missing model_state")
        model.load_state_dict(state, strict=True)
        model.eval()

        xs = _tile_starts(x, tile=tile, stride=stride)
        ys = _tile_starts(y, tile=tile, stride=stride)

        with torch.no_grad():
            for zi in iterator:
                stack_full = _stack_slices(vol, z_index=int(zi), num_slices=int(args.num_slices))  # [C,X,Y]

                sum_logits = np.zeros((x, y), dtype=np.float32)
                sum_w = np.zeros((x, y), dtype=np.float32)

                for x0 in xs:
                    for y0 in ys:
                        patch, xslice, yslice = _extract_patch_chw(stack_full, x0=x0, y0=y0, tile=tile)
                        inp = torch.from_numpy(patch[None, ...]).to(device)
                        logits = model(inp)[0, 0].detach().cpu().numpy()  # [tile,tile]
                        logits = logits[: xslice.stop - xslice.start, : yslice.stop - yslice.start]
                        sum_logits[xslice, yslice] += logits
                        sum_w[xslice, yslice] += 1.0

                logits_vol[:, :, int(zi)] = sum_logits / np.maximum(sum_w, 1.0)

    else:
        if not args.onnx.exists():
            raise FileNotFoundError(f"Missing ONNX model: {args.onnx}")

        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if str(args.ort_provider).upper() == "CUDA":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        sess = ort.InferenceSession(str(args.onnx), providers=providers)

        xs = _tile_starts(x, tile=tile, stride=stride)
        ys = _tile_starts(y, tile=tile, stride=stride)

        for zi in iterator:
            stack_full = _stack_slices(vol, z_index=int(zi), num_slices=int(args.num_slices))  # [C,X,Y]

            sum_logits = np.zeros((x, y), dtype=np.float32)
            sum_w = np.zeros((x, y), dtype=np.float32)

            for x0 in xs:
                for y0 in ys:
                    patch, xslice, yslice = _extract_patch_chw(stack_full, x0=x0, y0=y0, tile=tile)
                    inp = patch[None, ...].astype(np.float32, copy=False)
                    logits = sess.run(None, {"image": inp})[0][0, 0]  # [tile,tile]
                    logits = logits[: xslice.stop - xslice.start, : yslice.stop - yslice.start]
                    sum_logits[xslice, yslice] += logits
                    sum_w[xslice, yslice] += 1.0

            logits_vol[:, :, int(zi)] = sum_logits / np.maximum(sum_w, 1.0)

    prob = _sigmoid(logits_vol)
    mask = (prob >= float(args.threshold)).astype(np.uint8)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_nii = out_dir / f"{args.name}_pred_mask_preproc.nii.gz"
    _save_nifti(pred_nii, mask.astype(np.uint8, copy=False), affine=affine)

    image_nii = None
    if bool(args.save_preproc_image):
        image_nii = out_dir / f"{args.name}_image_preproc.nii.gz"
        _save_nifti(image_nii, vol.astype(np.float32, copy=False), affine=affine)

    qc_png = out_dir / f"{args.name}_qc.png"
    _save_qc_png(qc_png, image_xyz=vol, mask_xyz=mask, title=f"{args.name} | thr={args.threshold}")

    summary: dict[str, Any] = {
        "image": str(args.image),
        "backend": "pytorch" if args.ckpt is not None else "onnxruntime",
        "ckpt": str(args.ckpt) if args.ckpt is not None else None,
        "onnx": str(args.onnx) if args.onnx is not None else None,
        "mode": mode,
        "num_slices": int(args.num_slices),
        "threshold": float(args.threshold),
        "preprocess": asdict(cfg),
        "pred_mask": str(pred_nii),
        "image_preproc": str(image_nii) if image_nii is not None else None,
        "qc_png": str(qc_png),
        "mask_sum": int(mask.sum()),
    }

    if true_mask is not None:
        d = _dice(mask, true_mask)
        if mode == "roi":
            summary["dice_preproc_roi"] = float(d)
        else:
            summary["dice_preproc_fullres"] = float(d)

    (out_dir / f"{args.name}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    print("Inference done")
    print(f"  backend: {summary['backend']}")
    print(f"  pred: {pred_nii.resolve()}")
    print(f"  qc:   {qc_png.resolve()}")
    print(f"  mask_sum: {summary['mask_sum']}")
    print(f"  mode: {mode}")
    if "dice_preproc_fullres" in summary:
        print(f"  dice_preproc_fullres: {summary['dice_preproc_fullres']:.4f}")
    if "dice_preproc_roi" in summary:
        print(f"  dice_preproc_roi: {summary['dice_preproc_roi']:.4f}")


if __name__ == "__main__":
    main()
