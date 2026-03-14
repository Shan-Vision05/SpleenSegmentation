from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _find_npz(data_dir: Path, case_id: str) -> Path:
    p = data_dir / f"{case_id}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return p


def _render_slices(image: np.ndarray, label: np.ndarray, out_path: Path | None, show: bool) -> Path | None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if image.ndim != 4 or label.ndim != 4:
        raise ValueError(f"Expected image/label with shape [C,D,H,W], got {image.shape} / {label.shape}")
    if image.shape != label.shape:
        raise ValueError(f"Image/label shape mismatch: {image.shape} vs {label.shape}")

    # Squeeze channel (C=1). After MONAI transforms, spatial dims are typically [X, Y, Z]
    # (i.e., the same ordering as the loaded NIfTI array after reorientation/resampling).
    vol = np.asarray(image[0])
    msk = np.asarray(label[0])

    if vol.ndim != 3:
        raise ValueError(f"Expected spatial volume with 3 dims, got {vol.shape}")

    x, y, z = vol.shape

    z_mid = z // 2
    if int(msk.sum()) > 0:
        # Pick the axial slice (along Z) with maximum mask area.
        z_max = int(np.argmax(msk.sum(axis=(0, 1))))
    else:
        z_max = z_mid

    def _panel(ax, z_idx: int, title: str) -> None:
        # Axial slice is X-Y plane at a given Z.
        sl = vol[:, :, z_idx]
        m = msk[:, :, z_idx]

        # For normalized images, percentile window still helps contrast.
        vmin = float(np.percentile(sl, 1.0))
        vmax = float(np.percentile(sl, 99.0))

        ax.imshow(sl.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", interpolation="bilinear")
        ax.imshow(
            np.ma.masked_where(m.T <= 0, m.T),
            cmap="Reds",
            alpha=0.40,
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.axis("off")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150, constrained_layout=True)
    _panel(axes[0], z_mid, f"Axial mid slice z={z_mid}")
    _panel(axes[1], z_max, f"Axial max-mask slice z={z_max}")

    if show:
        plt.show()
        plt.close(fig)
        return None

    if out_path is None:
        raise ValueError("out_path must be provided when show=False")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a preprocessed Task09_Spleen .npz case.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_processed/task09_spleen"))
    parser.add_argument("--case-id", type=str, default="spleen_19")
    parser.add_argument("--npz-path", type=Path, default=None, help="Optional direct path to a .npz (overrides --data-dir/--case-id)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/step2_inspect.png"))
    parser.add_argument("--show", action="store_true", help="Show an interactive window instead of saving PNG")

    args = parser.parse_args()

    npz_path = args.npz_path if args.npz_path is not None else _find_npz(args.data_dir, args.case_id)

    d = np.load(npz_path)
    image = d["image"]
    label = d["label"]
    spacing = d.get("spacing", None)

    print(f"Loaded: {npz_path.resolve()}")
    print(f"image: shape={image.shape} dtype={image.dtype} min/max={float(image.min()):.4f}/{float(image.max()):.4f}")
    print(f"label: shape={label.shape} dtype={label.dtype} unique={np.unique(label).tolist()} sum={int(label.sum())}")
    if spacing is not None:
        spacing_list = spacing.tolist() if hasattr(spacing, "tolist") else list(spacing)
        print(f"spacing: {spacing_list}")

    # Heuristic checks
    if not (0.0 <= float(image.min()) <= 1.0 and 0.0 <= float(image.max()) <= 1.0):
        print("WARN: image is not in [0,1] range; check preprocessing.")
    if set(np.unique(label).tolist()) - {0, 1}:
        print("WARN: label is not binary; check interpolation/casting.")

    saved = _render_slices(image=image, label=label, out_path=args.output, show=bool(args.show))
    if saved is not None:
        print(f"Saved inspection figure: {saved.resolve()}")


if __name__ == "__main__":
    main()
