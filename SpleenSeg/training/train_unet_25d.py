from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import AsDiscrete, Compose, EnsureType, Activations
from torch.utils.data import DataLoader

from SpleenSeg.preprocessing.transforms import PreprocessConfig
from SpleenSeg.training.dataset_25d import DecathlonSpleen25DDataset, Slice25DConfig


def _split_cases(n_cases: int, val_fraction: float, test_fraction: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Split case indices into train/val/test.

    Splits are by case (not slices) to prevent leakage.
    Note: Decathlon Task09 provides a separate 'test' set without labels; this split
    is for internal evaluation using labeled training cases.
    """
    if n_cases <= 0:
        return [], [], []
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0,1)")
    if not (0.0 <= test_fraction < 1.0):
        raise ValueError("test_fraction must be in [0,1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    rng = np.random.default_rng(seed)
    idx = np.arange(n_cases)
    rng.shuffle(idx)

    # Ensure at least 1 train case.
    n_test = int(round(n_cases * test_fraction)) if test_fraction > 0 else 0
    n_val = max(1, int(round(n_cases * val_fraction)))

    # For very small n_cases, prioritize train/val.
    if n_cases <= 2:
        n_test = 0
        n_val = 1 if n_cases == 2 else 0

    # Clamp so that train is non-empty.
    n_test = min(n_test, max(0, n_cases - 2))
    n_val = min(n_val, max(0, n_cases - n_test - 1))

    test_idx = idx[:n_test].tolist() if n_test > 0 else []
    val_idx = idx[n_test : n_test + n_val].tolist() if n_val > 0 else []
    train_idx = idx[n_test + n_val :].tolist()
    return train_idx, val_idx, test_idx


def _evaluate_dice(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    post_pred = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([EnsureType(), AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    model.eval()
    dice_metric.reset()

    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            n_batches += 1
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            y_pred = post_pred(logits)
            y = post_label(labels)
            dice_metric(y_pred=y_pred, y=y)

    if n_batches == 0:
        return float("nan")

    dice_t = dice_metric.aggregate()
    if dice_t is None:
        return float("nan")
    return float(dice_t.item()) if hasattr(dice_t, "item") else float(dice_t)


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_dice: float, args: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "args": args,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: Train 2D UNet on 2.5D slice-stack dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("Task09_Spleen"))

    # Data
    parser.add_argument("--num-slices", type=int, default=5)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Held-out labeled test fraction (from training cases)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0, help="If >0, limit total cases used before splitting")

    # Preprocess
    parser.add_argument("--hu-min", type=float, default=-200.0)
    parser.add_argument("--hu-max", type=float, default=300.0)
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.5, 1.5, 1.5))
    parser.add_argument("--roi-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--axcodes", type=str, default="RAS")

    # Train
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only)")

    # Output
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs/unet25d"))

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp) and device.type == "cuda"

    # Build configs
    pre_cfg = PreprocessConfig(
        hu_min=float(args.hu_min),
        hu_max=float(args.hu_max),
        target_spacing=(float(args.spacing[0]), float(args.spacing[1]), float(args.spacing[2])),
        roi_size=(int(args.roi_size[0]), int(args.roi_size[1]), int(args.roi_size[2])),
        axcodes=str(args.axcodes),
    )
    slice_cfg = Slice25DConfig(num_slices=int(args.num_slices), positive_only=True, negative_ratio=float(args.negative_ratio))

    # Determine case split (by case index in dataset.json training list)
    dataset_json = args.dataset_root / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"Missing dataset.json at: {dataset_json}")
    info = json.loads(dataset_json.read_text())
    n_cases = len(info.get("training", []))
    if n_cases <= 0:
        raise ValueError("No training cases found in dataset.json")

    if args.max_cases and int(args.max_cases) > 0:
        n_cases = min(n_cases, int(args.max_cases))

    train_case_idx, val_case_idx, test_case_idx = _split_cases(
        n_cases=n_cases,
        val_fraction=float(args.val_fraction),
        test_fraction=float(args.test_fraction),
        seed=int(args.seed),
    )

    print(f"Device: {device} (amp={use_amp})")
    print(
        f"Cases: total={n_cases} train={len(train_case_idx)} val={len(val_case_idx)} test={len(test_case_idx)} "
        f"(val_fraction={args.val_fraction}, test_fraction={args.test_fraction})"
    )
    print(f"Case indices: train={train_case_idx} val={val_case_idx} test={test_case_idx}")

    # Recreate datasets with subset selection. (Yes, this preprocesses again; we keep this simple for now.)
    train_ds = DecathlonSpleen25DDataset(
        dataset_root=args.dataset_root,
        preprocess_config=pre_cfg,
        slice_config=slice_cfg,
        augment=True,
        case_indices=train_case_idx,
        verbose=False,
        seed=int(args.seed),
    )

    val_ds = None
    if val_case_idx:
        val_ds = DecathlonSpleen25DDataset(
            dataset_root=args.dataset_root,
            preprocess_config=pre_cfg,
            slice_config=slice_cfg,
            augment=False,
            case_indices=val_case_idx,
            verbose=False,
            seed=int(args.seed),
        )

    test_ds = None
    if test_case_idx:
        test_ds = DecathlonSpleen25DDataset(
            dataset_root=args.dataset_root,
            preprocess_config=pre_cfg,
            slice_config=slice_cfg,
            augment=False,
            case_indices=test_case_idx,
            verbose=False,
            seed=int(args.seed),
        )

    print(f"Slice samples: train={len(train_ds)} val={len(val_ds) if val_ds is not None else 0} test={len(test_ds) if test_ds is not None else 0}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
        )

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
        )

    # Model
    model = UNet(
        spatial_dims=2,
        in_channels=int(args.num_slices),
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    loss_fn = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    post_pred = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([EnsureType(), AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_dir = args.run_dir
    ckpt_best = run_dir / "checkpoints" / "best.pt"
    ckpt_last = run_dir / "checkpoints" / "last.pt"

    best_dice = -1.0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        for batch in train_loader:
            images = batch["image"].to(device)  # [B, C, H, W]
            labels = batch["label"].to(device).float()  # [B, 1, H, W]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().item())
            n_steps += 1

        epoch_loss /= max(1, n_steps)
        print(f"Epoch {epoch}/{args.epochs} | train loss={epoch_loss:.4f}")

        # Validation
        val_dice = None
        if val_loader is not None:
            val_dice = _evaluate_dice(model=model, loader=val_loader, device=device)
            if np.isnan(val_dice):
                val_dice = None
                print(f"Epoch {epoch}/{args.epochs} | val dice skipped (empty val dataset)")
            else:
                print(f"Epoch {epoch}/{args.epochs} | val dice={val_dice:.4f}")

        # Save last
        _save_checkpoint(
            ckpt_last,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_dice=best_dice,
            args={"cli": vars(args), "preprocess": asdict(pre_cfg), "slice": asdict(slice_cfg)},
        )

        # Save best
        if val_dice is not None and val_dice > best_dice:
            best_dice = val_dice
            _save_checkpoint(
                ckpt_best,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_dice=best_dice,
                args={"cli": vars(args), "preprocess": asdict(pre_cfg), "slice": asdict(slice_cfg)},
            )
            print(f"New best dice={best_dice:.4f} | saved: {ckpt_best}")

    print(f"Done. Best val dice={best_dice:.4f}")
    print(f"Last checkpoint: {ckpt_last.resolve()}")
    if ckpt_best.exists():
        print(f"Best checkpoint: {ckpt_best.resolve()}")

    # Final held-out test evaluation (once, after training) using best checkpoint.
    if test_loader is not None and ckpt_best.exists():
        best = torch.load(ckpt_best, map_location=device, weights_only=False)
        model.load_state_dict(best["model_state"], strict=True)
        test_dice = _evaluate_dice(model=model, loader=test_loader, device=device)
        if np.isnan(test_dice):
            print("Final test dice skipped: test split produced no batches (empty dataset).")
        else:
            print(f"Final test dice (held-out labeled split): {test_dice:.4f}")


if __name__ == "__main__":
    main()
