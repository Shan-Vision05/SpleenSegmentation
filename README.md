# SpleenSeg — 2.5D CT Spleen Segmentation Pipeline

Production-style pipeline for spleen segmentation on CT volumes from the MSD Decathlon Task09_Spleen dataset.

| | |
|---|---|
| **Model** | MONAI 2D UNet, ~1.6 M parameters |
| **Approach** | 2.5D — 5 adjacent axial slices stacked as channels, fed to a 2D UNet |
| **Best val Dice** | ~0.956 mean (8 held-out cases, ROI-mode evaluation) |
| **Inference backends** | PyTorch `.pt` checkpoint or ONNXRuntime `.onnx` |
| **Python** | 3.10+ |

---

## Installation

```bash
# Install PyTorch for your CUDA version first:
# https://pytorch.org/get-started/locally/

pip install -e .
```

---

## Dataset

Download **Task09_Spleen** from the [MSD Decathlon](http://medicaldecathlon.com/) and extract so the layout is:

```
Task09_Spleen/
├── dataset.json
├── imagesTr/    # 41 labelled training CT volumes
├── imagesTs/    # 20 unlabelled test volumes
└── labelsTr/    # 41 binary spleen masks
```

---

## Repo structure

```
SpleenSeg/
├── model.py          # Architecture + checkpoint helpers (shared by all steps)
├── preprocessing/    # MONAI transform pipelines
├── training/         # 2.5D dataset, training loop
├── export/           # ONNX export + numerical verification
└── inference/        # Full-volume slice-by-slice inference
```

---

## How each file works

### `SpleenSeg/model.py`

Single source of truth for the model and checkpoint reading. Two functions:

- **`build_unet_2d(num_slices)`** — builds the MONAI `UNet` with `spatial_dims=2`, `in_channels=num_slices` (default 5), `out_channels=1`, encoder widths `(16, 32, 64, 128, 256)`, strided convolutions, 2 residual units per block, batch normalisation. Output is a single-channel logit map.
- **`read_ckpt_meta(ckpt)`** — reads a `.pt` checkpoint and extracts `num_slices` and `roi_size` that were stored during training, so export/inference scripts don't need those values passed manually.

---

### `SpleenSeg/preprocessing/transforms.py`

Defines `PreprocessConfig` (frozen dataclass: HU window, target spacing, ROI size, orientation) and three MONAI `Compose` pipelines:

- **`build_preprocessing_transforms(config)`** — used at **training time** (image + label).  
  Steps: Load → add channel dim → reorient to RAS → resample to 1.5 mm isotropic → clip HU `[−200, 300]` and normalise to `[0, 1]` → crop to label foreground bounding box → pad/crop to 128³.
- **`build_preprocessing_transforms_inference_fullres(config)`** — used at **inference without a label**.  
  Same steps but no fixed-size crop; the volume keeps its natural shape for tiled inference.
- **`build_preprocessing_transforms_inference_fullres_with_label(config)`** — same as above but carries the label through so Dice can be computed.

---

### `SpleenSeg/training/dataset_25d.py`

`DecathlonSpleen25DDataset` — a PyTorch `Dataset` that:

1. Reads image/label paths from `dataset.json`.
2. Runs the full 3D preprocessing pipeline on each case and **caches everything in RAM** once at startup.
3. Builds an index of `(case_idx, z_index)` pairs — foreground slices always included, plus a sampled set of background slices controlled by `negative_ratio` (default 1.0 background per foreground slice).

Each `__getitem__` call returns:
- `image` — `[5, H, W]` float32 tensor: 5 adjacent axial slices stacked as channels (the 2.5D trick).
- `label` — `[1, H, W]` uint8 tensor: binary mask for the **centre** slice only.

Training augmentations (when `augment=True`): random flips, random 90° rotations, small intensity scale/shift — applied consistently to image and label.

---

### `SpleenSeg/training/train.py`

Full training loop. Key details:

- **Split** — `_split_train_val(n_cases, val_fraction=0.2, seed=0)` shuffles case indices with a fixed RNG seed and reserves 20% (8 of 41 cases) for validation. Reproducible every run.
- **Loss** — MONAI `DiceLoss` with `sigmoid=True` and `squared_pred=True`.
- **Optimiser** — `AdamW`, default `lr=1e-3`, `weight_decay=1e-4`.
- **Checkpointing** — saves `last.pt` every epoch and `best.pt` whenever val Dice improves. Both include model weights, optimiser state, epoch, best Dice, and the full CLI/preprocessing/slice config so nothing needs to be re-specified later.
- **MLflow** (optional) — logs all hyperparameters at start, `train.loss` and `val.dice` each epoch, and optionally uploads checkpoint files as artifacts.

---

### `SpleenSeg/export/export_onnx_unet25d.py`

Loads `best.pt`, reads `num_slices` and `roi_size` from checkpoint metadata, rebuilds the model, loads weights, creates a dummy `[1, 5, 128, 128]` input, and calls `torch.onnx.export` with `opset_version=17` and `do_constant_folding=True`. Supports `--dynamic-axes` for variable batch/spatial sizes.

---

### `SpleenSeg/export/verify_onnx_unet25d.py`

Runs the same random input through both PyTorch and ONNXRuntime and compares outputs with `np.allclose(rtol=1e-3, atol=1e-4)`. Reports `max_abs_err` and `mean_abs_err`. Typical result: `max_abs_err ≈ 1.9e-5` (tiny floating-point ordering difference). Exits with code 2 if the check fails.

---

### `SpleenSeg/inference/run_inference_25d.py`

Full-volume inference pipeline:

1. **Preprocess** the input CT using the same pipeline as training (`roi` mode) or the full-res variant (`fullres` mode).
2. **Inference loop** — for every axial Z slice, stacks 5 adjacent slices → runs the model (PyTorch or ONNXRuntime) → gets a `[1, 1, H, W]` logit. In `fullres` mode neighbouring tiles are averaged via an accumulator (overlap = 32 voxels).
3. **Postprocess** — sigmoid + threshold at 0.5 → binary mask.
4. **Outputs**:
   - `<name>_pred_mask_preproc.nii.gz` — binary mask in preprocessed space.
   - `<name>_image_preproc.nii.gz` — preprocessed CT (with `--save-preproc-image`); load both in 3D Slicer to overlay.
   - `<name>_qc/` — one PNG per axial slice in the foreground bounding box, each a **2×2 grid**:

     | CT (gray) | GT overlay (cyan) |
     |---|---|
     | **Prediction (magenta)** | **Comparison: lime=TP · tomato=FP · blue=FN** |

   - `<name>_qc/summary.png` — 10-slice mosaic for a quick overview.
   - `<name>_summary.json` — all run metadata and Dice score if a label was provided.

#### Inference modes

| Mode | Preprocessing | When to use |
|------|--------------|-------------|
| `roi` | Label-cropped 128³ (identical to training) | Evaluation with a ground-truth label |
| `fullres` | Full volume, tiled patches with overlap | Production / unlabelled scans |
| `auto` | `roi` if `--label` supplied, else `fullres` | Default |

---

## Data flow

```
.nii.gz CT scan
    │
    ▼  transforms.py
    │  Load → Orient (RAS) → Resample (1.5 mm) → Clip HU → Normalise [0,1] → Crop → Resize 128³
    │
128×128×128 float32 volume  ← cached in RAM by dataset_25d.py
    │
    ▼  _stack_slices()  (per Z-slice)
    │
[5, 128, 128] input
    │
    ▼  2D UNet (model.py)
    │
[1, 128, 128] logit → sigmoid → threshold 0.5
    │
    ▼  stack all Z-slices
    │
128×128×128 binary mask → .nii.gz + QC images
```

---

## Step-by-step commands

### Step 1 — Inspect a case

```bash
python -m SpleenSeg.data_loading.verify_dataset \
  --dataset-root Task09_Spleen \
  --split train --index 0 \
  --output artifacts/step1_mid_slice.png
```

### Step 2 — Preprocess dataset (optional on-disk cache)

```bash
python -m SpleenSeg.preprocessing.preprocess_dataset \
  --dataset-root Task09_Spleen \
  --out-dir data_processed/task09_spleen
```

### Step 3 — Verify 2.5D DataLoader

```bash
python -m SpleenSeg.training.verify_dataloader_25d \
  --dataset-root Task09_Spleen --max-cases 1
```

### Step 4 — Train

```bash
python -m SpleenSeg.training.train \
  --dataset-root Task09_Spleen \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-3 \
  --val-fraction 0.2 \
  --run-dir artifacts/runs/unet25d \
  --progress
```

Optional flags: `--amp` (mixed precision, GPU only), `--mlflow` (enable MLflow tracking).

### Step 5 — Export to ONNX

```bash
python -m SpleenSeg.export.export_onnx_unet25d \
  --ckpt     artifacts/runs/unet25d/checkpoints/best.pt \
  --onnx-out artifacts/onnx/unet25d.onnx
```

### Step 6 — Verify ONNX parity

```bash
python -m SpleenSeg.export.verify_onnx_unet25d \
  --ckpt artifacts/runs/unet25d/checkpoints/best.pt \
  --onnx artifacts/onnx/unet25d.onnx
```

### Step 7 — Run inference

**Evaluation on a labelled case (ROI mode):**

```bash
python -m SpleenSeg.inference.run_inference_25d \
  --image  Task09_Spleen/imagesTr/spleen_12.nii.gz \
  --label  Task09_Spleen/labelsTr/spleen_12.nii.gz \
  --onnx   artifacts/onnx/unet25d.onnx \
  --mode   roi \
  --name   spleen_12 \
  --save-preproc-image \
  --out-dir artifacts/inference
```

**Production / unlabelled case (full-resolution):**

```bash
python -m SpleenSeg.inference.run_inference_25d \
  --image  Task09_Spleen/imagesTs/spleen_1.nii.gz \
  --onnx   artifacts/onnx/unet25d.onnx \
  --name   spleen_1 \
  --out-dir artifacts/inference
```

---

## MLflow UI

```bash
mlflow ui --backend-store-uri mlruns/
# Open http://localhost:5000
```

---

## License

MIT — see [LICENSE](LICENSE).
