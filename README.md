# SpleenSeg – 2.5D CT Spleen Segmentation Pipeline

Production-style pipeline for spleen segmentation on CT volumes from the
[MSD Decathlon](http://medicaldecathlon.com/) Task09_Spleen dataset.

Uses a **2.5D** approach: adjacent axial slices are stacked as channels and fed
to a 2D UNet, giving volumetric context without the memory cost of full 3D
convolutions.

| | |
|---|---|
| **Model** | MONAI 2D UNet (5-channel 2.5D input) |
| **Best val Dice** | ~0.956 mean (8 held-out cases, ROI-mode evaluation) |
| **Inference backends** | PyTorch `.pt` checkpoint or ONNXRuntime `.onnx` |
| **Python** | 3.10+ |

---

## Pipeline overview

```
Step 1  verify_dataset       Load & visualise one case (NIfTI → PNG)
Step 2  preprocess_dataset   Resample / HU-clip / crop → .npz cache
Step 3  verify_dataloader    Smoke-test 2.5D DataLoader shapes
Step 4  train                Train 2D UNet with val split + optional MLflow
Step 5  export_onnx          Export best checkpoint → ONNX
Step 6  verify_onnx          Assert PyTorch ≈ ONNXRuntime outputs
Step 7  run_inference        Full-volume inference → NIfTI mask + QC PNG
```

## Repo structure

```
SpleenSeg/
├── model.py          # Shared architecture: build_unet_2d, read_ckpt_meta
├── data_loading/     # Step 1 – dataset inspection & QC overlay
├── preprocessing/    # Step 2 – MONAI transforms, preprocessing CLI, QC viewer
├── training/         # Steps 3–4 – 2.5D dataset, train loop, dataloader verifier
├── export/           # Steps 5–6 – ONNX export & numerical verification
└── inference/        # Step 7 – 3D slice-by-slice inference (PyTorch or ONNX)
```

---

## Installation

```bash
# 1. Install PyTorch for your CUDA version first:
#    https://pytorch.org/get-started/locally/

# 2. Install remaining dependencies:
pip install -e .
# or without editable install:
pip install -r requirements.txt
```

## Dataset

Download **Task09_Spleen** from the
[MSD Decathlon Google Drive](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--P)
and extract it so the layout is:

```
Task09_Spleen/
├── dataset.json
├── imagesTr/   (41 training CT volumes)
├── imagesTs/   (20 unlabelled test volumes)
└── labelsTr/   (41 training segmentation masks)
```

---

## Usage

### Step 1 – Inspect a case

```bash
python -m SpleenSeg.data_loading.verify_dataset \
  --dataset-root Task09_Spleen \
  --split train --index 0 \
  --output artifacts/step1_mid_slice.png
```

### Step 2 – Preprocess dataset (optional on-disk cache)

```bash
python -m SpleenSeg.preprocessing.preprocess_dataset \
  --dataset-root Task09_Spleen \
  --out-dir data_processed/task09_spleen
```

Inspect a preprocessed case:

```bash
python -m SpleenSeg.preprocessing.inspect_preprocessed \
  --data-dir data_processed/task09_spleen \
  --case-id spleen_19
```

### Step 3 – Verify 2.5D DataLoader

```bash
python -m SpleenSeg.training.verify_dataloader_25d \
  --dataset-root Task09_Spleen --max-cases 1
```

### Step 4 – Train

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

Add `--amp` for mixed-precision (GPU), `--mlflow` to enable MLflow tracking.

### Step 5 – Export checkpoint to ONNX

```bash
python -m SpleenSeg.export.export_onnx_unet25d \
  --ckpt  artifacts/runs/unet25d/checkpoints/best.pt \
  --onnx-out artifacts/onnx/unet25d.onnx
```

### Step 6 – Verify ONNX parity

```bash
python -m SpleenSeg.export.verify_onnx_unet25d \
  --ckpt artifacts/runs/unet25d/checkpoints/best.pt \
  --onnx artifacts/onnx/unet25d.onnx
```

### Step 7 – Run inference

**Evaluation on a labelled case** (ROI mode – preprocessing matches training):

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

**Production / unlabelled case** (full-volume tiled inference):

```bash
python -m SpleenSeg.inference.run_inference_25d \
  --image Task09_Spleen/imagesTs/spleen_1.nii.gz \
  --onnx  artifacts/onnx/unet25d.onnx \
  --name  spleen_1 \
  --out-dir artifacts/inference
```

Outputs per case: `*_pred_mask_preproc.nii.gz`, `*_image_preproc.nii.gz`
(with `--save-preproc-image`), `*_qc.png`, `*_summary.json`.

#### Inference modes

| Mode | Preprocessing | When to use |
|------|---------------|-------------|
| `roi` | Label-cropped 128³ ROI (identical to training) | Evaluation with a ground-truth label |
| `fullres` | Full volume, tiled 128×128 patches with overlap | Production / unlabelled scans |
| `auto` | `roi` if `--label` supplied, else `fullres` | Default |

#### Visualising outputs

Open outputs in **3D Slicer** or **ITK-SNAP**:
1. Load `*_image_preproc.nii.gz` as the background volume.
2. Load `*_pred_mask_preproc.nii.gz` as a label/segmentation layer.

Or open the auto-generated `*_qc.png` for a quick 2-panel axial-slice check.

---

## MLflow

```bash
mlflow ui --backend-store-uri mlruns/
```

Navigate to `http://localhost:5000` to browse runs, metrics, and artifacts.

---

## Preprocessing details

| Parameter | Value |
|-----------|-------|
| HU clip | [−200, 300] |
| Voxel spacing | 1.5 × 1.5 × 1.5 mm |
| Orientation | RAS |
| ROI (training) | 128 × 128 × 128 voxels (label-foreground cropped) |

---

## License

MIT – see [LICENSE](LICENSE).
