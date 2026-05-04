# Training Stage A — Full-body HALPE-26 pose model

This document describes the **Stage A** training recipe for the HALPE-26 pose model used in the gait analysis pipeline.  Stage A produces a full-body model (`yolo11s-pose`, 26 keypoints) with per-keypoint loss weighting that emphasises lower-limb and foot joints.

**Target population:** paediatric gait (children walking, side-view mobile phone video).

---

## 1. Prerequisites

| Requirement | Version / Notes |
|---|---|
| Python | 3.10+ |
| Poetry env | `poetry install -E yolo` |
| Ultralytics | ^8.0 (tested with 8.2+) |
| GPU | 12–16 GB VRAM recommended for `imgsz=768, batch=8` |
| HALPE-26 dataset | Prepared per `docs/train_halpe.md` §1–3 |

Ensure the YOLO-format dataset is ready at `data/halpe26_yolo_pose/` with `halpe26-pose.yaml`, including `flip_idx` and `kpt_shape: [26, 3]`.

---

## 2. Architecture

| Parameter | Value | Rationale |
|---|---|---|
| Base model | `yolo11s-pose.pt` | Small — good accuracy/speed trade-off for 26 keypoints |
| `imgsz` | 768 | Children's feet are small in pixels; higher resolution improves foot keypoint localisation |
| `multi_scale` | 0.25 | Jitters ±25 % per batch (576–960 px); improves robustness to varying child-camera distance |

The model head is automatically reconfigured for `kpt_shape: [26, 3]` by Ultralytics when the dataset YAML is loaded.

---

## 3. Per-keypoint loss weighting

Ultralytics does not natively support per-keypoint loss weights.  We provide a custom trainer (`src/gait_analysis/weighted_pose_trainer.py`) that subclasses `PoseTrainer → PoseModel → v8PoseLoss` to apply an element-wise weight vector.

**Default weights (HALPE-26 order):**

| Index | Keypoint | Weight | Rationale |
|---|---|---|---|
| 0–14 | Upper body + knees | 1.0 | Baseline |
| 15, 16 | LAnkle, RAnkle | **2.0** | Gait event anchor |
| 17–19 | Head, Neck, Hip | 1.0 | Baseline |
| 20–25 | Toes + Heels | **3.0** | Critical for heel-strike vs toe-walk classification |

Weights are defined in `config/train_halpe_stage_a.yaml` under `kpt_weights` and can be tuned without code changes.

> **Sensitivity note:** Do not over-weight toes if labels are noisy (e.g. occluded or ambiguously annotated).  Start with the defaults and increase incrementally if val foot-subset OKS improves.

---

## 4. Augmentation stack

| Augmentation | Value | Notes |
|---|---|---|
| `hsv_h / hsv_s / hsv_v` | 0.02 / 0.7 / 0.5 | Indoor + outdoor, phone cameras |
| `degrees` | 5.0 | Small rotation (walking is approximately level) |
| `translate` | 0.15 | ±15 % — keep feet in frame |
| `scale` | 0.4 | ±40 % zoom |
| `shear` | 2.0 | Light shear |
| `perspective` | 0.0002 | Subtle perspective warp |
| `fliplr` | 0.5 | Horizontal flip with `flip_idx` in dataset YAML |
| `mosaic` | 1.0 | Full mosaic |
| `mixup` | 0.15 | Light mixup |
| `copy_paste` | 0.1 | Light copy-paste |
| `close_mosaic` | 15 | Disable mosaic for last 15 epochs (stabilises small-object learning) |
| `erasing` | 0.3 | Moderate random erasing (simulates partial occlusion) |

**Motion blur / defocus:** Not natively available in Ultralytics augmentation.  To add motion blur, use an `albumentations` transform in a custom dataset class (see Ultralytics docs) or apply offline preprocessing.  This is documented as a limitation / future work.

---

## 5. Schedule and optimisation

| Parameter | Value | Notes |
|---|---|---|
| `epochs` | 200 | Sufficient for convergence with patience=20 |
| `patience` | 20 | Early stopping after 20 epochs with no val improvement |
| `cos_lr` | `true` | Cosine annealing LR schedule |
| `warmup_epochs` | 5.0 | Warm-up for stable early training |
| `lr0 / lrf` | 0.01 / 0.001 | Initial / final LR ratio |
| `optimizer` | `auto` | SGD for large batch, AdamW otherwise |
| `amp` | `true` | Mixed precision — disable if NaN gradients occur |
| `deterministic` | `true` | Reproducible results (minor GPU non-determinism possible) |
| `seed` | 42 | Fixed seed |

**EMA:** Enabled by default in Ultralytics (`ema=True`) for all YOLOv8/v11 training.  No special configuration needed.

---

## 6. Training commands

### Quick start

```bash
# From project root, poetry env active
python scripts/train_halpe_stage_a.py
```

### With GPU and custom batch size

```bash
python scripts/train_halpe_stage_a.py --device 0 --batch 16
```

### Resume from checkpoint

```bash
python scripts/train_halpe_stage_a.py --resume runs/pose/halpe26_stage_a/weights/last.pt
```

### Direct Ultralytics CLI (without per-keypoint weighting)

```bash
yolo pose train \
  data=data/halpe26_yolo_pose/halpe26-pose.yaml \
  model=yolo11s-pose.pt \
  epochs=200 imgsz=768 multi_scale=0.25 \
  cos_lr=true warmup_epochs=5 patience=20 \
  close_mosaic=15 erasing=0.3 \
  project=runs/pose name=halpe26_stage_a
```

> **Note:** The CLI command does not apply per-keypoint weighting.  Use `scripts/train_halpe_stage_a.py` for the full recipe.

---

## 7. Expected VRAM and timing

| GPU | imgsz | batch | VRAM (approx.) | Epochs / hour |
|---|---|---|---|---|
| RTX 3060 12 GB | 768 | 4 | ~10 GB | ~8 |
| RTX 3090 24 GB | 768 | 8 | ~14 GB | ~15 |
| RTX 4090 24 GB | 768 | 16 | ~18 GB | ~25 |

Reduce `batch` or `imgsz` if VRAM is limited.

---

## 8. Validation and monitoring

- Ultralytics logs validation metrics each epoch under `runs/pose/halpe26_stage_a/`.
- Key metrics to watch: `pose/P`, `pose/R`, `pose/mAP50`, and especially foot-subset OKS (plot manually from val predictions).
- Training plots (`results.png`, `confusion_matrix.png`) are saved automatically.

---

## 9. Pediatric-specific considerations

- **Smaller feet in pixels** → the higher `imgsz` (768) and `multi_scale` directly address this.
- **Child proportions:** HALPE dataset is mixed-age.  If child-specific images can be identified, stratified sampling during training would improve generalization.  Document this as a known limitation if not feasible.
- **Shorter stride frequency / higher instability:** Handled at inference time via temporal smoothing and event detection thresholds in `config/inference.yaml`, not via training augmentation.

---

## 10. Output

After training, the best model weights are at:

```
runs/pose/halpe26_stage_a/weights/best.pt
```

Copy to `models/halpe26_stage_a/best.pt` for deployment:

```bash
mkdir -p models/halpe26_stage_a
cp runs/pose/halpe26_stage_a/weights/best.pt models/halpe26_stage_a/best.pt
```

Then update `config/inference.yaml`:

```yaml
model_path: models/halpe26_stage_a/best.pt
```
