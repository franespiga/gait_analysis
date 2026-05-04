# Training Stage B — Foot ROI safeguard model

Stage B is a **secondary pose model** trained on cropped lower-leg / foot regions, used only when Stage A produces low foot confidence, inter-foot overlap, or high keypoint jitter.  It acts as a safeguard to resolve foot keypoints under challenging conditions (e.g. feet crossing, partial occlusion).

---

## 1. Rationale

| Problem | Stage B solution |
|---|---|
| Feet overlap in camera view | Crop isolates one foot region; model sees feet at higher resolution |
| Low foot keypoint confidence | Smaller model on crop is more focused, often higher conf |
| High keypoint jitter between frames | Fresh prediction on crop can break jitter cycles |

Stage B uses the **same HALPE-26 keypoint schema** and the same source dataset — only the *view* (cropped vs full-frame) differs.

---

## 2. Building the crop dataset

The crop builder reads the Stage A YOLO-format dataset and produces a new dataset of foot/lower-leg crops.

### Command

```bash
python scripts/build_foot_crops.py \
  --stage-a-dir data/halpe26_yolo_pose \
  --output-dir  data/halpe26_foot_crops \
  --margin 0.4 --min-crop 192 --max-crop 384
```

### Parameters

| Argument | Default | Description |
|---|---|---|
| `--stage-a-dir` | (required) | Path to the Stage A YOLO dataset (`images/` + `labels/`) |
| `--output-dir` | (required) | Output directory for cropped dataset |
| `--margin` | 0.4 | Margin fraction around the ankle span (controls crop size) |
| `--min-crop` | 192 | Minimum crop side in pixels |
| `--max-crop` | 384 | Maximum crop side in pixels |

### Crop policy

1. For each person annotation, locate ankles (indices 15, 16).
2. Compute a square region centred on the mid-ankle point, sized by the foot keypoint span + margin.
3. Shift the centre slightly downward (15 % of half-side) to capture toes.
4. Clamp to image boundaries.
5. Transform all 26 keypoints into crop-local coordinates; set visibility to 0 for points outside the crop.

### Output structure

```
data/halpe26_foot_crops/
  images/
    train/    # crop JPEGs  (e.g. COCO_train2015_000000001_p0.jpg)
    val/
  labels/
    train/    # YOLO pose labels (83 values per line)
    val/
  foot_crops.yaml   # Ultralytics dataset YAML with kpt_shape + flip_idx
```

---

## 3. Training

### Command

```bash
python scripts/train_halpe_stage_b.py
python scripts/train_halpe_stage_b.py --device 0 --batch 32
```

### Config: `config/train_halpe_stage_b.yaml`

| Parameter | Value | Notes |
|---|---|---|
| Model | `yolo11n-pose.pt` | Nano — crops are small, lighter model suffices |
| `imgsz` | 320 | Crops are 192–384 px |
| `multi_scale` | 0.25 | ±25 % jitter |
| `epochs` | 150 | |
| `patience` | 20 | Early stopping |
| `batch` | 32 | Small images allow larger batch |
| `pose` (loss) | 16.0 | Higher pose gain for crop model |
| `kpt_weights` | same as Stage A | Ankle 2.0, foot 3.0 |

VRAM requirement is low (~4–6 GB at `batch=32, imgsz=320`).

---

## 4. Inference contract

Stage B is triggered **per-frame** during inference when *any* of these conditions fire:

| Condition | Config key | Default |
|---|---|---|
| Mean foot-keypoint confidence < threshold | `stage_b.trigger_min_foot_conf` | 0.3 |
| Overlap flag (L/R foot centroids < 30 px apart) | `stage_b.trigger_overlap_flag` | `true` |
| Mean foot-keypoint jitter > threshold | `stage_b.trigger_max_jitter_px` | 15.0 px/frame |

When triggered, `stage_b_refinement.refine_foot_keypoints()`:

1. Computes a foot crop from the Stage A keypoints.
2. Runs Stage B model on the crop.
3. For each foot keypoint index (15, 16, 20–25), replaces the Stage A prediction **only if** Stage B confidence is higher.
4. Returns merged keypoints to the main pipeline.

### Enabling Stage B in inference

In `config/inference.yaml`:

```yaml
stage_b:
  enabled: true
  model_path: models/halpe26_stage_b/best.pt
  trigger_min_foot_conf: 0.3
  trigger_overlap_flag: true
  trigger_max_jitter_px: 15.0
  crop_margin: 0.4
  crop_min_px: 128
  crop_max_px: 384
```

Or via CLI flags (all flags are optional; config file values are the defaults):

```bash
python scripts/infer_video.py video.mp4 --config config/inference.yaml
```

### Default behaviour

Stage B is **disabled by default** (`stage_b.enabled: false`).  Single-stage inference (Stage A only) remains the default.  This ensures full backward compatibility.

---

## 5. Deploying Stage B weights

After training, copy the best weights:

```bash
mkdir -p models/halpe26_stage_b
cp runs/pose/halpe26_stage_b_foot/weights/best.pt models/halpe26_stage_b/best.pt
```

Then set `stage_b.enabled: true` and `stage_b.model_path: models/halpe26_stage_b/best.pt` in `config/inference.yaml`.

---

## 6. Limitations and future work

- Stage B adds latency (~5–15 ms per triggered frame on GPU).
- Crop policy is heuristic-based; a learned ROI proposal could improve it.
- The same 26-keypoint schema is used for crops — upper-body keypoints outside the crop are zeroed.  A 6-keypoint crop model could be more efficient but would require schema conversion.
- Motion blur simulation is not available natively in Ultralytics augmentation.  Consider `albumentations` integration or offline preprocessing.
