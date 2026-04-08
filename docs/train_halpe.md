## Training a 26‑keypoint HALPE pose model

This guide explains how to go from raw **Halpe Full‑Body** data to a **26‑keypoint body pose model** using Ultralytics YOLO and this repository’s helpers.

The 26‑keypoint subset follows the **AlphaPose HALPE‑26** convention: 26 body/foot keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles, head, neck, hip, and L/R big toe, small toe, heel).

> **Two-stage training:** For the full paediatric gait recipe with per-keypoint
> loss weighting and a two-stage inference safeguard, see:
>
> - **[docs/train_halpe_stage_a.md](train_halpe_stage_a.md)** — Stage A (full-body, yolo11s-pose, weighted loss)
> - **[docs/train_halpe_stage_b.md](train_halpe_stage_b.md)** — Stage B (foot-crop safeguard)
>
> The sections below cover the *basic* single-stage workflow.

### 1. Dataset overview and sources

- **Base dataset**: [Halpe Full‑Body](https://github.com/Fang-Haoshu/Halpe-FullBody)  
  Full whole‑body annotations with 136 keypoints (body, face, hands, feet).
- **Body subset**: HALPE‑26 – first 26 keypoints from the full annotation:

| Index | Name       | Index | Name        |
|-------|------------|-------|-------------|
| 0     | Nose       | 13    | LKnee       |
| 1     | LEye       | 14    | RKnee       |
| 2     | REye       | 15    | LAnkle      |
| 3     | LEar       | 16    | RAnkle      |
| 4     | REar       | 17    | Head        |
| 5     | LShoulder  | 18    | Neck        |
| 6     | RShoulder  | 19    | Hip         |
| 7     | LElbow     | 20    | LBigToe     |
| 8     | RElbow     | 21    | RBigToe     |
| 9     | LWrist     | 22    | LSmallToe   |
| 10    | RWrist     | 23    | RSmallToe   |
| 11    | LHip       | 24    | LHeel       |
| 12    | RHip       | 25    | RHeel       |

Each keypoint is stored in COCO format as **[x, y, visibility]**, so each person has **78 values** for the 26 keypoints.

**Images:**

- **Train images**: from **HICO‑DET**, usually under `hico_20160224_det/images/train2015/`  
- **Val images**: from **COCO val2017**, usually under a `val2017/` directory

### 2. Download and prepare HALPE‑26

Use `scripts/download_halpe26.py` to download annotations, images, and create a cleaned 26‑keypoint annotation set.

From the project root:

```bash
python scripts/download_halpe26.py --output-dir ./data/halpe26 --cleanup
```

This script will:

- Download (or accept existing) annotation JSONs:
  - `halpe_train_v1.json`
  - `halpe_val_v1.json`
- Download image archives:
  - HICO‑DET train archive
  - COCO `val2017.zip`
- Extract them into:

```text
data/halpe26/
  images/
    train/          # HICO‑DET train2015 images
    val/            # COCO val2017 images
  annotations/
    original/       # Full Halpe JSONs (136‑keypoint)
    halpe26/        # Filtered JSONs with only 26 body keypoints/person
  metadata/
    download_manifest.json
    keypoint_info.json
    class_info.json
```

Filtering rules:

- Keep only the first **26** keypoints (78 values) per person.
- Set `num_keypoints = 26`.
- Preserve all other COCO fields (bbox, image_id, id, category_id, area, iscrowd, etc.).
- Skip annotations with fewer than 78 keypoint values and count them in the summary.

If the Google Drive URLs are not accessible from your environment, you can download the files manually from the [Halpe Full‑Body](https://github.com/Fang-Haoshu/Halpe-FullBody) repo and place:

- `halpe_train_v1.json`, `halpe_val_v1.json` into `data/halpe26/annotations/original/`
- HICO‑DET images and COCO `val2017` into the appropriate `images/train` and `images/val` locations

Then re‑run `download_halpe26.py` (it is idempotent and will reuse existing files).

### 3. Convert HALPE‑26 to Ultralytics YOLO pose format

The Ultralytics trainer expects **YOLO pose labels** (`.txt` files) and a dataset YAML. Use:

```bash
python scripts/convert_halpe26_to_yolo_pose.py \
  --dataset-root ./data/halpe26 \
  --output-dir ./data/halpe26_yolo_pose
```

Key arguments:

- `--dataset-root`: root with `images/train`, `images/val`, `annotations/halpe26/*.json`
- `--output-dir`: YOLO‑format dataset root, e.g. `./data/halpe26_yolo_pose`
- `--copy-images`: copy images instead of symlinking
- `--check-images`: verify each referenced image exists before writing labels
- `--min-visible-kpts`: minimum visible keypoints per person (default 1)

After conversion:

```text
data/halpe26_yolo_pose/
  images/
    train/
    val/
  labels/
    train/
    val/
  manifests/
    train_manifest.csv
    val_manifest.csv
  halpe26-pose.yaml
  conversion_report.json
  visualize_samples.py
```

Each label line has **83 values**:

```text
class_id  cx  cy  w  h  kpt1_x kpt1_y kpt1_v ... kpt26_x kpt26_y kpt26_v
```

where `cx, cy, w, h, kpt*_x, kpt*_y` are normalized to \[0, 1\] and `kpt*_v` is the COCO visibility flag (0/1/2).

You can quickly visualize a few samples:

```bash
cd data/halpe26_yolo_pose
python visualize_samples.py
```

(Requires `opencv-python`.)

### 4. Train a HALPE‑26 pose model with Ultralytics

From the project root:

```bash
yolo pose train \
  data=data/halpe26_yolo_pose/halpe26-pose.yaml \
  model=yolo11n-pose.pt \
  epochs=100 \
  imgsz=640
```

Recommendations:

- Start from a **pretrained Ultralytics pose model** (`yolo11n-pose.pt`, `yolov8m-pose.pt`, etc.).
- Ensure the model head is configured for `kpt_shape: [26, 3]` (the generated YAML already declares this).
- Optionally set `project` and `name`:

```bash
yolo pose train \
  data=data/halpe26_yolo_pose/halpe26-pose.yaml \
  model=yolo11n-pose.pt \
  epochs=100 \
  imgsz=640 \
  project=runs/pose \
  name=halpe26
```

**AlphaPose checkpoints:** AlphaPose (and other non‑Ultralytics) models cannot be loaded directly into Ultralytics YOLO. Always fine‑tune from an Ultralytics pose checkpoint.

**flip_idx:** The generated YAML contains a comment reminding you to define `flip_idx` (left‑right keypoint index pairs) if you plan to use horizontal flip augmentation. You should add a 26‑element list that pairs left/right symmetric keypoints (eyes, ears, shoulders, elbows, wrists, hips, knees, ankles, toes, heels).

### 5. Run inference with the trained HALPE‑26 model

There are two main ways to test the trained model on video.

#### 5.1 Ultralytics `predict`

```bash
yolo pose predict \
  model=runs/pose/halpe26/weights/best.pt \
  source=path/to/video.mp4 \
  save=True \
  imgsz=640 \
  device=0
```

This produces an annotated video under `runs/pose/predict*` with keypoints and skeleton drawn.

#### 5.2 Project gait pipeline (`scripts/infer_video.py`)

To use the gait analysis pipeline (event detection + metrics) with your HALPE‑26 model:

```bash
python scripts/infer_video.py path/to/video.mp4 \
  --config config/inference.yaml \
  --model runs/pose/halpe26/weights/best.pt \
  --output-dir results_halpe26 \
  --annotated-video results_halpe26/video_annotated.mp4 \
  --device 0
```

Behaviour:

- The **drawn skeleton** uses all 26 keypoints and a HALPE‑26 skeleton definition (see `HALPE26_SKELETON` in `scripts/infer_video.py`).
- The **gait pipeline** still runs on **6 foot keypoints**, mapped from the 26‑keypoint output using `HALPE26_TO_6` (L/R heel, big toe, small toe). This keeps gait metrics compatible with the existing 6‑point foot logic while allowing richer visualization.

