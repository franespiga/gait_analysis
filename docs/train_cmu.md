## Training a 6‑keypoint CMU foot pose model

This guide walks through using the **CMU Human Foot Keypoint Dataset** (6 keypoints per foot region) to train a YOLO pose model for gait analysis and run inference with this project’s pipeline.

### 1. Dataset overview and sources

The [CMU Human Foot Keypoint Dataset](https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/) augments COCO images with **foot keypoints** to go beyond ankle‑only labels.

- Each annotated person has additional **foot keypoints** (right/left heel, big toe, small toe), giving 6 foot keypoints in total.
- Annotations are distributed as COCO‑style JSON:
  - `person_keypoints_train2017_foot_v1.json`
  - `person_keypoints_val2017_foot_v1.json`
- Images come from **COCO 2017**:
  - `train2017.zip`
  - `val2017.zip`

In this project we use a **6‑keypoint schema** focused on feet:

- L_HEEL, L_BIG_TOE, L_SMALL_TOE, R_HEEL, R_BIG_TOE, R_SMALL_TOE  
  (see `config/keypoint_schema.yaml`)

### 2. Download CMU foot annotations and COCO images

Use the helper script to download annotations from a GitHub mirror or print all URLs:

```bash
python scripts/download_foot_dataset.py --output-dir data/foot_cmu/annotations
```

This will fetch:

- `person_keypoints_train2017_foot_v1.json`
- `person_keypoints_val2017_foot_v1.json`

To see all sources (mirror + official + COCO image URLs), run:

```bash
python scripts/download_foot_dataset.py --print-urls
```

You still need **COCO 2017 images**. Download from:

- Official:
  - Train: `http://images.cocodataset.org/zips/train2017.zip`
  - Val:   `http://images.cocodataset.org/zips/val2017.zip`
- Or via Academic Torrents (see output of `--print-urls`).

Unzip them so you have folders like:

```text
data/foot_cmu/
  images/
    train2017/
    val2017/
  annotations/
    person_keypoints_train2017_foot_v1.json
    person_keypoints_val2017_foot_v1.json
```

(The exact layout doesn’t matter as long as you pass the correct `--images-dir`.)

### 3. Prepare a YOLO pose dataset for 6 foot keypoints

Use `scripts/prepare_dataset.py` to convert CMU foot annotations into **Ultralytics pose label files** (one `.txt` per image) and create a dataset YAML:

```bash
python scripts/prepare_dataset.py \
  --dataset-dir data/foot_pose \
  --coco-annot data/foot_cmu/annotations/person_keypoints_train2017_foot_v1.json \
  --images-dir data/foot_cmu/images/train2017 \
  --schema config/keypoint_schema.yaml \
  --train-split 0.9
```

This will:

- Read the COCO‑style annotation JSON and the images from `--images-dir`.
- Map CMU foot keypoints to the 6‑keypoint schema defined in `config/keypoint_schema.yaml`.
- Write YOLO‑style label files under:

```text
data/foot_pose/
  labels/
    train/
    val/
  images/           # if you used dataset_dir/images as the images root
  data.yaml         # Ultralytics dataset config
```

If you already have images + labels in a different layout, you can also run `prepare_dataset.py` without `--coco-annot` to just generate `data.yaml` pointing at existing folders (see the script docstring for details).

### 4. Train a 6‑keypoint CMU foot pose model

Train directly with Ultralytics YOLO:

```bash
yolo pose train \
  data=data/foot_pose/data.yaml \
  model=yolo11n-pose.pt \
  epochs=100 \
  imgsz=320 \
  batch=16 \
  device=0
```

Notes:

- `data/foot_pose/data.yaml` is the dataset YAML from the previous step (includes `kpt_shape: [6, 3]`).
- `model` should be a pose model (`yolo11n-pose.pt`, `yolov8n-pose.pt`, etc.).
- You can override hyperparameters directly on the CLI (e.g. `lr0=0.01`, `optimizer=adamw`).

To **resume** from the last checkpoint:

```bash
yolo pose train \
  data=data/foot_pose/data.yaml \
  model=runs/pose/train/weights/last.pt \
  resume=True \
  device=0
```

Adjust the `model=` path to whatever run directory Ultralytics used for your experiment (e.g. `runs/pose/foot_pose/weights/last.pt`).

### 5. Run inference and gait analysis

Use the production pipeline `scripts/infer_video.py` to apply the trained 6‑keypoint model to a video and compute gait metrics:

```bash
python scripts/infer_video.py path/to/video.mp4 \
  --config config/inference.yaml \
  --model runs/pose/foot_pose/weights/best.pt \
  --output-dir results_cmu \
  --annotated-video results_cmu/video_annotated.mp4 \
  --device 0
```

Behaviour:

- Uses your 6‑keypoint CMU foot model for pose estimation.
- Converts per‑frame raw keypoints to the 6‑keypoint representation expected by the gait pipeline.
- Runs temporal smoothing, event detection (IC/TO), and contact classification.
- Writes:
  - Per‑event CSV (`*_events.csv`)
  - Per‑step CSV (`*_steps.csv`)
  - Summary JSON (`*_summary.json`)
  - Optional annotated video with keypoints and a simple 6‑point foot skeleton (from `config/keypoint_schema.yaml`).

### 6. Notes and recommendations

- Make sure the image paths and annotation JSONs align (same filenames) before running `prepare_dataset.py`.
- If you change the keypoint schema or add more keypoints, update `config/keypoint_schema.yaml` and re‑run the preparation step so the YOLO labels and the gait pipeline stay consistent.
- For faster iteration you can start with smaller image size (`--imgsz 320`) and a small model (`yolov8n-pose.pt`), then scale up once you’re satisfied with the pipeline.

