# Gait Analysis

A computer vision project for analyzing walking patterns using pose estimation. This tool processes video footage of people walking and classifies their gait as correct (heel-toe walking), incorrect (toe walking), or suboptimal (flat foot).

## Features

- **YOLO-Based Pose Estimation**: Support for YOLO pose models with:
  - **HALPE 26 keypoints**: Full-body pose (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles, head, neck, hip, and L/R foot keypoints). See `docs/train_halpe.md`.
  - **CMU 6 keypoints**: Foot-focused (L/R heel, big toe, small toe). See `docs/train_cmu.md`.
  - **YOLO COCO**: Standard 17 keypoints (ankle only); **YOLO Lower Body**: 10 keypoints including heel and toe (optional).
- **Accurate Step Classification**: With heel/toe keypoints, precisely detects:
  - **Heel Strike** (Correct): Heel touches ground first, then toe
  - **Toe Strike** (Incorrect): Toe touches ground first (toe walking)
  - **Flat Foot**: Foot lands flat (suboptimal)
- **Comprehensive Metrics**:
  - Total steps per foot
  - Correct vs incorrect vs flat-foot step counts
  - Contact angles for heel/toe strikes
  - Stance time statistics (mean, min, max, median, std)
  - Knee flexion metrics for rigidity assessment
- **JSON Output**: Complete analysis results saved to JSON
- **Real-time Visualization**: Optional window showing keypoints and step classification
- **Video Output**: Save annotated videos with pose overlay

### Production pipeline (training + inference)

The project supports a **production-quality pipeline** for overground mobile phone videos (side or behind, with shoes):

- **Training (two-stage):**
  - **Stage A** — Full-body HALPE-26 model (`yolo11s-pose`, multi-scale, per-keypoint loss weighting for ankles/heels/toes). See [`docs/train_halpe_stage_a.md`](docs/train_halpe_stage_a.md).
  - **Stage B** — Foot-ROI safeguard model trained on lower-leg crops derived from the same HALPE labels. Activated only when Stage A flags low foot confidence or inter-foot overlap. See [`docs/train_halpe_stage_b.md`](docs/train_halpe_stage_b.md).
  - Basic single-stage training is also supported: [`docs/train_halpe.md`](docs/train_halpe.md), [`docs/train_cmu.md`](docs/train_cmu.md).
- **Inference**: Person detection + pose → optional **Stage B foot refinement** → **temporal smoothing** (Savitzky-Golay) → **gait event detection** (IC/TO) → **contact classification** (heel-strike / toe-walk / flat) → **quality-aware filtering** (occlusion/overlap).
- **Outputs**: Per-event CSV, per-step CSV, summary JSON (cadence, step/stride time, stance, swing, step/stride length, speed, symmetry, variability). Optional annotated video. Quality report (all steps vs primary-only).
- **Gait metrics**: Cadence (steps/min), step time & stride time (mean, std), stance time, swing time, duty factor, step length & stride length (with calibration), walking speed (m/s), symmetry indices (L vs R), variability (CV%). Toe-walking bout detection, severity index, heel-strike rate, foot-strike entropy.

**Example commands** (from project root):

```bash
# Stage A training (per-keypoint weighted)
python scripts/train_halpe_stage_a.py --device 0

# Stage B: build foot crops + train
python scripts/build_foot_crops.py --stage-a-dir data/halpe26_yolo_pose --output-dir data/halpe26_foot_crops
python scripts/train_halpe_stage_b.py --device 0

# Gait inference on video (Stage A only, default)
python scripts/infer_video.py path/to/video.mp4 --config config/inference.yaml --model path/to/best.pt

# Gait inference with Stage B enabled (set stage_b.enabled: true in config)
python scripts/infer_video.py path/to/video.mp4 --config config/inference.yaml
```

Configs: `config/inference.yaml`, `config/train_halpe_stage_a.yaml`, `config/train_halpe_stage_b.yaml`, `config/keypoint_schema_halpe.yaml`. Tech stack: Python 3.10+, Ultralytics, OpenCV, NumPy/SciPy, Pandas, YAML.

**Analysis outputs** from the app, CLI scripts, or notebooks are stored under the top-level **`analyses/`** folder. The next level is the source: **`APP`** (Streamlit), **`CLI`** (scripts / `gait-analyze*`), or **`OTHER`** (e.g. notebooks). Each run writes into a timestamped subfolder **`YYYYMMDD_HHMM`**; all generated files for that run (JSON, CSV, annotated video, etc.) go only in that folder. Override with `--output-dir` (or script-specific options) to use a custom directory instead.

## Project Structure

```
gait_analysis/
├── pyproject.toml                  # Poetry configuration and dependencies
├── README.md                       # This file
├── config/
│   ├── inference.yaml              # Inference pipeline config (events, smoothing, Stage B)
│   ├── training.yaml               # Legacy training config
│   ├── train_halpe_stage_a.yaml    # Stage A training (yolo11s-pose, weighted loss)
│   ├── train_halpe_stage_b.yaml    # Stage B training (foot-crop safeguard)
│   ├── keypoint_schema_halpe.yaml  # HALPE-26 keypoint schema + skeleton
│   └── keypoint_schema_cmu.yaml    # CMU 6-keypoint schema
├── src/
│   └── gait_analysis/
│       ├── __init__.py             # Package exports
│       ├── inference_pipeline.py   # Production pipeline (IC/TO, metrics)
│       ├── gait_events.py          # Gait event detection + quality annotation
│       ├── gait_metrics.py         # Step records + summary metrics
│       ├── smoothing.py            # Temporal smoothing (Savitzky-Golay)
│       ├── yolo_detector.py        # YOLO COCO and Lower Body detectors
│       ├── weighted_pose_trainer.py # Custom trainer with per-keypoint loss weights
│       ├── stage_b_refinement.py   # Stage B foot-crop refinement module
│       ├── bout_detection.py       # Toe-walking bout detection
│       ├── severity.py             # Toe-walking severity index
│       ├── extended_metrics.py     # Proxy metrics (heel rise, plantarflexion, etc.)
│       ├── stats_utils.py          # Descriptive stats, group comparisons
│       ├── batch_pipeline.py       # Manifest-driven batch processing
│       ├── eval/                   # Validation metrics (classification, ICC, ROC)
│       ├── keypoint_config.py      # Keypoint configurations for each model
│       ├── base_detector.py        # Abstract detector interface
│       ├── analyzer.py             # Basic gait analysis logic
│       └── ...                     # Legacy modules (angle_analyzer, visualizer, etc.)
├── scripts/
│   ├── infer_video.py              # Single-video inference (Stage A + optional B)
│   ├── train_halpe_stage_a.py      # Stage A training script
│   ├── train_halpe_stage_b.py      # Stage B training script
│   ├── build_foot_crops.py         # Build foot-crop dataset for Stage B
│   ├── convert_halpe26_to_yolo_pose.py  # HALPE COCO → YOLO format
│   ├── download_halpe26.py         # Download HALPE-26 dataset
│   ├── run_batch.py                # Batch pipeline (manifest-driven)
│   ├── evaluate.py                 # Evaluation script (classification, agreement)
│   └── summarize_quality.py        # Quality summary (all vs primary-only steps)
├── tests/                          # Unit tests
├── docs/
│   ├── installation.md             # Installation guide
│   ├── train_halpe.md              # Basic HALPE-26 training
│   ├── train_halpe_stage_a.md      # Stage A training guide
│   ├── train_halpe_stage_b.md      # Stage B training guide
│   └── train_cmu.md                # CMU 6-keypoint training
├── analyses/                       # All analysis outputs (APP/, CLI/, OTHER/)
├── deploy/                         # Docker and Streamlit Cloud deployment
├── notebooks/
│   └── exploration.ipynb           # Exploration notebook
└── data/
    └── .gitkeep                    # Sample videos directory
```


## Documentation

- **Installation & environment**: [docs/installation.md](docs/installation.md)
- **Train on CMU 6‑keypoint foot dataset**: [docs/train_cmu.md](docs/train_cmu.md)
- **Train on HALPE‑26 full‑body dataset (basic)**: [docs/train_halpe.md](docs/train_halpe.md)
- **Train Stage A (full-body, weighted loss)**: [docs/train_halpe_stage_a.md](docs/train_halpe_stage_a.md)
- **Train Stage B (foot-crop safeguard)**: [docs/train_halpe_stage_b.md](docs/train_halpe_stage_b.md)
- **Pipeline usage & gait metrics**: [PIPELINE_USAGE.md](PIPELINE_USAGE.md)
- **Metric definitions**: [METRICS_DEFINITIONS.md](METRICS_DEFINITIONS.md)
- **Deploy the Streamlit app** (Docker, Streamlit Community Cloud): [deploy/README.md](deploy/README.md)

## Usage

### Streamlit application

A web UI for gait analysis (upload video or webcam) lives in **`app/`**. It runs pose estimation and shows keypoints with a live legend (frames, steps, steps/min, average step length).

**Quick start** (from project root):

```bash
poetry install -E yolo -E app
poetry run streamlit run app/gait_streamlit.py
```

Then in the sidebar: choose **Upload video** or **Webcam**, select a **Pose model**, click **Load model**, then **Start Gait Analysis**. Use **Stop** to end.

**Full instructions** (model selection, custom models under `models/`, troubleshooting): see **[app/README.md](app/README.md)**.

### Command Line Interface

The following analysis entry points are available:

#### Basic Analysis (`gait-analyze`)

Basic usage (analyzes video and saves JSON results):
```bash
gait-analyze path/to/video.mp4
```

With visualization window:
```bash
gait-analyze path/to/video.mp4 --show
```

Specify output file:
```bash
gait-analyze path/to/video.mp4 --output path/to/analysis.json --show
```

#### Advanced Analysis (`gait-analyze-advanced`)

The advanced script provides additional features:
- **Angle of contact measurements** (heel-to-floor and toe-to-floor angles)
- **Extended metrics** (mean, min, max, median, std for angles and stance times)
- **Knee flexion metrics** for rigidity assessment
- **Video output** with keypoint visualizations
- **Timestamped results** in organized folders

Basic usage:
```bash
gait-analyze-advanced path/to/video.mp4
```

With visualization window:
```bash
gait-analyze-advanced path/to/video.mp4 --show
```

Custom output directory:
```bash
gait-analyze-advanced path/to/video.mp4 --output-dir my_results --show
```

Without video output (JSON only):
```bash
gait-analyze-advanced path/to/video.mp4 --no-video
```

Use a different YOLO model:
```bash
gait-analyze-advanced path/to/video.mp4 --model yolov8m-pose.pt --device cuda
```

#### Multi-Backend Analysis (`gait-analyze-multi`) ⭐ RECOMMENDED

The multi-backend script supports different pose estimation models for improved accuracy:

**Using Standard YOLO (default - ankle only):**
```bash
gait-analyze-multi path/to/video.mp4
```

**Using YOLO Lower Body (with heel/toe keypoints) - RECOMMENDED:**
```bash
gait-analyze-multi path/to/video.mp4 --backend yolo_lower
```
(Uses `models/yolo_lower/best.pt` by default; override with `--model` if needed.)

The YOLO Lower Body model provides 10 keypoints including heel and toe, enabling accurate heel-strike vs toe-walking detection. Download the model from [Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints](https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints).

With visualization and custom output:
```bash
gait-analyze-multi path/to/video.mp4 --backend yolo_lower --show --output-dir results
```

### Available Options

#### Basic Script (`gait-analyze`)

| Option | Description | Default |
|--------|-------------|---------|
| `video` | Path to input video file | Required |
| `-o, --output` | Path to output JSON file | Same as video with .json extension |
| `--show` | Show visualization window during analysis | False |
| `--model` | YOLO pose model to use | yolov8n-pose.pt |
| `--device` | Inference device (auto, cpu, cuda, cuda:0) | auto |

#### Advanced Script (`gait-analyze-advanced`)

| Option | Description | Default |
|--------|-------------|---------|
| `video` | Path to input video file | Required |
| `-o, --output-dir` | Run directory for results | analyses/CLI/YYYYMMDD_HHMM |
| `--show` | Show visualization window during analysis | False |
| `--no-video` | Don't save annotated output video | False |
| `--model` | YOLO pose model to use | yolov8n-pose.pt |
| `--device` | Inference device (auto, cpu, cuda, cuda:0) | auto |

#### Multi-Backend Script (`gait-analyze-multi`)

| Option | Description | Default |
|--------|-------------|---------|
| `video` | Path to input video file | Required |
| `-o, --output-dir` | Run directory for results | analyses/CLI/YYYYMMDD_HHMM |
| `--show` | Show visualization window during analysis | False |
| `--no-video` | Don't save annotated output video | False |
| `--backend` | Pose estimation backend (see below) | yolo_coco |
| `--model` | Model path/name | yolov8n-pose.pt |
| `--device` | Inference device (auto, cpu, cuda) | auto |

**Available Backends (YOLO-based only):**

| Backend | Keypoints | Heel/Toe | Description |
|---------|-----------|----------|-------------|
| `yolo_coco` / `yolov8` | 17 | ❌ | Standard YOLO COCO format (ankle only) |
| `yolo_lower` | 10 | ✅ | Fine-tuned for lower body with heel/toe |

### Python API

You can also use the package programmatically:

```python
from gait_analysis import KeypointDetector, GaitAnalyzer, GaitVisualizer
from gait_analysis.main import analyze_video

# Simple analysis
results = analyze_video(
    video_path="video.mp4",
    output_path="results.json",
    show_visualization=True
)

# Access results
print(f"Total steps: {results['summary']['total_steps']}")
print(f"Correct percentage: {results['summary']['correct_percentage']}%")
```

## Output Format

### Basic Analysis Output

The basic analysis produces a JSON file with the following structure:

```json
{
  "video_info": {
    "path": "video.mp4",
    "total_frames": 300,
    "fps": 30.0,
    "duration_seconds": 10.0
  },
  "summary": {
    "total_steps": 12,
    "correct_steps": 9,
    "incorrect_steps": 3,
    "correct_percentage": 75.0
  },
  "timing": {
    "avg_step_time_ms": 550.5,
    "avg_correct_step_time_ms": 520.3,
    "avg_incorrect_step_time_ms": 640.1
  },
  "per_foot": {
    "left": {
      "total_steps": 6,
      "correct_steps": 5,
      "incorrect_steps": 1
    },
    "right": {
      "total_steps": 6,
      "correct_steps": 4,
      "incorrect_steps": 2
    }
  },
  "detailed_steps": {
    "left": [
      {
        "foot": "left",
        "step_type": "heel_strike",
        "start_frame": 15,
        "end_frame": 32,
        "start_time_ms": 500.0,
        "end_time_ms": 1066.7,
        "duration_ms": 566.7
      }
    ],
    "right": [...]
  }
}
```

### Advanced Analysis Output

The advanced analysis creates a timestamped folder (`yyyymmdd_hhmm_videoname/`) containing:

1. **`<videoname>_results.json`** - Extended analysis results
2. **`<videoname>_analyzed.mp4`** - Annotated video with keypoint visualizations

The JSON includes comprehensive metrics:

```json
{
  "video_info": {
    "path": "video.mp4",
    "total_frames": 300,
    "fps": 30.0,
    "duration_seconds": 10.0,
    "output_video_path": "results/20240115_1430_video/video_analyzed.mp4"
  },
  "summary": {
    "total_steps": 12,
    "correct_steps": 9,
    "incorrect_steps": 3,
    "correct_percentage": 75.0
  },
  "metrics": {
    "left_foot": {
      "correct_steps": {
        "angles": { "mean": 25.3, "min": 18.5, "max": 32.1, "median": 24.8, "count": 5 },
        "stance_time_ms": { "mean": 450.2, "min": 380.0, "max": 520.0, "median": 445.0, "count": 5 }
      },
      "incorrect_steps": {
        "angles": { "mean": 65.2, "min": 58.0, "max": 72.5, "median": 64.8, "count": 1 },
        "stance_time_ms": { "mean": 320.0, "min": 320.0, "max": 320.0, "median": 320.0, "count": 1 }
      },
      "all_steps": { ... }
    },
    "right_foot": { ... },
    "overall": {
      "correct_steps": { ... },
      "incorrect_steps": { ... },
      "all_steps": { ... }
    }
  },
  "detailed_steps": {
    "left": [
      {
        "foot": "left",
        "step_type": "heel_strike",
        "start_frame": 15,
        "end_frame": 32,
        "start_time_ms": 500.0,
        "end_time_ms": 1066.7,
        "duration_ms": 566.7,
        "contact_angle_degrees": 25.3,
        "stance_time_ms": 450.2
      }
    ],
    "right": [...]
  }
}
```

### Angle Measurements

- **Heel Strike Angle**: The angle of the foot's approach to the ground when the heel contacts first. Measured as the angle between the foot trajectory and the horizontal plane. Lower angles indicate a more proper heel-first gait.

- **Toe Strike Angle**: For incorrect (toe walking) steps, measures the angle at the toe vertex (foot-toe-pavement). Higher angles indicate more pronounced toe walking.




## Algorithm Details

### Keypoint Detection

The system supports multiple pose estimation backends:

#### YOLO COCO (17 keypoints)
Standard YOLOv8-pose with COCO keypoints. Only provides ankle positions, requiring trajectory-based step classification.

#### YOLO fine-tuned with HALPE dataset (26 keypoints) ⭐ RECOMMENDED

Fine-tuned `yolo11s-pose` with HALPE-26 full-body keypoints and per-keypoint loss weighting for ankles/heels/toes (Stage A).  Optional Stage B foot-crop safeguard for overlap / low-confidence scenarios.  See [`docs/train_halpe_stage_a.md`](docs/train_halpe_stage_a.md) and [`docs/train_halpe_stage_b.md`](docs/train_halpe_stage_b.md) for training instructions.




#### YOLO Lower Body (10 keypoints) 
Fine-tuned model from [yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints](https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints):
- Left/Right Hip (0, 1)
- Left/Right Knee (2, 3)
- Left/Right Ankle (4, 5)
- Left/Right Heel (6, 7)
- Left/Right Foot/Toe (8, 9)

Having actual heel and toe positions enables precise step classification.




### Step Detection

Steps are detected by tracking foot phases:
1. **Swing Phase**: Foot is in the air (high vertical velocity)
2. **Stance Phase**: Foot is on the ground (low vertical velocity)

A step is registered when transitioning from swing to stance.

### Step Classification

**With Heel/Toe Keypoints (YOLO Lower Body):**

Classification is based on actual heel and toe Y-positions at contact:
- **Heel Strike**: Heel Y > Toe Y (heel lower/contacts first) - CORRECT
- **Toe Strike**: Toe Y > Heel Y (toe lower/contacts first) - INCORRECT
- **Flat Foot**: |Heel Y - Toe Y| < threshold (simultaneous contact) - SUBOPTIMAL

**Without Heel/Toe Keypoints (YOLO COCO):**

Classification is based on ankle trajectory during landing:
- **Heel Strike**: More gradual descent with horizontal movement
- **Toe Strike**: Rapid, uniform descent with minimal horizontal variation
- **Flat Foot**: Mixed characteristics

## Pose Estimation Models

### YOLO Models

Available standard pose models (download automatically on first use):
- `yolov8n-pose.pt` - Nano (fastest, least accurate)
- `yolov8s-pose.pt` - Small
- `yolov8m-pose.pt` - Medium
- `yolov8l-pose.pt` - Large
- `yolov8x-pose.pt` - Extra Large (slowest, most accurate)

### YOLO Lower Body Model

Download from [GitHub repository](https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints) and place in `models/yolo_lower/`:
```bash
mkdir -p models/yolo_lower
wget -O models/yolo_lower/best.pt https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints/raw/main/best.pt
```

The `yolo_lower` backend uses `models/yolo_lower/best.pt` by default:
```bash
gait-analyze-multi video.mp4 --backend yolo_lower
```

### Datasets

The project supports training with two keypoint setups:

- **CMU 6 keypoints**: Foot keypoints (L/R heel, big toe, small toe) from the [CMU Human Foot Keypoint Dataset](https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/). Download annotations and COCO 2017 images, then prepare labels with `scripts/download_foot_dataset.py` and `scripts/prepare_dataset.py`. Full steps: **[`docs/train_cmu.md`](docs/train_cmu.md)**.
- **HALPE 26 keypoints**: Full-body keypoints from [Halpe Full-Body](https://github.com/Fang-Haoshu/Halpe-FullBody) (26 body/foot keypoints). Download and convert with `scripts/download_halpe26.py` and `scripts/convert_halpe26_to_yolo_pose.py`. Full steps: **[`docs/train_halpe.md`](docs/train_halpe.md)**.
  - **Stage A** (recommended): `yolo11s-pose` with per-keypoint loss weighting → **[`docs/train_halpe_stage_a.md`](docs/train_halpe_stage_a.md)**.
  - **Stage B** (optional safeguard): Foot-crop model → **[`docs/train_halpe_stage_b.md`](docs/train_halpe_stage_b.md)**.

## Development

### Running Tests

#### Basic Test Commands

```bash
# Run all unit tests
poetry run pytest tests/

# Run with verbose output
poetry run pytest tests/ -v

# Run with coverage
poetry run pytest tests/ --cov=gait_analysis
```

### Compatibility Testing

The compatibility test suite runs all pose backends over sample videos and generates annotated output videos with keypoint overlays.

#### Setup

Place your test video(s) in the `data/` folder:
```
data/
├── sample_video.mp4
└── test/
    └── walking_video.mp4
```

#### Running All Backends on a Video

```bash
# Run all available backends on auto-detected video
poetry run pytest tests/test_compatibility.py -v

# Run all backends on a specific video
python -m tests.test_compatibility --video data/test/my_video.mp4
```

#### Running Specific Backends

```bash
# Test only YOLOv8 COCO backend
poetry run pytest tests/test_compatibility.py -v -k "yolov8"

# Test only YOLO Lower Body backend
poetry run pytest tests/test_compatibility.py -v -k "yolo_lower"

# Test only YOLO Lower backend
poetry run pytest tests/test_compatibility.py -v -k "yolo_lower"

# Test multiple specific backends
poetry run pytest tests/test_compatibility.py -v -k "yolov8 or yolo_lower"
```

#### Running with CLI (More Control)

```bash
# Run specific backends via CLI
python -m tests.test_compatibility --backends yolov8 yolo_lower

# Run with specific video and backends
python -m tests.test_compatibility --video data/test/walk.mp4 --backends yolov8 yolo_lower

# Custom output directory
python -m tests.test_compatibility --output-dir checks/my_test
```

#### Specifying Model Paths

For backends that require custom model files (like `yolo_lower`):

```bash
# Via pytest command line (default is models/yolo_lower/best.pt)
poetry run pytest tests/test_compatibility.py -v --yolo-lower-model=models/yolo_lower/best.pt

# Via environment variable (Windows CMD)
set GAIT_YOLO_LOWER_MODEL=models/yolo_lower/best.pt
poetry run pytest tests/test_compatibility.py -v

# Via environment variable (PowerShell)
$env:GAIT_YOLO_LOWER_MODEL="models/yolo_lower/best.pt"
poetry run pytest tests/test_compatibility.py -v

# Via CLI with model paths
python -m tests.test_compatibility --model-path yolo_lower=models/yolo_lower/best.pt
```

#### Environment Variables for Model Paths

| Variable | Backend | Description |
|----------|---------|-------------|
| `GAIT_YOLOV8_MODEL` | `yolov8`, `yolo_coco` | Path to YOLOv8 pose model |
| `GAIT_YOLO_LOWER_MODEL` | `yolo_lower` | Path to YOLO lower body model (default: `models/yolo_lower/best.pt`) |

#### Compatibility Test Output

Results are saved to `checks/YYYYMMDD_HHMM/`:

```
checks/20260218_1430/
├── yolov8/
│   ├── gait_analysis.json       # Comprehensive metrics
│   └── annotated_video.mp4      # Video with skeleton overlay
├── yolo_lower/
│   ├── gait_analysis.json
│   └── annotated_video.mp4
└── compatibility_report.json    # Summary of all backends
```

#### Skipping Slow Tests

```bash
# Skip compatibility tests (they process full videos)
poetry run pytest tests/ -v -m "not slow"

# Run only fast unit tests
poetry run pytest tests/test_backend_registry.py tests/test_schema_mapping.py -v
```

## Visualization Controls

When running with `--show`:
- Press `q` to quit early
- The window shows:
  - Detected skeleton and keypoints
  - Ankle markers (colored by last step type)
  - Real-time statistics overlay
  - Legend for step type colors

## Tips for Best Results

1. **Video Quality**: Use good lighting and clear view of the person's legs
2. **Camera Angle**: Side view or slight angle works best for gait analysis
3. **Walking Surface**: Flat surface with consistent background helps detection
4. **Frame Rate**: Higher FPS (30+) provides more accurate step detection
5. **Model Selection**: Use larger models for difficult videos

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLOv8 pose estimation model
- [Yankai Zhao](https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints) for the fine-tuned YOLO lower body model
- OpenCV for video processing and visualization

