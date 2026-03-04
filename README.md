# Gait Analysis

A computer vision project for analyzing walking patterns using pose estimation. This tool processes video footage of people walking and classifies their gait as correct (heel-toe walking), incorrect (toe walking), or suboptimal (flat foot).

## Features

- **YOLO-Based Pose Estimation**: Support for YOLO pose detection models:
  - **YOLO COCO**: Standard YOLOv8-pose with 17 keypoints (ankle only)
  - **YOLO Lower Body**: Fine-tuned model with 10 keypoints including heel and toe
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

- **Training**: Train a YOLO-family keypoint model (e.g. YOLOv8 pose) with **foot keypoints** (COCO-WholeBody style: L_HEEL, L_BIG_TOE, L_SMALL_TOE, R_HEEL, R_BIG_TOE, R_SMALL_TOE). Scripts: dataset preparation, training with configurable hyperparameters, evaluation, optional ONNX export.
- **Inference**: Person detection + pose → **temporal smoothing** (Savitzky-Golay) → **gait event detection** (Initial Contact IC, Toe-Off TO) → **contact classification** (heel-strike / toe-walk / flat) using foot pitch and contact timing.
- **Outputs**: Per-event CSV, per-step CSV, summary JSON (cadence, step/stride time, stance, swing, step/stride length, speed, symmetry, variability). Optional annotated video.
- **Gait metrics**: Cadence (steps/min), step time & stride time (mean, std), stance time, swing time, duty factor, step length & stride length (with calibration), walking speed (m/s), symmetry indices (L vs R), variability (CV%).

**CLI commands** (from project root):

```bash
python scripts/prepare_dataset.py [--dataset-dir ...] [--coco-annot ...]
python scripts/train_pose.py [--config config/training.yaml] [--data ...] [--resume] [--export-onnx]
# Or Ultralytics-aligned training (device/resume/batch):
python scripts/train_pose_ultralytics.py [--config config/training.yaml] [--data ...] [--resume] [--export-onnx] [--device 0] [--batch -1]
python scripts/infer_video.py path/to/video.mp4 [--config config/inference.yaml] [--output-dir results]
```

Configs: `config/training.yaml`, `config/inference.yaml`, `config/keypoint_schema.yaml`. Tech stack: Python 3.10+, Ultralytics, OpenCV, NumPy/SciPy, Pandas, YAML.

#### Ultralytics-aligned training (`train_pose_ultralytics.py`)

`train_pose_ultralytics.py` uses the same config and dataset as `train_pose.py` but follows [Ultralytics Train mode](https://docs.ultralytics.com/modes/train/) for device, resume, and batch:

- **`--device`**: `0` (single GPU), `0,1` (multi-GPU), `-1` (idle GPU), `mps` (Apple Silicon), `cpu`, or `auto` (omit; default).
- **`--resume`**: Loads `{project}/{name}/weights/last.pt` and calls `model.train(resume=True)`. Use after an interrupted or previous run.
- **`--batch`**: Integer (e.g. `16`), `-1` (auto batch size, e.g. 60% GPU memory), or float (e.g. `0.7` for utilization fraction). Config keys `batch` and `device` are supported in `config/training.yaml`.

## Installation

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation) package manager

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gait_analysis
   ```

2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Usage

### Command Line Interface

There are two analysis scripts available:

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
gait-analyze path/to/video.mp4 --output results/analysis.json --show
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
| `-o, --output-dir` | Base directory for results | results |
| `--show` | Show visualization window during analysis | False |
| `--no-video` | Don't save annotated output video | False |
| `--model` | YOLO pose model to use | yolov8n-pose.pt |
| `--device` | Inference device (auto, cpu, cuda, cuda:0) | auto |

#### Multi-Backend Script (`gait-analyze-multi`)

| Option | Description | Default |
|--------|-------------|---------|
| `video` | Path to input video file | Required |
| `-o, --output-dir` | Base directory for results | results |
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
```

## Project Structure

```
gait_analysis/
├── pyproject.toml                  # Poetry configuration and dependencies
├── README.md                       # This file
├── src/
│   └── gait_analysis/
│       ├── __init__.py             # Package exports
│       ├── keypoint_config.py      # Keypoint configurations for each model
│       ├── base_detector.py        # Abstract detector interface
│       ├── yolo_detector.py        # YOLO COCO and Lower Body detectors
│       ├── detector.py             # Legacy YOLO COCO detector
│       ├── analyzer.py             # Basic gait analysis logic
│       ├── angle_analyzer.py       # Enhanced analysis with angles
│       ├── heel_toe_analyzer.py    # Analyzer for heel/toe keypoints
│       ├── visualizer.py           # Basic visualization utilities
│       ├── advanced_visualizer.py  # Enhanced visualization with angles
│       ├── main.py                 # Basic CLI entry point
│       ├── advanced_main.py        # Advanced CLI entry point
│       └── multi_backend_main.py   # Multi-backend CLI entry point
├── tests/
│   ├── __init__.py
│   ├── test_analyzer.py            # Basic analyzer tests
│   ├── test_angle_analyzer.py      # Enhanced analyzer tests
│   └── test_multi_backend.py       # Multi-backend tests
├── notebooks/
│   └── exploration.ipynb           # Exploration notebook
└── data/
    └── .gitkeep                    # Sample videos directory
```

## Algorithm Details

### Keypoint Detection

The system supports multiple pose estimation backends:

#### YOLO COCO (17 keypoints)
Standard YOLOv8-pose with COCO keypoints. Only provides ankle positions, requiring trajectory-based step classification.

#### YOLO Lower Body (10 keypoints) ⭐ RECOMMENDED
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

### Foot keypoint dataset (CMU) – alternative downloads

The [CMU Human Foot Keypoint Dataset](https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/) (6 keypoints: heel, big toe, small toe per foot) is used for training foot keypoint models. The official CMU download URLs are often unavailable. Use the following alternatives.

**1) Download annotations via script (recommended)**

The project script fetches annotation JSONs from a GitHub mirror:

```bash
# Download train + val annotation JSONs to data/foot_cmu/annotations/
python scripts/download_foot_dataset.py --output-dir data/foot_cmu/annotations

# Or only print URLs for manual download
python scripts/download_foot_dataset.py --print-urls
```

**2) Annotation sources**

| Source | Format | Notes |
|--------|--------|--------|
| **GitHub mirror** | JSON | [Eva20150932/coco-foot-and-leg](https://github.com/Eva20150932/coco-foot-and-leg) – repo contains `person_keypoints_train2017_foot_v1.json` and `person_keypoints_val2017_foot_v1.json` (raw files or clone). |
| **Official CMU** | ZIP | `http://posefs1.perception.cs.cmu.edu/OpenPose/datasets/foot/` – often down. |

**3) COCO 2017 images (required)**

Annotations refer to COCO 2017 images. Download from one of:

- **Official:** [Train (18GB)](http://images.cocodataset.org/zips/train2017.zip), [Val (1GB)](http://images.cocodataset.org/zips/val2017.zip)
- **Academic Torrents:** [COCO 2017](https://academictorrents.com/details/74dec1dd21ae4994dfd9069f9cb0443eb960c962)

Place images so your dataset has `images/train/` and `images/val/` (e.g. `train2017/` and `val2017/` contents). Then run `scripts/prepare_dataset.py` with `--coco-annot` pointing to the train JSON and `--images-dir` to the folder containing the image subfolders. Note: CMU keypoint order (L big toe, L small toe, L heel, R big toe, R small toe, R heel) may require a reorder step to match `config/keypoint_schema.yaml` (L_HEEL, L_BIG_TOE, L_SMALL_TOE, R_*); see the dataset docs or add a CMU-specific conversion if needed.

### Streamlit app

A web UI for gait analysis (upload video or use webcam) is in `app/`. It shows keypoints from the waist down and a live legend with frame count, step count, steps per minute, and average step length (pixels).

1. Install the app extra (includes Streamlit):
   ```bash
   poetry install -E yolo -E app
   ```

2. From the project root, run:
   ```bash
   poetry run streamlit run app/gait_streamlit.py
   ```

3. In the sidebar: choose **Upload video** (then use “Browse” to select a file) or **Webcam** (if multiple cameras exist, pick one). Click **Start Gait Analysis** to run. Use **Stop** to end webcam or video playback.

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

