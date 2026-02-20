# Gait Analysis

A computer vision project for analyzing walking patterns using pose estimation. This tool processes video footage of people walking and classifies their gait as correct (heel-toe walking), incorrect (toe walking), or suboptimal (flat foot).

## Features

- **Multi-Backend Pose Estimation**: Support for multiple pose detection models:
  - **YOLO COCO**: Standard YOLOv8-pose with 17 keypoints (ankle only)
  - **YOLO Lower Body**: Fine-tuned model with 10 keypoints including heel and toe
  - **OpenPose**: CMU OpenPose with 25+ body keypoints including detailed foot keypoints
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
gait-analyze-multi path/to/video.mp4 --backend yolo_lower --model path/to/best.pt
```

The YOLO Lower Body model provides 10 keypoints including heel and toe, enabling accurate heel-strike vs toe-walking detection. Download the model from [Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints](https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints).

**Using OpenPose (detailed foot keypoints):**
```bash
gait-analyze-multi path/to/video.mp4 --backend openpose --openpose-path C:/path/to/openpose
```

OpenPose provides 25 body keypoints plus detailed foot keypoints (heels, big toes, small toes). See [OpenPose installation](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md).

With visualization and custom output:
```bash
gait-analyze-multi path/to/video.mp4 --backend yolo_lower --model best.pt --show --output-dir results
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
| `--openpose-path` | Path to OpenPose installation | None |

**Available Backends:**

| Backend | Keypoints | Heel/Toe | Description |
|---------|-----------|----------|-------------|
| `yolo_coco` | 17 | ❌ | Standard YOLO COCO format (ankle only) |
| `yolo_lower` | 10 | ✅ | Fine-tuned for lower body with heel/toe |
| `openpose` | 25+ | ✅ | CMU OpenPose with detailed foot keypoints |
| `alphapose` | 136 | ✅ | HALPE-136 whole-body with detailed feet |
| `alphapose_body` | 26 | ✅ | HALPE-26 body-focused with feet |

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
│       ├── openpose_detector.py    # OpenPose detector wrapper
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

#### OpenPose (25+ keypoints)
CMU OpenPose from [CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose):
- Hips: 9 (right), 12 (left)
- Knees: 10 (right), 13 (left)
- Ankles: 11 (right), 14 (left)
- Heels: 21 (left), 24 (right)
- Big Toe: 19 (left), 22 (right)
- Small Toe: 20 (left), 23 (right)

See [keypoint documentation](https://chingswy.github.io/easymocap-public-doc/database/2_keypoints.html) for details.

### Step Detection

Steps are detected by tracking foot phases:
1. **Swing Phase**: Foot is in the air (high vertical velocity)
2. **Stance Phase**: Foot is on the ground (low vertical velocity)

A step is registered when transitioning from swing to stance.

### Step Classification

**With Heel/Toe Keypoints (YOLO Lower Body, OpenPose):**

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

Download from [GitHub repository](https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints):
```bash
# Download best.pt from the repository
wget https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints/raw/main/best.pt
```

Use with:
```bash
gait-analyze-multi video.mp4 --backend yolo_lower --model best.pt
```

### OpenPose Installation

OpenPose must be installed separately. Follow the [installation guide](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md).

After installation, set the environment variable or use the `--openpose-path` flag:
```bash
# Set environment variable
set OPENPOSE_PATH=C:\path\to\openpose

# Or use flag
gait-analyze-multi video.mp4 --backend openpose --openpose-path C:\path\to\openpose
```

### AlphaPose Installation

AlphaPose provides HALPE whole-body keypoints (136 points) including detailed foot keypoints (heel, big toe, small toe). This is the recommended backend for accurate gait analysis.

#### Option 1: Use Pre-downloaded Models (Recommended)

The models are pre-downloaded in `models/AlphaPose/`. AlphaPose still requires the Python package:

```bash
# Install AlphaPose from source (required for model loading)
git clone https://github.com/MVIG-SJTU/AlphaPose.git
cd AlphaPose
pip install -e .
```

The backend will automatically find models in:
- `models/AlphaPose/pretrained_models/halpe136_fast50_regression_256x192.pth` (HALPE-136, whole-body)
- `models/AlphaPose/pretrained_models/halpe26_fast_res50_256x192.pth` (HALPE-26, body only)
- `models/AlphaPose/detector/yolo/data/yolov3-spp.weights` (person detector)

#### Option 2: Set Environment Variable

```bash
# Point to your own AlphaPose installation
set ALPHAPOSE_DIR=C:\path\to\AlphaPose

# Run with AlphaPose
gait-analyze-multi video.mp4 --backend alphapose
```

#### Option 3: Full Installation (Linux/WSL2)

For the full AlphaPose experience with all features:

```bash
# Create conda environment
conda create -n alphapose python=3.7 -y
conda activate alphapose

# Install PyTorch
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Clone and build
git clone https://github.com/MVIG-SJTU/AlphaPose.git
cd AlphaPose
pip install cython
python setup.py build develop

# Download models from Model Zoo
# https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
```

#### AlphaPose Keypoints

| Model | Keypoints | Feet | Use Case |
|-------|-----------|------|----------|
| HALPE-136 | 136 | ✅ Heel, big toe, small toe | Full body + hands + face + feet |
| HALPE-26 | 26 | ✅ Heel, big toe, small toe | Body + feet (faster) |

**Note**: If AlphaPose is not properly installed, the backend falls back to YOLOv8-pose which does NOT have foot keypoints.

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

# Test only OpenPose backend
poetry run pytest tests/test_compatibility.py -v -k "openpose"

# Test multiple specific backends
poetry run pytest tests/test_compatibility.py -v -k "yolov8 or alphapose"
```

#### Running with CLI (More Control)

```bash
# Run specific backends via CLI
python -m tests.test_compatibility --backends yolov8 yolo_lower

# Run with specific video and backends
python -m tests.test_compatibility --video data/test/walk.mp4 --backends yolov8 openpose

# Custom output directory
python -m tests.test_compatibility --output-dir checks/my_test
```

#### Specifying Model Paths

For backends that require custom model files (like `yolo_lower`):

```bash
# Via pytest command line
poetry run pytest tests/test_compatibility.py -v --yolo-lower-model=models/best.pt

# Via environment variable (Windows CMD)
set GAIT_YOLO_LOWER_MODEL=models/best.pt
poetry run pytest tests/test_compatibility.py -v

# Via environment variable (PowerShell)
$env:GAIT_YOLO_LOWER_MODEL="models/best.pt"
poetry run pytest tests/test_compatibility.py -v

# Via CLI with model paths
python -m tests.test_compatibility --model-path yolo_lower=models/best.pt
python -m tests.test_compatibility --model-path yolo_lower=models/best.pt --model-path openpose=C:/openpose
```

#### Environment Variables for Model Paths

| Variable | Backend | Description |
|----------|---------|-------------|
| `GAIT_YOLO_LOWER_MODEL` | `yolo_lower` | Path to YOLO lower body model (`best.pt`) |
| `GAIT_YOLOV8_MODEL` | `yolov8`, `yolo_coco` | Path to YOLOv8 pose model |
| `OPENPOSE_PATH` | `openpose` | Path to OpenPose installation directory |
| `GAIT_POCKETPOSE_MODEL` | `pocketpose` | PocketPose model name |
| `GAIT_SDPOSE_MODEL` | `sdpose` | SDPose HuggingFace model name |
| `GAIT_ALPHAPOSE_MODEL` | `alphapose` | AlphaPose model path |

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
├── openpose/
│   └── ...
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
- [CMU Perceptual Computing Lab](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for OpenPose
- [Yankai Zhao](https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints) for the fine-tuned YOLO lower body model
- OpenCV for video processing and visualization

