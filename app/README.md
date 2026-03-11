# Gait Analysis Web App

Streamlit-based web UI for real-time and video gait analysis. Run pose estimation on uploaded video or webcam and see keypoints (skeleton + points) with a live metrics legend.

## Requirements

- Python 3.10+
- Project dependencies with the **yolo** and **app** extras (includes Ultralytics and Streamlit)

## Installation

From the **project root**:

```bash
poetry install -E yolo -E app
```

## Running the app

From the **project root**:

```bash
poetry run streamlit run app/gait_streamlit.py
```

Streamlit will open in your browser (default: http://localhost:8501).

## How to use

### 1. Input (sidebar)

- **Upload video**: Choose a video file (e.g. MP4, AVI, MOV, MKV) via “Choose a video file” and **Browse**.
- **Webcam**: Select **Webcam** as source; if you have multiple cameras, pick one from the **Webcam** dropdown.

### 2. Model (sidebar)

- **Pose model**: Select a pose model from the dropdown.
  - **Built-in**: YOLO COCO (nano/small/medium) or YOLO Lower Body (heel/toe) if available.
  - **Custom**: Any model under `models/` is listed automatically. Each top-level folder (e.g. `models/foot_pose/`, `models/yolo_lower/`) that contains `weights/best.pt` appears as an option (e.g. `foot_pose (models/foot_pose)`).
- **Load model**: After choosing a model, click **Load model**. The app loads the selected weights (and may try YOLOv8 then YOLO11 for built-in names). Wait for the “Model loaded: …” message before starting analysis.

### 3. Start analysis

- Click **Start Gait Analysis** in the sidebar.
- For **Upload video**: playback runs with pose overlay and metrics; click **Stop** to end.
- For **Webcam**: live feed runs until you click **Stop**.

### 4. On-screen legend (top-right)

During analysis the overlay shows:

- **Frames**: Processed frame count  
- **Steps**: Total steps detected (left + right)  
- **Steps/min**: Cadence (steps per minute)  
- **Avg step length**: Average step length in pixels (when at least two steps are available)

## Custom models

Place your trained YOLO pose weights so that:

```
models/
  <your_model_name>/
    weights/
      best.pt
```

The app discovers all such folders and adds them to the **Pose model** dropdown. Select your model, click **Load model**, then **Start Gait Analysis**.

## Troubleshooting

- **“Please choose a model and click Load model”**: Select a model and click **Load model** before **Start Gait Analysis**.
- **“No webcams detected”**: Check camera permissions and that no other app is using the camera.
- **Model load fails**: Ensure the selected model path exists (e.g. `models/yolo_lower/weights/best.pt`). For built-in names (e.g. `yolov8n-pose.pt`), the app will try the corresponding YOLO11 name if YOLOv8 fails.
