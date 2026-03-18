# Deploying the Gait Analysis Streamlit App

This folder contains everything needed to run the app **locally with Docker** or on **Streamlit Community Cloud** (streamlit.io).

## Contents

| File | Purpose |
|------|--------|
| **README.md** | This guide |
| **config.yaml** | Model paths, inference settings, and deployment options (reference and env overrides) |
| **Dockerfile** | Image for running the app in Docker |
| **docker-compose.yml** | Run the app with optional volume mounts for `models/` and `analyses/` |
| **requirements-streamlit.txt** | Pip dependencies for the app (used by Docker and Streamlit Cloud) |
| **.streamlit/config.toml** | Optional Streamlit server and theme config |
| **.dockerignore** | Template to reduce Docker build context (optional: copy to repo root) |

---

## 1. Deploy locally with Docker

### Prerequisites

- Docker (and optionally Docker Compose)
- Repository cloned; custom models (if any) under `models/` at repo root

### Build and run with Docker Compose (recommended)

From the **repository root**:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

- App: **http://localhost:8501**
- `analyses/` and `models/` on the host are mounted so outputs and custom models are persisted.

### Build and run with Docker only

From the **repository root**:

```bash
docker build -f deploy/Dockerfile -t gait-analysis-app .
docker run -p 8501:8501 -v "$(pwd)/analyses:/app/analyses" -v "$(pwd)/models:/app/models:ro" gait-analysis-app
```

Then open **http://localhost:8501**.

### Optional: smaller build context

To speed up builds and shrink context, copy the ignore list to the repo root:

```bash
cp deploy/.dockerignore .dockerignore
```

Then run the same `docker build` as above.

### Custom models in Docker

- Place weights on the host under `models/<name>/weights/best.pt`.
- Use the `docker-compose.yml` volume: `./models:/app/models:ro`.
- The app will list those models in the dropdown.

### Configuration

- **config.yaml** in `deploy/` documents model directory, default model, device, and output paths.
- Override at runtime with environment variables (see **config.yaml** comments), e.g. in `docker-compose.yml`:

  ```yaml
  environment:
    - GAIT_DEVICE=cpu
    - GAIT_MODELS_DIR=/app/models
  ```

---

## 2. Deploy on Streamlit Community Cloud (streamlit.io)

### Prerequisites

- GitHub (or GitLab) repo with this project
- [Streamlit Community Cloud](https://share.streamlit.io) account

### Steps

1. **Push the repo** to GitHub (or GitLab).

2. **Requirements file**  
   Streamlit Cloud looks for a requirements file in the **repository root** or in the **same directory as the app** (`app/`). Do one of the following:
   - **Option A:** Copy the deploy requirements to the root:
     ```bash
     cp deploy/requirements-streamlit.txt requirements.txt
     git add requirements.txt && git commit -m "Add requirements for Streamlit Cloud" && git push
     ```
   - **Option B:** Copy into `app/` so the app directory is self-contained:
     ```bash
     cp deploy/requirements-streamlit.txt app/requirements.txt
     git add app/requirements.txt && git commit -m "Add requirements for Streamlit Cloud" && git push
     ```

3. **Connect the app on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io), sign in, and click **New app**.
   - **Repository:** `your-username/gait_analysis` (or your repo URL).
   - **Branch:** e.g. `main`.
   - **Main file path:** `app/gait_streamlit.py`.
   - **App URL:** choose a subdomain (e.g. `gait-analysis`).

4. **Deploy**  
   Streamlit will install from `requirements.txt` and run:
   ```text
   streamlit run app/gait_streamlit.py
   ```

### Limitations on Streamlit Cloud

- **No GPU** on the free tier; set device to **CPU** (e.g. via `GAIT_DEVICE=cpu` in Cloud secrets if the app reads it, or use a small default model).
- **Memory** is limited (~1 GB on free tier). Prefer `yolov8n-pose.pt`; larger models or many custom models may cause restarts.
- **No persistent disk** for `models/` or `analyses/`. Custom weights are not available unless you bundle them in the repo (not ideal for large files) or download them at startup from a URL (you’d need to add that logic).
- **Webcam** may not work in the browser for the deployed app; **uploaded video** is the main use case.

### Optional: Streamlit config and secrets

- To use the same server/theme settings as in **deploy/.streamlit/config.toml**, copy that file to the repo root:
  ```bash
  mkdir -p .streamlit
  cp deploy/.streamlit/config.toml .streamlit/config.toml
  git add .streamlit && git commit -m "Add Streamlit config" && git push
  ```
- In the Cloud dashboard you can set **Secrets** (e.g. `GAIT_DEVICE: cpu`) if the app is updated to read them.

---

## 3. Configuration reference (deploy/config.yaml)

- **models.directory** – Directory under project root for custom models (`models/*/weights/best.pt`).
- **models.default** – Default model name or path when no custom models exist.
- **inference.confidence_threshold** – Detection/keypoint confidence.
- **inference.device** – `auto`, `cpu`, or `cuda` (use `cpu` for Cloud).
- **output.analyses_root** – Root for run outputs (`analyses/APP/YYYYMMDD_HHMM`, etc.).
- **streamlit.port / address** – Port and bind address for the Streamlit server.

Environment overrides (for Docker or Cloud) are documented at the bottom of **config.yaml**.

---

## 4. Quick reference

| Target | Command / action |
|--------|-------------------|
| **Docker Compose (local)** | `docker compose -f deploy/docker-compose.yml up --build` |
| **Docker only (local)** | `docker build -f deploy/Dockerfile -t gait-analysis-app .` then `docker run -p 8501:8501 ...` |
| **Streamlit Cloud** | Push repo, add `requirements.txt` (from `deploy/requirements-streamlit.txt`), connect app at share.streamlit.io, main file `app/gait_streamlit.py` |
| **Config** | Edit `deploy/config.yaml`; override with env vars in Docker/Cloud as needed |
