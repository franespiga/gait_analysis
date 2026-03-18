# Gait Analysis Streamlit app - Docker image
# Build from repository root: docker build -f deploy/Dockerfile -t gait-analysis-app .
# Run: docker run -p 8501:8501 gait-analysis-app

FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV (optional, headless works without)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# App dependencies (mirrors pyproject.toml app + yolo extras)
COPY deploy/requirements-streamlit.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy package, app, config, and deploy config reference (from repo root)
COPY src/ /app/src/
COPY app/ /app/app/
COPY config/ /app/config/
COPY deploy/config.yaml /app/deploy/config.yaml
# Optional: create analyses and models dirs so app can write
RUN mkdir -p /app/analyses/APP /app/analyses/CLI /app/analyses/OTHER /app/models

# Streamlit runs on 8501; run from project root so _project_root = /app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
EXPOSE 8501

# Run the app (working directory = /app)
CMD ["streamlit", "run", "app/gait_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
