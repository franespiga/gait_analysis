"""
Central output location for all gait analysis runs.

All analysis outputs (from app, CLI scripts, or notebooks) are stored under:
  analyses/
    APP/           # Streamlit app
      YYYYMMDD_HHMM/
    CLI/           # Scripts (infer_video, gait-analyze, etc.)
      YYYYMMDD_HHMM/
    OTHER/         # Notebooks or any other source
      YYYYMMDD_HHMM/

Each run writes only into its timestamped folder.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

ANALYSES_DIR = "analyses"
SOURCE_APP = "APP"
SOURCE_CLI = "CLI"
SOURCE_OTHER = "OTHER"


def get_analysis_run_dir(project_root: Path, source: str) -> Path:
    """
    Return the timestamped run directory for this analysis: analyses/<source>/YYYYMMDD_HHMM.
    Creates the directory and parent structure. Use this for all outputs of a single run.

    Args:
        project_root: Repository root (parent of src/, app/, etc.).
        source: One of SOURCE_APP, SOURCE_CLI, SOURCE_OTHER.

    Returns:
        Path to analyses/<source>/YYYYMMDD_HHMM (created).
    """
    if source not in (SOURCE_APP, SOURCE_CLI, SOURCE_OTHER):
        source = SOURCE_OTHER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = Path(project_root) / ANALYSES_DIR / source / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
