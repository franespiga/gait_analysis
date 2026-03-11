## Installation

### Prerequisites

- **Python**: 3.10 or newer
- **Poetry**: recommended package manager for this project  
  See the official docs: `https://python-poetry.org/docs/#installation`

### Setup

From a terminal:

```bash
git clone <repository-url>
cd gait_analysis
```

Install dependencies with Poetry:

```bash
poetry install
```

Activate the virtual environment:

```bash
poetry shell
```

At this point you can:

- Run the gait analysis CLIs (`gait-analyze`, `gait-analyze-advanced`, `gait-analyze-multi`)
- Use the training and inference scripts under `scripts/`
- Launch the optional Streamlit app in `app/` (see project README for details)

