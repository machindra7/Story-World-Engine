# Story World Engine

Multi-chapter fiction studio with canon tracking, knowledge graph visualization, and book exports.

## Setup & Run

```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py