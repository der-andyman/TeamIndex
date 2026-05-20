from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/media/data/duman/teamindex")

SOURCE_INDEX_CONFIG = BASE_DIR / "toy_index.json"
INDEX_CONFIG = BASE_DIR / "study_index_data.json"
INDEX_DATA_DIR = DATA_ROOT / "index_data"
DATA_PATH = DATA_ROOT / "uniform_toy_data.parquet"
