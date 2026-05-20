from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/media/data/duman/teamindex")

SOURCE_INDEX_CONFIG = BASE_DIR / "toy_index.json"
DATA_INDEX_CONFIG = BASE_DIR / "toy_index_data.json"
LOCAL_INDEX_CONFIG = BASE_DIR / "toy_index_local.json"
SCRATCH_INDEX_CONFIG = BASE_DIR / "toy_index_scratch.json"
DATA_INDEX_DATA_DIR = DATA_ROOT / "index_data"
LOCAL_INDEX_DATA_DIR = BASE_DIR / "index_data"
SCRATCH_INDEX_DATA_DIR = Path("/media/scratch/duman/teamindex/index_data")

DATA_PATH = DATA_ROOT / "uniform_toy_data.parquet"

# The large benchmark artifacts live on the dedicated 1.9 TB SSD.
INDEX_CONFIG = DATA_INDEX_CONFIG
INDEX_DATA_DIR = DATA_INDEX_DATA_DIR

# Backward-compatible aliases for older scripts/imports.
LOCAL_ROOT = BASE_DIR
SCRATCH_ROOT = DATA_ROOT
