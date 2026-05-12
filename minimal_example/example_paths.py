from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SCRATCH_ROOT = Path("/media/scratch/duman/teamindex")

SOURCE_INDEX_CONFIG = BASE_DIR / "toy_index.json"
LOCAL_INDEX_CONFIG = BASE_DIR / "toy_index_local.json"
SCRATCH_INDEX_CONFIG = BASE_DIR / "toy_index_scratch.json"
LOCAL_INDEX_DATA_DIR = BASE_DIR / "index_data"
SCRATCH_INDEX_DATA_DIR = SCRATCH_ROOT / "index_data"

DATA_PATH = SCRATCH_ROOT / "uniform_toy_data.parquet"

# The index is always expected on the local SSD-backed project directory.
# Only the large Parquet data file remains on scratch for now.
INDEX_CONFIG = LOCAL_INDEX_CONFIG
INDEX_DATA_DIR = LOCAL_INDEX_DATA_DIR

# Backward-compatible aliases for older scripts/imports.
LOCAL_ROOT = BASE_DIR
