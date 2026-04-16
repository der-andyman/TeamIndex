from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SCRATCH_ROOT = Path("/media/scratch/duman/teamindex")

SOURCE_INDEX_CONFIG = BASE_DIR / "toy_index.json"
SCRATCH_INDEX_CONFIG = BASE_DIR / "toy_index_scratch.json"
INDEX_CONFIG = SCRATCH_INDEX_CONFIG

DATA_PATH = SCRATCH_ROOT / "uniform_toy_data.parquet"
SCRATCH_INDEX_DATA_DIR = SCRATCH_ROOT / "index_data"

# Backward-compatible alias for scripts that create the scratch index.
INDEX_DATA_DIR = SCRATCH_INDEX_DATA_DIR
