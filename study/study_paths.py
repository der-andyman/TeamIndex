from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/media/data/duman/teamindex")

SOURCE_INDEX_CONFIG = BASE_DIR / "toy_index.json"
INDEX_CONFIG = BASE_DIR / "study_index_data.json"
INDEX_DATA_DIR = DATA_ROOT / "index_data"
DATA_PATH = DATA_ROOT / "uniform_toy_data.parquet"

TEAM_BENCH_DATA_ROOT = DATA_ROOT / "team_bench"
TEAM_BENCH_RESULTS_ROOT = BASE_DIR / "team_bench_results"
TEAM_BENCH_EXPERIMENTS_DIR = BASE_DIR / "experiments"
