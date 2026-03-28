from TeamIndex import evaluation as eva
import pandas as pd
from pathlib import Path

INDEX_CONFIG = "./toy_index.json"
DATA_PATH = Path("./uniform_toy_data.parquet")

table = pd.read_parquet(DATA_PATH)
index = eva.TeamIndex(INDEX_CONFIG)

queries = [
    "B < 30",
    "I < 30",
    "F < 30",
    "H < 30",
    "B < 20 and I < 20",
    "B < 40 and I < 40",
    "B < 60 and I < 60",
    "F < 20 and H < 20",
    "F < 40 and H < 40",
    "F < 60 and H < 60",
]



for q in queries:
    ref = set(table.query(q).index)
    res = set(index.run_query(q)[0])

    print("\n", q)
    print("ref:", ref)
    print("idx:", res)
    print("missing:", len(ref - res))
    print("extra:", len(res - ref))