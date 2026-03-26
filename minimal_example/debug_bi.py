from TeamIndex import evaluation as eva
import pandas as pd
from pathlib import Path

INDEX_CONFIG = "./toy_index.json"
DATA_PATH = Path("./uniform_toy_data.parquet")

table = pd.read_parquet(DATA_PATH)
index = eva.TeamIndex(INDEX_CONFIG)

print("\n--- DEBUG QUERY 6: TEAM SPLIT ---")

q_bi = "B < 30 and I < 30"
q_fh = "F < 30 and H < 30"
q_bifh = "B < 30 and I < 30 and F < 30 and H < 30"

ref_bi = set(table.query(q_bi).index)
ref_fh = set(table.query(q_fh).index)
ref_bifh = set(table.query(q_bifh).index)

res_bi = set(index.run_query(q_bi)[0])
res_fh = set(index.run_query(q_fh)[0])
res_bifh = set(index.run_query(q_bifh)[0])

print("Ref BI:", len(ref_bi))
print("Idx BI:", len(res_bi))
print("BI missing:", len(ref_bi - res_bi))
print("BI extra:", len(res_bi - ref_bi))

print("Ref FH:", len(ref_fh))
print("Idx FH:", len(res_fh))
print("FH missing:", len(ref_fh - res_fh))
print("FH extra:", len(res_fh - ref_fh))

print("Ref BI∩FH:", len(ref_bifh))
print("Idx BI∩FH:", len(res_bifh))
print("BI∩FH missing:", len(ref_bifh - res_bifh))
print("BI∩FH extra:", len(res_bifh - ref_bifh))

manual_intersection = res_bi & res_fh
print("Manual intersection from single-team results:", len(manual_intersection))
print("Manual missing:", len(ref_bifh - manual_intersection))
print("Manual extra:", len(manual_intersection - ref_bifh))

print("\n--- DEBUG QUERY 6: ORDER CHECK ---")

q1 = "B < 30 and I < 30 and F < 30 and H < 30"
q2 = "F < 30 and H < 30 and B < 30 and I < 30"

r1 = set(index.run_query(q1)[0])
r2 = set(index.run_query(q2)[0])

print("q1 size:", len(r1))
print("q2 size:", len(r2))
print("q1 == q2:", r1 == r2)

print("\n--- DEBUG B-I ONLY ---")

bi_queries = [
    "B < 10 and I < 10",
    "B < 19 and I < 19",
    "B < 25 and I < 25",
    "B < 30 and I < 30",
    "B < 40 and I < 20",
]

for q in bi_queries:
    ref = set(table.query(q).index)
    res = set(index.run_query(q)[0])
    print(q)
    print("  ref:", len(ref))
    print("  idx:", len(res))
    print("  missing:", len(ref - res))
    print("  extra:", len(res - ref))