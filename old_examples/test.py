from TeamIndex import evaluation as eva
import pandas as pd
from example_paths import DATA_PATH, INDEX_CONFIG

OUTPUT_CSV = "benchmark_results.csv"
OUTPUT_META_CSV = "benchmark_metadata.csv"

table = pd.read_parquet(DATA_PATH)
index = eva.TeamIndex(INDEX_CONFIG)

def optimize(mopts):
    assert len(mopts) >= 1, "Empty result?"

    mopts[0][1]["is_expanded"] = True
    if mopts[0][1]["max_group_count"] > 128:
        mopts[0][1]["group_count"] = eva.po2_near_sqrt(mopts[0][1]["max_group_count"])

    limit = 128
    if mopts[0][1]["group_count"] < 16 and len(mopts) > 1:
        mopts[1][1]["is_expanded"] = True
        limit = 16

    for i in range(1, len(mopts)):
        if mopts[i][1]["max_group_count"] > limit:
            mopts[i][1]["group_count"] = min(
                eva.po2_near_sqrt(mopts[i][1]["max_group_count"]),
                limit
            )

    return mopts

queries = [
    "A < 10 and E < 10 and C < 10",
    "A < 15 and E < 15 and C < 15",
    "A < 19 and E < 19 and C < 19",
    "A < 25 and E < 25 and C < 25",
    "A < 30 and E < 30 and C < 30",
    "A < 38 and E < 38 and C < 19",
    "A < 40 and E < 20 and C < 10",
    "A < 50 and E < 20 and C < 10",
    "A < 60 and E < 20 and C < 10",
    "A < 70 and E < 15 and C < 10",

    "J < 10 and D < 10 and G < 10",
    "J < 15 and D < 15 and G < 15",
    "J < 19 and D < 19 and G < 19",
    "J < 25 and D < 25 and G < 25",
    "J < 30 and D < 30 and G < 30",
    "J < 38 and D < 38 and G < 19",
    "J < 40 and D < 20 and G < 10",
    "J < 50 and D < 20 and G < 10",
    "J < 60 and D < 20 and G < 10",
    "J < 70 and D < 15 and G < 10",

    "B < 10 and I < 10",
    "B < 19 and I < 19",
    "B < 25 and I < 25",
    "B < 30 and I < 30",
    "B < 40 and I < 20",
    "B < 50 and I < 20",
    "B < 60 and I < 30",
    "B < 70 and I < 30",

    "F < 10 and H < 10",
    "F < 19 and H < 19",
    "F < 25 and H < 25",
    "F < 30 and H < 30",
    "F < 40 and H < 20",
    "F < 50 and H < 20",
    "F < 60 and H < 30",
    "F < 70 and H < 30",

    "A < 19 and E < 19 and C < 19 and B < 19",
    "A < 25 and E < 25 and C < 25 and B < 25",
    "A < 38 and E < 38 and C < 19 and B < 38",
    "A < 38 and E < 38 and C < 19 and B < 38 and F < 40",
    "A < 60 and E < 20 and C < 10 and B < 50",
    "J < 19 and D < 19 and G < 19 and F < 19",
    "J < 20 and D < 20 and G < 20 and F < 20",
    "J < 30 and D < 30 and G < 15 and F < 30",
    "B < 19 and I < 19 and F < 19 and H < 19",
    "B < 25 and I < 25 and F < 25 and H < 25",
    "B < 30 and I < 30 and F < 30 and H < 30",
    "B < 40 and I < 20 and F < 40 and H < 20",
    "A < 20 and E < 20 and C < 20 and J < 20 and D < 20 and G < 20",
    "A < 30 and E < 15 and C < 10 and J < 30 and D < 15 and G < 10",
]

rows = []
rows_meta = []

def get_plan_runtime_s(stats):
    if hasattr(stats, "plan_construction_runtime"):
        return stats.plan_construction_runtime / 1_000_000_000
    return None

for qid, query in enumerate(queries, start=1):
    print(f"Running query {qid}/{len(queries)}")
    ref = set(table.query(query).index)

    res = index.run_query(query)
    result_ids = set(res[0])
    stats = res[1]

    mopts = index.prepare_optimization(query=query)

    for team_name, d in mopts:
        rows_meta.append({
            "query_id": qid,
            "query": query,
            "team": team_name,
            "union_cardinality": d["union_cardinality"],
            "selectivity": d["selectivity"],
            "group_count_before": int(d["group_count"]),
            "max_group_count": int(d["max_group_count"]),
            "netto_data_volume_KiB": d["netto_data_volume_KiB"],
            "io_volume_KiB": d["io_volume_KiB"],
            "is_included": d["is_included"],
            "is_expanded_before": d["is_expanded"],
        })

    opt_mopts = optimize(mopts)
    res_opt = index.run_query(query, manual_optimizations=opt_mopts)
    result_ids_opt = set(res_opt[0])
    stats_opt = res_opt[1]

    rows.append({
        "query_id": qid,
        "query": query,
        "optimized": False,
        "correct_subset": ref.issubset(result_ids),
        "missing_true_hits": len(ref - result_ids),
        "extra_hits": len(result_ids - ref),
        "result_size": len(result_ids),
        "ref_size": len(ref),
        "executor_runtime_s": stats.executor_runtime / 1_000_000_000,
        "plan_runtime_s": get_plan_runtime_s(stats),
    })

    rows.append({
        "query_id": qid,
        "query": query,
        "optimized": True,
        "correct_subset": ref.issubset(result_ids_opt),
        "missing_true_hits": len(ref - result_ids_opt),
        "extra_hits": len(result_ids_opt - ref),
        "result_size": len(result_ids_opt),
        "ref_size": len(ref),
        "executor_runtime_s": stats_opt.executor_runtime / 1_000_000_000,
        "plan_runtime_s": get_plan_runtime_s(stats_opt),
    })

df = pd.DataFrame(rows)
meta_df = pd.DataFrame(rows_meta)

df.to_csv(OUTPUT_CSV, index=False)
meta_df.to_csv(OUTPUT_META_CSV, index=False)

print(df)
print(meta_df)
print(f"\nSaved to {OUTPUT_CSV}")
print(f"Saved to {OUTPUT_META_CSV}")

# Speedup-Auswertung
default_df = df[df["optimized"] == False].copy()
opt_df = df[df["optimized"] == True].copy()

merged = default_df.merge(
    opt_df,
    on=["query_id", "query"],
    suffixes=("_default", "_opt")
)

merged["speedup"] = merged["executor_runtime_s_default"] / merged["executor_runtime_s_opt"]
merged["saved_time_s"] = merged["executor_runtime_s_default"] - merged["executor_runtime_s_opt"]
merged["saved_percent"] = (
    (merged["executor_runtime_s_default"] - merged["executor_runtime_s_opt"])
    / merged["executor_runtime_s_default"]
) * 100.0

merged.to_csv("benchmark_comparison.csv", index=False)

print("\nComparison:")
print(merged[[
    "query_id",
    "executor_runtime_s_default",
    "executor_runtime_s_opt",
    "speedup",
    "saved_percent",
    "correct_subset_default",
    "correct_subset_opt",
    "missing_true_hits_default",
    "missing_true_hits_opt"
]])

print("\nSaved to benchmark_comparison.csv")

# Query-6-Debug
debug_query = "B < 30 and I < 30 and F < 30 and H < 30"

print("\n--- DEBUG QUERY 6 ---")
ref_debug = set(table.query(debug_query).index)

res_debug = index.run_query(debug_query)
result_debug = set(res_debug[0])

mopts_debug = index.prepare_optimization(query=debug_query)
res_debug_opt = index.run_query(debug_query, manual_optimizations=optimize(mopts_debug))
result_debug_opt = set(res_debug_opt[0])

missing_default = ref_debug - result_debug
missing_opt = ref_debug - result_debug_opt

extra_default = result_debug - ref_debug
extra_opt = result_debug_opt - ref_debug

print("Referenzgröße:", len(ref_debug))
print("Indexgröße default:", len(result_debug))
print("Indexgröße optimized:", len(result_debug_opt))

print("Fehlende echte Treffer default:", len(missing_default))
print("Fehlende echte Treffer optimized:", len(missing_opt))

print("Zusätzliche Treffer default:", len(extra_default))
print("Zusätzliche Treffer optimized:", len(extra_opt))

print("Erste 20 fehlende IDs default:", sorted(list(missing_default))[:20])
print("Erste 20 fehlende IDs optimized:", sorted(list(missing_opt))[:20])
