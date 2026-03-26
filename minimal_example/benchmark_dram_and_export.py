from TeamIndex import evaluation as eva
from pathlib import Path
import copy

INDEX_CONFIG = "./toy_index.json"
OUT_DIR = Path("./stats_and_configs")
OUT_DIR.mkdir(exist_ok=True)

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
    "A < 19 and E < 19 and C < 19 and B < 19",
    "A < 38 and E < 38 and C < 19 and B < 38 and F < 40",
    "J < 20 and D < 20 and G < 20 and F < 20",
    "B < 30 and I < 30 and F < 30 and H < 30",
]

for qid, query in enumerate(queries, start=1):
    print(f"\n=== Query {qid} ===")
    print(query)

    config = eva.get_new_default_runtime_config()
    config["backend"] = "dram"
    config["verbose_runtime"] = False
    config["return_result"] = True
    config["StorageConfig"]["queue_pair_count"] = 3

    config["print_execution_plan"] = str(OUT_DIR / f"q{qid:02d}_default_execution_plan.dot")
    config["print_task_stats"] = str(OUT_DIR / f"q{qid:02d}_default_task_stats.json")
    config["print_result_stats"] = str(OUT_DIR / f"q{qid:02d}_default_result_stats.json")
    config["task_graph_path"] = str(OUT_DIR / f"q{qid:02d}_default_task_graph.json")

    result_default = index.run_query(query, config=config)
    print("Default result size:", len(result_default[0]))

    default_plan_path = OUT_DIR / f"q{qid:02d}_default_exported_plan.json"
    index.run_query(query, config=config, dry_run=default_plan_path)
    print("Default plan written to:", default_plan_path)

    mopts = index.prepare_optimization(query=query)
    opt_mopts = optimize(copy.deepcopy(mopts))

    config_opt = eva.get_new_default_runtime_config()
    config_opt["backend"] = "dram"
    config_opt["verbose_runtime"] = False
    config_opt["return_result"] = True
    config_opt["StorageConfig"]["queue_pair_count"] = 3

    config_opt["print_execution_plan"] = str(OUT_DIR / f"q{qid:02d}_optimized_execution_plan.dot")
    config_opt["print_task_stats"] = str(OUT_DIR / f"q{qid:02d}_optimized_task_stats.json")
    config_opt["print_result_stats"] = str(OUT_DIR / f"q{qid:02d}_optimized_result_stats.json")
    config_opt["task_graph_path"] = str(OUT_DIR / f"q{qid:02d}_optimized_task_graph.json")

    result_opt = index.run_query(query, config=config_opt, manual_optimizations=opt_mopts)
    print("Optimized result size:", len(result_opt[0]))

    opt_plan_path = OUT_DIR / f"q{qid:02d}_optimized_exported_plan.json"
    index.run_query(query, config=config_opt, manual_optimizations=opt_mopts, dry_run=opt_plan_path)
    print("Optimized plan written to:", opt_plan_path)

print("\nDone.")
print("Standalone example:")
print("teamindexstandalone stats_and_configs/q01_default_exported_plan.json")