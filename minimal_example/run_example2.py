from TeamIndex import evaluation as eva

from pathlib import Path

##################################################
## Load existing index (run other example to create it first!)
##################################################

index = eva.TeamIndex("./toy_index.json")

# create subfolder for results etc.
subfolder = Path("./stats_and_configs/")
if not subfolder.exists():
    subfolder.mkdir(exist_ok=True)

##################################################
## Configure execution options
##################################################
# We can change several aspects of the execution via this dictionary:
execution_configuration = eva.get_new_default_runtime_config()

# We can use a DRAM-based backend. Here, we will first load all indices to main memory, before starting processing.
# May take longer on first start for large indices than just running the query (which only load parts of the index) on demand.
# This is primarily used to avoid (most) storage I/O overhead when measuring index performance.
execution_configuration["backend"] = "dram"

## Set the number of worker threads to use
#execution_configuration["worker_count"] = 8  # adjust or use default value determined by get_new_default_runtime_config

## Without this flag, no information will be printed. Useful when running many benchmarks
execution_configuration["verbose_runtime"] = True  # include more details in the plan

## Enable printing of the execution plan. Needs to be rendered to pdf with a graph visualization tool, e.g., Graphviz:
## Will write the plan to the specified file path. Name will be adjust to include timestamp!
## Linux:
# > dot -Tpdf execution_plan-2026_01_14-18_45_46.dot > plan.pdf
# > okular plan.pdf
## Web-based viewer:
# https://dreampuf.github.io/GraphvizOnline/
execution_configuration["print_execution_plan"] = subfolder.joinpath("execution_plan.dot")  

## Enable printing of detailed per-task runtime statistics after query execution. Can impact runtime performance!
execution_configuration["print_task_stats"] = subfolder.joinpath("task_stats.json")  # print detailed per-task runtime statistics after query execution
 
## Enable printing of detailed stats after query execution.
## Contains information such as result size and execution time. Useful for benchmarking.
execution_configuration["print_result_stats"] = subfolder.joinpath("result_stats.json")
# ensure that the actual result is returned.
# We may skip this if we dont want to measure the time it takes to convert the roaring bitmap containing the result to a python list
execution_configuration["return_result"] = True

# This function prints the execution times of Taskflow tasks on a time axis for each worker thread
# This allows to visualize idle times and parallelism.
# The result is a json file that can be rendered (for small graphs) using a web-based viewer:
# https://taskflow.github.io/tfprof/
## THIS FILE WILL BE OVERWRITTEN ON EACH RUN!
execution_configuration["task_graph_path"] = subfolder.joinpath("task_graph.json")


# There are more options available, such as for liburing. Note that not all options/combinations are tested.
# See .code/python/TeamIndex/evaluation.py for details.
# The runtime is defined in
# code/cpp/include/runtime/runtime.hpp
# code/cpp/src/runtime/runtime.cpp


##################################################
## Run the query with the specified configuration
##################################################

query = "A < 19 and E < 19 and C < 19 and B < 19" # restricts two indices, so we need an intersection
print("Running query:", query)
index_result, runtime_stats, request_info, global_info = index.run_query(query, config=execution_configuration)

print("Done. Index result cardinality:", len(index_result))
print("First 10 result tuple IDs:", index_result[:10])


# We also support a non-python, "standalone" execution mode, where the query plan is fully optimized and prepared in python.
# The plan is then exported to a file and can be executed using a compiled C++ binary.
# This avoids python overhead during execution and allows for more accurate performance measurements.
# To use this mode, use the dry_run option, which will then return the full query plan or directly write to a specified file.
# E.g.: .run_query(..., dry_run="exported_plan.json")
# The resulting plan explicitly defines every necessary I/O request so it may become large (depending on the index complexity)!
# This runtime is defined here:
# code/cpp/src/runtime/standalone_runtime.cpp

print("Exporting Plan:")
plan_path = subfolder.joinpath("exported_plan.json")
index.run_query(query, config=execution_configuration, dry_run=plan_path)
print("Plan written to:", plan_path)

print("To execute the exported plan, use the following command:")
print()
print("teamindexstandalone stats_and_configs/exported_plan.json")
print()
print("Note: requires installing the optional target as well, e.g., via ")
print()
print("> python -m pip install -vvv ./code \\")
print("  --no-clean \\")
print("  --no-build-isolation \\")
print("  --config-settings=cmake.build-type=Debug \\")
print("  --config-settings=cmake.define.ENABLE_STANDALONE=ON")
