#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
STUDY_DIR = (BASE_DIR / "mopts_study").resolve()
PLANS_DIR = STUDY_DIR / "plans"
PYTHON_RESULTS_CSV = STUDY_DIR / "results.csv"
OUT_DIR = STUDY_DIR / "standalone"
LOGS_DIR = OUT_DIR / "logs"
PLOTS_DIR = OUT_DIR / "plots"
RUNTIME_PLANS_DIR = OUT_DIR / "runtime_plans"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute exported TeamIndex standalone plans and compare them to Python runs."
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for exported plans inside mopts_study/plans.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of plans to execute.",
    )
    return parser.parse_args()


def ensure_dirs():
    for subdir in [OUT_DIR, LOGS_DIR, PLOTS_DIR, RUNTIME_PLANS_DIR]:
        subdir.mkdir(parents=True, exist_ok=True)


def latest_artifact(base_path: Path) -> Path | None:
    pattern = f"{base_path.stem}-*{base_path.suffix}"
    matches = sorted(base_path.parent.glob(pattern))
    return matches[-1] if matches else None


def snapshot_artifacts(base_path: Path) -> set[Path]:
    pattern = f"{base_path.stem}-*{base_path.suffix}"
    return set(base_path.parent.glob(pattern))


def find_new_artifact(base_path: Path, before: set[Path]) -> Path | None:
    pattern = f"{base_path.stem}-*{base_path.suffix}"
    after = set(base_path.parent.glob(pattern))
    new_files = sorted(after - before)
    if new_files:
        return new_files[-1]
    return latest_artifact(base_path)


def find_standalone_binary() -> Path:
    binary = shutil.which("teamindexstandalone")
    if binary is not None:
        return Path(binary).resolve()

    venv_binary = (BASE_DIR.parent / "venv" / "bin" / "teamindexstandalone").resolve()
    if venv_binary.exists():
        return venv_binary

    raise FileNotFoundError(
        "teamindexstandalone was not found in PATH or at venv/bin/teamindexstandalone."
    )


def parse_plan(plan_path: Path) -> dict:
    with plan_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_runtime_plan(plan_path: Path, plan: dict) -> Path:
    runtime_plan = copy.deepcopy(plan)
    runtime_plan["executor_config"]["verbose"] = True
    runtime_plan["executor_config"]["return_result"] = True

    runtime_plan_path = RUNTIME_PLANS_DIR / plan_path.name
    with runtime_plan_path.open("w", encoding="utf-8") as handle:
        json.dump(runtime_plan, handle, indent=4, separators=(",", ": "))
    return runtime_plan_path


def parse_result_stats(result_stats_path: Path | None) -> dict:
    if result_stats_path is None or not result_stats_path.exists():
        return {}
    with result_stats_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_task_stats(task_stats_path: Path | None) -> dict:
    if task_stats_path is None or not task_stats_path.exists():
        return {}
    with task_stats_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_task_stats(task_stats: dict) -> dict:
    metadata = task_stats.get("metadata", {})
    task_counts = task_stats.get("task_counts", {})
    task_statistics = task_stats.get("task_statistics", [])

    durations_ns = [
        task["stop_ns"] - task["start_ns"]
        for task in task_statistics
        if task.get("start_ns") is not None and task.get("stop_ns") is not None
    ]
    worker_ids = {
        task["worker_id"] for task in task_statistics if task.get("worker_id") is not None
    }

    summary = {
        "standalone_task_total_count": int(sum(task_counts.values())),
        "standalone_task_type_count": int(len(task_counts)),
        "standalone_active_worker_count": int(len(worker_ids)),
        "standalone_task_graph_runtime_ms": metadata.get("execution_time_ns", 0) / 1_000_000,
        "standalone_longest_task_ms": (max(durations_ns) / 1_000_000) if durations_ns else None,
        "standalone_avg_task_ms": (
            (sum(durations_ns) / len(durations_ns) / 1_000_000) if durations_ns else None
        ),
    }
    for task_type, count in sorted(task_counts.items()):
        safe_name = (
            task_type.replace("Task::", "")
            .replace("::", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )
        summary[f"count_{safe_name}"] = int(count)
    return summary


def execute_plan(binary: Path, plan_path: Path) -> dict:
    plan = parse_plan(plan_path)
    runtime_plan_path = write_runtime_plan(plan_path, plan)
    executor_config = plan["executor_config"]

    result_stats_base = Path(executor_config["print_result_stats"]).resolve()
    task_stats_base = Path(executor_config["print_task_stats"]).resolve()
    execution_plan_base = Path(executor_config["print_execution_plan"]).resolve()

    before_result = snapshot_artifacts(result_stats_base)
    before_task = snapshot_artifacts(task_stats_base)
    before_dot = snapshot_artifacts(execution_plan_base)

    proc = subprocess.run(
        [str(binary), str(runtime_plan_path)],
        cwd=BASE_DIR.parent,
        text=True,
        capture_output=True,
        check=False,
    )

    log_path = LOGS_DIR / f"{plan_path.stem}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(proc.stdout)
        if proc.stderr:
            handle.write("\n[stderr]\n")
            handle.write(proc.stderr)

    result_stats_path = find_new_artifact(result_stats_base, before_result)
    task_stats_path = find_new_artifact(task_stats_base, before_task)
    execution_plan_dot = find_new_artifact(execution_plan_base, before_dot)

    result_stats = parse_result_stats(result_stats_path)
    task_stats = parse_task_stats(task_stats_path)
    task_summary = summarize_task_stats(task_stats)

    return {
        "query": plan.get("query"),
        "team_count_from_plan": len(plan.get("team_workload_infos", [])),
        "standalone_runtime_plan_path": str(runtime_plan_path),
        "standalone_verbose_enabled": True,
        "standalone_returncode": proc.returncode,
        "standalone_stdout_log": str(log_path),
        "standalone_result_stats_json": str(result_stats_path) if result_stats_path else None,
        "standalone_task_stats_json": str(task_stats_path) if task_stats_path else None,
        "standalone_execution_plan_dot": str(execution_plan_dot) if execution_plan_dot else None,
        "standalone_executor_runtime_ms": result_stats.get("executor_runtime", 0) / 1_000_000,
        "standalone_plan_runtime_ms": result_stats.get("plan_construction_runtime", 0)
        / 1_000_000,
        "standalone_result_size": result_stats.get("result_cardinality"),
        "standalone_input_cardinality": result_stats.get("input_cardinality"),
        "standalone_ise_count": result_stats.get("ise_count"),
        "standalone_team_count": result_stats.get("team_count"),
        "standalone_expanded_team_count": result_stats.get("expanded_team_count"),
        **task_summary,
    }


def plot_standalone_runtime(df: pd.DataFrame, figure_path: Path):
    pivot = df.pivot(index="query_name", columns="variant", values="standalone_executor_runtime_ms")
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Standalone Runtime (ms)")
    ax.set_xlabel("Query")
    ax.set_title("Standalone Runtime Comparison")
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def plot_python_vs_standalone(df: pd.DataFrame, figure_path: Path):
    compare = df[
        [
            "query_name",
            "variant",
            "executor_runtime_ms",
            "standalone_executor_runtime_ms",
        ]
    ].copy()
    compare["query_variant"] = compare["query_name"] + "\n" + compare["variant"]
    plot_df = compare.set_index("query_variant")[
        ["executor_runtime_ms", "standalone_executor_runtime_ms"]
    ]

    ax = plot_df.plot(kind="bar", figsize=(14, 7))
    ax.set_ylabel("Runtime (ms)")
    ax.set_xlabel("Query / Variant")
    ax.set_title("Python vs Standalone Runtime per Query and Variant")
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.legend(["Python", "Standalone"])
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def plot_runtime_delta(df: pd.DataFrame, figure_path: Path):
    plot_df = df[["query_name", "variant", "runtime_delta_ms"]].copy()
    plot_df["query_variant"] = plot_df["query_name"] + "\n" + plot_df["variant"]
    plot_df = plot_df.set_index("query_variant")

    colors = ["#c44e52" if value > 0 else "#55a868" for value in plot_df["runtime_delta_ms"]]
    ax = plot_df.plot(
        kind="bar",
        y="runtime_delta_ms",
        figsize=(14, 7),
        legend=False,
        color=colors,
    )
    ax.set_ylabel("Runtime Delta (Standalone - Python) in ms")
    ax.set_xlabel("Query / Variant")
    ax.set_title("Runtime Difference: Positive Means Standalone Is Slower")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def plot_speedup_vs_python(df: pd.DataFrame, figure_path: Path):
    plot_df = df.copy()
    plot_df["standalone_speedup_vs_python"] = (
        plot_df["executor_runtime_ms"] / plot_df["standalone_executor_runtime_ms"]
    )
    pivot = plot_df.pivot(
        index="query_name",
        columns="variant",
        values="standalone_speedup_vs_python",
    )

    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Speedup (Python / Standalone)")
    ax.set_xlabel("Query")
    ax.set_title("Standalone Speedup Relative to Python Execution (> 1 means Standalone faster)")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def main():
    args = parse_args()
    ensure_dirs()

    binary = find_standalone_binary()
    python_results = pd.read_csv(PYTHON_RESULTS_CSV)

    plan_paths = sorted(PLANS_DIR.glob(args.pattern))
    if args.limit is not None:
        plan_paths = plan_paths[: args.limit]

    if not plan_paths:
        raise FileNotFoundError(f"No plan files found in {PLANS_DIR} for pattern {args.pattern!r}.")

    standalone_rows = []

    for plan_path in plan_paths:
        print(f"Executing standalone plan: {plan_path.name}")
        run_data = execute_plan(binary=binary, plan_path=plan_path.resolve())

        row = python_results[
            python_results["exported_plan_path"] == str(plan_path.resolve())
        ].copy()
        if row.empty:
            raise KeyError(f"Could not find matching Python result row for {plan_path}.")

        base = row.iloc[0].to_dict()
        merged = {
            **base,
            **run_data,
        }
        merged["result_size_matches_python"] = (
            merged["result_size"] == merged["standalone_result_size"]
        )
        merged["runtime_delta_ms"] = (
            merged["standalone_executor_runtime_ms"] - merged["executor_runtime_ms"]
        )
        merged["standalone_speedup_vs_python"] = (
            merged["executor_runtime_ms"] / merged["standalone_executor_runtime_ms"]
            if merged["standalone_executor_runtime_ms"]
            else None
        )
        standalone_rows.append(merged)

    standalone_df = pd.DataFrame(standalone_rows)
    standalone_results_csv = OUT_DIR / "standalone_results.csv"
    comparison_csv = OUT_DIR / "standalone_comparison_vs_python.csv"
    summary_csv = OUT_DIR / "standalone_runtime_summary.csv"

    standalone_df.to_csv(standalone_results_csv, index=False)
    standalone_df.to_csv(comparison_csv, index=False)
    standalone_df[
        [
            "query_name",
            "variant",
            "executor_runtime_ms",
            "standalone_executor_runtime_ms",
            "runtime_delta_ms",
            "standalone_speedup_vs_python",
            "result_size",
            "standalone_result_size",
            "result_size_matches_python",
            "standalone_returncode",
            "standalone_stdout_log",
        ]
    ].to_csv(summary_csv, index=False)

    plot_standalone_runtime(standalone_df, PLOTS_DIR / "standalone_runtime_comparison.pdf")
    plot_python_vs_standalone(standalone_df, PLOTS_DIR / "python_vs_standalone_runtime.pdf")
    plot_runtime_delta(standalone_df, PLOTS_DIR / "runtime_delta_standalone_minus_python.pdf")
    plot_speedup_vs_python(standalone_df, PLOTS_DIR / "standalone_speedup_vs_python.pdf")

    print("\nSaved:")
    print(standalone_results_csv)
    print(comparison_csv)
    print(summary_csv)
    print(PLOTS_DIR / "standalone_runtime_comparison.pdf")
    print(PLOTS_DIR / "python_vs_standalone_runtime.pdf")
    print(PLOTS_DIR / "runtime_delta_standalone_minus_python.pdf")
    print(PLOTS_DIR / "standalone_speedup_vs_python.pdf")


if __name__ == "__main__":
    main()
