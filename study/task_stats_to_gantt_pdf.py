#!/usr/bin/env python3
"""
Erzeuge Gantt-PDFs aus TeamIndex-Task-Stats.

Eingabe ist ein Ordner, der von `export_team_bench_taskflow_trace.py`
erzeugt wurde. Das Skript sucht pro Strategie nach `task_stats-*.json`
und erstellt ein mehrseitiges PDF mit einer Gantt-Ansicht pro Strategie.

Die Darstellung ist fuer qualitative Fallstudien gedacht:
- Welche Worker sind ausgelastet?
- Gibt es Luecken/Wartezeiten?
- Gibt es viele sehr kleine Tasks?
- Dominiert eine Task-Art die Laufzeit?
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch


TASK_COLORS = {
    "Task::Leaf": "#9ecae1",
    "Task::LeafUnion": "#3182bd",
    "Task::TeamUnion": "#08519c",
    "Task::ExpandedInit": "#fdd0a2",
    "Task::DistributedIntersection": "#e6550d",
    "Task::BigInnerIntersection": "#31a354",
    "Task::BigOuterIntersection": "#006d2c",
    "Task::BigInnerUnion": "#756bb1",
    "Task::BigOuterUnion": "#54278f",
    "Task::Materialize": "#636363",
}
DEFAULT_COLOR = "#bdbdbd"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Erzeuge ein Gantt-PDF aus TeamIndex task_stats-JSON-Dateien."
    )
    parser.add_argument(
        "trace_dir",
        type=Path,
        help="Ordner eines export_team_bench_taskflow_trace.py-Laufs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ziel-PDF. Default: <trace_dir>/gantt_overview.pdf",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=3000,
        help=(
            "Maximale Anzahl Tasks pro Strategie fuer die Gantt-Darstellung. "
            "Die CSV-Auslastungsmetriken werden trotzdem aus allen Tasks berechnet."
        ),
    )
    return parser.parse_args()


def newest_task_stats(variant_dir: Path) -> Path | None:
    files = sorted(variant_dir.glob("task_stats-*.json"))
    return files[-1] if files else None


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_variant_data(variant_dir: Path):
    task_stats_path = newest_task_stats(variant_dir)
    if task_stats_path is None:
        return None
    data = load_json(task_stats_path)
    summary_path = variant_dir / "run_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    return task_stats_path, data, summary


def maybe_filter_tasks(tasks: list[dict], max_tasks: int):
    if len(tasks) <= max_tasks:
        return tasks, ""
    ranked = sorted(
        tasks,
        key=lambda task: int(task["stop_ns"]) - int(task["start_ns"]),
        reverse=True,
    )
    kept = ranked[:max_tasks]
    note = f"Filtered: showing {len(kept)} longest of {len(tasks)} tasks"
    return kept, note


def compute_utilization_metrics(variant_name: str, data: dict, summary: dict) -> dict[str, object]:
    tasks = data.get("task_statistics", [])
    if not tasks:
        return {"variant": variant_name, "task_count": 0}

    min_start = min(int(task["start_ns"]) for task in tasks)
    max_stop = max(int(task["stop_ns"]) for task in tasks)
    wall_ns = max(1, max_stop - min_start)
    workers_seen = sorted({int(task["worker_id"]) for task in tasks})
    configured_workers = int(summary.get("worker_count") or (max(workers_seen) + 1 if workers_seen else len(workers_seen)))
    configured_workers = max(configured_workers, len(workers_seen), 1)

    busy_by_worker = defaultdict(int)
    busy_by_type = defaultdict(int)
    longest_task_ns = 0
    for task in tasks:
        duration_ns = max(0, int(task["stop_ns"]) - int(task["start_ns"]))
        worker = int(task["worker_id"])
        task_type = task.get("type", "unknown")
        busy_by_worker[worker] += duration_ns
        busy_by_type[task_type] += duration_ns
        longest_task_ns = max(longest_task_ns, duration_ns)

    busy_ns = sum(busy_by_worker.values())
    capacity_ns = wall_ns * configured_workers
    idle_ns = max(0, capacity_ns - busy_ns)
    worker_busy_values = [busy_by_worker.get(worker, 0) for worker in range(configured_workers)]
    mean_worker_busy_ns = busy_ns / configured_workers if configured_workers else 0
    max_worker_busy_ns = max(worker_busy_values) if worker_busy_values else 0
    worker_load_imbalance = (
        max_worker_busy_ns / mean_worker_busy_ns
        if mean_worker_busy_ns > 0 else 0.0
    )
    top_type, top_type_busy_ns = max(busy_by_type.items(), key=lambda item: item[1])

    return {
        "variant": variant_name,
        "worker_count": configured_workers,
        "workers_used": len(workers_seen),
        "task_count": len(tasks),
        "executor_runtime_ms": float(summary.get("executor_runtime_ms", 0.0) or 0.0),
        "task_wall_ms": wall_ns / 1_000_000,
        "busy_cpu_ms": busy_ns / 1_000_000,
        "idle_cpu_ms": idle_ns / 1_000_000,
        "utilization_fraction": busy_ns / capacity_ns if capacity_ns else 0.0,
        "idle_fraction": idle_ns / capacity_ns if capacity_ns else 0.0,
        "effective_parallelism": busy_ns / wall_ns if wall_ns else 0.0,
        "worker_load_imbalance": worker_load_imbalance,
        "longest_task_ms": longest_task_ns / 1_000_000,
        "top_task_type": top_type.replace("Task::", ""),
        "top_task_type_fraction": top_type_busy_ns / busy_ns if busy_ns else 0.0,
        "result_size": summary.get("result_size"),
        "ise_count": summary.get("ise_count"),
        "total_request_count": summary.get("total_request_count"),
        "total_input_cardinality": summary.get("total_input_cardinality"),
    }


def write_utilization_csv(rows: list[dict[str, object]], output_path: Path):
    if not rows:
        return
    fieldnames = [
        "variant",
        "worker_count",
        "workers_used",
        "task_count",
        "executor_runtime_ms",
        "task_wall_ms",
        "busy_cpu_ms",
        "idle_cpu_ms",
        "utilization_fraction",
        "idle_fraction",
        "effective_parallelism",
        "worker_load_imbalance",
        "longest_task_ms",
        "top_task_type",
        "top_task_type_fraction",
        "result_size",
        "ise_count",
        "total_request_count",
        "total_input_cardinality",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def draw_summary_page(pdf: PdfPages, rows: list[dict[str, object]], trace_dir: Path):
    if not rows:
        return
    rows = sorted(rows, key=lambda row: float(row.get("executor_runtime_ms") or 0.0))
    columns = [
        "Strategie",
        "Runtime [ms]",
        "Util. [%]",
        "Idle [%]",
        "eff. Parallel.",
        "Tasks",
        "Top-Task",
    ]
    table_rows = []
    for row in rows:
        table_rows.append([
            str(row.get("variant", "")),
            f"{float(row.get('executor_runtime_ms') or 0.0):.1f}",
            f"{100 * float(row.get('utilization_fraction') or 0.0):.1f}",
            f"{100 * float(row.get('idle_fraction') or 0.0):.1f}",
            f"{float(row.get('effective_parallelism') or 0.0):.1f}x",
            f"{int(row.get('task_count') or 0):,}",
            f"{row.get('top_task_type', '')} ({100 * float(row.get('top_task_type_fraction') or 0.0):.0f}%)",
        ])

    fig, ax = plt.subplots(figsize=(13.5, max(4.5, 1.0 + 0.45 * len(table_rows))))
    ax.axis("off")
    ax.set_title("Worker-Auslastung und Idle-Anteil", fontsize=15, pad=18)
    fig.text(
        0.01,
        0.94,
        f"Trace: {trace_dir.name}\n"
        "Utilization = aufsummierte Taskzeit / (Wall-Time * Worker). "
        "Hoher Idle-Anteil spricht fuer Wartezeiten, zu wenig Parallelitaet oder Lastungleichgewicht.",
        fontsize=9,
        color="#333333",
    )
    table = ax.table(
        cellText=table_rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#e5e7eb")
            cell.set_text_props(weight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_variant_page(pdf: PdfPages, variant_name: str, task_stats_path: Path, data: dict, summary: dict, max_tasks: int, utilization: dict[str, object] | None = None):
    tasks = data.get("task_statistics", [])
    tasks, filter_note = maybe_filter_tasks(tasks, max_tasks)
    metadata = data.get("metadata", {})
    task_counts = data.get("task_counts", {})

    if not tasks:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, f"No tasks found for {variant_name}", ha="center", va="center")
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    min_start = min(int(task["start_ns"]) for task in tasks)
    max_stop = max(int(task["stop_ns"]) for task in tasks)
    total_ms = (max_stop - min_start) / 1_000_000
    workers = sorted({int(task["worker_id"]) for task in tasks})
    worker_to_y = {worker: idx for idx, worker in enumerate(workers)}

    fig_height = max(5, min(12, 2 + 0.35 * len(workers)))
    fig, ax = plt.subplots(figsize=(14, fig_height))

    for task in tasks:
        start_ms = (int(task["start_ns"]) - min_start) / 1_000_000
        duration_ms = max(0.0005, (int(task["stop_ns"]) - int(task["start_ns"])) / 1_000_000)
        worker = int(task["worker_id"])
        task_type = task.get("type", "unknown")
        ax.barh(
            worker_to_y[worker],
            duration_ms,
            left=start_ms,
            height=0.72,
            color=TASK_COLORS.get(task_type, DEFAULT_COLOR),
            edgecolor="none",
            alpha=0.9,
        )

    ax.set_yticks(list(worker_to_y.values()))
    ax.set_yticklabels([str(worker) for worker in workers])
    ax.set_xlabel("Zeit seit erstem Task [ms]")
    ax.set_ylabel("Worker")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_xlim(0, max(total_ms * 1.02, 0.001))

    runtime_ms = summary.get("executor_runtime_ms")
    result_size = summary.get("result_size")
    ise_count = summary.get("ise_count")
    title = f"{variant_name}"
    if runtime_ms is not None:
        title += f" | runtime={float(runtime_ms):.3f} ms"
    if utilization:
        title += (
            f" | util={100 * float(utilization.get('utilization_fraction') or 0.0):.1f}%"
            f" | idle={100 * float(utilization.get('idle_fraction') or 0.0):.1f}%"
            f" | par={float(utilization.get('effective_parallelism') or 0.0):.1f}x"
        )
    if result_size is not None:
        title += f" | result={result_size}"
    if ise_count is not None:
        title += f" | ISE={ise_count}"
    ax.set_title(title)

    present_types = sorted({task.get("type", "unknown") for task in tasks})
    legend_items = [
        Patch(color=TASK_COLORS.get(task_type, DEFAULT_COLOR), label=task_type.replace("Task::", ""))
        for task_type in present_types
    ]
    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)

    counts_text = ", ".join(
        f"{name.replace('Task::', '')}: {count}" for name, count in sorted(task_counts.items())
    )
    note_parts = [
        f"task_stats: {task_stats_path.name}",
        f"execution_time_ns: {metadata.get('execution_time_ns', 'n/a')}",
    ]
    if filter_note:
        note_parts.append(filter_note)
    note_parts.append(counts_text)
    fig.text(0.01, 0.01, " | ".join(note_parts), fontsize=7, color="#444444")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    trace_dir = args.trace_dir
    output = args.output or trace_dir / "gantt_overview.pdf"

    variant_dirs = [path for path in sorted(trace_dir.iterdir()) if path.is_dir()]
    if not variant_dirs:
        raise RuntimeError(f"No variant subdirectories found in {trace_dir}")

    loaded_variants = []
    utilization_rows = []
    for variant_dir in variant_dirs:
        loaded = load_variant_data(variant_dir)
        if loaded is None:
            continue
        task_stats_path, data, summary = loaded
        utilization = compute_utilization_metrics(variant_dir.name, data, summary)
        loaded_variants.append((variant_dir.name, task_stats_path, data, summary, utilization))
        utilization_rows.append(utilization)

    utilization_csv = trace_dir / "worker_utilization_summary.csv"
    write_utilization_csv(utilization_rows, utilization_csv)

    with PdfPages(output) as pdf:
        draw_summary_page(pdf, utilization_rows, trace_dir)
        for variant_name, task_stats_path, data, summary, utilization in loaded_variants:
            draw_variant_page(pdf, variant_name, task_stats_path, data, summary, args.max_tasks, utilization)

    print(f"Wrote {output}")
    print(f"Wrote {utilization_csv}")


if __name__ == "__main__":
    main()
