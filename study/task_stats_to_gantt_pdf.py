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
import json
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
        default=20000,
        help="Maximale Anzahl Tasks pro Strategie, bevor die kuerzesten Tasks gefiltert werden.",
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


def draw_variant_page(pdf: PdfPages, variant_name: str, task_stats_path: Path, data: dict, summary: dict, max_tasks: int):
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

    with PdfPages(output) as pdf:
        for variant_dir in variant_dirs:
            loaded = load_variant_data(variant_dir)
            if loaded is None:
                continue
            task_stats_path, data, summary = loaded
            draw_variant_page(pdf, variant_dir.name, task_stats_path, data, summary, args.max_tasks)

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
