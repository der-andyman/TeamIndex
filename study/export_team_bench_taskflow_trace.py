#!/usr/bin/env python3
"""
Erzeuge Taskflow-/Gantt-Artefakte fuer einzelne team_bench-Szenarien.

Dieses Skript ist fuer qualitative Fallstudien gedacht, nicht fuer grosse
Benchmark-Laeufe. Es fuehrt ein ausgewaehltes Szenario mit einer oder mehreren
Strategien einmal aus und speichert pro Strategie:

- task_graph.json: Taskflow-Profiler-Datei fuer https://taskflow.github.io/tfprof/
- execution_plan-<timestamp>.dot: Taskgraph als DOT-Datei
- task_stats-<timestamp>.json: detaillierte Task-Laufzeiten
- result_stats-<timestamp>.json: Ergebnis-/Runtime-Statistiken
- manual_optimizations.json: tatsaechlich verwendete Planparameter
- run_summary.json: kompakte Zusammenfassung des Laufs

Damit lassen sich Worker-Auslastung, Wartezeiten, Gruppierung und
Parallelitaetsunterschiede zwischen Strategien visuell vergleichen.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from TeamIndex import evaluation as eva
from mopts_strategies import VARIANTS, clone_mopts
from mopts_study import summarize_bin_selection, summarize_query_structure
from team_bench_workflow import DEFAULT_CONFIG_PATH, expand_experiment_scenarios, load_experiment_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Erzeuge Taskflow/Gantt-Artefakte fuer ein einzelnes team_bench-Szenario."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Pfad zur JSON-Konfiguration des team_bench-Experiments.",
    )
    parser.add_argument(
        "--scenario-id",
        required=True,
        help="Exakte scenario_id aus der expandierten team_bench-Konfiguration.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Strategie-Name. Mehrfach nutzbar. Default: baseline_union_first und union_first_parallel.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help="Optionaler Override fuer worker_count.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("study/taskflow_gantt"),
        help="Zielordner fuer die erzeugten Artefakte.",
    )
    parser.add_argument(
        "--verbose-runtime",
        action="store_true",
        help="Verbose-Ausgaben der TeamIndex-Runtime aktivieren.",
    )
    return parser.parse_args()


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_variant_registry(selected_names: list[str]):
    by_name = {variant["name"]: variant for variant in VARIANTS}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise RuntimeError(f"Unknown strategy names: {missing}")
    return [by_name[name] for name in selected_names]


def build_runtime_config(worker_count: int, verbose_runtime: bool, variant_dir: Path):
    config = eva.get_new_default_runtime_config()
    config["backend"] = "dram"
    config["verbose_runtime"] = verbose_runtime
    config["return_result"] = True
    config["worker_count"] = worker_count
    config["task_graph_path"] = variant_dir / "task_graph.json"
    config["print_execution_plan"] = variant_dir / "execution_plan.dot"
    config["print_task_stats"] = variant_dir / "task_stats.json"
    config["print_result_stats"] = variant_dir / "result_stats.json"
    return config


def find_scenario(config_path: Path, scenario_id: str):
    config = load_experiment_config(config_path)
    scenarios = expand_experiment_scenarios(config)
    matches = [scenario for scenario in scenarios if scenario["scenario_id"] == scenario_id]
    if not matches:
        available = "\n".join(scenario["scenario_id"] for scenario in scenarios[:50])
        raise RuntimeError(
            f"Unknown scenario_id '{scenario_id}'. First available scenario IDs:\n{available}"
        )
    return config, matches[0]


def main():
    args = parse_args()
    config, scenario = find_scenario(args.config, args.scenario_id)
    selected_names = args.variant or ["baseline_union_first", "union_first_parallel"]
    variants = build_variant_registry(selected_names)

    index_config_path = Path(scenario["index_config_path"])
    if not index_config_path.exists():
        raise RuntimeError(
            f"Missing generated team_bench index for {scenario['scenario_id']}: {index_config_path}. "
            "Run study/generate_team_bench_data.py first."
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    worker_count = int(args.worker_count or scenario["worker_count"])
    run_dir = args.output_root / f"{config['name']}_{scenario['scenario_id']}_w{worker_count}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    index = eva.TeamIndex(index_config_path)
    index.default_runtime_config["OptimizerConfig"]["ignore_empty_teams"] = False
    index.default_runtime_config["OptimizerConfig"]["allow_exclusion"] = False
    index.default_runtime_config["worker_count"] = worker_count

    with (run_dir / "scenario.json").open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(scenario), fh, indent=2)

    summaries = []
    print(f"Scenario: {scenario['scenario_id']}")
    print(f"Query: {scenario['query']}")
    print(f"Output: {run_dir}")

    for variant in variants:
        variant_name = variant["name"]
        variant_dir = run_dir / variant_name
        variant_dir.mkdir()

        manual_mopts = variant["builder"](index, scenario["query"]) if variant["builder"] is not None else None
        executed_mopts = clone_mopts(manual_mopts) if manual_mopts is not None else clone_mopts(
            index.prepare_optimization(query=scenario["query"])
        )
        query_structure = summarize_query_structure(executed_mopts)
        _, bin_summary = summarize_bin_selection(index, scenario["query"], executed_mopts)
        query_structure.update(bin_summary)

        runtime_config = build_runtime_config(worker_count, args.verbose_runtime, variant_dir)
        result_ids, runtime_stats, request_info, global_info = index.run_query(
            scenario["query"],
            config=runtime_config,
            manual_optimizations=manual_mopts,
            experiment_name=f"{scenario['scenario_id']}::{variant_name}",
        )

        summary = {
            "scenario_id": scenario["scenario_id"],
            "variant": variant_name,
            "variant_description": variant["description"],
            "worker_count": worker_count,
            "result_size": len(result_ids),
            "executor_runtime_ms": runtime_stats.executor_runtime / 1_000_000,
            "task_stats_path": runtime_stats.task_stats_path,
            "total_request_count": global_info["total_request_count"],
            "total_input_cardinality": global_info["total_input_cardinality"],
            "total_read_volume_KiB": global_info["total_read_volume_KiB"],
            "ise_count": global_info["ise_count"],
            "outer_union_term_count": global_info["outer_union_term_count"],
            "outer_intersection_term_count": global_info["outer_intersection_term_count"],
            **query_structure,
        }
        summaries.append(summary)

        with (variant_dir / "manual_optimizations.json").open("w", encoding="utf-8") as fh:
            json.dump(_json_safe(executed_mopts), fh, indent=2)
        with (variant_dir / "request_info.json").open("w", encoding="utf-8") as fh:
            json.dump(_json_safe(request_info), fh, indent=2)
        with (variant_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(_json_safe(summary), fh, indent=2)

        print(
            f"  {variant_name}: {summary['executor_runtime_ms']:.3f} ms, "
            f"result={summary['result_size']}, ise={summary['ise_count']}, "
            f"task_graph={variant_dir / 'task_graph.json'}"
        )

    with (run_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(summaries), fh, indent=2)

    print("\nDone. Open task_graph.json files with:")
    print("  https://taskflow.github.io/tfprof/")


if __name__ == "__main__":
    main()
