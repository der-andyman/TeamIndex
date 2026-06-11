#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from TeamIndex import evaluation as eva
from generate_comparison_pdfs import generate_team_bench_plots
from mopts_strategies import VARIANTS, clone_mopts, product
from mopts_study import (
    summarize_bin_selection,
    summarize_mopts,
    summarize_query_structure,
)
from team_bench_workflow import DEFAULT_CONFIG_PATH, expand_experiment_scenarios, load_experiment_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuehre die definierten Strategien auf synthetischen team_bench-Indizes aus."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Pfad zur JSON-Konfiguration des team_bench-Experiments.",
    )
    parser.add_argument(
        "--scenario-filter",
        action="append",
        default=[],
        help="Nur Szenarien ausfuehren, deren ID dieses Teilstueck enthaelt. Mehrfach nutzbar.",
    )
    parser.add_argument(
        "--variant-filter",
        action="append",
        default=[],
        help="Nur Varianten ausfuehren, deren Name dieses Teilstueck enthaelt. Mehrfach nutzbar.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Optionaler Override fuer die Anzahl der Wiederholungen pro Szenario und Variante.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help="Optionaler Override fuer worker_count.",
    )
    parser.add_argument(
        "--verbose-runtime",
        action="store_true",
        help="Verbose-Runtime-Ausgaben von TeamIndex aktivieren.",
    )
    return parser.parse_args()


def build_variant_registry(selected_names: list[str]):
    variants = []
    for variant in VARIANTS:
        if variant["name"] in selected_names:
            variants.append(variant)
    missing = [name for name in selected_names if name not in {variant["name"] for variant in variants}]
    if missing:
        raise RuntimeError(f"Unknown strategy names in config/filter: {missing}")
    return variants


def build_runtime_config(worker_count: int, verbose_runtime: bool):
    config = eva.get_new_default_runtime_config()
    config["backend"] = "dram"
    config["verbose_runtime"] = verbose_runtime
    config["return_result"] = True
    config["worker_count"] = worker_count
    return config


def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def determine_effective_worker_count(args, config: dict, scenarios: list[dict]) -> int | None:
    if args.worker_count is not None:
        return int(args.worker_count)
    default_worker = config.get("defaults", {}).get("worker_count")
    scenario_workers = {int(scenario["worker_count"]) for scenario in scenarios if "worker_count" in scenario}
    if default_worker is not None:
        scenario_workers.add(int(default_worker))
    if len(scenario_workers) == 1:
        return next(iter(scenario_workers))
    return None


def create_run_output_dir(experiment_root: Path, experiment_name: str, worker_count: int | None) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    worker_label = f"_w{worker_count}" if worker_count is not None else "_wmixed"
    run_dir = experiment_root / f"{experiment_name}{worker_label}_{timestamp}"
    ensure_output_dir(run_dir)
    return run_dir


def write_outputs(output_dir: Path, result_rows, mopts_rows, baseline_variant: str, skipped_rows=None):
    results_df = pd.DataFrame(result_rows)
    mopts_df = pd.DataFrame(mopts_rows)
    skipped_df = pd.DataFrame(skipped_rows or [])

    results_csv = output_dir / "results.csv"
    mopts_csv = output_dir / "mopts_per_team.csv"
    results_df.to_csv(results_csv, index=False)
    mopts_df.to_csv(mopts_csv, index=False)
    if skipped_rows:
        skipped_df.to_csv(output_dir / "skipped_variants.csv", index=False)

    if results_df.empty:
        return results_df, mopts_df, pd.DataFrame(), pd.DataFrame()

    group_cols = [
        "scenario_id",
        "family_name",
        "team_count",
        "dimension",
        "t_rel",
        "distribution_profile",
        "distribution_strength",
        "n",
        "worker_count",
        "variant",
    ]
    summary_df = (
        results_df
        .groupby(group_cols, as_index=False)
        .agg(
            repetition_count=("repetition", "nunique"),
            runtime_ms_mean=("executor_runtime_ms", "mean"),
            runtime_ms_std=("executor_runtime_ms", "std"),
            ids_per_second_mean=("ids_per_second", "mean"),
            ids_per_second_std=("ids_per_second", "std"),
            read_mib_per_second_mean=("read_mib_per_second", "mean"),
            read_mib_per_second_std=("read_mib_per_second", "std"),
            result_size_mean=("result_size", "mean"),
            ise_count_mean=("ise_count", "mean"),
            total_request_count_mean=("total_request_count", "mean"),
        )
    )

    for metric_prefix in ["runtime_ms", "ids_per_second", "read_mib_per_second"]:
        mean_col = f"{metric_prefix}_mean"
        std_col = f"{metric_prefix}_std"
        rel_std_col = f"{metric_prefix}_rel_std"
        summary_df[rel_std_col] = summary_df[std_col] / summary_df[mean_col]

    baseline_df = summary_df[
        summary_df["variant"] == baseline_variant
    ][["scenario_id", "runtime_ms_mean", "ids_per_second_mean", "read_mib_per_second_mean"]].rename(
        columns={
            "runtime_ms_mean": "baseline_runtime_ms_mean",
            "ids_per_second_mean": "baseline_ids_per_second_mean",
            "read_mib_per_second_mean": "baseline_read_mib_per_second_mean",
        }
    )
    summary_df = summary_df.merge(baseline_df, on="scenario_id", how="left")
    summary_df["speedup_vs_baseline"] = summary_df["baseline_runtime_ms_mean"] / summary_df["runtime_ms_mean"]
    summary_df["throughput_ratio_vs_baseline"] = (
        summary_df["ids_per_second_mean"] / summary_df["baseline_ids_per_second_mean"]
    )
    summary_df["bandwidth_ratio_vs_baseline"] = (
        summary_df["read_mib_per_second_mean"] / summary_df["baseline_read_mib_per_second_mean"]
    )

    best_idx = summary_df.groupby("scenario_id")["runtime_ms_mean"].idxmin()
    best_df = summary_df.loc[best_idx].sort_values(["team_count", "dimension", "t_rel", "scenario_id"])

    summary_df = summary_df.sort_values(["team_count", "dimension", "t_rel", "scenario_id", "variant"]).reset_index(drop=True)
    summary_df.to_csv(output_dir / "summary_by_variant.csv", index=False)
    best_df.to_csv(output_dir / "best_strategy_by_scenario.csv", index=False)
    return results_df, mopts_df, summary_df, best_df


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    scenarios = expand_experiment_scenarios(config)
    if args.scenario_filter:
        scenarios = [
            scenario for scenario in scenarios
            if any(token in scenario["scenario_id"] for token in args.scenario_filter)
        ]
    if not scenarios:
        raise RuntimeError("No team_bench scenarios matched the current filters.")

    selected_names = list(config["defaults"]["strategies"])
    if args.variant_filter:
        selected_names = [
            name for name in selected_names
            if any(token in name for token in args.variant_filter)
        ]
    if not selected_names:
        raise RuntimeError("No strategy names matched the current filters.")
    variants = build_variant_registry(selected_names)

    experiment_root = Path(scenarios[0]["benchmark_output_root"])
    ensure_output_dir(experiment_root)
    effective_worker_count = determine_effective_worker_count(args, config, scenarios)
    output_dir = create_run_output_dir(experiment_root, config["name"], effective_worker_count)
    with (output_dir / "config_snapshot.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    with (output_dir / "scenarios_snapshot.json").open("w", encoding="utf-8") as fh:
        json.dump(scenarios, fh, indent=2)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "experiment_name": config["name"],
                "effective_worker_count": effective_worker_count,
                "worker_count_override": args.worker_count,
                "repetitions_override": args.repetitions,
                "scenario_filters": args.scenario_filter,
                "variant_filters": args.variant_filter,
            },
            fh,
            indent=2,
        )

    result_rows = []
    mopts_rows = []
    skipped_rows = []
    repetitions = int(args.repetitions or config["defaults"]["repetitions"])
    safety_config = config.get("safety", {})
    max_ise_count = safety_config.get("max_ise_count")
    max_ise_count = int(max_ise_count) if max_ise_count is not None else None

    for scenario_index, scenario in enumerate(scenarios, start=1):
        scenario_id = scenario["scenario_id"]
        index_config_path = Path(scenario["index_config_path"])
        if not index_config_path.exists():
            raise RuntimeError(
                f"Missing generated team_bench index for {scenario_id}: {index_config_path}. "
                f"Run study/generate_team_bench_data.py first."
            )

        print(f"\n=== [{scenario_index}/{len(scenarios)}] {scenario_id} ===")
        print(scenario["query"])
        print(scenario["query_note"])

        index = eva.TeamIndex(index_config_path)
        # team_bench-Indizes enthalten bewusst nur die Daten innerhalb der
        # konstruierten Benchmark-Region. Fuer diese synthetischen Szenarien
        # duerfen Teams daher nicht als "selektiert nichts/alles" verworfen
        # oder direkt auf Komplement umgestellt werden.
        index.default_runtime_config["OptimizerConfig"]["ignore_empty_teams"] = False
        index.default_runtime_config["OptimizerConfig"]["allow_exclusion"] = False
        worker_count = int(args.worker_count or scenario["worker_count"])
        runtime_config = build_runtime_config(worker_count, args.verbose_runtime)
        index.default_runtime_config["worker_count"] = worker_count

        for repetition in range(1, repetitions + 1):
            print(f"  repetition {repetition}/{repetitions}")
            for variant in variants:
                variant_name = variant["name"]
                manual_mopts = variant["builder"](index, scenario["query"]) if variant["builder"] is not None else None
                executed_mopts = clone_mopts(manual_mopts) if manual_mopts is not None else clone_mopts(index.prepare_optimization(query=scenario["query"]))
                query_structure = summarize_query_structure(executed_mopts)
                query_structure["ise_count_estimate_manual"] = product(
                    [int(opts["group_count"]) for _, opts in executed_mopts if opts["is_expanded"]]
                ) if query_structure["expanded_team_count_manual"] > 0 else 0

                if (
                    max_ise_count is not None
                    and int(query_structure["ise_count_estimate_manual"]) > max_ise_count
                ):
                    skipped_rows.append(
                        {
                            "scenario_id": scenario_id,
                            "family_name": scenario["family_name"],
                            "team_count": scenario["team_count"],
                            "dimension": scenario["dimension"],
                            "t_rel": scenario["t_rel"],
                            "distribution_profile": scenario.get("distribution_profile", "uniform"),
                            "distribution_strength": scenario.get("distribution_strength", 8.0),
                            "variant": variant_name,
                            "repetition": repetition,
                            "worker_count": runtime_config["worker_count"],
                            "reason": "ise_count_estimate_above_limit",
                            "ise_count_estimate_manual": int(query_structure["ise_count_estimate_manual"]),
                            "max_ise_count": max_ise_count,
                        }
                    )
                    print(
                        f"    {variant_name}: skipped, estimated ISE "
                        f"{query_structure['ise_count_estimate_manual']} > limit {max_ise_count}"
                    )
                    continue

                bin_info_by_team, bin_summary = summarize_bin_selection(index, scenario["query"], executed_mopts)
                query_structure.update(bin_summary)

                result_ids, runtime_stats, request_info, global_info = index.run_query(
                    scenario["query"],
                    config=runtime_config,
                    manual_optimizations=manual_mopts,
                )

                result_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "family_name": scenario["family_name"],
                        "team_count": scenario["team_count"],
                        "dimension": scenario["dimension"],
                        "t_rel": scenario["t_rel"],
                        "distribution_profile": scenario.get("distribution_profile", "uniform"),
                        "distribution_strength": scenario.get("distribution_strength", 8.0),
                        "n": scenario["n"],
                        "query": scenario["query"],
                        "query_note": scenario["query_note"],
                        "variant": variant_name,
                        "variant_description": variant["description"],
                        "repetition": repetition,
                        **query_structure,
                        "worker_count": runtime_config["worker_count"],
                        "result_size": len(result_ids),
                        "executor_runtime_ms": runtime_stats.executor_runtime / 1_000_000,
                        "total_request_count": global_info["total_request_count"],
                        "total_input_cardinality": global_info["total_input_cardinality"],
                        "total_read_volume_KiB": global_info["total_read_volume_KiB"],
                        "ids_per_second": global_info["total_input_cardinality"] / (runtime_stats.executor_runtime / 1_000_000_000),
                        "read_mib_per_second": (global_info["total_read_volume_KiB"] / 1024) / (runtime_stats.executor_runtime / 1_000_000_000),
                        "ise_count": global_info["ise_count"],
                        "outer_union_term_count": global_info["outer_union_term_count"],
                        "outer_intersection_term_count": global_info["outer_intersection_term_count"],
                    }
                )
                mopts_rows.extend(
                    summarize_mopts(
                        query_id=scenario_index,
                        query_name=scenario_id,
                        query=scenario["query"],
                        variant_name=variant_name,
                        mopts=executed_mopts,
                        request_info=request_info,
                        bin_info_by_team=bin_info_by_team,
                    )
                )
                print(
                    f"    {variant_name}: runtime={runtime_stats.executor_runtime / 1_000_000:.3f} ms, "
                    f"result={len(result_ids)}, ise={global_info['ise_count']}"
                )

            write_outputs(output_dir, result_rows, mopts_rows, scenario["baseline_variant"], skipped_rows)

    _, _, summary_df, best_df = write_outputs(output_dir, result_rows, mopts_rows, scenarios[0]["baseline_variant"], skipped_rows)
    plot_paths = generate_team_bench_plots(
        output_dir,
        pd.read_csv(output_dir / "results.csv"),
        scenarios[0]["baseline_variant"],
    )
    print("\nSaved team_bench results to:")
    print(output_dir / "results.csv")
    print(output_dir / "mopts_per_team.csv")
    print(output_dir / "summary_by_variant.csv")
    print(output_dir / "best_strategy_by_scenario.csv")
    if skipped_rows:
        print(output_dir / "skipped_variants.csv")
    print("\nCreated plots:")
    for plot_path in plot_paths:
        print(plot_path)
    if not best_df.empty:
        print("\nCurrent best strategy per scenario:")
        for row in best_df.itertuples(index=False):
            print(f"  {row.scenario_id}: {row.variant} ({row.runtime_ms_mean:.3f} ms)")


if __name__ == "__main__":
    main()
