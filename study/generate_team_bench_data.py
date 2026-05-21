#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import team_bench as tb
from team_bench_workflow import (
    DEFAULT_CONFIG_PATH,
    expand_experiment_scenarios,
    family_query,
    family_query_slices,
    family_quantiles,
    family_team_specs,
    group_scenarios_by_family,
    load_experiment_config,
    scenario_manifest_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Erzeuge synthetische team_bench-Indizes aus einer Experiment-Konfiguration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Pfad zur JSON-Konfiguration des team_bench-Experiments.",
    )
    parser.add_argument(
        "--family-filter",
        action="append",
        default=[],
        help="Nur Familien ausfuehren, deren Name dieses Teilstueck enthaelt. Mehrfach nutzbar.",
    )
    return parser.parse_args()


def build_uniform_distributions(team_specs):
    team_dists = {}
    for team, shape in team_specs.items():
        dist = np.ones(shape, dtype=float)
        dist /= dist.sum()
        team_dists[team] = dist
    return team_dists


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    scenarios = expand_experiment_scenarios(config)
    manifest_path = scenario_manifest_path(config)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(scenarios, fh, indent=2)
    print(f"Expanded scenario manifest written to: {manifest_path}")

    grouped = group_scenarios_by_family(scenarios)
    for family_name, family_scenarios in sorted(grouped.items()):
        if args.family_filter and not any(token in family_name for token in args.family_filter):
            continue

        team_specs = family_team_specs(family_scenarios)
        query_slices = family_query_slices(family_scenarios)
        quantiles = family_quantiles(family_scenarios)
        query = family_query(family_scenarios)
        destination_folder = Path(family_scenarios[0]["data_root"]).parent
        destination_folder.mkdir(parents=True, exist_ok=True)

        t_rel_values = [scenario["t_rel"] for scenario in family_scenarios]
        n = int(family_scenarios[0]["n"])
        worker_count = int(family_scenarios[0]["worker_count"])

        print(f"\n=== Generating family {family_name} ===")
        print(f"Teams: {[list(team) for team in team_specs]}")
        print(f"Query: {query}")
        print(f"Output root: {destination_folder}")
        print(f"T_rel values: {t_rel_values}")

        team_dists = build_uniform_distributions(team_specs)
        tb.generate_indices(
            N=n,
            T_rel_list=t_rel_values,
            team_dists=team_dists,
            team_queries=query_slices,
            destination_folder=destination_folder,
            quantiles=quantiles,
            query=query,
            n_jobs=worker_count,
        )


if __name__ == "__main__":
    main()
