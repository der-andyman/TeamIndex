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
    parser.add_argument(
        "--scenario-filter",
        action="append",
        default=[],
        help="Nur Szenarien ausfuehren, deren ID dieses Teilstueck enthaelt. Mehrfach nutzbar.",
    )
    return parser.parse_args()


def _normalize_distribution(weights):
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Distribution has no positive probability mass.")
    return weights / total


def _small_query_corner(query_slices):
    hotspot_slices = []
    for slc in query_slices:
        start = int(slc.start or 0)
        stop = int(slc.stop)
        width = max(1, stop - start)
        hotspot_slices.append(slice(start, start + max(1, width // 4)))
    return tuple(hotspot_slices)


def _build_profile_distribution(shape, query_slices, profile, strength, team_position):
    profile = profile.lower()
    weights = np.ones(shape, dtype=np.float64)

    if profile == "uniform":
        return _normalize_distribution(weights)

    if profile == "query_hotspot":
        weights[query_slices] += strength
        return _normalize_distribution(weights)

    if profile == "anti_query_hotspot":
        weights *= 1.0 + strength * 0.25
        weights[query_slices] = 1.0
        return _normalize_distribution(weights)

    if profile == "cell_skew":
        # Same query window, but very uneven cell/list cardinalities inside it.
        weights[query_slices] += strength * 0.25
        weights[_small_query_corner(query_slices)] += strength * 4.0
        return _normalize_distribution(weights)

    if profile == "mixed_team_imbalance":
        if team_position == 0:
            weights[query_slices] += strength * 2.0
        elif team_position == 1:
            weights *= 1.0 + strength * 0.05
            weights[query_slices] = 1.0
        elif team_position % 2 == 0:
            weights[_small_query_corner(query_slices)] += strength * 4.0
        else:
            weights[query_slices] += strength * 0.5
        return _normalize_distribution(weights)

    raise ValueError(
        f"Unknown distribution_profile '{profile}'. "
        "Supported profiles: uniform, query_hotspot, anti_query_hotspot, "
        "cell_skew, mixed_team_imbalance."
    )


def build_team_distributions(team_specs, query_slices, profile, strength):
    team_dists = {}
    for team_position, (team, shape) in enumerate(team_specs.items()):
        dist = _build_profile_distribution(
            shape=shape,
            query_slices=query_slices[team],
            profile=profile,
            strength=strength,
            team_position=team_position,
        )
        team_dists[team] = dist
        query_mass = float(dist[query_slices[team]].sum())
        non_zero = dist[dist > 0]
        skew_ratio = float(non_zero.max() / non_zero.min()) if len(non_zero) else 0.0
        print(
            f"  distribution {profile:>20} | team {team_position + 1}: {list(team)} | "
            f"query_mass={query_mass:.6f} | cell_skew_ratio={skew_ratio:.2f}",
            flush=True,
        )
    return team_dists


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
    manifest_path = scenario_manifest_path(config)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(scenarios, fh, indent=2)
    print(f"Expanded scenario manifest written to: {manifest_path}", flush=True)

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
        distribution_profile = family_scenarios[0].get("distribution_profile", "uniform")
        distribution_strength = float(family_scenarios[0].get("distribution_strength", 8.0))
        min_query_hits_warning = int(family_scenarios[0].get("min_query_hits_warning", 500))

        print(f"\n=== Generating family {family_name} ===", flush=True)
        print(f"Teams: {[list(team) for team in team_specs]}", flush=True)
        print(f"Query: {query}", flush=True)
        total_index_bin_cells = sum(int(np.prod(shape)) for shape in team_specs.values())
        selected_bin_cells_by_team = {
            team: int(np.prod([len(range(*slc.indices(dim))) for slc, dim in zip(query_slices[team], team_specs[team])]))
            for team in team_specs
        }
        total_selected_bin_cells = sum(selected_bin_cells_by_team.values())
        selected_fraction = (
            total_selected_bin_cells / total_index_bin_cells
            if total_index_bin_cells else 0.0
        )

        print(f"Output root: {destination_folder}", flush=True)
        print(f"T_rel values: {t_rel_values}", flush=True)
        print(f"Distribution profile: {distribution_profile} (strength={distribution_strength:g})", flush=True)
        print(
            "Bin-Geometrie: "
            f"{list(team_specs.values())[0][0]} Bins/Dimension, "
            f"{len(range(*next(iter(query_slices.values()))[0].indices(list(team_specs.values())[0][0])))} Query-Bins/Dimension, "
            f"{next(iter(selected_bin_cells_by_team.values())):,} getroffene Zellen pro Team, "
            f"{total_selected_bin_cells:,} getroffene Zellen gesamt "
            f"von {total_index_bin_cells:,} Index-Zellen "
            f"({selected_fraction:.2%})",
            flush=True,
        )

        team_dists = build_team_distributions(
            team_specs=team_specs,
            query_slices=query_slices,
            profile=distribution_profile,
            strength=distribution_strength,
        )
        min_query_mass = min(float(dist[query_slices[team]].sum()) for team, dist in team_dists.items())
        expected_min_hits = min_query_mass * n
        max_t_rel = max(t_rel_values)
        expected_max_intersection = expected_min_hits * max_t_rel
        if expected_min_hits < min_query_hits_warning:
            print(
                "  WARNING: sehr wenige erwartete Query-Treffer im kleinsten Team "
                f"({expected_min_hits:.1f} < {min_query_hits_warning}). "
                "Laufzeiten koennen stark varianzgepraegt sein.",
                flush=True,
            )
        print(
            f"  smallest_team_query_hits≈{expected_min_hits:.1f}, "
            f"intersection_hits_at_max_Trel≈{expected_max_intersection:.1f}",
            flush=True,
        )
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
