from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from study_paths import TEAM_BENCH_DATA_ROOT, TEAM_BENCH_EXPERIMENTS_DIR, TEAM_BENCH_RESULTS_ROOT


DEFAULT_CONFIG_PATH = TEAM_BENCH_EXPERIMENTS_DIR / "team_bench_bins20_hit8_dims234_n50k.json"


def load_experiment_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    config["_config_path"] = str(path.resolve())
    return config


def _build_attribute_names(count: int, prefix: str = "A") -> list[str]:
    if prefix == "A":
        alphabet = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
        if count <= len(alphabet):
            return alphabet[:count]
    return [f"{prefix}{idx + 1}" for idx in range(count)]


def _build_quantiles(attributes: list[str], bins_per_dimension: int, value_domain_max: int) -> dict[str, list[float]]:
    step = value_domain_max / bins_per_dimension
    boundaries = [round(step * idx, 6) for idx in range(1, bins_per_dimension)]
    return {attr: boundaries[:] for attr in attributes}


def _build_query(attributes: list[str], selected_bins_per_dimension: int, bins_per_dimension: int, value_domain_max: int) -> str:
    threshold = round((value_domain_max / bins_per_dimension) * selected_bins_per_dimension, 6)
    threshold_str = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
    return " and ".join(f"{attr} < {threshold_str}" for attr in attributes)


def _build_team_layout(team_count: int, dimension: int, bins_per_dimension: int, selected_bins_per_dimension: int, value_domain_max: int, attribute_prefix: str) -> dict[str, Any]:
    attr_count = team_count * dimension
    attributes = _build_attribute_names(attr_count, prefix=attribute_prefix)
    teams = []
    for offset in range(0, attr_count, dimension):
        teams.append(tuple(attributes[offset:offset + dimension]))

    query = _build_query(attributes, selected_bins_per_dimension, bins_per_dimension, value_domain_max)
    quantiles = _build_quantiles(attributes, bins_per_dimension, value_domain_max)
    query_slices = {team: tuple(slice(0, selected_bins_per_dimension) for _ in range(dimension)) for team in teams}

    return {
        "attributes": attributes,
        "teams": teams,
        "query": query,
        "quantiles": quantiles,
        "query_slices": query_slices,
    }


def _t_rel_token(t_rel: float) -> str:
    return str(t_rel).replace(".", "")


def _query_note(team_count: int, dimension: int, t_rel: float) -> str:
    return (
        f"{team_count} Teams, {dimension}D, T_rel={t_rel:.2f}: "
        f"kontrollierte synthetische Query mit disjunkten Teams"
    )


def expand_experiment_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = config["defaults"]
    experiment_name = config["name"]
    output_root = Path(config.get("output_root", TEAM_BENCH_DATA_ROOT / experiment_name))
    benchmark_output_root = Path(config.get("benchmark_output_root", TEAM_BENCH_RESULTS_ROOT / experiment_name))

    bins_per_dimension = int(defaults["bins_per_dimension"])
    selected_bins_per_dimension = int(defaults["selected_bins_per_dimension"])
    value_domain_max = int(defaults.get("value_domain_max", 100))
    attribute_prefix = defaults.get("attribute_prefix", "A")
    n = int(defaults["n"])
    worker_count = int(defaults["worker_count"])
    repetitions = int(defaults["repetitions"])
    baseline_variant = defaults["baseline_variant"]
    strategies = list(defaults["strategies"])

    scenarios: list[dict[str, Any]] = []
    for team_count in defaults["team_counts"]:
        for dimension in defaults["dimensions"]:
            layout = _build_team_layout(
                team_count=int(team_count),
                dimension=int(dimension),
                bins_per_dimension=bins_per_dimension,
                selected_bins_per_dimension=selected_bins_per_dimension,
                value_domain_max=value_domain_max,
                attribute_prefix=attribute_prefix,
            )
            family_name = f"teams{int(team_count)}_dim{int(dimension)}"
            family_output_root = output_root / family_name

            for t_rel in defaults["t_rel_values"]:
                scenario_id = f"tb_{int(team_count)}t_{int(dimension)}d_t{int(round(float(t_rel) * 100)):03d}"
                t_rel_folder = f"selectivity_Trel{_t_rel_token(float(t_rel))}_N{n}"
                scenario_data_root = family_output_root / t_rel_folder
                scenarios.append(
                    {
                        "experiment_name": experiment_name,
                        "scenario_id": scenario_id,
                        "family_name": family_name,
                        "team_count": int(team_count),
                        "dimension": int(dimension),
                        "t_rel": float(t_rel),
                        "n": n,
                        "bins_per_dimension": bins_per_dimension,
                        "selected_bins_per_dimension": selected_bins_per_dimension,
                        "value_domain_max": value_domain_max,
                        "worker_count": worker_count,
                        "repetitions": repetitions,
                        "baseline_variant": baseline_variant,
                        "strategies": strategies,
                        "query": layout["query"],
                        "query_note": _query_note(int(team_count), int(dimension), float(t_rel)),
                        "attributes": layout["attributes"],
                        "teams": [list(team) for team in layout["teams"]],
                        "quantiles": layout["quantiles"],
                        "query_slices": {
                            "-".join(team): [[slc.start, slc.stop, slc.step] for slc in slices]
                            for team, slices in layout["query_slices"].items()
                        },
                        "data_root": str(scenario_data_root),
                        "index_config_path": str(scenario_data_root / "index.json"),
                        "benchmark_output_root": str(benchmark_output_root),
                    }
                )
    return scenarios


def group_scenarios_by_family(scenarios: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        grouped.setdefault(scenario["family_name"], []).append(scenario)
    return grouped


def family_team_specs(scenarios: list[dict[str, Any]]) -> dict[tuple[str, ...], tuple[int, ...]]:
    first = scenarios[0]
    dim = int(first["dimension"])
    bins = int(first["bins_per_dimension"])
    return {
        tuple(team): tuple([bins] * dim)
        for team in first["teams"]
    }


def family_query_slices(scenarios: list[dict[str, Any]]) -> dict[tuple[str, ...], tuple[slice, ...]]:
    first = scenarios[0]
    query_slices = {}
    for team_name, encoded_slices in first["query_slices"].items():
        query_slices[tuple(team_name.split("-"))] = tuple(
            slice(start, stop, step) for start, stop, step in encoded_slices
        )
    return query_slices


def family_quantiles(scenarios: list[dict[str, Any]]) -> dict[str, list[float]]:
    return scenarios[0]["quantiles"]


def family_query(scenarios: list[dict[str, Any]]) -> str:
    return scenarios[0]["query"]


def scenario_manifest_path(config: dict[str, Any]) -> Path:
    experiment_name = config["name"]
    return TEAM_BENCH_EXPERIMENTS_DIR / f"{experiment_name}-expanded.json"
