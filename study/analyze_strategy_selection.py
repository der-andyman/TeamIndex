#!/usr/bin/env python3
"""
Analysiert team_bench-Laeufe mit Blick auf Strategiewahl.

Das Skript beantwortet nicht nur "welche Strategie war am schnellsten?",
sondern sammelt Hinweise fuer die eigentliche Optimizer-Frage:

- In welchen Szenariofamilien gewinnt welche Strategie?
- Wie stabil ist der Sieg im Vergleich zur gemessenen Varianz?
- Welche Planmerkmale unterscheiden Gewinner und Verlierer?
- Wo fehlen vermutlich noch Strategien zwischen den bisherigen Baselines?

Die Eingabe sind bereits erzeugte team_bench-Ergebnisordner mit mindestens
results.csv und summary_by_variant.csv. Es werden keine Benchmarks ausgefuehrt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_ROOT = Path("study/team_bench_results")
SUMMARY_FILE = "summary_by_variant.csv"
RESULTS_FILE = "results.csv"
SKIPPED_FILE = "skipped_variants.csv"

PLAN_METRIC_COLUMNS = [
    "included_team_count_manual",
    "expanded_team_count_manual",
    "sum_max_group_count",
    "sum_group_count",
    "min_max_group_count",
    "max_max_group_count",
    "imbalance_group_count",
    "sum_union_cardinality",
    "min_union_cardinality",
    "max_union_cardinality",
    "imbalance_union_cardinality",
    "ise_count_estimate_manual",
    "total_selected_bin_cells",
    "total_selected_attribute_bins",
    "total_input_cardinality",
    "total_read_volume_KiB",
    "total_request_count",
    "result_size",
    "ise_count",
    "outer_union_term_count",
    "outer_intersection_term_count",
]

META_COLUMNS = [
    "scenario_id",
    "family_name",
    "team_count",
    "dimension",
    "t_rel",
    "n",
    "worker_count",
]


@dataclass(frozen=True)
class RunData:
    run_dir: Path
    run_key: str
    outcomes: pd.DataFrame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analysiere vorhandene team_bench-Ergebnisse fuer Strategieauswahl."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Ein konkreter Ergebnisordner mit results.csv und summary_by_variant.csv. "
            "Mehrfach nutzbar. Ohne Angabe werden aktuelle Laeufe gesucht."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root-Ordner fuer automatische Suche nach Ergebnislaeufen.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Zielordner fuer Report und CSVs. Standard: study/team_bench_results/strategy_selection_analysis/<timestamp>.",
    )
    parser.add_argument(
        "--experiment-filter",
        action="append",
        default=[],
        help="Nur automatisch gefundene Experimente beruecksichtigen, deren Name dieses Teilstueck enthaelt.",
    )
    return parser.parse_args()


def run_has_required_files(run_dir: Path) -> bool:
    return (run_dir / RESULTS_FILE).exists() and (run_dir / SUMMARY_FILE).exists()


def infer_worker_count(run_dir: Path) -> str:
    summary_path = run_dir / SUMMARY_FILE
    try:
        summary_df = pd.read_csv(summary_path, usecols=["worker_count"])
    except Exception:
        return "unknown"
    values = sorted(str(int(value)) for value in summary_df["worker_count"].dropna().unique())
    return values[0] if len(values) == 1 else "mixed"


def discover_latest_result_dirs(results_root: Path, experiment_filters: list[str]) -> list[Path]:
    if not results_root.exists():
        raise RuntimeError(f"Results root does not exist: {results_root}")

    run_dirs = [
        path
        for path in results_root.glob("*/*")
        if path.is_dir() and run_has_required_files(path)
    ]
    if experiment_filters:
        run_dirs = [
            path for path in run_dirs
            if any(token in path.parent.name for token in experiment_filters)
        ]

    latest_by_experiment_and_worker: dict[tuple[str, str], Path] = {}
    for run_dir in run_dirs:
        worker_label = infer_worker_count(run_dir)
        key = (run_dir.parent.name, worker_label)
        current = latest_by_experiment_and_worker.get(key)
        if current is None or run_dir.name > current.name:
            latest_by_experiment_and_worker[key] = run_dir

    return sorted(latest_by_experiment_and_worker.values())


def read_run_dirs(args) -> list[Path]:
    if args.results_dir:
        run_dirs = [path.resolve() for path in args.results_dir]
    else:
        run_dirs = discover_latest_result_dirs(args.results_root.resolve(), args.experiment_filter)

    missing = [path for path in run_dirs if not run_has_required_files(path)]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise RuntimeError(f"Diese Ergebnisordner sind unvollstaendig:\n{formatted}")
    if not run_dirs:
        raise RuntimeError("Keine passenden team_bench-Ergebnisordner gefunden.")
    return run_dirs


def add_run_columns(df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["experiment_name"] = run_dir.parent.name
    df["run_name"] = run_dir.name
    df["run_dir"] = str(run_dir)
    df["run_key"] = f"{run_dir.parent.name}/{run_dir.name}"
    return df


def aggregate_plan_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    available_metrics = [col for col in PLAN_METRIC_COLUMNS if col in results_df.columns]
    group_cols = ["scenario_id", "variant"]
    agg_spec = {col: (col, "mean") for col in available_metrics}
    return results_df.groupby(group_cols, as_index=False).agg(**agg_spec)


def build_skipped_outcomes(skipped_df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    if skipped_df.empty:
        return pd.DataFrame()

    group_cols = [
        col for col in [
            "scenario_id",
            "family_name",
            "team_count",
            "dimension",
            "t_rel",
            "worker_count",
            "variant",
        ]
        if col in skipped_df.columns
    ]
    skipped_summary = (
        skipped_df
        .groupby(group_cols, as_index=False)
        .agg(
            skipped_count=("repetition", "nunique"),
            skip_reason=("reason", lambda values: ",".join(sorted(set(map(str, values))))),
            ise_count_estimate_manual=("ise_count_estimate_manual", "max"),
            max_ise_count=("max_ise_count", "max"),
        )
    )
    skipped_summary["is_skipped"] = True
    skipped_summary["runtime_ms_mean"] = pd.NA
    skipped_summary["runtime_ms_std"] = pd.NA
    skipped_summary["runtime_ms_rel_std"] = pd.NA
    skipped_summary["ids_per_second_mean"] = pd.NA
    skipped_summary["read_mib_per_second_mean"] = pd.NA
    skipped_summary["n"] = pd.NA
    return add_run_columns(skipped_summary, run_dir)


def load_run(run_dir: Path) -> RunData:
    results_df = add_run_columns(pd.read_csv(run_dir / RESULTS_FILE), run_dir)
    summary_df = add_run_columns(pd.read_csv(run_dir / SUMMARY_FILE), run_dir)

    plan_df = aggregate_plan_metrics(results_df)
    outcomes = summary_df.merge(plan_df, on=["scenario_id", "variant"], how="left")
    outcomes["is_skipped"] = False
    outcomes["skipped_count"] = 0
    outcomes["skip_reason"] = ""
    outcomes["max_ise_count"] = pd.NA

    skipped_path = run_dir / SKIPPED_FILE
    if skipped_path.exists() and skipped_path.stat().st_size > 0:
        try:
            skipped_df = pd.read_csv(skipped_path)
        except pd.errors.EmptyDataError:
            skipped_df = pd.DataFrame()
        skipped_outcomes = build_skipped_outcomes(skipped_df, run_dir)
        if not skipped_outcomes.empty:
            outcomes = pd.concat([outcomes, skipped_outcomes], ignore_index=True, sort=False)

    run_key = f"{run_dir.parent.name}/{run_dir.name}"
    return RunData(run_dir=run_dir, run_key=run_key, outcomes=outcomes)


def winner_rows(outcomes: pd.DataFrame) -> pd.DataFrame:
    runnable = outcomes[
        (~outcomes["is_skipped"].fillna(False))
        & outcomes["runtime_ms_mean"].notna()
    ].copy()
    rows = []

    for (run_key, scenario_id), group in runnable.groupby(["run_key", "scenario_id"], sort=False):
        ordered = group.sort_values("runtime_ms_mean").reset_index(drop=True)
        winner = ordered.iloc[0]
        runner = ordered.iloc[1] if len(ordered) > 1 else None
        runner_runtime = pd.NA if runner is None else runner["runtime_ms_mean"]
        margin = pd.NA if runner is None else runner["runtime_ms_mean"] / winner["runtime_ms_mean"]
        winner_std = winner.get("runtime_ms_std", pd.NA)
        runner_std = pd.NA if runner is None else runner.get("runtime_ms_std", pd.NA)
        std_gap = pd.NA
        variance_clear = False
        if runner is not None and pd.notna(winner_std) and pd.notna(runner_std):
            std_gap = runner["runtime_ms_mean"] - winner["runtime_ms_mean"] - winner_std - runner_std
            variance_clear = bool(std_gap > 0)

        row = {
            "run_key": run_key,
            "experiment_name": winner["experiment_name"],
            "run_name": winner["run_name"],
            "scenario_id": scenario_id,
            "family_name": winner.get("family_name", ""),
            "team_count": winner.get("team_count", pd.NA),
            "dimension": winner.get("dimension", pd.NA),
            "t_rel": winner.get("t_rel", pd.NA),
            "n": winner.get("n", pd.NA),
            "worker_count": winner.get("worker_count", pd.NA),
            "winner_variant": winner["variant"],
            "winner_runtime_ms": winner["runtime_ms_mean"],
            "winner_runtime_ms_std": winner_std,
            "winner_runtime_rel_std": winner.get("runtime_ms_rel_std", pd.NA),
            "runner_up_variant": pd.NA if runner is None else runner["variant"],
            "runner_up_runtime_ms": runner_runtime,
            "runner_up_runtime_ms_std": runner_std,
            "winner_margin": margin,
            "variance_clear": variance_clear,
            "std_gap_ms": std_gap,
            "winner_expanded_team_count": winner.get("expanded_team_count_manual", pd.NA),
            "winner_ise_estimate": winner.get("ise_count_estimate_manual", pd.NA),
            "winner_sum_group_count": winner.get("sum_group_count", pd.NA),
            "winner_total_selected_bin_cells": winner.get("total_selected_bin_cells", pd.NA),
            "winner_total_input_cardinality": winner.get("total_input_cardinality", pd.NA),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def family_strategy_summary(outcomes: pd.DataFrame, winners: pd.DataFrame) -> pd.DataFrame:
    win_counts = (
        winners
        .groupby(
            ["experiment_name", "worker_count", "team_count", "dimension", "winner_variant"],
            as_index=False,
        )
        .agg(win_count=("scenario_id", "count"))
        .rename(columns={"winner_variant": "variant"})
    )

    runtime_summary = (
        outcomes[
            (~outcomes["is_skipped"].fillna(False))
            & outcomes["runtime_ms_mean"].notna()
        ]
        .groupby(
            ["experiment_name", "worker_count", "team_count", "dimension", "variant"],
            as_index=False,
        )
        .agg(
            scenario_count=("scenario_id", "nunique"),
            runtime_ms_mean=("runtime_ms_mean", "mean"),
            runtime_ms_rel_std_mean=("runtime_ms_rel_std", "mean"),
            ise_estimate_mean=("ise_count_estimate_manual", "mean"),
            expanded_team_count_mean=("expanded_team_count_manual", "mean"),
            total_selected_bin_cells_mean=("total_selected_bin_cells", "mean"),
            total_input_cardinality_mean=("total_input_cardinality", "mean"),
        )
    )
    skipped_counts = (
        outcomes[outcomes["is_skipped"].fillna(False)]
        .groupby(
            ["experiment_name", "worker_count", "team_count", "dimension", "variant"],
            as_index=False,
        )
        .agg(skipped_scenario_count=("scenario_id", "nunique"))
    )

    summary = runtime_summary.merge(
        win_counts,
        on=["experiment_name", "worker_count", "team_count", "dimension", "variant"],
        how="left",
    )
    summary = summary.merge(
        skipped_counts,
        on=["experiment_name", "worker_count", "team_count", "dimension", "variant"],
        how="left",
    )
    summary["win_count"] = summary["win_count"].fillna(0).astype(int)
    summary["skipped_scenario_count"] = summary["skipped_scenario_count"].fillna(0).astype(int)
    return summary.sort_values(["experiment_name", "worker_count", "team_count", "dimension", "variant"])


def pivot_runtime(outcomes: pd.DataFrame) -> pd.DataFrame:
    runnable = outcomes[
        (~outcomes["is_skipped"].fillna(False))
        & outcomes["runtime_ms_mean"].notna()
    ].copy()
    pivot = runnable.pivot_table(
        index=["run_key", "scenario_id"],
        columns="variant",
        values="runtime_ms_mean",
        aggfunc="first",
    )
    return pivot.reset_index()


def compare_to_handcrafted(outcomes: pd.DataFrame) -> pd.DataFrame:
    runtime_pivot = pivot_runtime(outcomes)
    if "current_handcrafted" not in runtime_pivot.columns:
        return pd.DataFrame()

    meta_cols = [
        "run_key",
        "scenario_id",
        "experiment_name",
        "run_name",
        "family_name",
        "team_count",
        "dimension",
        "t_rel",
        "worker_count",
    ]
    meta = outcomes[meta_cols].drop_duplicates(["run_key", "scenario_id"])
    comparison = runtime_pivot.merge(meta, on=["run_key", "scenario_id"], how="left")

    for variant in [
        "baseline_union_first",
        "baseline_minimal_intersection",
        "dynamic_selective_expansion",
        "expand_all_adaptive_grouping",
    ]:
        if variant in comparison.columns:
            comparison[f"{variant}_runtime_ratio_vs_handcrafted"] = (
                comparison[variant] / comparison["current_handcrafted"]
            )
    return comparison


def scenario_feature_table(outcomes: pd.DataFrame, winners: pd.DataFrame) -> pd.DataFrame:
    runnable = outcomes[
        (~outcomes["is_skipped"].fillna(False))
        & outcomes["runtime_ms_mean"].notna()
    ].copy()
    if runnable.empty:
        return pd.DataFrame()

    neutral = runnable[runnable["variant"] == "baseline_union_first"].copy()
    if neutral.empty:
        neutral = runnable.sort_values("variant").drop_duplicates(["run_key", "scenario_id"])

    feature_cols = [
        "run_key",
        "experiment_name",
        "run_name",
        "scenario_id",
        "family_name",
        "team_count",
        "dimension",
        "t_rel",
        "n",
        "worker_count",
        "sum_union_cardinality",
        "min_union_cardinality",
        "max_union_cardinality",
        "imbalance_union_cardinality",
        "total_selected_bin_cells",
        "total_selected_attribute_bins",
        "total_input_cardinality",
        "total_read_volume_KiB",
        "total_request_count",
        "result_size",
    ]
    feature_cols = [col for col in feature_cols if col in neutral.columns]
    features = neutral[feature_cols].drop_duplicates(["run_key", "scenario_id"]).copy()
    features["selected_cells_per_team"] = (
        features["total_selected_bin_cells"] / features["team_count"]
    )
    features["union_ids_per_selected_cell"] = (
        features["sum_union_cardinality"] / features["total_selected_bin_cells"]
    )
    features["input_ids_per_selected_cell"] = (
        features["total_input_cardinality"] / features["total_selected_bin_cells"]
    )

    winner_cols = [
        "run_key",
        "scenario_id",
        "winner_variant",
        "runner_up_variant",
        "winner_margin",
        "variance_clear",
        "winner_runtime_ms",
        "runner_up_runtime_ms",
    ]
    return features.merge(winners[winner_cols], on=["run_key", "scenario_id"], how="left")


def format_float(value, digits=3) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def append_winner_map(lines: list[str], winners: pd.DataFrame):
    lines.append("## 1. Gewinnerkarte")
    lines.append("")
    grouped = (
        winners
        .groupby(["experiment_name", "worker_count", "team_count", "dimension", "winner_variant"], as_index=False)
        .agg(win_count=("scenario_id", "count"))
        .sort_values(["experiment_name", "worker_count", "team_count", "dimension", "winner_variant"])
    )
    for row in grouped.itertuples(index=False):
        lines.append(
            f"- `{row.experiment_name}`, w{int(row.worker_count)}, "
            f"{int(row.team_count)}T-{int(row.dimension)}D: "
            f"`{row.winner_variant}` gewinnt {int(row.win_count)} Szenarien"
        )
    lines.append("")


def append_margin_and_variance(lines: list[str], winners: pd.DataFrame):
    lines.append("## 2. Wie belastbar sind die Siege?")
    lines.append("")
    if winners.empty:
        lines.append("- Keine Gewinnerzeilen vorhanden.")
        lines.append("")
        return

    fragile = winners[
        (winners["winner_margin"].notna())
        & (
            (winners["winner_margin"] < 1.05)
            | (~winners["variance_clear"].fillna(False))
        )
    ].copy()
    stable_count = int(winners["variance_clear"].fillna(False).sum())
    lines.append(
        f"- {stable_count}/{len(winners)} Gewinner liegen mit ihrem Mittelwert weiter vorne als "
        "die Summe der Standardabweichungen von Gewinner und Runner-up."
    )
    lines.append(
        f"- {len(fragile)}/{len(winners)} Faelle sind knapp oder varianzempfindlich "
        "(Margin < 1.05 oder Varianz ueberlappt)."
    )
    if not fragile.empty:
        lines.append("- Knappste/unsicherste Faelle:")
        for row in fragile.sort_values(["winner_margin", "std_gap_ms"], na_position="last").head(10).itertuples(index=False):
            lines.append(
                f"- `{row.scenario_id}` in `{row.run_name}`: "
                f"`{row.winner_variant}` vor `{row.runner_up_variant}`, "
                f"Margin={format_float(row.winner_margin)}, "
                f"Std-Gap={format_float(row.std_gap_ms)} ms"
            )
    lines.append("")


def append_plan_signal(lines: list[str], family_summary: pd.DataFrame):
    lines.append("## 3. Plan-Signale")
    lines.append("")
    compact = family_summary.sort_values(
        ["experiment_name", "worker_count", "team_count", "dimension", "runtime_ms_mean"]
    )
    for row in compact.itertuples(index=False):
        lines.append(
            f"- `{row.experiment_name}`, w{int(row.worker_count)}, "
            f"{int(row.team_count)}T-{int(row.dimension)}D, `{row.variant}`: "
            f"Wins={int(row.win_count)}/{int(row.scenario_count)}, "
            f"Runtime={format_float(row.runtime_ms_mean)} ms, "
            f"ISE={format_float(row.ise_estimate_mean, 1)}, "
            f"expanded={format_float(row.expanded_team_count_mean, 1)}, "
            f"CV={format_float(row.runtime_ms_rel_std_mean)}"
        )
    lines.append("")


def append_handcrafted_comparison(lines: list[str], comparison: pd.DataFrame):
    lines.append("## 4. Handcrafted gegen Union/Dynamic")
    lines.append("")
    if comparison.empty:
        lines.append("- Kein Vergleich moeglich, weil `current_handcrafted` fehlt.")
        lines.append("")
        return

    ratio_cols = [
        col for col in comparison.columns
        if col.endswith("_runtime_ratio_vs_handcrafted")
    ]
    summary = comparison.groupby(["experiment_name", "worker_count", "team_count", "dimension"], as_index=False).agg(
        **{col: (col, "mean") for col in ratio_cols}
    )
    lines.append("Interpretation: Wert > 1 bedeutet, dass die Vergleichsstrategie langsamer ist als `current_handcrafted`.")
    for row in summary.sort_values(["experiment_name", "worker_count", "team_count", "dimension"]).itertuples(index=False):
        values = []
        for col in ratio_cols:
            label = col.replace("_runtime_ratio_vs_handcrafted", "")
            values.append(f"{label}={format_float(getattr(row, col))}")
        lines.append(
            f"- `{row.experiment_name}`, w{int(row.worker_count)}, "
            f"{int(row.team_count)}T-{int(row.dimension)}D: "
            + ", ".join(values)
        )
    lines.append("")


def append_strategy_gaps(lines: list[str], outcomes: pd.DataFrame, comparison: pd.DataFrame, winners: pd.DataFrame):
    lines.append("## 5. Was uns das fuer neue Strategien sagt")
    lines.append("")

    if not comparison.empty and "dynamic_selective_expansion_runtime_ratio_vs_handcrafted" in comparison.columns:
        dynamic_bad = comparison[
            comparison["dynamic_selective_expansion_runtime_ratio_vs_handcrafted"] > 1.10
        ]
        if not dynamic_bad.empty:
            lines.append(
                f"- In {len(dynamic_bad)} Faellen ist `dynamic_selective_expansion` mehr als 10 Prozent "
                "langsamer als `current_handcrafted`. Das spricht fuer eine Strategie zwischen beiden: "
                "`bounded_selective_expansion`, also selektiv expandieren, aber mit klarer ISE-/Gruppengrenze."
            )

    if not comparison.empty and "baseline_union_first_runtime_ratio_vs_handcrafted" in comparison.columns:
        union_better = comparison[
            comparison["baseline_union_first_runtime_ratio_vs_handcrafted"] < 0.98
        ]
        if not union_better.empty:
            examples = ", ".join(f"`{sid}`" for sid in union_better["scenario_id"].head(5))
            lines.append(
                f"- In {len(union_better)} Faellen ist Union First messbar schneller als `current_handcrafted` "
                f"(z.B. {examples}). Daraus folgt: Die Strategiewahl braucht einen echten No-Expansion-Zweig."
            )

    skipped = outcomes[outcomes["is_skipped"].fillna(False)]
    if not skipped.empty:
        skipped_scenarios = skipped.drop_duplicates(["run_key", "scenario_id", "variant"])
        lines.append(
            f"- {len(skipped_scenarios)} Szenario/Strategie-Kombinationen wurden aus Sicherheitsgruenden "
            "uebersprungen. Ungebremste Expansion ist damit kein realistischer Kandidat fuer groessere Dimensionen."
        )

    minimal_wins = winners[winners["winner_variant"] == "baseline_minimal_intersection"]
    if minimal_wins.empty:
        lines.append(
            "- `baseline_minimal_intersection` gewinnt in den betrachteten Hauptlaeufen praktisch nicht. "
            "Als Idee bleibt sie nuetzlich, aber nur mit Gruppierung oder Safety-Limit."
        )

    lines.append("")
    lines.append("Erster Regelentwurf fuer die spaetere Strategiewahl:")
    lines.append("- 2D: als Kontroll-/Randfall behandeln; nur dann hart entscheiden, wenn Margin und Varianz klar sind.")
    lines.append("- 3D bis 5D mit vielen getroffenen Zellen: eine gebremste Ein-Team-Expansion ist aktuell der staerkste Kandidat.")
    lines.append("- Wenn eine Strategie sehr hohe ISE Counts erzeugt, vor der Ausfuehrung pruefen und notfalls verwerfen.")
    lines.append("- Wenn Union First knapp oder klar gewinnt, nicht expandieren; Expansion ist dann nur Overhead.")
    lines.append("")


def build_report(outcomes: pd.DataFrame, winners: pd.DataFrame, family_summary: pd.DataFrame, comparison: pd.DataFrame) -> str:
    lines = []
    lines.append("# Strategieauswahl: Analyse vorhandener team_bench-Laeufe")
    lines.append("")
    lines.append(f"Erzeugt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Ausgewertete Runs:")
    for run_name in sorted(outcomes["run_key"].unique()):
        lines.append(f"- `{run_name}`")
    lines.append("")
    append_winner_map(lines, winners)
    append_margin_and_variance(lines, winners)
    append_plan_signal(lines, family_summary)
    append_handcrafted_comparison(lines, comparison)
    append_strategy_gaps(lines, outcomes, comparison, winners)
    return "\n".join(lines)


def write_outputs(output_dir: Path, outcomes: pd.DataFrame, winners: pd.DataFrame):
    output_dir.mkdir(parents=True, exist_ok=True)
    family_summary = family_strategy_summary(outcomes, winners)
    comparison = compare_to_handcrafted(outcomes)
    features = scenario_feature_table(outcomes, winners)
    report = build_report(outcomes, winners, family_summary, comparison)

    outcomes.to_csv(output_dir / "combined_variant_outcomes.csv", index=False)
    winners.to_csv(output_dir / "scenario_winners.csv", index=False)
    family_summary.to_csv(output_dir / "family_strategy_summary.csv", index=False)
    if not features.empty:
        features.to_csv(output_dir / "scenario_feature_table.csv", index=False)
    if not comparison.empty:
        comparison.to_csv(output_dir / "handcrafted_runtime_comparison.csv", index=False)
    (output_dir / "strategy_selection_report.md").write_text(report, encoding="utf-8")


def main():
    args = parse_args()
    run_dirs = read_run_dirs(args)
    runs = [load_run(run_dir) for run_dir in run_dirs]
    outcomes = pd.concat([run.outcomes for run in runs], ignore_index=True, sort=False)
    winners = winner_rows(outcomes)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = DEFAULT_RESULTS_ROOT / "strategy_selection_analysis" / timestamp
    else:
        output_dir = args.output_dir

    write_outputs(output_dir.resolve(), outcomes, winners)
    print("Analysierte Runs:")
    for run_dir in run_dirs:
        print(f"  {run_dir}")
    print("\nErzeugt:")
    print(output_dir.resolve() / "strategy_selection_report.md")
    print(output_dir.resolve() / "combined_variant_outcomes.csv")
    print(output_dir.resolve() / "scenario_winners.csv")
    print(output_dir.resolve() / "family_strategy_summary.csv")
    print(output_dir.resolve() / "scenario_feature_table.csv")


if __name__ == "__main__":
    main()
