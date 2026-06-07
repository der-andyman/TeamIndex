#!/usr/bin/env python3
"""
Analysiert, warum current_handcrafted in einem team_bench-Lauf gut oder schlecht
abschneidet.

Das Skript liest vorhandene Ergebnis-CSVs und erzeugt daraus eine kompakte
Markdown-Notiz, die vor allem diese Fragen beleuchtet:

- In welchen Szenariofamilien gewinnt current_handcrafted?
- Wie gross ist der Laufzeitabstand zu Union First und Dynamic?
- Wie sehen die dabei gewaehlten Planformen aus?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGET_VARIANT = "current_handcrafted"
COMPARE_VARIANTS = ["baseline_union_first", "dynamic_selective_expansion"]


def parse_args():
    parser = argparse.ArgumentParser(description="Analysiere current_handcrafted auf Basis vorhandener team_bench-Ergebnisse.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Ordner mit results.csv, mopts_per_team.csv und summary_by_variant.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optionale Zieldatei fuer eine Markdown-Zusammenfassung. Standard: handcrafted_analysis.md im Ergebnisordner.",
    )
    return parser.parse_args()


def speedup_columns(summary_df: pd.DataFrame, base_variant: str, target_variant: str) -> pd.DataFrame:
    pivot = summary_df.pivot(index="scenario_id", columns="variant", values="runtime_ms_mean")
    if base_variant not in pivot.columns or target_variant not in pivot.columns:
        return pd.DataFrame(columns=["scenario_id", "speedup"])
    data = pivot[[base_variant, target_variant]].dropna().reset_index()
    data["speedup"] = data[base_variant] / data[target_variant]
    return data[["scenario_id", "speedup"]]


def scenario_meta(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df[
            ["scenario_id", "family_name", "team_count", "dimension", "t_rel", "worker_count", "n"]
        ]
        .drop_duplicates()
        .sort_values(["team_count", "dimension", "t_rel", "scenario_id"])
    )


def handcrafted_plan_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    handcrafted = results_df[results_df["variant"] == TARGET_VARIANT].copy()
    return (
        handcrafted
        .groupby(["scenario_id", "family_name", "team_count", "dimension", "t_rel"], as_index=False)
        .agg(
            runtime_ms_mean=("executor_runtime_ms", "mean"),
            runtime_ms_std=("executor_runtime_ms", "std"),
            expanded_team_count_mean=("expanded_team_count_manual", "mean"),
            ise_estimate_mean=("ise_count_estimate_manual", "mean"),
            total_selected_bin_cells_mean=("total_selected_bin_cells", "mean"),
            total_input_cardinality_mean=("total_input_cardinality", "mean"),
        )
        .sort_values(["team_count", "dimension", "t_rel", "scenario_id"])
    )


def handcrafted_team_level_summary(mopts_df: pd.DataFrame) -> pd.DataFrame:
    handcrafted = mopts_df[mopts_df["variant"] == TARGET_VARIANT].copy()
    handcrafted["expanded_team_name"] = handcrafted.apply(
        lambda row: row["team"] if bool(row["is_expanded"]) else "",
        axis=1,
    )
    return (
        handcrafted
        .groupby(["query_name", "variant"], as_index=False)
        .agg(
            expanded_team_count=("is_expanded", "sum"),
            expanded_teams=("expanded_team_name", lambda values: ",".join(sorted(filter(None, set(values))))),
            total_group_count=("group_count", "sum"),
            total_max_group_count=("max_group_count", "sum"),
        )
        .rename(columns={"query_name": "scenario_id"})
    )


def family_win_summary(summary_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    best_df = summary_df.loc[summary_df.groupby("scenario_id")["runtime_ms_mean"].idxmin()].copy()
    missing_cols = [col for col in ["family_name", "team_count", "dimension"] if col not in best_df.columns]
    if missing_cols:
        best_df = best_df.merge(
            scenario_meta(results_df)[["scenario_id", "family_name", "team_count", "dimension"]],
            on="scenario_id",
            how="left",
        )
    best_df["handcrafted_wins"] = best_df["variant"] == TARGET_VARIANT
    return (
        best_df
        .groupby(["family_name", "team_count", "dimension"], as_index=False)
        .agg(
            scenario_count=("scenario_id", "count"),
            handcrafted_win_count=("handcrafted_wins", "sum"),
        )
        .sort_values(["team_count", "dimension", "family_name"])
    )


def family_speedup_summary(summary_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    meta = scenario_meta(results_df)
    handcrafted = summary_df[summary_df["variant"] == TARGET_VARIANT][["scenario_id", "runtime_ms_mean"]].rename(
        columns={"runtime_ms_mean": "handcrafted_runtime_ms_mean"}
    )
    merged = meta.merge(handcrafted, on="scenario_id", how="left")

    for compare_variant in COMPARE_VARIANTS:
        compare_df = speedup_columns(summary_df, compare_variant, TARGET_VARIANT).rename(
            columns={"speedup": f"{compare_variant}_speedup_over_handcrafted"}
        )
        merged = merged.merge(compare_df, on="scenario_id", how="left")

    return (
        merged
        .groupby(["family_name", "team_count", "dimension"], as_index=False)
        .agg(
            handcrafted_runtime_ms_mean=("handcrafted_runtime_ms_mean", "mean"),
            baseline_union_first_speedup_over_handcrafted=("baseline_union_first_speedup_over_handcrafted", "mean"),
            dynamic_selective_expansion_speedup_over_handcrafted=("dynamic_selective_expansion_speedup_over_handcrafted", "mean"),
        )
        .sort_values(["team_count", "dimension", "family_name"])
    )


def build_report(results_df: pd.DataFrame, mopts_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    win_df = family_win_summary(summary_df, results_df)
    speedup_df = family_speedup_summary(summary_df, results_df)
    plan_df = handcrafted_plan_summary(results_df)
    team_df = handcrafted_team_level_summary(mopts_df)
    merged_plan_df = plan_df.merge(team_df, on="scenario_id", how="left").sort_values(
        ["team_count", "dimension", "t_rel", "scenario_id"]
    )

    lines = []
    lines.append("# Analyse: current_handcrafted")
    lines.append("")
    lines.append("## 1. In welchen Familien gewinnt current_handcrafted?")
    lines.append("")
    for row in win_df.itertuples(index=False):
        lines.append(
            f"- `{row.family_name}` ({row.team_count} Teams, {row.dimension}D): "
            f"{int(row.handcrafted_win_count)}/{int(row.scenario_count)} Siege"
        )

    lines.append("")
    lines.append("## 2. Mittlerer Vergleich zu Union First und Dynamic")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Wert > 1.0 bedeutet: Vergleichsstrategie ist langsamer als current_handcrafted.")
    lines.append("- Wert < 1.0 bedeutet: Vergleichsstrategie ist schneller als current_handcrafted.")
    lines.append("")
    for row in speedup_df.itertuples(index=False):
        lines.append(
            f"- `{row.family_name}`: "
            f"Union/Handcrafted={row.baseline_union_first_speedup_over_handcrafted:.3f}, "
            f"Dynamic/Handcrafted={row.dynamic_selective_expansion_speedup_over_handcrafted:.3f}, "
            f"Handcrafted-Laufzeit={row.handcrafted_runtime_ms_mean:.3f} ms"
        )

    lines.append("")
    lines.append("## 3. Welche Planformen waehlt current_handcrafted konkret?")
    lines.append("")
    for row in merged_plan_df.itertuples(index=False):
        lines.append(
            f"- `{row.scenario_id}`: expanded_teams={row.expanded_teams or '-'}, "
            f"expanded_team_count={row.expanded_team_count_mean:.2f}, "
            f"ise_estimate={row.ise_estimate_mean:.2f}, "
            f"group_sum={int(row.total_group_count)}, "
            f"max_group_sum={int(row.total_max_group_count)}, "
            f"selected_cells={row.total_selected_bin_cells_mean:.0f}, "
            f"input_ids={row.total_input_cardinality_mean:.0f}, "
            f"runtime={row.runtime_ms_mean:.3f} +- {0.0 if pd.isna(row.runtime_ms_std) else row.runtime_ms_std:.3f} ms"
        )

    lines.append("")
    lines.append("## 4. Wofuer diese Datei gut ist")
    lines.append("")
    lines.append("- Sie hilft dabei, current_handcrafted nicht nur als Gewinner/Verlierer zu sehen,")
    lines.append("  sondern als konkrete Folge von Expandierungs- und Gruppierungsentscheidungen.")
    lines.append("- Genau daraus koennen spaeter robustere Auswahlregeln fuer bestimmte Szenariofamilien entstehen.")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_path = args.output.resolve() if args.output is not None else results_dir / "handcrafted_analysis.md"

    results_df = pd.read_csv(results_dir / "results.csv")
    mopts_df = pd.read_csv(results_dir / "mopts_per_team.csv")
    summary_df = pd.read_csv(results_dir / "summary_by_variant.csv")

    report = build_report(results_df, mopts_df, summary_df)
    output_path.write_text(report, encoding="utf-8")

    print("Saved handcrafted analysis to:")
    print(output_path)


if __name__ == "__main__":
    main()
