#!/usr/bin/env python3
"""
Erzeugt Vergleichs-PDFs aus bereits vorhandenen Ergebnis-CSVs.

Das Skript fuehrt selbst keinen Benchmark aus. Es laedt nur eine vorhandene
`results.csv` und erstellt daraus die zusammenfassenden Plots neu.

Unterstuetzte Modi:
- mopts-Studie unter `study/results/`
- team_bench-Ergebnisse unter `study/team_bench_results/<experiment>/`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Erzeuge Vergleichs-PDFs aus vorhandenen Ergebnis-CSVs.")
    parser.add_argument(
        "--mode",
        choices=["auto", "mopts", "team_bench"],
        default="auto",
        help="Welcher Ergebnistyp geplottet werden soll. 'auto' erkennt den Typ aus der CSV.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Ordner mit results.csv. Standard: mopts -> study/results, team_bench -> explizit angeben.",
    )
    parser.add_argument(
        "--baseline-variant",
        default=None,
        help="Optionale Baseline fuer Speedup-Plots. Standard: baseline_union_first, sonst baseline_minimal_intersection.",
    )
    return parser.parse_args()


def detect_mode(results_df: pd.DataFrame) -> str:
    if "scenario_id" in results_df.columns:
        return "team_bench"
    return "mopts"


def choose_baseline_variant(variants: list[str], requested: str | None) -> str:
    if requested is not None:
        if requested not in variants:
            raise RuntimeError(f"Requested baseline variant not found: {requested}")
        return requested
    if "baseline_union_first" in variants:
        return "baseline_union_first"
    if "baseline_minimal_intersection" in variants:
        return "baseline_minimal_intersection"
    return sorted(variants)[0]


def sanitize_filename(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_")


def short_scenario_label(row: pd.Series) -> str:
    return f"{int(row['team_count'])}T-{int(row['dimension'])}D-{float(row['t_rel']):.2f}"


def team_bench_tick_notes(summary_df: pd.DataFrame) -> str:
    scenario_meta = (
        summary_df[["scenario_id", "family_name", "team_count", "dimension", "t_rel"]]
        .drop_duplicates()
        .sort_values(["family_name", "t_rel"])
    )
    notes = [
        f"{short_scenario_label(row)} = {row['family_name']}, T_rel={float(row['t_rel']):.2f}"
        for _, row in scenario_meta.iterrows()
    ]
    lines = ["   |   ".join(notes[i:i + 3]) for i in range(0, len(notes), 3)]
    return "\n".join(lines)


def plot_grouped_metric(summary_df: pd.DataFrame, metric_col: str, title: str, ylabel: str, output_path: Path):
    pivot = summary_df.pivot(index="scenario_id", columns="variant", values=metric_col)
    scenario_meta = (
        summary_df[["scenario_id", "family_name", "t_rel", "team_count", "dimension"]]
        .drop_duplicates()
        .set_index("scenario_id")
        .loc[pivot.index]
    )
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.index) * 0.9), 7))
    for idx, variant in enumerate(pivot.columns):
        values = pivot[variant].to_numpy()
        ax.bar(x + idx * width - ((len(pivot.columns) - 1) * width / 2), values, width=width, label=variant)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Szenario")
    ax.set_xticks(x)
    ax.set_xticklabels([short_scenario_label(scenario_meta.loc[name]) for name in pivot.index], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    note_text = team_bench_tick_notes(summary_df)
    fig.text(0.5, 0.01, note_text, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_best_strategy_heatmap(summary_df: pd.DataFrame, output_path: Path):
    best_df = summary_df.loc[summary_df.groupby("scenario_id")["runtime_ms_mean"].idxmin()].copy()
    families = sorted(best_df["family_name"].unique())
    t_rel_values = sorted(best_df["t_rel"].unique())
    strategies = sorted(best_df["variant"].unique())
    strategy_to_id = {name: idx for idx, name in enumerate(strategies)}

    matrix = np.full((len(families), len(t_rel_values)), np.nan)
    label_matrix = [["" for _ in t_rel_values] for _ in families]

    for _, row in best_df.iterrows():
        y = families.index(row["family_name"])
        x = t_rel_values.index(row["t_rel"])
        matrix[y, x] = strategy_to_id[row["variant"]]
        label_matrix[y][x] = row["variant"].replace("baseline_", "b_").replace("current_", "cur_").replace("dynamic_", "dyn_")

    cmap = plt.cm.get_cmap("tab10", len(strategies))
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-0.5, vmax=len(strategies) - 0.5)
    ax.set_title("Beste Strategie pro team_bench-Szenario")
    ax.set_xlabel("T_rel")
    ax.set_ylabel("Szenariofamilie")
    ax.set_xticks(np.arange(len(t_rel_values)))
    ax.set_xticklabels([f"{value:.2f}" for value in t_rel_values])
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels(families)

    for y in range(len(families)):
        for x in range(len(t_rel_values)):
            if label_matrix[y][x]:
                ax.text(x, y, label_matrix[y][x], ha="center", va="center", fontsize=8, color="black")

    patches = [mpatches.Patch(color=cmap(idx), label=name) for name, idx in strategy_to_id.items()]
    ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_team_bench_plots(results_dir: Path, results_df: pd.DataFrame, baseline_variant: str | None):
    summary_df = (
        results_df
        .groupby(["scenario_id", "family_name", "team_count", "dimension", "t_rel", "variant"], as_index=False)
        .agg(
            runtime_ms_mean=("executor_runtime_ms", "mean"),
            ids_per_second_mean=("ids_per_second", "mean"),
            read_mib_per_second_mean=("read_mib_per_second", "mean"),
        )
    )

    baseline = choose_baseline_variant(summary_df["variant"].unique().tolist(), baseline_variant)
    baseline_df = summary_df[summary_df["variant"] == baseline][["scenario_id", "runtime_ms_mean"]].rename(
        columns={"runtime_ms_mean": "baseline_runtime_ms_mean"}
    )
    summary_df = summary_df.merge(baseline_df, on="scenario_id", how="left")
    summary_df["speedup_vs_baseline_runtime"] = summary_df["baseline_runtime_ms_mean"] / summary_df["runtime_ms_mean"]

    plot_paths = []
    plot_paths.append(
        plot_grouped_metric(
            summary_df,
            metric_col="runtime_ms_mean",
            title="team_bench: Laufzeitvergleich",
            ylabel="Laufzeit [ms]",
            output_path=results_dir / "runtime_comparison.pdf",
        )
    )
    plot_paths.append(
        plot_grouped_metric(
            summary_df,
            metric_col="speedup_vs_baseline_runtime",
            title=f"team_bench: Speedup relativ zu {baseline}",
            ylabel="Speedup relativ zur Baseline",
            output_path=results_dir / "speedup_vs_baseline_runtime.pdf",
        )
    )
    plot_paths.append(
        plot_grouped_metric(
            summary_df,
            metric_col="ids_per_second_mean",
            title="team_bench: IDs pro Sekunde",
            ylabel="IDs pro Sekunde",
            output_path=results_dir / "ids_per_second_comparison.pdf",
        )
    )
    plot_paths.append(
        plot_grouped_metric(
            summary_df,
            metric_col="read_mib_per_second_mean",
            title="team_bench: MiB pro Sekunde",
            ylabel="MiB pro Sekunde",
            output_path=results_dir / "mib_per_second_comparison.pdf",
        )
    )
    plot_paths.append(
        plot_best_strategy_heatmap(
            summary_df,
            output_path=results_dir / "best_strategy_heatmap.pdf",
        )
    )
    return plot_paths


def generate_mopts_plots(results_dir: Path, results_df: pd.DataFrame):
    import mopts_study as ms
    if results_dir != ms.OUT_DIR:
        # Temporarily point the plotting helper to a different directory if needed.
        ms.OUT_DIR = results_dir
    return ms.generate_all_summary_plots(results_df)


def main():
    args = parse_args()

    if args.results_dir is None:
        import mopts_study as ms
        results_dir = ms.OUT_DIR
    else:
        results_dir = args.results_dir.resolve()

    results_csv = results_dir / "results.csv"
    if not results_csv.exists():
        raise RuntimeError(f"Missing results file: {results_csv}")

    results_df = pd.read_csv(results_csv)
    mode = args.mode if args.mode != "auto" else detect_mode(results_df)

    if mode == "mopts":
        plot_paths = generate_mopts_plots(results_dir, results_df)
    else:
        plot_paths = generate_team_bench_plots(results_dir, results_df, args.baseline_variant)

    print("Regenerated comparison PDFs from:")
    print(results_csv)
    print(f"Detected mode: {mode}")
    print("\nCreated:")
    for plot_path in plot_paths:
        print(plot_path)


if __name__ == "__main__":
    main()
