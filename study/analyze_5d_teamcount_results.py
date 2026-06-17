#!/usr/bin/env python3
"""
Konsolidiert den 5D-Teamanzahl-Block aus einzelnen team_bench-Runs.

Das Skript fuehrt keine Benchmarks aus. Es liest alle vorhandenen
`summary_by_variant.csv`-Dateien eines Experiments, waehlt pro Szenario den
neuesten Run und erzeugt daraus:

- eine zusammengefuehrte CSV,
- thesisfaehige LaTeX-Tabellen,
- PDF-Plots fuer Laufzeiten, Speedups und Gewinner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = (
    BASE_DIR
    / "study"
    / "team_bench_results"
    / "team_bench_bins20_hit8_5d_teams2345_n1m"
)
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_ROOT / "combined_5d_teamcount"
DEFAULT_THESIS_DIR = BASE_DIR / "thesis"

VARIANT_ORDER = [
    "baseline_union_first",
    "union_first_parallel",
    "current_handcrafted",
    "bounded_selective_expansion",
    "dynamic_selective_expansion",
]

VARIANT_LABELS = {
    "baseline_union_first": "Union First",
    "union_first_parallel": "UF Parallel",
    "current_handcrafted": "Handcrafted",
    "bounded_selective_expansion": "Bounded Expansion",
    "dynamic_selective_expansion": "Dynamic Selective",
}

VARIANT_COLORS = {
    "baseline_union_first": "#8b8f97",
    "union_first_parallel": "#0f766e",
    "current_handcrafted": "#c2410c",
    "bounded_selective_expansion": "#ca8a04",
    "dynamic_selective_expansion": "#2563eb",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Erzeuge konsolidierte Tabellen und Plots fuer den 5D-Teamanzahl-Block."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Ordner mit Run-Unterordnern des 5D-Teamanzahl-Experiments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Zielordner fuer konsolidierte CSVs und PDFs.",
    )
    parser.add_argument(
        "--thesis-dir",
        type=Path,
        default=DEFAULT_THESIS_DIR,
        help="Optionaler Thesis-Ordner, in den Figuren und Tabellen kopiert werden.",
    )
    parser.add_argument(
        "--no-thesis-copy",
        action="store_true",
        help="Nur in output-dir schreiben, nichts nach thesis/ kopieren.",
    )
    return parser.parse_args()


def load_latest_summaries(experiment_root: Path) -> pd.DataFrame:
    frames = []
    pattern = r"tb_[2345]t_5d_t(010|035|060|085)_uniform"
    for path in sorted(experiment_root.glob("*/summary_by_variant.csv")):
        df = pd.read_csv(path)
        if df.empty or "scenario_id" not in df.columns:
            continue
        df = df[df["scenario_id"].astype(str).str.match(pattern)].copy()
        if df.empty:
            continue
        df["run_dir"] = path.parent.name
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No matching 5D team-count summaries found below {experiment_root}")

    all_rows = pd.concat(frames, ignore_index=True)
    latest_runs = (
        all_rows[["scenario_id", "run_dir"]]
        .drop_duplicates()
        .sort_values("run_dir")
        .groupby("scenario_id", as_index=False)
        .tail(1)
    )
    latest = all_rows.merge(latest_runs, on=["scenario_id", "run_dir"], how="inner")
    latest = latest[latest["variant"].isin(VARIANT_ORDER)].copy()
    latest["variant"] = pd.Categorical(latest["variant"], categories=VARIANT_ORDER, ordered=True)
    latest = latest.sort_values(["team_count", "t_rel", "variant"]).reset_index(drop=True)
    return latest


def latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def write_latex_table(path: Path, caption: str, label: str, columns: list[str], rows: list[list[object]]):
    align = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_runtime_table(summary: pd.DataFrame, output_path: Path):
    pivot = summary.pivot_table(
        index=["team_count", "t_rel"],
        columns="variant",
        values="runtime_ms_mean",
        aggfunc="first",
    )
    pivot = pivot[VARIANT_ORDER]
    rows = []
    for (team_count, t_rel), row in pivot.iterrows():
        rows.append(
            [
                f"{int(team_count)}",
                f"{float(t_rel):.2f}",
                *[f"{float(row[variant]):.1f}" for variant in VARIANT_ORDER],
            ]
        )
    write_latex_table(
        output_path,
        "Mittlere Laufzeiten im 5D-Teamanzahl-Block bei N=1,000,000 und 32 Workern.",
        "tab:team-bench-5d-teamcount-runtime",
        ["Teams", r"$T_{rel}$", "UF", "UF Parallel", "Handcrafted", "Bounded", "Dynamic"],
        rows,
    )


def write_speedup_table(summary: pd.DataFrame, output_path: Path):
    pivot = summary.pivot_table(
        index=["team_count", "t_rel"],
        columns="variant",
        values="runtime_ms_mean",
        aggfunc="first",
    )
    rows = []
    for (team_count, t_rel), row in pivot.iterrows():
        speedup = float(row["baseline_union_first"]) / float(row["union_first_parallel"])
        rows.append([f"{int(team_count)}", f"{float(t_rel):.2f}", f"{speedup:.2f}"])
    write_latex_table(
        output_path,
        "Speedup von Union First Parallel gegenueber Union First im 5D-Teamanzahl-Block.",
        "tab:team-bench-5d-teamcount-speedup",
        ["Teams", r"$T_{rel}$", "Speedup"],
        rows,
    )


def plot_runtime_lines(summary: pd.DataFrame, output_path: Path):
    t_rel_values = sorted(summary["t_rel"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for axis, t_rel in zip(axes, t_rel_values):
        subset = summary[summary["t_rel"] == t_rel]
        for variant in VARIANT_ORDER:
            variant_df = subset[subset["variant"] == variant].sort_values("team_count")
            axis.errorbar(
                variant_df["team_count"],
                variant_df["runtime_ms_mean"],
                yerr=variant_df["runtime_ms_std"].fillna(0.0),
                marker="o",
                linewidth=2,
                capsize=3,
                label=VARIANT_LABELS[variant],
                color=VARIANT_COLORS[variant],
            )
        axis.set_title(rf"$T_{{rel}}={t_rel:.2f}$")
        axis.set_xticks([2, 3, 4, 5])
        axis.grid(axis="y", linestyle=":", alpha=0.45)

    fig.suptitle("5D-Teamanzahl-Block: Laufzeit nach Teamanzahl und Strategie")
    fig.supxlabel("Anzahl beteiligter Teams")
    fig.supylabel("Mittlere Laufzeit [ms]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.09, 1, 0.94])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ufp_speedup_heatmap(summary: pd.DataFrame, output_path: Path):
    pivot = summary.pivot_table(
        index=["team_count", "t_rel"],
        columns="variant",
        values="runtime_ms_mean",
        aggfunc="first",
    )
    speedup = (pivot["baseline_union_first"] / pivot["union_first_parallel"]).reset_index(name="speedup")
    matrix = speedup.pivot(index="team_count", columns="t_rel", values="speedup")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    image = ax.imshow(matrix.to_numpy(), cmap="YlGn", aspect="auto", vmin=1.0)
    ax.set_title("Speedup von Union First Parallel gegenueber Union First")
    ax.set_xlabel(r"$T_{rel}$")
    ax.set_ylabel("Anzahl Teams")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([f"{value:.2f}" for value in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([str(int(value)) for value in matrix.index])
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, f"{matrix.iloc[y, x]:.2f}x", ha="center", va="center", fontsize=10)
    fig.colorbar(image, ax=ax, label="Speedup")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_best_strategy_heatmap(summary: pd.DataFrame, output_path: Path):
    best = summary.loc[summary.groupby("scenario_id")["runtime_ms_mean"].idxmin()].copy()
    team_counts = sorted(best["team_count"].unique())
    t_rel_values = sorted(best["t_rel"].unique())
    strategies = [variant for variant in VARIANT_ORDER if variant in set(best["variant"])]
    strategy_to_id = {variant: idx for idx, variant in enumerate(strategies)}
    cmap = matplotlib.colormaps["Set2"].resampled(max(1, len(strategies)))
    matrix = np.full((len(team_counts), len(t_rel_values)), np.nan)

    for _, row in best.iterrows():
        y = team_counts.index(row["team_count"])
        x = t_rel_values.index(row["t_rel"])
        matrix[y, x] = strategy_to_id[row["variant"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-0.5, vmax=max(0.5, len(strategies) - 0.5))
    ax.set_title("Beste Strategie im 5D-Teamanzahl-Block")
    ax.set_xlabel(r"$T_{rel}$")
    ax.set_ylabel("Anzahl Teams")
    ax.set_xticks(np.arange(len(t_rel_values)))
    ax.set_xticklabels([f"{value:.2f}" for value in t_rel_values])
    ax.set_yticks(np.arange(len(team_counts)))
    ax.set_yticklabels([str(int(value)) for value in team_counts])
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            variant = strategies[int(matrix[y, x])]
            ax.text(x, y, VARIANT_LABELS[variant], ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def copy_to_thesis(output_dir: Path, thesis_dir: Path):
    fig_dir = thesis_dir / "fig"
    table_dir = thesis_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "team_bench_5d_teamcount_runtime.pdf",
        "team_bench_5d_teamcount_ufp_speedup.pdf",
        "team_bench_5d_teamcount_best_strategy.pdf",
    ]:
        (fig_dir / filename).write_bytes((output_dir / filename).read_bytes())
    for filename in [
        "team_bench_5d_teamcount_runtime.tex",
        "team_bench_5d_teamcount_speedup.tex",
    ]:
        (table_dir / filename).write_bytes((output_dir / filename).read_bytes())


def main():
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_latest_summaries(experiment_root)
    summary.to_csv(output_dir / "summary_5d_teamcount_latest.csv", index=False)

    write_runtime_table(summary, output_dir / "team_bench_5d_teamcount_runtime.tex")
    write_speedup_table(summary, output_dir / "team_bench_5d_teamcount_speedup.tex")
    plot_runtime_lines(summary, output_dir / "team_bench_5d_teamcount_runtime.pdf")
    plot_ufp_speedup_heatmap(summary, output_dir / "team_bench_5d_teamcount_ufp_speedup.pdf")
    plot_best_strategy_heatmap(summary, output_dir / "team_bench_5d_teamcount_best_strategy.pdf")

    if not args.no_thesis_copy and args.thesis_dir.exists():
        copy_to_thesis(output_dir, args.thesis_dir.resolve())

    best = summary.loc[summary.groupby("scenario_id")["runtime_ms_mean"].idxmin()]
    print(f"Read latest summaries from: {experiment_root}")
    print(f"Covered scenarios: {best['scenario_id'].nunique()}")
    print(f"Output dir: {output_dir}")
    print("\nBest strategies:")
    print(
        best.sort_values(["team_count", "t_rel"])[
            ["scenario_id", "variant", "runtime_ms_mean", "runtime_ms_std"]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )


if __name__ == "__main__":
    main()
