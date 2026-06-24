#!/usr/bin/env python3
"""
Konsolidiert den 5D-Teamanzahl-Block aus einzelnen team_bench-Runs.

Das Skript fuehrt keine Benchmarks aus. Es liest alle vorhandenen
`summary_by_variant.csv`-Dateien eines Experiments, waehlt pro Szenario den
neuesten Run und erzeugt daraus:

- eine zusammengefuehrte CSV,
- thesisfaehige LaTeX-Tabellen,
- PDF-Plots fuer Laufzeiten, Speedups und den Strategiewechsel bei Imbalance.
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
DEFAULT_UNBALANCED_ANALYSIS_DIR = (
    BASE_DIR
    / "study"
    / "team_bench_results"
    / "strategy_selection_analysis"
    / "nonuniform_3t5d_mixed_w32_2026-06-18"
)

VARIANT_ORDER = [
    "baseline_union_first",
    "union_first_parallel",
    "current_handcrafted",
    "bounded_selective_expansion",
    "dynamic_selective_expansion",
]

VARIANT_LABELS = {
    "baseline_union_first": "UF",
    "union_first_parallel": "UF Par",
    "current_handcrafted": "Handcr.",
    "bounded_selective_expansion": "Begr. Exp.",
    "dynamic_selective_expansion": "Dynamic",
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
        "--unbalanced-analysis-dir",
        type=Path,
        default=DEFAULT_UNBALANCED_ANALYSIS_DIR,
        help="Analyseordner des nicht-uniformen 3T/5D-Imbalance-Blocks.",
    )
    parser.add_argument(
        "--no-thesis-copy",
        action="store_true",
        help="Nur in output-dir schreiben, nichts nach thesis/ kopieren.",
    )
    return parser.parse_args()


def load_latest_summaries(experiment_root: Path, profile_token: str = "uniform") -> pd.DataFrame:
    frames = []
    pattern = rf"tb_[2345]t_5d_t(010|035|060|085)_{profile_token}"
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
        raise RuntimeError(f"No matching 5D team-count summaries for profile '{profile_token}' found below {experiment_root}")

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows[all_rows["variant"].isin(VARIANT_ORDER)].copy()

    # Ignore smoke tests or partial reruns. The consolidated thesis plots need
    # one complete strategy set per scenario, otherwise a short verification run
    # could accidentally replace a full benchmark just because it is newer.
    required_variants = set(VARIANT_ORDER)
    complete_runs = []
    for (scenario_id, run_dir), group in all_rows.groupby(["scenario_id", "run_dir"]):
        if required_variants.issubset(set(group["variant"])):
            complete_runs.append({"scenario_id": scenario_id, "run_dir": run_dir})
    if not complete_runs:
        raise RuntimeError("No complete 5D team-count runs found; only partial/smoke runs matched.")
    complete_runs_df = pd.DataFrame(complete_runs)

    latest_runs = (
        complete_runs_df
        .sort_values("run_dir")
        .groupby("scenario_id", as_index=False)
        .tail(1)
    )
    latest = all_rows.merge(latest_runs, on=["scenario_id", "run_dir"], how="inner")
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

    for axis in axes[2:]:
        axis.set_xlabel("Anzahl beteiligter Teams")

    fig.suptitle("5D-Teamanzahl-Block: Laufzeit nach Teamanzahl und Strategie", y=0.985)
    fig.supylabel("Mittlere Laufzeit [ms]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3)
    fig.tight_layout(rect=[0, 0.04, 1, 0.88])
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



def plot_profile_teamcount_winners(
    uniform_summary: pd.DataFrame,
    mixed_summary: pd.DataFrame,
    output_path: Path,
):
    """Show how the winning strategy changes between uniform and imbalanced data."""
    rows: list[dict[str, object]] = []
    profiles = [
        ("uniform", "uniform", uniform_summary),
        ("mixed", "mixed imbalance", mixed_summary),
    ]
    for team_count in [4, 5]:
        for profile_key, profile_label, summary in profiles:
            subset = summary[summary["team_count"] == team_count].copy()
            if subset.empty:
                continue
            best = subset.loc[subset.groupby("scenario_id")["runtime_ms_mean"].idxmin()]
            for _, row in best.iterrows():
                rows.append(
                    {
                        "row_key": f"{team_count}t_{profile_key}",
                        "row_label": f"{team_count} Teams\n{profile_label}",
                        "team_count": int(team_count),
                        "profile": profile_key,
                        "t_rel": float(row["t_rel"]),
                        "variant": str(row["variant"]),
                    }
                )

    winner_df = pd.DataFrame(rows)
    if winner_df.empty:
        return

    row_order = ["4t_uniform", "4t_mixed", "5t_uniform", "5t_mixed"]
    row_order = [row for row in row_order if row in set(winner_df["row_key"])]
    row_labels = [
        winner_df[winner_df["row_key"] == row]["row_label"].iloc[0]
        for row in row_order
    ]
    t_rel_values = sorted(winner_df["t_rel"].unique())
    present_variants = [
        variant for variant in VARIANT_ORDER if variant in set(winner_df["variant"])
    ]
    variant_to_id = {variant: idx for idx, variant in enumerate(present_variants)}
    matrix = np.full((len(row_order), len(t_rel_values)), np.nan)

    for _, row in winner_df.iterrows():
        y = row_order.index(row["row_key"])
        x = t_rel_values.index(float(row["t_rel"]))
        matrix[y, x] = variant_to_id[row["variant"]]

    cmap = matplotlib.colors.ListedColormap([VARIANT_COLORS[variant] for variant in present_variants])
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-0.5, vmax=len(present_variants) - 0.5)
    ax.set_title("Gewinnerstrategie: uniforme vs. imbalanced 5D-Szenarien", pad=12)
    ax.set_xlabel(r"$T_{rel}$")
    ax.set_ylabel("Szenario")
    ax.set_xticks(np.arange(len(t_rel_values)))
    ax.set_xticklabels([f"{value:.2f}" for value in t_rel_values])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            variant = present_variants[int(matrix[y, x])]
            ax.text(
                x,
                y,
                VARIANT_LABELS[variant],
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )

    patches = [
        matplotlib.patches.Patch(color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant])
        for variant in present_variants
    ]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Gewinner")
    fig.text(
        0.02,
        0.01,
        "Alle Szenarien: 5D-Teams, N=1M, 20 Bins/Dimension, 8 Query-Bins/Dimension, 32 Worker.",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.08, 0.82, 1])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_shift_comparison(
    balanced_summary: pd.DataFrame,
    unbalanced_analysis_dir: Path,
    output_path: Path,
):
    winners: list[dict[str, object]] = []

    balanced = balanced_summary[balanced_summary["team_count"] == 3].copy()
    balanced_best = balanced.loc[balanced.groupby("scenario_id")["runtime_ms_mean"].idxmin()]
    for _, row in balanced_best.sort_values("t_rel").iterrows():
        winners.append(
            {
                "profile": "balanced",
                "row_label": "balanced / uniform\nImbalance ca. 1x",
                "t_rel": float(row["t_rel"]),
                "variant": str(row["variant"]),
            }
        )

    unbalanced_path = unbalanced_analysis_dir / "scenario_winners.csv"
    if unbalanced_path.exists():
        unbalanced = pd.read_csv(unbalanced_path)
        unbalanced = unbalanced[unbalanced["team_count"] == 3].copy()
        for _, row in unbalanced.sort_values("t_rel").iterrows():
            winners.append(
                {
                    "profile": "unbalanced",
                    "row_label": "unbalanced / mixed\nImbalance 28-85x",
                    "t_rel": float(row["t_rel"]),
                    "variant": str(row["winner_variant"]),
                }
            )

    winner_df = pd.DataFrame(winners)
    if winner_df.empty or winner_df["profile"].nunique() < 2:
        return

    row_order = ["balanced", "unbalanced"]
    row_labels = [
        winner_df[winner_df["profile"] == profile]["row_label"].iloc[0]
        for profile in row_order
    ]
    t_rel_values = sorted(winner_df["t_rel"].unique())
    present_variants = [
        variant for variant in VARIANT_ORDER if variant in set(winner_df["variant"])
    ]
    variant_to_id = {variant: idx for idx, variant in enumerate(present_variants)}
    matrix = np.full((len(row_order), len(t_rel_values)), np.nan)

    for _, row in winner_df.iterrows():
        y = row_order.index(row["profile"])
        x = t_rel_values.index(float(row["t_rel"]))
        matrix[y, x] = variant_to_id[row["variant"]]

    colors = [VARIANT_COLORS[variant] for variant in present_variants]
    cmap = matplotlib.colors.ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-0.5, vmax=len(present_variants) - 0.5)
    ax.set_title("Strategiewechsel durch Team-Imbalance", pad=12)
    ax.set_xlabel(r"$T_{rel}$")
    ax.set_ylabel("Datenprofil")
    ax.set_xticks(np.arange(len(t_rel_values)))
    ax.set_xticklabels([f"{value:.2f}" for value in t_rel_values])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            variant = present_variants[int(matrix[y, x])]
            ax.text(
                x,
                y,
                VARIANT_LABELS[variant],
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )

    patches = [
        matplotlib.patches.Patch(color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant])
        for variant in present_variants
    ]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Gewinner")
    fig.text(
        0.02,
        0.01,
        "Vergleich: 3 Teams, 5D, N=1M, 32 Worker. Imbalance = groesste Team-Union / kleinste Team-Union.",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.08, 0.82, 1])
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
        "team_bench_5d_mixed_teamcount_runtime.pdf",
        "team_bench_5d_mixed_teamcount_ufp_speedup.pdf",
        "team_bench_5d_profile_teamcount_winners.pdf",
        "team_bench_5d_strategy_shift_balanced_vs_unbalanced.pdf",
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

    summary = load_latest_summaries(experiment_root, profile_token="uniform")
    summary.to_csv(output_dir / "summary_5d_teamcount_latest.csv", index=False)

    mixed_summary = load_latest_summaries(experiment_root, profile_token="mixed_team_imbalance")
    mixed_summary.to_csv(output_dir / "summary_5d_mixed_teamcount_latest.csv", index=False)

    write_runtime_table(summary, output_dir / "team_bench_5d_teamcount_runtime.tex")
    write_speedup_table(summary, output_dir / "team_bench_5d_teamcount_speedup.tex")
    plot_runtime_lines(summary, output_dir / "team_bench_5d_teamcount_runtime.pdf")
    plot_ufp_speedup_heatmap(summary, output_dir / "team_bench_5d_teamcount_ufp_speedup.pdf")

    plot_runtime_lines(mixed_summary, output_dir / "team_bench_5d_mixed_teamcount_runtime.pdf")
    plot_ufp_speedup_heatmap(mixed_summary, output_dir / "team_bench_5d_mixed_teamcount_ufp_speedup.pdf")
    plot_profile_teamcount_winners(
        summary,
        mixed_summary,
        output_dir / "team_bench_5d_profile_teamcount_winners.pdf",
    )
    plot_strategy_shift_comparison(
        summary,
        args.unbalanced_analysis_dir.resolve(),
        output_dir / "team_bench_5d_strategy_shift_balanced_vs_unbalanced.pdf",
    )

    if not args.no_thesis_copy and args.thesis_dir.exists():
        copy_to_thesis(output_dir, args.thesis_dir.resolve())

    best = summary.loc[summary.groupby("scenario_id")["runtime_ms_mean"].idxmin()]
    mixed_best = mixed_summary.loc[mixed_summary.groupby("scenario_id")["runtime_ms_mean"].idxmin()]
    print(f"Read latest summaries from: {experiment_root}")
    print(f"Covered uniform scenarios: {best['scenario_id'].nunique()}")
    print(f"Covered mixed-team-imbalance scenarios: {mixed_best['scenario_id'].nunique()}")
    print(f"Output dir: {output_dir}")
    print("\nBest strategies, uniform:")
    print(
        best.sort_values(["team_count", "t_rel"])[
            ["scenario_id", "variant", "runtime_ms_mean", "runtime_ms_std"]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print("\nBest strategies, mixed-team-imbalance:")
    print(
        mixed_best.sort_values(["team_count", "t_rel"])[
            ["scenario_id", "variant", "runtime_ms_mean", "runtime_ms_std"]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )


if __name__ == "__main__":
    main()
