#!/usr/bin/env python3
"""
Erzeugt visuelle Auswertungen aus strategy_selection_analysis-CSV-Dateien.

Das Skript fuehrt keine Benchmarks aus. Es liest einen vorhandenen Ordner aus
`study/team_bench_results/strategy_selection_analysis/...` und erstellt daraus
PDFs, die die Strategiewahl besser erklaeren:

- welche Strategie in welchem Szenario gewinnt,
- wie stark Union First Parallel gegen Union First ist,
- wie nah die Strategien pro Szenariofamilie beieinander liegen,
- ob ISE Count, Workload und Varianz die Beobachtungen erklaeren.

Beispiel:

    venv/bin/python study/strategy_selection_to_plots.py \
        --analysis-dir study/team_bench_results/strategy_selection_analysis/full_union_first_parallel_core
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd


VARIANT_ALIASES = {
    "worker_balanced_union_grouping": "union_first_parallel",
}

VARIANT_LABELS = {
    "baseline_union_first": "Union First",
    "union_first_parallel": "UF Parallel",
    "current_handcrafted": "Handcrafted",
    "bounded_selective_expansion": "Bounded Exp.",
    "dynamic_selective_expansion": "Dynamic",
    "baseline_minimal_intersection": "Minimal Int.",
    "expand_all_adaptive_grouping": "Expand All",
}

VARIANT_COLORS = {
    "baseline_union_first": "#8a8f98",
    "union_first_parallel": "#1f77b4",
    "current_handcrafted": "#ff7f0e",
    "bounded_selective_expansion": "#2ca02c",
    "dynamic_selective_expansion": "#9467bd",
    "baseline_minimal_intersection": "#d62728",
    "expand_all_adaptive_grouping": "#17becf",
}

VARIANT_ORDER = [
    "baseline_union_first",
    "union_first_parallel",
    "current_handcrafted",
    "bounded_selective_expansion",
    "dynamic_selective_expansion",
    "baseline_minimal_intersection",
    "expand_all_adaptive_grouping",
]


@dataclass(frozen=True)
class AnalysisData:
    analysis_dir: Path
    outcomes: pd.DataFrame
    winners: pd.DataFrame
    family_summary: pd.DataFrame
    scenario_features: pd.DataFrame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Erzeuge Strategieauswahl-PDFs aus vorhandenen strategy_selection_analysis-CSVs."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        action="append",
        required=True,
        help="Ordner mit combined_variant_outcomes.csv, scenario_winners.csv usw. Mehrfach nutzbar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optionaler Zielordner. Nur sinnvoll bei genau einem --analysis-dir. Standard: <analysis-dir>/plots.",
    )
    return parser.parse_args()


def normalize_variant_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["variant", "winner_variant", "runner_up_variant"]:
        if column in df.columns:
            df[column] = df[column].replace(VARIANT_ALIASES)
    return df


def ensure_distribution_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "distribution_profile" not in df.columns:
        df["distribution_profile"] = "uniform"
    if "distribution_strength" not in df.columns:
        df["distribution_strength"] = 0.0
    return df


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Missing required CSV: {path}")
    return ensure_distribution_columns(normalize_variant_columns(pd.read_csv(path)))


def load_analysis(analysis_dir: Path) -> AnalysisData:
    analysis_dir = analysis_dir.resolve()
    return AnalysisData(
        analysis_dir=analysis_dir,
        outcomes=read_csv_required(analysis_dir / "combined_variant_outcomes.csv"),
        winners=read_csv_required(analysis_dir / "scenario_winners.csv"),
        family_summary=read_csv_required(analysis_dir / "family_strategy_summary.csv"),
        scenario_features=read_csv_required(analysis_dir / "scenario_feature_table.csv"),
    )


def variant_label(variant: str) -> str:
    return VARIANT_LABELS.get(str(variant), str(variant))


def variant_short(variant: str) -> str:
    return {
        "baseline_union_first": "UF",
        "union_first_parallel": "UFP",
        "current_handcrafted": "HC",
        "bounded_selective_expansion": "BSE",
        "dynamic_selective_expansion": "DYN",
        "baseline_minimal_intersection": "MI",
        "expand_all_adaptive_grouping": "EAG",
    }.get(str(variant), str(variant)[:8])


def variants_in_order(variants) -> list[str]:
    present = list(dict.fromkeys(str(v) for v in variants if pd.notna(v)))
    ordered = [variant for variant in VARIANT_ORDER if variant in present]
    ordered += sorted(variant for variant in present if variant not in ordered)
    return ordered


def experiment_scope(name: str) -> str:
    match = re.search(r"dims([0-9]+)", str(name))
    if match:
        return f"dims{match.group(1)}"
    return str(name).replace("team_bench_", "")


def profile_label(profile: str) -> str:
    return {
        "uniform": "uniform",
        "mixed_team_imbalance": "mixed",
        "cell_skew": "cell_skew",
        "query_hotspot": "query_hot",
        "anti_query_hotspot": "anti_query",
    }.get(str(profile), str(profile))


def family_key(row) -> tuple:
    return (
        str(row.experiment_name),
        int(row.team_count),
        int(row.dimension),
        str(getattr(row, "distribution_profile", "uniform")),
    )


def family_label(row) -> str:
    return (
        f"{experiment_scope(row.experiment_name)} | "
        f"{int(row.team_count)}T-{int(row.dimension)}D | "
        f"{profile_label(getattr(row, 'distribution_profile', 'uniform'))}"
    )


def scenario_label(row) -> str:
    return (
        f"{experiment_scope(row.experiment_name)} "
        f"{int(row.team_count)}T-{int(row.dimension)}D "
        f"{profile_label(getattr(row, 'distribution_profile', 'uniform'))} "
        f"T={float(row.t_rel):.2f}"
    )


def scenario_sort_cols(df: pd.DataFrame) -> list[str]:
    return [
        col for col in ["experiment_name", "team_count", "dimension", "distribution_profile", "t_rel", "scenario_id"]
        if col in df.columns
    ]


def sorted_families(df: pd.DataFrame) -> list[tuple[tuple, str]]:
    meta = (
        df[["experiment_name", "team_count", "dimension", "distribution_profile"]]
        .drop_duplicates()
        .sort_values(["experiment_name", "team_count", "dimension", "distribution_profile"])
    )
    return [(family_key(row), family_label(row)) for row in meta.itertuples(index=False)]


def sorted_t_rel_values(df: pd.DataFrame) -> list[float]:
    return sorted(float(value) for value in df["t_rel"].dropna().unique())


def ensure_output_dir(data: AnalysisData, requested_output: Path | None, multiple_inputs: bool) -> Path:
    if requested_output is not None:
        if multiple_inputs:
            output_dir = requested_output / data.analysis_dir.name
        else:
            output_dir = requested_output
    else:
        output_dir = data.analysis_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def finish_figure(fig, output_path: Path, dashboard: PdfPages | None):
    fig.savefig(output_path, bbox_inches="tight")
    if dashboard is not None:
        dashboard.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_key_findings(data: AnalysisData, output_path: Path, dashboard: PdfPages | None):
    outcomes = data.outcomes
    winners = data.winners

    pivot = outcomes.pivot_table(
        index=["run_key", "scenario_id"],
        columns="variant",
        values="runtime_ms_mean",
        aggfunc="mean",
    )

    uf_speedups = pd.Series(dtype=float)
    if {"baseline_union_first", "union_first_parallel"}.issubset(pivot.columns):
        uf_speedups = (pivot["baseline_union_first"] / pivot["union_first_parallel"]).dropna()

    hc_speedups = pd.Series(dtype=float)
    if {"current_handcrafted", "union_first_parallel"}.issubset(pivot.columns):
        hc_speedups = (pivot["current_handcrafted"] / pivot["union_first_parallel"]).dropna()

    winner_counts = winners["winner_variant"].value_counts()
    variance_clear_count = int(winners.get("variance_clear", pd.Series(dtype=bool)).fillna(False).sum())
    total_winners = len(winners)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle(f"Strategieauswahl: Kernaussagen\n{data.analysis_dir.name}", fontsize=15)

    ax = axes[0, 0]
    ax.axis("off")
    lines = [
        ("Szenarien", f"{total_winners}"),
        ("UF Parallel >= 10% schneller als UF", f"{int((uf_speedups > 1.10).sum())}/{len(uf_speedups)}"),
        ("Mittlerer Faktor vs. UF", f"{uf_speedups.mean():.3f}x" if not uf_speedups.empty else "n/a"),
        ("UF Parallel >= 2% schneller als HC", f"{int((hc_speedups > 1.02).sum())}/{len(hc_speedups)}" if not hc_speedups.empty else "n/a"),
        ("Varianz-klarer Gewinner", f"{variance_clear_count}/{total_winners}"),
    ]
    y = 0.9
    for label, value in lines:
        ax.text(0.03, y, label, fontsize=12, color="#333333", ha="left", va="center")
        ax.text(0.96, y, value, fontsize=17, weight="bold", color="#111111", ha="right", va="center")
        y -= 0.16
    ax.set_title("Was man aus dem Run mitnehmen soll", loc="left")

    ax = axes[0, 1]
    if not winner_counts.empty:
        variants = variants_in_order(winner_counts.index)
        values = [winner_counts.get(variant, 0) for variant in variants]
        colors = [VARIANT_COLORS.get(variant, "#777777") for variant in variants]
        ax.barh([variant_label(v) for v in variants], values, color=colors)
        ax.set_xlabel("Anzahl gewonnener Szenarien")
        ax.set_title("Gewinner-Verteilung")
        ax.grid(axis="x", linestyle=":", alpha=0.35)
    else:
        ax.axis("off")

    ax = axes[1, 0]
    if not uf_speedups.empty:
        ax.hist(uf_speedups, bins=min(12, max(4, len(uf_speedups) // 2)), color=VARIANT_COLORS["union_first_parallel"], alpha=0.85)
        ax.axvline(1.0, color="#333333", linewidth=1.0)
        ax.axvline(1.10, color="#b23a48", linewidth=1.2, linestyle="--", label="10% Schwelle")
        ax.set_xlabel("Speedup von Union First Parallel gegen Union First")
        ax.set_ylabel("Szenarien")
        ax.set_title("Wie deutlich wird Union First geschlagen?")
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.35)
    else:
        ax.axis("off")

    ax = axes[1, 1]
    margin = winners["winner_margin"].dropna()
    if not margin.empty:
        ax.hist(margin, bins=min(12, max(4, len(margin) // 2)), color="#c49a00", alpha=0.85)
        ax.axvline(1.05, color="#b23a48", linewidth=1.2, linestyle="--", label="5% Abstand")
        ax.set_xlabel("Runner-up Laufzeit / Gewinner Laufzeit")
        ax.set_ylabel("Szenarien")
        ax.set_title("Wie knapp sind die Siege?")
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.35)
    else:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return finish_figure(fig, output_path, dashboard)


def plot_winner_map(data: AnalysisData, output_path: Path, dashboard: PdfPages | None):
    winners = data.winners.sort_values(scenario_sort_cols(data.winners))
    families = sorted_families(winners)
    t_rel_values = sorted_t_rel_values(winners)
    variants = variants_in_order(winners["winner_variant"].unique())

    strategy_to_id = {variant: idx for idx, variant in enumerate(variants)}
    colors = [VARIANT_COLORS.get(variant, "#777777") for variant in variants]
    cmap = ListedColormap(["#eeeeee"] + colors)
    norm = BoundaryNorm(np.arange(-1.5, len(variants) + 0.5, 1), cmap.N)

    matrix = np.full((len(families), len(t_rel_values)), -1.0)
    labels = [["" for _ in t_rel_values] for _ in families]

    family_index = {key: idx for idx, (key, _) in enumerate(families)}
    t_rel_index = {value: idx for idx, value in enumerate(t_rel_values)}

    for row in winners.itertuples(index=False):
        y = family_index.get(family_key(row))
        x = t_rel_index.get(float(row.t_rel))
        if y is None or x is None:
            continue
        matrix[y, x] = strategy_to_id[str(row.winner_variant)]
        suffix = "" if bool(row.variance_clear) else "*"
        labels[y][x] = f"{variant_short(row.winner_variant)}\n{float(row.winner_margin):.2f}x{suffix}"

    fig, ax = plt.subplots(figsize=(max(9, len(t_rel_values) * 1.3), max(5.5, len(families) * 0.52)))
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_title(
        "Gewinnerkarte pro Szenario\n"
        "Zahl = Abstand Runner-up/Gewinner; * = Varianz ueberlappt"
    )
    ax.set_xlabel("T_rel")
    ax.set_ylabel("Szenariofamilie")
    ax.set_xticks(np.arange(len(t_rel_values)))
    ax.set_xticklabels([f"{value:.2f}" for value in t_rel_values])
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels([label for _, label in families])

    for y in range(len(families)):
        for x in range(len(t_rel_values)):
            if labels[y][x]:
                ax.text(x, y, labels[y][x], ha="center", va="center", fontsize=8)

    patches = [mpatches.Patch(color=VARIANT_COLORS.get(v, "#777777"), label=f"{variant_short(v)} = {variant_label(v)}") for v in variants]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    return finish_figure(fig, output_path, dashboard)


def plot_union_first_parallel_speedup_heatmap(data: AnalysisData, output_path: Path, dashboard: PdfPages | None):
    outcomes = data.outcomes
    pivot = outcomes.pivot_table(
        index=["run_key", "scenario_id"],
        columns="variant",
        values="runtime_ms_mean",
        aggfunc="mean",
    )
    if not {"baseline_union_first", "union_first_parallel"}.issubset(pivot.columns):
        return None

    meta_cols = [
        "run_key",
        "scenario_id",
        "experiment_name",
        "team_count",
        "dimension",
        "distribution_profile",
        "t_rel",
    ]
    meta = outcomes[meta_cols].drop_duplicates().set_index(["run_key", "scenario_id"])
    speedup = (pivot["baseline_union_first"] / pivot["union_first_parallel"]).rename("speedup")
    df = meta.join(speedup).reset_index().dropna(subset=["speedup"])
    df = df.sort_values(scenario_sort_cols(df))

    families = sorted_families(df)
    t_rel_values = sorted_t_rel_values(df)
    matrix = np.full((len(families), len(t_rel_values)), np.nan)
    labels = [["" for _ in t_rel_values] for _ in families]
    family_index = {key: idx for idx, (key, _) in enumerate(families)}
    t_rel_index = {value: idx for idx, value in enumerate(t_rel_values)}

    for row in df.itertuples(index=False):
        y = family_index.get(family_key(row))
        x = t_rel_index.get(float(row.t_rel))
        if y is None or x is None:
            continue
        matrix[y, x] = float(row.speedup)
        labels[y][x] = f"{float(row.speedup):.2f}x"

    vmax = max(1.15, np.nanquantile(matrix, 0.95))
    fig, ax = plt.subplots(figsize=(max(8, len(t_rel_values) * 1.3), max(5.5, len(families) * 0.52)))
    cmap = matplotlib.colormaps["YlGnBu"].copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=1.0, vmax=vmax)
    ax.set_title("Speedup von Union First Parallel gegen Union First\nWert > 1 bedeutet: UFP ist schneller")
    ax.set_xlabel("T_rel")
    ax.set_ylabel("Szenariofamilie")
    ax.set_xticks(np.arange(len(t_rel_values)))
    ax.set_xticklabels([f"{value:.2f}" for value in t_rel_values])
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels([label for _, label in families])
    for y in range(len(families)):
        for x in range(len(t_rel_values)):
            if labels[y][x]:
                ax.text(x, y, labels[y][x], ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Speedup vs. Union First")
    fig.tight_layout()
    return finish_figure(fig, output_path, dashboard)


def plot_win_counts(data: AnalysisData, output_path: Path, dashboard: PdfPages | None):
    winners = data.winners.copy()
    winners["family_label"] = winners.apply(family_label, axis=1)
    family_order = (
        winners[["experiment_name", "team_count", "dimension", "distribution_profile", "family_label"]]
        .drop_duplicates()
        .sort_values(["experiment_name", "team_count", "dimension", "distribution_profile"])["family_label"]
        .tolist()
    )
    variants = variants_in_order(winners["winner_variant"].unique())
    counts = (
        winners.groupby(["family_label", "winner_variant"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=family_order, columns=variants, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(11, max(5.5, len(counts) * 0.55)))
    left = np.zeros(len(counts))
    y = np.arange(len(counts))
    for variant in variants:
        values = counts[variant].to_numpy()
        ax.barh(y, values, left=left, label=variant_label(variant), color=VARIANT_COLORS.get(variant, "#777777"))
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index)
    ax.invert_yaxis()
    ax.set_xlabel("Gewonnene Szenarien")
    ax.set_title("Wer gewinnt in welcher Szenariofamilie?")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return finish_figure(fig, output_path, dashboard)


def plot_relative_runtime_by_family(data: AnalysisData, output_path: Path, dashboard: PdfPages | None):
    summary = data.family_summary.copy()
    summary = summary[summary["runtime_ms_mean"].notna()]
    summary["family_label"] = summary.apply(family_label, axis=1)
    summary["best_runtime"] = summary.groupby(
        ["experiment_name", "worker_count", "team_count", "dimension", "distribution_profile"]
    )["runtime_ms_mean"].transform("min")
    summary["runtime_vs_family_best"] = summary["runtime_ms_mean"] / summary["best_runtime"]
    summary = summary.sort_values(["experiment_name", "team_count", "dimension", "distribution_profile", "variant"])

    family_order = (
        summary[["experiment_name", "team_count", "dimension", "distribution_profile", "family_label"]]
        .drop_duplicates()
        .sort_values(["experiment_name", "team_count", "dimension", "distribution_profile"])["family_label"]
        .tolist()
    )
    y_positions = {label: idx for idx, label in enumerate(family_order)}
    variants = variants_in_order(summary["variant"].unique())
    variant_offsets = {
        variant: (idx - (len(variants) - 1) / 2) * 0.08
        for idx, variant in enumerate(variants)
    }

    fig, ax = plt.subplots(figsize=(12, max(5.5, len(family_order) * 0.55)))
    for variant in variants:
        subset = summary[summary["variant"] == variant]
        y = [y_positions[label] + variant_offsets[variant] for label in subset["family_label"]]
        ax.scatter(
            subset["runtime_vs_family_best"],
            y,
            label=variant_label(variant),
            color=VARIANT_COLORS.get(variant, "#777777"),
            s=60,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.axvline(1.0, color="#333333", linewidth=1.0)
    ax.axvline(1.05, color="#b23a48", linewidth=1.0, linestyle="--", label="5% langsamer")
    ax.axvline(1.20, color="#b23a48", linewidth=1.0, linestyle=":", label="20% langsamer")
    ax.set_yticks(np.arange(len(family_order)))
    ax.set_yticklabels(family_order)
    ax.invert_yaxis()
    ax.set_xlabel("Laufzeit relativ zur besten Strategie derselben Familie")
    ax.set_title("Wie nah liegen die Strategien pro Familie beieinander?")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return finish_figure(fig, output_path, dashboard)


def plot_ise_vs_speedup(data: AnalysisData, output_path: Path, dashboard: PdfPages | None):
    df = data.outcomes.copy()
    df = df[(~df["is_skipped"].fillna(False)) & df["runtime_ms_mean"].notna()]
    df = df[df["speedup_vs_baseline"].notna()]
    if df.empty:
        return None

    variants = variants_in_order(df["variant"].unique())
    offsets = {variant: (idx - (len(variants) - 1) / 2) * 0.045 for idx, variant in enumerate(variants)}
    fig, ax = plt.subplots(figsize=(11, 7))
    for variant in variants:
        subset = df[df["variant"] == variant]
        x = np.log2(subset["ise_count_estimate_manual"].fillna(0).clip(lower=0) + 1) + offsets[variant]
        sizes = 35 + 120 * (
            subset["total_selected_bin_cells"].fillna(0)
            / max(1.0, df["total_selected_bin_cells"].fillna(0).max())
        )
        ax.scatter(
            x,
            subset["speedup_vs_baseline"],
            s=sizes,
            alpha=0.72,
            color=VARIANT_COLORS.get(variant, "#777777"),
            label=variant_label(variant),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.axhline(1.0, color="#333333", linewidth=1.0)
    ax.axhline(1.10, color="#b23a48", linewidth=1.0, linestyle="--", label="10% schneller als UF")
    ax.set_xlabel("log2(geschaetzte ISE Count + 1)")
    ax.set_ylabel("Speedup relativ zu Union First")
    ax.set_title("Erklaerplot: ISE Count vs. Speedup\nPunktgroesse = getroffene Zellen")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return finish_figure(fig, output_path, dashboard)


def plot_workload_vs_speedup(data: AnalysisData, output_path: Path, dashboard: PdfPages | None):
    df = data.outcomes.copy()
    df = df[(~df["is_skipped"].fillna(False)) & df["runtime_ms_mean"].notna()]
    df = df[df["speedup_vs_baseline"].notna()]
    if df.empty:
        return None

    variants = variants_in_order(df["variant"].unique())
    fig, ax = plt.subplots(figsize=(11, 7))
    for variant in variants:
        subset = df[df["variant"] == variant]
        x = subset["total_selected_bin_cells"].fillna(0).clip(lower=1)
        ax.scatter(
            x,
            subset["speedup_vs_baseline"],
            s=60,
            alpha=0.75,
            color=VARIANT_COLORS.get(variant, "#777777"),
            label=variant_label(variant),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xscale("log", base=2)
    ax.axhline(1.0, color="#333333", linewidth=1.0)
    ax.axhline(1.10, color="#b23a48", linewidth=1.0, linestyle="--", label="10% schneller als UF")
    ax.set_xlabel("getroffene Zellen ueber alle Teams (log2)")
    ax.set_ylabel("Speedup relativ zu Union First")
    ax.set_title("Erklaerplot: Workload vs. Speedup\nWann lohnt bessere Arbeitsteilung?")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return finish_figure(fig, output_path, dashboard)


def plot_margin_variance(data: AnalysisData, output_path: Path, dashboard: PdfPages | None):
    winners = data.winners.copy()
    winners = winners.sort_values("winner_margin", ascending=True).reset_index(drop=True)
    winners["label"] = winners.apply(scenario_label, axis=1)
    winners["winner_gap_percent"] = (winners["winner_margin"] - 1.0) * 100.0
    colors = np.where(winners["variance_clear"].fillna(False), "#2ca02c", "#c49a00")

    fig, ax = plt.subplots(figsize=(12, max(6, len(winners) * 0.25)))
    y = np.arange(len(winners))
    ax.barh(y, winners["winner_gap_percent"], color=colors, alpha=0.9)
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.axvline(5.0, color="#b23a48", linewidth=1.1, linestyle="--", label="5% Abstand")
    ax.set_yticks(y)
    ax.set_yticklabels(winners["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Abstand Runner-up zu Gewinner [%]")
    ax.set_title("Wie belastbar sind die Gewinner?\nGruen = Abstand groesser als beide Standardabweichungen, Gelb = varianzempfindlich")
    for idx, row in winners.iterrows():
        ax.text(
            float(row.winner_gap_percent) + 0.25,
            idx,
            f"{variant_short(row.winner_variant)}",
            va="center",
            fontsize=8,
        )
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    return finish_figure(fig, output_path, dashboard)


def generate_plots(data: AnalysisData, output_dir: Path) -> list[Path]:
    plot_paths: list[Path] = []
    dashboard_path = output_dir / "strategy_selection_dashboard.pdf"
    with PdfPages(dashboard_path) as dashboard:
        for plotter, filename in [
            (plot_key_findings, "key_findings.pdf"),
            (plot_winner_map, "winner_map.pdf"),
            (plot_union_first_parallel_speedup_heatmap, "union_first_parallel_speedup_heatmap.pdf"),
            (plot_win_counts, "win_counts_by_family.pdf"),
            (plot_relative_runtime_by_family, "relative_runtime_by_family.pdf"),
            (plot_ise_vs_speedup, "ise_vs_speedup.pdf"),
            (plot_workload_vs_speedup, "workload_vs_speedup.pdf"),
            (plot_margin_variance, "winner_margin_variance.pdf"),
        ]:
            result = plotter(data, output_dir / filename, dashboard)
            if result is not None:
                plot_paths.append(result)
    plot_paths.insert(0, dashboard_path)
    return plot_paths


def main():
    args = parse_args()
    multiple_inputs = len(args.analysis_dir) > 1
    if args.output_dir is not None and multiple_inputs:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    all_paths: list[Path] = []
    for analysis_dir in args.analysis_dir:
        data = load_analysis(analysis_dir)
        output_dir = ensure_output_dir(data, args.output_dir, multiple_inputs)
        plot_paths = generate_plots(data, output_dir)
        all_paths.extend(plot_paths)
        print(f"\nAnalyseordner: {data.analysis_dir}")
        print(f"Plot-Ordner:    {output_dir}")
        for path in plot_paths:
            print(path)

    print(f"\nErzeugt: {len(all_paths)} PDF-Dateien")


if __name__ == "__main__":
    main()
