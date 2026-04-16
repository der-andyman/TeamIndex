#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from TeamIndex import benchmark as tib
from TeamIndex import evaluation as eva
from example_paths import DATA_PATH, INDEX_CONFIG


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = (BASE_DIR / "mopts_study").resolve()
GENERATE_WORKER_PLOTS = False


# A compact, explainable query set that covers:
# - one 3D team
# - one 2D team
# - mixed 3D/2D intersections
# - two 2D teams together
# - larger multi-team plans
QUERIES = [
    ("q01_single_3d", "A < 19 and E < 19 and C < 19"),
    ("q02_single_2d", "B < 30 and I < 30"),
    ("q03_two_teams", "A < 19 and E < 19 and C < 19 and B < 19"),
    ("q04_two_2d_teams", "B < 30 and I < 30 and F < 30 and H < 30"),
    ("q05_three_teams", "A < 38 and E < 38 and C < 19 and B < 38 and F < 40"),
    ("q06_two_3d_teams", "A < 20 and E < 20 and C < 20 and J < 20 and D < 20 and G < 20"),
]


def clone_mopts(mopts):
    return [(team_name, copy.deepcopy(opts)) for team_name, opts in mopts]


def manual_baseline_naive(index: eva.TeamIndex, query: str):
    # Naivste noch sinnvolle Baseline:
    # - uebernehme nur die von prepare_optimization bestimmte Team-Reihenfolge
    # - keine Expansion
    # - keine manuelle Aenderung der group_count-Werte
    #
    # Dadurch entsteht ein gueltiger, aber bewusst "unsmarter" Plan:
    # jedes Team wird im Wesentlichen lokal vereinigt und erst danach
    # mit den anderen Teams kombiniert.
    return clone_mopts(index.prepare_optimization(query=query))


def manual_current_handcrafted(index: eva.TeamIndex, query: str):
    # Aktuelle handgeschriebene Heuristik aus run_example.py:
    # - expandiere immer das erste / kleinste Team
    # - falls dieses Team danach nur wenige Gruppen hat, expandiere auch Team 2
    # - begrenze group_count mit festen Schranken (128 bzw. 16)
    #
    # Idee:
    # kleine/selective Teams frueh aufspalten, damit nachgelagerte
    # Schnitte feingranularer und oft guenstiger werden.
    mopts = clone_mopts(index.prepare_optimization(query=query))
    assert len(mopts) >= 1, "Empty result?"

    mopts[0][1]["is_expanded"] = True
    if mopts[0][1]["max_group_count"] > 128:
        mopts[0][1]["group_count"] = eva.po2_near_sqrt(mopts[0][1]["max_group_count"])

    limit = 128
    if mopts[0][1]["group_count"] < 16 and len(mopts) > 1:
        mopts[1][1]["is_expanded"] = True
        limit = 16

    for i in range(1, len(mopts)):
        if mopts[i][1]["max_group_count"] > limit:
            mopts[i][1]["group_count"] = min(
                eva.po2_near_sqrt(mopts[i][1]["max_group_count"]),
                limit,
            )

    return mopts


def manual_overhead_aware(index: eva.TeamIndex, query: str):
    # Overhead-aware Heuristik:
    # - niemals expandieren
    # - nur dann gruppieren, wenn ein Team wirklich viele Blaetter trifft
    # - Gruppierung bewusst konservativ halten
    #
    # Idee:
    # Wenn nur wenige Blaetter / Listen betroffen sind, ist zusaetzliche
    # Optimierungslogik oft nur Overhead. Diese Strategie versucht,
    # unnoetige Parallelisierung und zusaetzliche Task-Aufblaehung zu vermeiden.
    mopts = clone_mopts(index.prepare_optimization(query=query))
    for _, opt in mopts:
        opt["is_expanded"] = False

        # Nur bei groesseren Blattzahlen eingreifen.
        if opt["max_group_count"] >= 64:
            opt["group_count"] = min(eva.po2_near_sqrt(opt["max_group_count"]), 64)
        elif opt["max_group_count"] >= 24:
            opt["group_count"] = min(eva.po2_near_sqrt(opt["max_group_count"]), 24)

    return mopts


def manual_imbalance_aware(index: eva.TeamIndex, query: str):
    # Imbalance-aware Heuristik:
    # - sortiere Teams explizit nach union_cardinality und Blattzahl
    # - expandiere nur das kleinste Team
    # - halte die restlichen Teams konservativ gruppiert
    #
    # Idee:
    # Wenn Teams sehr unterschiedlich gross sind, sollte ein kleines Team
    # moeglichst frueh "dominieren", damit grosse Zwischenmengen vermieden werden.
    mopts = clone_mopts(index.prepare_optimization(query=query))
    if not mopts:
        return mopts

    mopts = sorted(
        mopts,
        key=lambda item: (
            item[1]["union_cardinality"],
            item[1]["max_group_count"],
        ),
    )

    # Einfache Imbalance-Abschaetzung auf Basis der Team-Kardinalitaeten.
    union_cards = [opt["union_cardinality"] for _, opt in mopts]
    imbalance = (max(union_cards) / min(union_cards)) if min(union_cards) > 0 else None

    mopts[0][1]["is_expanded"] = True
    if mopts[0][1]["max_group_count"] > 64:
        mopts[0][1]["group_count"] = min(eva.po2_near_sqrt(mopts[0][1]["max_group_count"]), 64)

    # Bei starker Imbalance sehr konservativ bleiben, damit das groesste Team
    # nicht noch mehr Overhead erzeugt.
    remaining_limit = 16 if imbalance is not None and imbalance >= 8 else 32
    for i in range(1, len(mopts)):
        mopts[i][1]["is_expanded"] = False
        if mopts[i][1]["max_group_count"] > remaining_limit:
            mopts[i][1]["group_count"] = min(
                eva.po2_near_sqrt(mopts[i][1]["max_group_count"]),
                remaining_limit,
            )

    return mopts


def manual_leaf_count_aware(index: eva.TeamIndex, query: str):
    # Leaf-count-aware Heuristik:
    # - sortiere Teams primaer nach Zahl getroffener Blaetter (max_group_count)
    # - bei Gleichstand nach union_cardinality
    # - expandiere nur das erste Team
    # - gruppiere weitere Teams moderat
    #
    # Unterschied zur Standardsortierung:
    # prepare_optimization ordnet primär nach union_cardinality.
    # Diese Heuristik priorisiert dagegen Teams mit wenigen betroffenen
    # Blaettern/Listen und ist damit eher overhead-orientiert.
    mopts = clone_mopts(index.prepare_optimization(query=query))
    if not mopts:
        return mopts

    mopts = sorted(
        mopts,
        key=lambda item: (item[1]["max_group_count"], item[1]["union_cardinality"]),
    )

    mopts[0][1]["is_expanded"] = True
    if mopts[0][1]["max_group_count"] > 64:
        mopts[0][1]["group_count"] = min(eva.po2_near_sqrt(mopts[0][1]["max_group_count"]), 64)

    for i in range(1, len(mopts)):
        if mopts[i][1]["max_group_count"] > 32:
            mopts[i][1]["group_count"] = min(eva.po2_near_sqrt(mopts[i][1]["max_group_count"]), 32)

    return mopts


VARIANTS = [
    {
        "name": "baseline_naive",
        "description": "Prepared team order only, no expansion and no manual regrouping",
        "builder": manual_baseline_naive,
    },
    {
        "name": "current_handcrafted",
        "description": "Current handwritten optimizer from run_example.py",
        "builder": manual_current_handcrafted,
    },
    {
        "name": "overhead_aware",
        "description": "Avoid unnecessary expansion and only group when many leaves are touched",
        "builder": manual_overhead_aware,
    },
    {
        "name": "imbalance_aware",
        "description": "Prioritize very small teams and stay conservative on strongly imbalanced queries",
        "builder": manual_imbalance_aware,
    },
    {
        "name": "leaf_count_aware",
        "description": "Prioritize teams with fewer touched leaves before applying expansion and grouping",
        "builder": manual_leaf_count_aware,
    },
]


def ensure_dirs():
    for sub in [
        OUT_DIR,
        OUT_DIR / "plans",
        OUT_DIR / "graphs",
        OUT_DIR / "stats",
        OUT_DIR / "plots",
    ]:
        sub.mkdir(parents=True, exist_ok=True)


def latest_artifact(base_path: Path) -> Path | None:
    pattern = f"{base_path.stem}-*{base_path.suffix}"
    matches = sorted(base_path.parent.glob(pattern))
    return matches[-1] if matches else None


def convert_dot_to_pdf(dot_path: Path):
    if shutil.which("dot") is None:
        raise RuntimeError(
            "Graphviz 'dot' was not found in PATH. Install graphviz or run the script in an environment where 'dot' is available."
        )
    pdf_path = (OUT_DIR / "plots" / f"{dot_path.stem}.pdf").resolve()
    subprocess.run(["dot", "-Tpdf", str(dot_path), "-o", str(pdf_path)], check=True)
    return pdf_path


def convert_all_execution_plans():
    pdf_paths = []
    for dot_path in sorted((OUT_DIR / "graphs").glob("*execution_plan-*.dot")):
        pdf_path = convert_dot_to_pdf(dot_path)
        if pdf_path is not None:
            pdf_paths.append(pdf_path)
    return pdf_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Run and evaluate mopts variants.")
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert existing execution_plan DOT files to PDFs.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip loading the large parquet file and do not compute pandas-based correctness references.",
    )
    return parser.parse_args()


def build_runtime_config(experiment_name: str):
    config = eva.get_new_default_runtime_config()
    config["backend"] = "dram"
    config["verbose_runtime"] = False
    config["return_result"] = True
    config["StorageConfig"]["queue_pair_count"] = 3
    config["experiment_name"] = experiment_name

    graph_base = (OUT_DIR / "graphs" / f"{experiment_name}-execution_plan.dot").resolve()
    task_stats_base = (OUT_DIR / "stats" / f"task_stats-{experiment_name}.json").resolve()
    result_stats_base = (OUT_DIR / "stats" / f"result_stats-{experiment_name}.json").resolve()
    task_graph_base = (OUT_DIR / "graphs" / f"{experiment_name}-task_graph.json").resolve()

    config["print_execution_plan"] = graph_base
    config["print_task_stats"] = task_stats_base
    config["print_result_stats"] = result_stats_base
    config["task_graph_path"] = task_graph_base
    return config, graph_base, task_stats_base, result_stats_base, task_graph_base


def save_worker_plot(task_stats_path: Path, figure_path: Path):
    task_data, _ = tib.import_benchmark_data(task_stats_path)
    tib.plot_worker_tasks(task_data, figure_path)


def summarize_mopts(query_id, query_name, query, variant_name, mopts, request_info):
    rows = []
    if mopts is None:
        return rows

    for team_name, opts in mopts:
        team_request_info = request_info.get(team_name, {})
        rows.append(
            {
                "query_id": query_id,
                "query_name": query_name,
                "query": query,
                "variant": variant_name,
                "team": team_name,
                "union_cardinality": opts.get("union_cardinality"),
                "selectivity": opts.get("selectivity"),
                "is_included": opts.get("is_included"),
                "is_expanded": opts.get("is_expanded"),
                "group_count": opts.get("group_count"),
                "max_group_count": opts.get("max_group_count"),
                "netto_data_volume_KiB": opts.get("netto_data_volume_KiB"),
                "io_volume_KiB": opts.get("io_volume_KiB"),
                "request_count": team_request_info.get("request_count"),
                "combined_id_count": team_request_info.get("combined_id_count"),
                "combined_list_count": team_request_info.get("combined_list_count"),
                "combined_request_size": team_request_info.get("combined_request_size"),
            }
        )
    return rows


def ratio_or_none(values):
    if not values:
        return None
    min_value = min(values)
    max_value = max(values)
    if min_value == 0:
        return None
    return max_value / min_value


def classify_query_domain(team_count, total_leaf_hits, imbalance_group_count):
    if team_count <= 1:
        if total_leaf_hits <= 8:
            return "single_team_small"
        return "single_team_wide"
    if team_count == 2:
        if imbalance_group_count is not None and imbalance_group_count >= 4:
            return "two_team_imbalanced"
        if total_leaf_hits <= 16:
            return "two_team_small"
        return "two_team_balanced"
    if total_leaf_hits <= 24:
        return "multi_team_medium"
    return "multi_team_wide"


def summarize_query_structure(mopts):
    if not mopts:
        return {
            "team_count": 0,
            "included_team_count_manual": 0,
            "expanded_team_count_manual": 0,
            "sum_max_group_count": 0,
            "sum_group_count": 0,
            "min_max_group_count": None,
            "max_max_group_count": None,
            "imbalance_group_count": None,
            "sum_union_cardinality": 0,
            "min_union_cardinality": None,
            "max_union_cardinality": None,
            "imbalance_union_cardinality": None,
            "query_domain": "empty",
        }

    max_group_counts = [int(opts["max_group_count"]) for _, opts in mopts]
    group_counts = [int(opts["group_count"]) for _, opts in mopts]
    union_cards = [int(opts["union_cardinality"]) for _, opts in mopts]
    included_count = sum(1 for _, opts in mopts if opts["is_included"])
    expanded_count = sum(1 for _, opts in mopts if opts["is_expanded"])
    imbalance_group_count = ratio_or_none(max_group_counts)

    return {
        "team_count": len(mopts),
        "included_team_count_manual": included_count,
        "expanded_team_count_manual": expanded_count,
        "sum_max_group_count": sum(max_group_counts),
        "sum_group_count": sum(group_counts),
        "min_max_group_count": min(max_group_counts),
        "max_max_group_count": max(max_group_counts),
        "imbalance_group_count": imbalance_group_count,
        "sum_union_cardinality": sum(union_cards),
        "min_union_cardinality": min(union_cards),
        "max_union_cardinality": max(union_cards),
        "imbalance_union_cardinality": ratio_or_none(union_cards),
        "query_domain": classify_query_domain(
            team_count=len(mopts),
            total_leaf_hits=sum(max_group_counts),
            imbalance_group_count=imbalance_group_count,
        ),
    }


def plot_runtime_comparison(df: pd.DataFrame, figure_path: Path):
    pivot = df.pivot(index="query_name", columns="variant", values="executor_runtime_ms")
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Executor Runtime (ms)")
    ax.set_xlabel("Query")
    ax.set_title("mopts Runtime Comparison")
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def plot_speedup_vs_baseline(df: pd.DataFrame, figure_path: Path):
    baseline = (
        df[df["variant"] == "baseline_naive"][["query_name", "executor_runtime_ms"]]
        .rename(columns={"executor_runtime_ms": "baseline_runtime_ms"})
    )
    merged = df.merge(baseline, on="query_name", how="left")
    merged = merged[merged["baseline_runtime_ms"].notna()].copy()
    merged["speedup_vs_baseline"] = merged["baseline_runtime_ms"] / merged["executor_runtime_ms"]

    pivot = merged.pivot(index="query_name", columns="variant", values="speedup_vs_baseline")
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Speedup vs baseline_naive")
    ax.set_xlabel("Query")
    ax.set_title("Speedup Relative to the Naive Baseline")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def main():
    args = parse_args()
    ensure_dirs()

    converted_plan_pdfs = convert_all_execution_plans()
    if args.convert_only:
        if converted_plan_pdfs:
            print("Converted execution-plan PDFs:")
            for pdf_path in converted_plan_pdfs:
                print(pdf_path)
        else:
            print("No execution_plan DOT files found in graphs/.")
        return

    table = None if args.no_reference else pd.read_parquet(DATA_PATH)
    index = eva.TeamIndex(INDEX_CONFIG)

    result_rows = []
    mopts_rows = []

    for query_id, (query_name, query) in enumerate(QUERIES, start=1):
        ref = set(table.query(query).index) if table is not None else None
        print(f"\n=== {query_name} ===")
        print(query)
        if ref is not None:
            print("Reference result size:", len(ref))
        else:
            print("Reference result size: skipped (--no-reference)")

        for variant in VARIANTS:
            variant_name = variant["name"]
            experiment_name = f"{query_name}-{variant_name}"
            config, graph_base, task_stats_base, result_stats_base, task_graph_base = build_runtime_config(experiment_name)

            if variant["builder"] is None:
                manual_mopts = None
                executed_mopts = clone_mopts(index.prepare_optimization(query=query))
            else:
                manual_mopts = variant["builder"](index, query)
                executed_mopts = clone_mopts(manual_mopts)

            query_structure = summarize_query_structure(executed_mopts)

            result_ids, runtime_stats, request_info, global_info = index.run_query(
                query,
                config=config,
                manual_optimizations=manual_mopts,
            )
            result_ids = set(result_ids)

            exported_plan_path = (OUT_DIR / "plans" / f"{experiment_name}-exported_plan.json").resolve()
            index.run_query(
                query,
                config=config,
                manual_optimizations=manual_mopts,
                dry_run=exported_plan_path,
            )

            latest_dot = latest_artifact(graph_base)
            latest_task_stats = latest_artifact(task_stats_base)
            latest_result_stats = latest_artifact(result_stats_base)
            task_graph_json = task_graph_base if task_graph_base.exists() else None

            worker_plot_path = None
            execution_plan_pdf = None
            if latest_dot is not None:
                execution_plan_pdf = convert_dot_to_pdf(latest_dot)
            if GENERATE_WORKER_PLOTS and latest_task_stats is not None:
                worker_plot_path = (OUT_DIR / "plots" / f"{experiment_name}-worker_tasks.pdf").resolve()
                save_worker_plot(latest_task_stats, worker_plot_path)

            result_rows.append(
                {
                    "query_id": query_id,
                    "query_name": query_name,
                    "query": query,
                    "variant": variant_name,
                    "variant_description": variant["description"],
                    **query_structure,
                    "ref_size": len(ref) if ref is not None else None,
                    "result_size": len(result_ids),
                    "correct_subset": ref.issubset(result_ids) if ref is not None else None,
                    "missing_true_hits": len(ref - result_ids) if ref is not None else None,
                    "extra_hits": len(result_ids - ref) if ref is not None else None,
                    "executor_runtime_ms": runtime_stats.executor_runtime / 1_000_000,
                    "plan_runtime_ms": runtime_stats.plan_construction_runtime / 1_000_000,
                    "total_request_count": global_info["total_request_count"],
                    "total_input_cardinality": global_info["total_input_cardinality"],
                    "total_read_volume_KiB": global_info["total_read_volume_KiB"],
                    "total_compressed_size_KB": global_info["total_compressed_size_KB"],
                    "ise_count": global_info["ise_count"],
                    "outer_union_term_count": global_info["outer_union_term_count"],
                    "outer_intersection_term_count": global_info["outer_intersection_term_count"],
                    "exported_plan_path": str(exported_plan_path),
                    "execution_plan_dot": str(latest_dot) if latest_dot else None,
                    "execution_plan_pdf": str(execution_plan_pdf) if execution_plan_pdf else None,
                    "task_graph_json": str(task_graph_json) if task_graph_json else None,
                    "task_stats_json": str(latest_task_stats) if latest_task_stats else None,
                    "result_stats_json": str(latest_result_stats) if latest_result_stats else None,
                    "worker_plot_pdf": str(worker_plot_path) if worker_plot_path else None,
                }
            )

            mopts_rows.extend(
                summarize_mopts(
                    query_id=query_id,
                    query_name=query_name,
                    query=query,
                    variant_name=variant_name,
                    mopts=executed_mopts,
                    request_info=request_info,
                )
            )

            print(
                f"{variant_name}: runtime={runtime_stats.executor_runtime / 1_000_000:.3f} ms, "
                f"missing={(len(ref - result_ids) if ref is not None else 'n/a')}, "
                f"extra={(len(result_ids - ref) if ref is not None else 'n/a')}, "
                f"result={len(result_ids)}"
            )

    results_df = pd.DataFrame(result_rows)
    mopts_df = pd.DataFrame(mopts_rows)

    baseline_df = results_df[results_df["variant"] == "baseline_naive"][
        ["query_name", "executor_runtime_ms"]
    ].rename(columns={"executor_runtime_ms": "baseline_runtime_ms"})
    comparison_df = results_df.merge(baseline_df, on="query_name", how="left")
    comparison_df["speedup_vs_baseline"] = (
        comparison_df["baseline_runtime_ms"] / comparison_df["executor_runtime_ms"]
    )

    results_csv = OUT_DIR / "results.csv"
    mopts_csv = OUT_DIR / "mopts_per_team.csv"
    comparison_csv = OUT_DIR / "comparison_vs_baseline.csv"

    results_df.to_csv(results_csv, index=False)
    mopts_df.to_csv(mopts_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)

    plot_runtime_comparison(results_df, OUT_DIR / "plots" / "runtime_comparison.pdf")
    plot_speedup_vs_baseline(results_df, OUT_DIR / "plots" / "speedup_vs_baseline.pdf")

    converted_plan_pdfs = convert_all_execution_plans()

    print("\nSaved:")
    print(results_csv)
    print(mopts_csv)
    print(comparison_csv)
    print(OUT_DIR / "plots" / "runtime_comparison.pdf")
    print(OUT_DIR / "plots" / "speedup_vs_baseline.pdf")
    if converted_plan_pdfs:
        print("\nExecution-plan PDFs:")
        for pdf_path in converted_plan_pdfs:
            print(pdf_path)
    print("\nExecution-plan PDFs and Taskflow JSONs are in:")
    print(OUT_DIR / "graphs")
    print("\nStandalone plans are in:")
    print(OUT_DIR / "plans")


if __name__ == "__main__":
    main()
