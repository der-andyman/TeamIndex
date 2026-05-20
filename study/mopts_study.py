#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from TeamIndex import benchmark as tib
from TeamIndex import evaluation as eva
from study_paths import DATA_PATH, INDEX_CONFIG


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = (BASE_DIR / "results").resolve()
GENERATE_WORKER_PLOTS = False
MAIN_BASELINE_VARIANT = "baseline_minimal_intersection"
DANGEROUS_ISE_WARNING = 5_000
DANGEROUS_VOLUME_IDS_WARNING = 1_000_000_000
DANGEROUS_LEAF_WARNING = 100


# A compact, explainable query set that covers:
# - one 3D team
# - one 2D team
# - mixed 3D/2D intersections
# - two 2D teams together
# - larger multi-team plans
# - wider stress queries that touch many leaves and produce longer runtimes
QUERIES = [
    ("q01_single_3d", "A < 19 and E < 19 and C < 19"),
    ("q02_single_2d", "B < 30 and I < 30"),
    ("q03_two_teams", "A < 19 and E < 19 and C < 19 and B < 19"),
    ("q04_two_2d_teams", "B < 30 and I < 30 and F < 30 and H < 30"),
    ("q05_three_teams", "A < 38 and E < 38 and C < 19 and B < 38 and F < 40"),
    ("q06_two_3d_teams", "A < 20 and E < 20 and C < 20 and J < 20 and D < 20 and G < 20"),
    ("q07_wide_2d_teams_75", "B < 75 and I < 75 and F < 75 and H < 75"),
    ("q08_wide_3d_teams_75", "A < 75 and E < 75 and C < 75 and J < 75 and D < 75 and G < 75"),
    ("q09_mixed_big_result", "A < 75 and E < 75 and B < 75 and I < 75 and F < 75 and H < 75"),
]

QUERY_PLOT_DESCRIPTIONS = {
    "q01_single_3d": "q01: einzelnes 3D-Team, kleine Query",
    "q02_single_2d": "q02: einzelnes 2D-Team, breite Ergebnisliste",
    "q03_two_teams": "q03: 3D+2D, kleine Zwei-Team-Intersection",
    "q04_two_2d_teams": "q04: zwei 2D-Teams, mittlere Intersection",
    "q05_three_teams": "q05: drei Teams, groesseres Working Set",
    "q06_two_3d_teams": "q06: zwei 3D-Teams, selektive Intersection",
    "q07_wide_2d_teams_75": "q07: zwei breite 2D-Teams, sehr viele Treffer",
    "q08_wide_3d_teams_75": "q08: zwei breite 3D-Teams, viele Blaetter",
    "q09_mixed_big_result": "q09: gemischte breite Query, sehr grosses Ergebnis",
}


def build_query_tick_labels(query_names):
    tick_labels = []
    notes = []
    for query_name in query_names:
        short_label = query_name.split("_", 1)[0]
        tick_labels.append(short_label)
        notes.append(QUERY_PLOT_DESCRIPTIONS.get(query_name, f"{short_label}: {query_name}"))
    return tick_labels, notes


def add_query_notes(ax, query_names):
    tick_labels, notes = build_query_tick_labels(list(query_names))
    ax.set_xticklabels(tick_labels, rotation=0)
    note_lines = ["   |   ".join(notes[i:i + 3]) for i in range(0, len(notes), 3)]
    note_text = "\n".join(note_lines)
    ax.figure.text(
        0.5,
        0.02,
        note_text,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    return note_text


def clone_mopts(mopts):
    return [(team_name, copy.deepcopy(opts)) for team_name, opts in mopts]


def product(values):
    return math.prod(values) if values else 0


def manual_baseline_union_first(index: eva.TeamIndex, query: str):
    # Baseline: Union First
    # ---------------------
    # Was passiert?
    # - uebernehme nur die von prepare_optimization bestimmte Team-Reihenfolge
    # - keine Expansion
    # - keine zusaetzliche Gruppierung
    #
    # Idee dahinter:
    # Diese Variante ist der minimale gueltige Referenzplan:
    # Innerhalb jedes Teams werden zuerst alle getroffenen Blaetter vereinigt.
    # Danach werden die fertigen Team-Ergebnisse miteinander geschnitten.
    #
    # Wichtig:
    # Das ist bewusst keine "gute" Optimierung, sondern eine einfache
    # theoretische Referenz. Sie zeigt, was passiert, wenn keine logische
    # Intersection-Optimierung vorgenommen wird.
    return clone_mopts(index.prepare_optimization(query=query))


def manual_baseline_minimal_intersection(index: eva.TeamIndex, query: str):
    # Baseline: Minimal Intersection
    # ------------------------------
    # Was passiert?
    # - sortiere wie prepare_optimization nach union_cardinality
    # - expandiere genau das kleinste/selectivste Team
    # - keine weitere Gruppierung
    # - kein zweites expandiertes Team
    #
    # Idee dahinter:
    # Das ist die einfachste Intersection-orientierte Baseline:
    # Ein kleines Team wird in die Intersections hineingezogen, damit
    # Zwischenergebnisse frueher reduziert werden koennen. Gleichzeitig
    # bleibt die Strategie bewusst minimal und wird nicht zu einer Kopie
    # der handgeschriebenen Heuristik aus dem alten Demo-Skript run_example.py.
    mopts = clone_mopts(index.prepare_optimization(query=query))
    if not mopts:
        return mopts

    # Bei Single-Team-Queries gibt es zwischen Teams nichts zu schneiden.
    if len(mopts) == 1:
        return mopts

    mopts[0][1]["is_expanded"] = True
    return mopts


def manual_current_handcrafted(index: eva.TeamIndex, query: str):
    # Optimizer: Current Handcrafted
    # --------------------------------
    # Was passiert?
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


def manual_expand_all_adaptive_grouping(index: eva.TeamIndex, query: str):
    # Optimizer: Expand All Adaptive Grouping
    # ---------------------------------------
    # Was passiert?
    # - expandiere grundsaetzlich alle beteiligten Teams
    # - gruppiere nicht bei trivial kleinen Plaenen
    # - wenn die rohe ISE-Komplexitaet hoch ist, gruppiere dynamisch
    # - gruppiere grosse/unselektive Teams staerker als kleine/selective Teams
    #
    # Idee:
    # Fruehes Intersecten ist oft gut, aber ungebremstes Expandieren kann
    # zu viele ISEs erzeugen. Diese Strategie versucht deshalb, alle Teams
    # fuer fruehe Intersections nutzbar zu machen, reduziert aber die
    # Granularitaet dort, wo sie voraussichtlich wenig bringt:
    # bei Teams mit hoher union_cardinality und vielen getroffenen Blaettern.
    #
    # Es gibt bewusst kein festes ISE_LIMIT. Stattdessen wird die
    # Gruppierungsstaerke aus raw_ise_count, Teamanzahl und Teamgroessen
    # abgeleitet. Je groesser die rohe ISE Count, desto staerker wird
    # gruppiert.
    mopts = clone_mopts(index.prepare_optimization(query=query))
    if not mopts:
        return mopts

    for _, opt in mopts:
        opt["is_expanded"] = True

    # Use the currently feasible group_count as upper bound. For excluded
    # Teams, prepare_optimization may switch to the complement selection,
    # which can have fewer usable leaves than max_group_count suggests.
    max_group_counts = [max(1, int(opt["group_count"])) for _, opt in mopts]
    raw_ise_count = product(max_group_counts)
    team_count = len(mopts)

    # Bei sehr kleinen Plaenen wuerde Gruppierung den Vorteil der Expansion
    # oft zerstoeren. Dort bleibt die Variante expand_all ohne Gruppierung.
    if team_count <= 1 or raw_ise_count <= 16:
        return mopts

    union_cards = [max(1, int(opt["union_cardinality"])) for _, opt in mopts]
    min_union_card = min(union_cards)

    # Dynamische Daempfung:
    # - raw_ise_factor waechst mit der rohen ISE-Komplexitaet pro Team
    # - alpha nahe 1 bedeutet wenig Gruppierung
    # - alpha kleiner bedeutet staerkere Gruppierung
    raw_ise_factor = max(1.0, math.log10(raw_ise_count) / team_count)
    alpha = max(0.35, min(0.90, 1.0 / raw_ise_factor))

    for _, opt in mopts:
        max_groups = max(1, int(opt["group_count"]))
        if max_groups <= 2:
            continue

        union_card = max(1, int(opt["union_cardinality"]))
        relative_size = union_card / min_union_card

        # Grosse/unselektive Teams bekommen einen etwas kleineren Exponenten
        # und werden dadurch staerker gruppiert. Kleine/selective Teams
        # behalten mehr Granularitaet, weil sie frueh gut filtern koennen.
        team_alpha = alpha / min(1.8, relative_size ** 0.15)
        team_alpha = max(0.25, min(0.90, team_alpha))

        grouped_count = int(round(max_groups ** team_alpha))
        grouped_count = max(1, min(grouped_count, max_groups - 1))
        opt["group_count"] = grouped_count

    return mopts


def manual_dynamic_selective_expansion(index: eva.TeamIndex, query: str):
    # Optimizer: Dynamic Selective Expansion
    # --------------------------------------
    # Was passiert?
    # - analysiere zuerst die Form der aktuellen Query
    # - beruecksichtige Teams, relevante Blaetter, gewaehlte Bins und
    #   Team-Volumina gemeinsam
    # - expandiere nicht blind, sondern nur Teams mit guenstigem
    #   Kosten-/Nutzen-Profil
    # - gruppiere expandierte Teams nur dann moderat, wenn die resultierende
    #   ISE Count fuer die aktuelle Query-Form sonst zu stark anwachsen wuerde
    #
    # Idee:
    # Diese Strategie soll nicht mehr einem festen Startpunkt wie
    # "Union First" oder "Expand First" folgen. Stattdessen versucht sie,
    # aus der Query-Struktur selbst abzuleiten, ob fruehe Intersections
    # ueberhaupt lohnend erscheinen. Kleine Single-Team- oder Low-Leaf-Queries
    # bleiben deshalb implizit bei einer union-nahen Ausfuehrung, waehrend
    # multi-teamige, strukturreichere Queries gezielt Expansion erhalten.
    mopts = clone_mopts(index.prepare_optimization(query=query))
    if not mopts:
        return mopts

    team_count = len(mopts)
    if team_count <= 1:
        return mopts

    slices_dict = index.query_to_slices(query, optimizations=mopts)
    total_leaf_hits = sum(int(opt["max_group_count"]) for _, opt in mopts)
    total_union_card = sum(int(opt["union_cardinality"]) for _, opt in mopts)
    min_union_card = min(max(1, int(opt["union_cardinality"])) for _, opt in mopts)
    worker_budget = max(1, int(index.default_runtime_config.get("worker_count", 1)))
    total_selected_bin_cells = 0
    candidate_infos = []

    for pos, (team_name, opt) in enumerate(mopts):
        if not bool(opt.get("is_included", True)):
            continue
        dims = index.cardinalities[team_name].shape
        slices = slices_dict[team_name]
        selected_bins_per_attribute = [
            _slice_len(slc, dim)
            for slc, dim in zip(slices, dims)
        ]
        selected_bin_cells = product(selected_bins_per_attribute)
        total_selected_bin_cells += selected_bin_cells

        feasible_groups = max(1, int(opt["group_count"]))
        union_card = max(1, int(opt["union_cardinality"]))
        relative_size = union_card / min_union_card
        avg_ids_per_leaf = union_card / feasible_groups

        # Hoher Score bedeutet:
        # - wenige Blaetter / ISE-gunstig
        # - kleines Team / hohe potenzielle Filterwirkung
        # - feine Bin-Ausdehnung statt extrem breiter geometrischer Query
        leaf_score = 1.0 / feasible_groups
        size_score = 1.0 / relative_size
        geometry_score = 1.0 / max(1.0, math.sqrt(selected_bin_cells))
        density_penalty = max(1.0, math.log10(max(avg_ids_per_leaf, 10)))
        expansion_score = (leaf_score * size_score * geometry_score) / density_penalty

        candidate_infos.append(
            {
                "pos": pos,
                "team_name": team_name,
                "feasible_groups": feasible_groups,
                "union_cardinality": union_card,
                "relative_size": relative_size,
                "selected_bin_cells": selected_bin_cells,
                "selected_bins_per_attribute": selected_bins_per_attribute,
                "expansion_score": expansion_score,
            }
        )

    # Sehr kleine oder geometrisch triviale Plaene profitieren selten von
    # fruehen Team-Intersections.
    if total_leaf_hits < max(6, team_count + 2):
        return mopts
    if total_selected_bin_cells < max(8, team_count * 2):
        return mopts

    # Nur Queries mit genug Arbeitspotenzial sollten ueberhaupt expandieren.
    query_large_enough = (
        total_union_card >= 50_000_000
        or total_leaf_hits >= 12
        or total_selected_bin_cells >= 24
    )
    if not query_large_enough:
        return mopts

    candidate_infos = sorted(
        candidate_infos,
        key=lambda item: (
            -item["expansion_score"],
            item["relative_size"],
            item["feasible_groups"],
        ),
    )

    selected_positions = []
    current_ise = 1
    ise_budget = max(worker_budget * 3, 24)

    for info in candidate_infos:
        # Gute Expansionskandidaten sollen
        # - nicht riesig sein
        # - nicht zu viele Blaetter haben
        # - geometrisch nicht zu breit sein
        good_local_shape = (
            info["relative_size"] <= 6.0
            and info["feasible_groups"] <= max(16, worker_budget)
            and info["selected_bin_cells"] <= max(64, worker_budget * 2)
        )
        if not good_local_shape:
            continue

        # Gleichzeitig muss der Team-Score hoch genug sein, damit wir nicht
        # aus Prinzip expandieren.
        if info["expansion_score"] < 0.003:
            continue

        projected_ise = current_ise * info["feasible_groups"]
        if projected_ise > ise_budget:
            continue

        selected_positions.append(info["pos"])
        current_ise = projected_ise

        # Nicht alle Teams expandieren: wir stoppen spaetestens nach zwei
        # Teams oder wenn die ISE Count schon einen sinnvollen Arbeitsvorrat
        # fuer die Threads erzeugt.
        if len(selected_positions) >= 2:
            break
        if current_ise >= worker_budget:
            break

    # Wenn kein Team ein gutes Profil hat, bleibt die Query implizit bei einer
    # union-nahen Ausfuehrung.
    if not selected_positions:
        return mopts

    for pos in selected_positions:
        _, opt = mopts[pos]
        opt["is_expanded"] = True

    # Nur expandierte Teams werden gegebenenfalls gruppiert.
    # Kleine ISE-Werte bleiben unberuehrt, groessere werden moderat gedaempft.
    for pos in selected_positions:
        _, opt = mopts[pos]
        max_groups = max(1, int(opt["group_count"]))
        if max_groups <= 2:
            continue

        union_card = max(1, int(opt["union_cardinality"]))
        relative_size = union_card / min_union_card

        # Je groesser die aktuelle ISE Count und je groesser das Team,
        # desto staerker gruppieren wir.
        alpha = 0.85 if current_ise <= worker_budget else 0.70
        alpha /= min(1.5, relative_size ** 0.10)
        alpha = max(0.45, min(0.95, alpha))

        grouped_count = int(round(max_groups ** alpha))
        grouped_count = max(1, min(grouped_count, max_groups))
        opt["group_count"] = grouped_count

    return mopts


VARIANTS = [
    {
        "name": "baseline_union_first",
        "description": "Union-first reference: no expansion and no manual grouping",
        "builder": manual_baseline_union_first,
    },
    {
        "name": "baseline_minimal_intersection",
        "description": "Minimal intersection baseline: expand only the smallest team, no grouping",
        "builder": manual_baseline_minimal_intersection,
    },
    {
        "name": "current_handcrafted",
        "description": "Current handwritten optimizer from the old run_example.py demo",
        "builder": manual_current_handcrafted,
    },
    {
        "name": "dynamic_selective_expansion",
        "description": "Start from union-first and expand only teams with favorable cost/benefit",
        "builder": manual_dynamic_selective_expansion,
    },
    {
        "name": "expand_all_adaptive_grouping",
        "description": "Expand every team and dynamically group large/unselective teams",
        "builder": manual_expand_all_adaptive_grouping,
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


def archive_previous_outputs():
    run_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    moved_anything = False

    for subdir_name in ["plans", "graphs", "stats", "plots"]:
        subdir = OUT_DIR / subdir_name
        archive_dir = subdir / run_stamp
        files_to_move = [path for path in subdir.iterdir() if path.is_file()]
        if not files_to_move:
            continue
        archive_dir.mkdir(parents=True, exist_ok=True)
        for path in files_to_move:
            shutil.move(str(path), str(archive_dir / path.name))
        moved_anything = True

    summary_archive_dir = OUT_DIR / "archives" / run_stamp
    summary_files = [
        OUT_DIR / "results.csv",
        OUT_DIR / "mopts_per_team.csv",
        OUT_DIR / "comparison_vs_baseline.csv",
        OUT_DIR / "runtime_comparison.pdf",
        OUT_DIR / "speedup_vs_baseline.pdf",
        OUT_DIR / "speedup_vs_baseline_runtime.pdf",
        OUT_DIR / "ids_per_second_comparison.pdf",
        OUT_DIR / "mib_per_second_comparison.pdf",
    ]
    existing_summary_files = [path for path in summary_files if path.exists()]
    if existing_summary_files:
        summary_archive_dir.mkdir(parents=True, exist_ok=True)
        for path in existing_summary_files:
            shutil.move(str(path), str(summary_archive_dir / path.name))
        moved_anything = True

    return run_stamp if moved_anything else None


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
        "--plots-only",
        action="store_true",
        help="Only regenerate summary plots from the existing results.csv without running benchmarks.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip loading the large parquet file and do not compute pandas-based correctness references.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help="Override runtime worker_count. Useful for thread-scaling experiments.",
    )
    parser.add_argument(
        "--queue-pair-count",
        type=int,
        default=None,
        help="Override StorageConfig.queue_pair_count. Defaults to the library runtime default.",
    )
    parser.add_argument(
        "--verbose-runtime",
        action="store_true",
        help="Enable verbose runtime output from TeamIndex.",
    )
    parser.add_argument(
        "--query-filter",
        action="append",
        default=[],
        help="Only run queries whose name contains the given substring. Can be passed multiple times.",
    )
    parser.add_argument(
        "--skip-dangerous",
        action="store_true",
        help="Skip variants that trigger the built-in stress warnings for very large ISE/volume combinations.",
    )
    return parser.parse_args()


def build_runtime_config(experiment_name: str, args):
    config = eva.get_new_default_runtime_config()
    config["backend"] = "dram"
    config["verbose_runtime"] = args.verbose_runtime
    config["return_result"] = True
    config["experiment_name"] = experiment_name
    if args.worker_count is not None:
        config["worker_count"] = args.worker_count
    if args.queue_pair_count is not None:
        config["StorageConfig"]["queue_pair_count"] = args.queue_pair_count

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


def _slice_len(slc, dim_size):
    if isinstance(slc, slice):
        start, stop, step = slc.indices(dim_size)
        return len(range(start, stop, step))
    if isinstance(slc, int):
        return 1
    if hasattr(slc, "dtype") and hasattr(slc, "shape"):
        if getattr(slc, "dtype", None) == bool:
            return int(slc.sum())
        return int(len(slc))
    raise TypeError(f"Unsupported slicer type: {type(slc)!r}")


def summarize_bin_selection(index: eva.TeamIndex, query: str, mopts):
    if not mopts:
        return {}, {
            "total_selected_bin_cells": 0,
            "total_selected_attribute_bins": 0,
        }

    slices_dict = index.query_to_slices(query, optimizations=mopts)
    per_team = {}
    total_selected_bin_cells = 0
    total_selected_attribute_bins = 0

    for team_name, opts in mopts:
        dims = index.cardinalities[team_name].shape
        attributes = index.teams[team_name]
        slices = slices_dict[team_name]
        per_attr = {
            attr: _slice_len(slc, dim)
            for attr, slc, dim in zip(attributes, slices, dims)
        }
        selected_bin_cells = product(per_attr.values())
        total_selected_bin_cells += selected_bin_cells
        total_selected_attribute_bins += sum(per_attr.values())
        per_team[team_name] = {
            "team_dimension_count": len(attributes),
            "selected_bin_count_product": selected_bin_cells,
            "selected_bin_counts_per_attribute": per_attr,
        }

    return per_team, {
        "total_selected_bin_cells": total_selected_bin_cells,
        "total_selected_attribute_bins": total_selected_attribute_bins,
    }


def summarize_mopts(query_id, query_name, query, variant_name, mopts, request_info, bin_info_by_team):
    rows = []
    if mopts is None:
        return rows

    for team_name, opts in mopts:
        team_request_info = request_info.get(team_name, {})
        team_bin_info = bin_info_by_team.get(team_name, {})
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
                "team_dimension_count": team_bin_info.get("team_dimension_count"),
                "selected_bin_count_product": team_bin_info.get("selected_bin_count_product"),
                "selected_bin_counts_per_attribute": json.dumps(
                    team_bin_info.get("selected_bin_counts_per_attribute", {}),
                    sort_keys=True,
                ),
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


def estimate_runtime_stress(query_name, variant_name, query_structure):
    warnings = []
    if query_structure["team_count"] >= 2 and query_structure["sum_max_group_count"] >= DANGEROUS_LEAF_WARNING:
        warnings.append(
            f"many leaves selected ({query_structure['sum_max_group_count']})"
        )
    if query_structure["ise_count_estimate_manual"] >= DANGEROUS_ISE_WARNING:
        warnings.append(
            f"high estimated ISE count ({query_structure['ise_count_estimate_manual']})"
        )
    if (
        query_structure["expanded_team_count_manual"] >= 2
        and query_structure["sum_union_cardinality"] >= DANGEROUS_VOLUME_IDS_WARNING
        and query_structure["sum_max_group_count"] >= DANGEROUS_LEAF_WARNING
    ):
        warnings.append(
            f"large expanded volume ({query_structure['sum_union_cardinality']} ids over all teams)"
        )
    if warnings:
        return (
            f"WARNING {query_name}/{variant_name}: "
            + "; ".join(warnings)
        )
    return None


def plot_runtime_comparison(df: pd.DataFrame, figure_path: Path):
    pivot = df.pivot(index="query_name", columns="variant", values="executor_runtime_ms")
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Executor Runtime (ms)")
    ax.set_xlabel("Query")
    ax.set_title("mopts Runtime Comparison")
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    add_query_notes(ax, pivot.index)
    plt.tight_layout(rect=(0, 0.10, 1, 1))
    plt.savefig(figure_path)
    plt.close()


def plot_speedup_vs_baseline(df: pd.DataFrame, figure_path: Path):
    baseline = (
        df[df["variant"] == MAIN_BASELINE_VARIANT][["query_name", "executor_runtime_ms"]]
        .rename(columns={"executor_runtime_ms": "baseline_runtime_ms"})
    )
    merged = df.merge(baseline, on="query_name", how="left")
    merged = merged[merged["baseline_runtime_ms"].notna()].copy()
    merged["speedup_vs_baseline"] = merged["baseline_runtime_ms"] / merged["executor_runtime_ms"]

    pivot = merged.pivot(index="query_name", columns="variant", values="speedup_vs_baseline")
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel(f"Speedup vs {MAIN_BASELINE_VARIANT}")
    ax.set_xlabel("Query")
    ax.set_title("Speedup Relative to the Main Baseline")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    add_query_notes(ax, pivot.index)
    plt.tight_layout(rect=(0, 0.10, 1, 1))
    plt.savefig(figure_path)
    plt.close()


def plot_metric_comparison(
    df: pd.DataFrame,
    metric_column: str,
    figure_path: Path,
    ylabel: str,
    title: str,
):
    pivot = df.pivot(index="query_name", columns="variant", values=metric_column)
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Query")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    add_query_notes(ax, pivot.index)
    plt.tight_layout(rect=(0, 0.10, 1, 1))
    plt.savefig(figure_path)
    plt.close()


def generate_all_summary_plots(results_df: pd.DataFrame):
    runtime_pdf = OUT_DIR / "runtime_comparison.pdf"
    speedup_runtime_pdf = OUT_DIR / "speedup_vs_baseline_runtime.pdf"
    ids_pdf = OUT_DIR / "ids_per_second_comparison.pdf"
    mib_pdf = OUT_DIR / "mib_per_second_comparison.pdf"

    plot_runtime_comparison(results_df, runtime_pdf)
    plot_speedup_vs_baseline(results_df, speedup_runtime_pdf)
    plot_metric_comparison(
        results_df,
        metric_column="ids_per_second",
        figure_path=ids_pdf,
        ylabel="IDs per Second",
        title="IDs per Second Comparison",
    )
    plot_metric_comparison(
        results_df,
        metric_column="read_mib_per_second",
        figure_path=mib_pdf,
        ylabel="Read MiB per Second",
        title="Read MiB per Second Comparison",
    )
    return [
        runtime_pdf,
        speedup_runtime_pdf,
        ids_pdf,
        mib_pdf,
    ]


def write_current_outputs(result_rows, mopts_rows):
    results_df = pd.DataFrame(result_rows)
    mopts_df = pd.DataFrame(mopts_rows)

    baseline_df = results_df[results_df["variant"] == MAIN_BASELINE_VARIANT][
        ["query_name", "executor_runtime_ms"]
    ].rename(columns={"executor_runtime_ms": "baseline_runtime_ms"})
    comparison_df = results_df.merge(baseline_df, on="query_name", how="left")
    comparison_df["speedup_vs_baseline"] = (
        comparison_df["baseline_runtime_ms"] / comparison_df["executor_runtime_ms"]
    )
    comparison_df["relative_runtime_vs_baseline"] = (
        comparison_df["executor_runtime_ms"] / comparison_df["baseline_runtime_ms"]
    )

    results_csv = OUT_DIR / "results.csv"
    mopts_csv = OUT_DIR / "mopts_per_team.csv"
    comparison_csv = OUT_DIR / "comparison_vs_baseline.csv"

    results_df.to_csv(results_csv, index=False)
    mopts_df.to_csv(mopts_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)
    return results_df, mopts_df, comparison_df, results_csv, mopts_csv, comparison_csv


def main():
    args = parse_args()
    ensure_dirs()

    if args.convert_only:
        converted_plan_pdfs = convert_all_execution_plans()
        if converted_plan_pdfs:
            print("Converted execution-plan PDFs:")
            for pdf_path in converted_plan_pdfs:
                print(pdf_path)
        else:
            print("No execution_plan DOT files found in graphs/.")
        return

    if args.plots_only:
        results_csv = OUT_DIR / "results.csv"
        if not results_csv.exists():
            raise RuntimeError(f"Missing results file for --plots-only: {results_csv}")
        results_df = pd.read_csv(results_csv)
        plot_paths = generate_all_summary_plots(results_df)
        print("Regenerated summary plots from existing results.csv:")
        for plot_path in plot_paths:
            print(plot_path)
        return

    archived_run_stamp = archive_previous_outputs()
    if archived_run_stamp is not None:
        print(f"Archived previous outputs under timestamp: {archived_run_stamp}")

    table = None if args.no_reference else pd.read_parquet(DATA_PATH)
    index = eva.TeamIndex(INDEX_CONFIG)
    selected_queries = [
        item for item in QUERIES
        if not args.query_filter
        or any(pattern in item[0] for pattern in args.query_filter)
    ]
    if not selected_queries:
        raise RuntimeError("No queries matched --query-filter.")

    result_rows = []
    mopts_rows = []

    for query_id, (query_name, query) in enumerate(selected_queries, start=1):
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
            config, graph_base, task_stats_base, result_stats_base, task_graph_base = build_runtime_config(experiment_name, args)

            if variant["builder"] is None:
                manual_mopts = None
                executed_mopts = clone_mopts(index.prepare_optimization(query=query))
            else:
                manual_mopts = variant["builder"](index, query)
                executed_mopts = clone_mopts(manual_mopts)

            query_structure = summarize_query_structure(executed_mopts)
            query_structure["ise_count_estimate_manual"] = product(
                [int(opts["group_count"]) for _, opts in executed_mopts if opts["is_expanded"]]
            ) if query_structure["expanded_team_count_manual"] > 0 else 0
            bin_info_by_team, bin_summary = summarize_bin_selection(index, query, executed_mopts)
            query_structure.update(bin_summary)
            stress_warning = estimate_runtime_stress(query_name, variant_name, query_structure)
            if stress_warning:
                print(stress_warning)
                if args.skip_dangerous:
                    print(f"Skipping {query_name}/{variant_name} because --skip-dangerous is active.")
                    continue

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
                    "worker_count": config["worker_count"],
                    "queue_pair_count": config["StorageConfig"]["queue_pair_count"],
                    "total_request_count": global_info["total_request_count"],
                    "total_input_cardinality": global_info["total_input_cardinality"],
                    "total_read_volume_KiB": global_info["total_read_volume_KiB"],
                    "total_compressed_size_KB": global_info["total_compressed_size_KB"],
                    "ids_per_second": (
                        global_info["total_input_cardinality"] / (runtime_stats.executor_runtime / 1_000_000_000)
                    ),
                    "million_ids_per_second": (
                        global_info["total_input_cardinality"] / (runtime_stats.executor_runtime / 1_000_000_000) / 1_000_000
                    ),
                    "read_mib_per_second": (
                        (global_info["total_read_volume_KiB"] / 1024)
                        / (runtime_stats.executor_runtime / 1_000_000_000)
                    ),
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
                    bin_info_by_team=bin_info_by_team,
                )
            )

            write_current_outputs(result_rows, mopts_rows)

            print(
                f"{variant_name}: runtime={runtime_stats.executor_runtime / 1_000_000:.3f} ms, "
                f"missing={(len(ref - result_ids) if ref is not None else 'n/a')}, "
                f"extra={(len(result_ids - ref) if ref is not None else 'n/a')}, "
                f"result={len(result_ids)}"
            )

    results_df, mopts_df, comparison_df, results_csv, mopts_csv, comparison_csv = write_current_outputs(
        result_rows,
        mopts_rows,
    )

    plot_paths = generate_all_summary_plots(results_df)

    converted_plan_pdfs = convert_all_execution_plans()

    print("\nSaved:")
    print(results_csv)
    print(mopts_csv)
    print(comparison_csv)
    for plot_path in plot_paths:
        print(plot_path)
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
