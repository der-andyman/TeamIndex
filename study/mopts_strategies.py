#!/usr/bin/env python3
"""
Strategiedefinitionen fuer die mopts- und team_bench-Studien.

Diese Datei enthaelt nur die eigentlichen Optimierungsstrategien:
- die Builder-Funktionen fuer manual_optimizations
- eine gemeinsame VARIANTS-Registry

Damit koennen sowohl mopts_study.py als auch run_team_bench_experiments.py
dieselben Strategien nutzen, ohne dass die Benchmark-Logik selbst daran
gekoppelt bleibt.
"""

from __future__ import annotations

import copy
import math

from TeamIndex import evaluation as eva


def clone_mopts(mopts):
    return [(team_name, copy.deepcopy(opts)) for team_name, opts in mopts]


def product(values):
    return math.prod(values) if values else 0


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

    if len(mopts) == 1:
        return mopts

    mopts[0][1]["is_expanded"] = True
    return mopts


def manual_union_first_parallel(index: eva.TeamIndex, query: str):
    # Optimizer: Union First Parallel
    # -------------------------------
    # Was passiert?
    # - keine Expansion, also bleibt der Plan logisch Union-First-aehnlich
    # - grosse Team-Unions werden aber in eine worker-orientierte Anzahl
    #   balancierter Gruppen zerlegt
    # - die Runtime verteilt die Blaetter innerhalb dieser Gruppen bereits
    #   greedily nach Listengroesse/Cardinality
    #
    # Idee:
    # Mehr physische Parallelitaet fuer grosse Unions, ohne die ISE Count
    # durch weitere logische Expansion zu erhoehen.
    mopts = clone_mopts(index.prepare_optimization(query=query))
    if not mopts:
        return mopts

    worker_budget = max(1, int(index.default_runtime_config.get("worker_count", 16) or 16))
    included = [(team_name, opt) for team_name, opt in mopts if bool(opt.get("is_included", True))]
    if not included:
        return mopts

    total_union_cardinality = sum(max(1, int(opt["union_cardinality"])) for _, opt in included)
    total_group_budget = max(worker_budget, len(included) * 2)

    for _, opt in mopts:
        opt["is_expanded"] = False
        max_groups = max(1, int(opt.get("max_group_count", opt.get("group_count", 1))))
        if max_groups <= 1:
            opt["group_count"] = 1
            continue

        union_cardinality = max(1, int(opt["union_cardinality"]))
        share = union_cardinality / total_union_cardinality
        target_groups = max(1, int(math.ceil(total_group_budget * share)))

        # Keep very large teams from collapsing into too few giant unions.
        if max_groups > worker_budget * 8:
            target_groups = max(target_groups, worker_budget)

        opt["group_count"] = min(max_groups, target_groups)

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


def _team_dimension(index: eva.TeamIndex, team_name: str) -> int:
    return len(index.cardinalities[team_name].shape)


def _bounded_group_count(max_group_count: int, worker_budget: int) -> int:
    if max_group_count <= 1:
        return 1
    if max_group_count <= 128:
        return max_group_count

    # sqrt grouping keeps large unions manageable without exploding ISE count.
    target = eva.po2_near_sqrt(max_group_count)
    return max(1, min(max_group_count, target, max(128, worker_budget * 4)))


def manual_bounded_selective_expansion(index: eva.TeamIndex, query: str):
    # Optimizer: Bounded Selective Expansion
    # --------------------------------------
    # Was passiert?
    # - 2D-Faelle bleiben bewusst Union-First-aehnlich, weil dort Overhead
    #   und Varianz schnell den Nutzen der Expansion auffressen.
    # - ab 3D werden grosse Team-Ergebnisse zuerst auf eine begrenzte
    #   Gruppenzahl reduziert
    # - expandiert wird nur ein selektives Team, damit fruehes Pruning
    #   moeglich ist, ohne ungebremste ISE Counts zu erzeugen
    mopts = clone_mopts(index.prepare_optimization(query=query))
    if len(mopts) <= 1:
        return mopts

    worker_budget = max(1, int(index.default_runtime_config.get("worker_count", 16) or 16))
    team_dims = [_team_dimension(index, team_name) for team_name, _ in mopts]
    if max(team_dims, default=0) <= 2:
        return mopts

    candidates = []
    for pos, (team_name, opt) in enumerate(mopts):
        if not bool(opt.get("is_included", True)):
            continue

        dimension = _team_dimension(index, team_name)
        max_groups = max(1, int(opt.get("max_group_count", opt.get("group_count", 1))))
        bounded_groups = _bounded_group_count(max_groups, worker_budget)
        opt["group_count"] = bounded_groups

        if dimension < 3 or bounded_groups <= 1:
            continue

        candidates.append(
            {
                "pos": pos,
                "bounded_groups": bounded_groups,
                "union_cardinality": max(1, int(opt["union_cardinality"])),
                "dimension": dimension,
            }
        )

    if not candidates:
        return mopts

    candidates = sorted(
        candidates,
        key=lambda item: (
            item["union_cardinality"],
            -item["dimension"],
            item["bounded_groups"],
        ),
    )
    first = candidates[0]
    mopts[first["pos"]][1]["is_expanded"] = True

    # Ein zweites Team nur bei sehr feiner erster Expansion. Das verhindert
    # den alten "expandiere alles"-Fehler, erlaubt aber mehr Parallelitaet,
    # wenn die erste Expansion kaum ISEs erzeugt.
    current_ise = first["bounded_groups"]
    second_limit = max(worker_budget, 32)
    if current_ise < max(8, worker_budget // 2):
        for candidate in candidates[1:]:
            projected_ise = current_ise * candidate["bounded_groups"]
            if projected_ise <= second_limit:
                mopts[candidate["pos"]][1]["is_expanded"] = True
                break

    return mopts


def manual_expand_all_adaptive_grouping(index: eva.TeamIndex, query: str):
    # Optimizer: Expand All Adaptive Grouping
    # ---------------------------------------
    # Was passiert?
    # - expandiere grundsaetzlich alle beteiligten Teams
    # - gruppiere nicht bei trivial kleinen Plaenen
    # - wenn die rohe ISE-Komplexitaet hoch ist, gruppiere dynamisch
    # - gruppiere grosse/unselektive Teams staerker als kleine/selective Teams
    mopts = clone_mopts(index.prepare_optimization(query=query))
    if not mopts:
        return mopts

    for _, opt in mopts:
        opt["is_expanded"] = True

    max_group_counts = [max(1, int(opt["group_count"])) for _, opt in mopts]
    raw_ise_count = product(max_group_counts)
    team_count = len(mopts)
    if team_count <= 1 or raw_ise_count <= 16:
        return mopts

    union_cards = [max(1, int(opt["union_cardinality"])) for _, opt in mopts]
    min_union_card = min(union_cards)
    raw_ise_factor = max(1.0, math.log10(raw_ise_count) / team_count)
    alpha = max(0.35, min(0.90, 1.0 / raw_ise_factor))

    for _, opt in mopts:
        max_groups = max(1, int(opt["group_count"]))
        if max_groups <= 2:
            continue

        union_card = max(1, int(opt["union_cardinality"]))
        relative_size = union_card / min_union_card
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

    if total_leaf_hits < max(6, team_count + 2):
        return mopts
    if total_selected_bin_cells < max(8, team_count * 2):
        return mopts

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
        good_local_shape = (
            info["relative_size"] <= 6.0
            and info["feasible_groups"] <= max(16, worker_budget)
            and info["selected_bin_cells"] <= max(64, worker_budget * 2)
        )
        if not good_local_shape:
            continue
        if info["expansion_score"] < 0.003:
            continue

        projected_ise = current_ise * info["feasible_groups"]
        if projected_ise > ise_budget:
            continue

        selected_positions.append(info["pos"])
        current_ise = projected_ise

        if len(selected_positions) >= 2:
            break
        if current_ise >= worker_budget:
            break

    if not selected_positions:
        return mopts

    for pos in selected_positions:
        _, opt = mopts[pos]
        opt["is_expanded"] = True

    for pos in selected_positions:
        _, opt = mopts[pos]
        max_groups = max(1, int(opt["group_count"]))
        if max_groups <= 2:
            continue

        union_card = max(1, int(opt["union_cardinality"]))
        relative_size = union_card / min_union_card
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
        "name": "union_first_parallel",
        "description": "Union-first plan with parallelized per-team union groups",
        "builder": manual_union_first_parallel,
    },
    {
        "name": "current_handcrafted",
        "description": "Current handwritten optimizer from the old run_example.py demo",
        "builder": manual_current_handcrafted,
    },
    {
        "name": "bounded_selective_expansion",
        "description": "Selective one-team expansion with bounded grouping and no 2D expansion",
        "builder": manual_bounded_selective_expansion,
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
