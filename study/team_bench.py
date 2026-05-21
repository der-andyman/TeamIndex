"""Hilfsfunktionen fuer kontrollierte synthetische TeamIndex-Benchmarks.

Dieses Modul konzentriert sich auf die Datengenerierung fuer definierte Query-Regionen.
TeamIndex selbst wird nur fuer den optionalen LHCb-Helfer am Ende benoetigt.
`numba` ist ebenfalls optional und dient lediglich als Beschleuniger fuer eine
Sampling-Funktion.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, lsq_linear

try:
    from numba import njit
except ImportError:
    def njit(func=None, **_kwargs):
        if func is None:
            def decorator(inner):
                return inner
            return decorator
        return func


def aggregate_dicts(dict_list):
    aggregated = defaultdict(int)
    for d in dict_list:
        for key, value in d.items():
            aggregated[key] += value
    return dict(aggregated)


def count_bins_by_shell(d, b):
    """Count bins by rounded Euclidean shell index around the grid center."""
    center = (b - 1) / 2.0
    shell_counts = defaultdict(int)
    for coords in product(range(b), repeat=d):
        dist_sq = 0.0
        for c in coords:
            diff = c - center
            dist_sq += diff * diff
        shell_idx = int(round(math.sqrt(dist_sq)))
        shell_counts[shell_idx] += 1
    return dict(sorted(shell_counts.items()))


def radial_shell_uniform_marginals(d, b, max_shell=None):
    """Build a radially symmetric distribution with uniform 1D marginals."""
    center = (b - 1) / 2.0
    bin_to_shell = {}
    shell_to_bins = defaultdict(list)

    def dist_euclidean(coords):
        return math.sqrt(sum((c - center) ** 2 for c in coords))

    for coords in product(range(b), repeat=d):
        shell_idx = int(round(dist_euclidean(coords)))
        bin_to_shell[coords] = shell_idx
        shell_to_bins[shell_idx].append(coords)

    if max_shell is not None:
        for shell_idx in list(shell_to_bins.keys()):
            if shell_idx > max_shell:
                for coords in shell_to_bins[shell_idx]:
                    bin_to_shell[coords] = None
                del shell_to_bins[shell_idx]

    shells = sorted(shell_to_bins.keys())
    r_to_varidx = {r: i for i, r in enumerate(shells)}
    num_vars = len(shells)

    A_eq = np.zeros((d * b + 1, num_vars), dtype=float)
    b_eq = np.zeros(d * b + 1, dtype=float)
    slice_bin_count = defaultdict(int)

    for r in shells:
        for coords in shell_to_bins[r]:
            for axis in range(d):
                slice_bin_count[(axis, coords[axis], r)] += 1

    row_idx = 0
    for axis in range(d):
        for alpha in range(b):
            for r in shells:
                A_eq[row_idx, r_to_varidx[r]] = slice_bin_count.get((axis, alpha, r), 0)
            b_eq[row_idx] = 1.0 / b
            row_idx += 1

    for r in shells:
        A_eq[row_idx, r_to_varidx[r]] = len(shell_to_bins[r])
    b_eq[row_idx] = 1.0

    res = linprog(np.zeros(num_vars), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * num_vars, method="highs")
    if not res.success:
        raise ValueError("No feasible shell distribution found.")

    p = {}
    for coords in product(range(b), repeat=d):
        shell_idx = bin_to_shell[coords]
        if shell_idx is None:
            p[coords] = 0.0
        else:
            p[coords] = res.x[r_to_varidx[shell_idx]]
    return p


def preprocess_grid_distribution(p_matrix, query_slices):
    """Condition one grid distribution on the query region and normalize it."""
    p_query = p_matrix[query_slices]
    p_query = np.nan_to_num(p_query, nan=0.0)
    query_mass = np.sum(p_query)
    if query_mass <= 0:
        raise ValueError("The query region has zero total mass; check p_matrix and query_slices.")
    p_conditioned = p_query / query_mass
    return p_conditioned, query_mass


def preprocess_all_grids(queries, distributions):
    """Condition all grid distributions on their respective query region."""
    conditioned_data = {}
    for grid_id, query_slices in queries.items():
        p_matrix = distributions[grid_id]
        p_conditioned, query_mass = preprocess_grid_distribution(p_matrix, query_slices)
        conditioned_data[grid_id] = {
            "p_conditioned": p_conditioned,
            "w": query_mass,
            "unconditioned_shape": p_matrix.shape,
            "query_slices": query_slices,
        }
    return conditioned_data


def distribute_tasks(n_threads, intersection_count, drag_count):
    tasks = []
    total_work = intersection_count + drag_count
    if total_work == 0:
        return tasks

    total_chunks = min(n_threads, total_work)

    if intersection_count > 0 and drag_count > 0:
        tasks_intersection = max(1, round(total_chunks * intersection_count / total_work))
        tasks_drag = total_chunks - tasks_intersection
        if tasks_drag == 0:
            tasks_drag = 1
            tasks_intersection = total_chunks - 1
    elif intersection_count > 0:
        tasks_intersection = total_chunks
        tasks_drag = 0
    else:
        tasks_drag = total_chunks
        tasks_intersection = 0

    def split_work(count, chunks):
        base = count // chunks
        remainder = count % chunks
        return [base + 1 if i < remainder else base for i in range(chunks)]

    if tasks_intersection:
        for chunk in split_work(intersection_count, tasks_intersection):
            tasks.append(("intersection", chunk))
    if tasks_drag:
        for chunk in split_work(drag_count, tasks_drag):
            tasks.append(("drag", chunk))
    return tasks


def gumbel_top_k(weights, k, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    weights = np.asarray(weights)
    log_weights = np.log(weights)
    gumbels = -np.log(-np.log(rng.uniform(size=len(weights))))
    scores = log_weights + gumbels
    return np.argpartition(-scores, k)[:k]


def efraimidis_spirakis(weights, k, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    weights = np.asarray(weights)
    u = rng.uniform(size=len(weights))
    keys = u ** (1.0 / weights)
    return np.argpartition(-keys, k)[:k]


def numpy_weighted_subset(weights, k, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice(range(weights.size), size=k, replace=False, p=weights)


def choose_k_grids_weighted(k, grid_ids, weights, method=numpy_weighted_subset):
    assert len(grid_ids) == len(weights), "Weights need to correspond to grid_ids!"
    assert all(weights), "All weights need to be > 0!"
    rng = np.random.default_rng()
    ids = method(weights, k, rng)
    return [grid_ids[i] for i in ids]


@njit

def sample_cell_from_conditioned(cdf, query_shape, query_offset):
    """Sample one full-grid cell coordinate from a conditioned query-region CDF."""
    r = np.random.random()
    flat_idx = np.searchsorted(cdf, r, side="right")

    ndim = len(query_shape)
    local_coords = np.empty(ndim, dtype=np.int64)
    for dim in range(ndim - 1, -1, -1):
        local_coords[dim] = flat_idx % query_shape[dim]
        flat_idx //= query_shape[dim]

    full_coords = np.empty(ndim, dtype=np.int64)
    for dim in range(ndim):
        full_coords[dim] = local_coords[dim] + query_offset[dim]
    return full_coords


def _generate_intersection_chunk(count, conditioned_data):
    local_results = []
    stats = {g: 0 for g in conditioned_data}
    for _ in range(count):
        sample_assignment = []
        for g in conditioned_data:
            query_slices = conditioned_data[g]["query_slices"]
            query_shape = conditioned_data[g]["p_conditioned"].shape
            query_offset = [s.start for s in query_slices]
            cell_idx = sample_cell_from_conditioned(conditioned_data[g]["cdf"], query_shape, query_offset)
            sample_assignment.extend(cell_idx)
            stats[g] += 1
        local_results.append(tuple(sample_assignment))
    return local_results, stats


def _generate_drag_chunk(count, grid_ids, conditioned_data, P_k, relative_grid_weights, method):
    local_results = []
    stats = {g: 0 for g in grid_ids}
    assert len(grid_ids) == len(relative_grid_weights)
    for _ in range(count):
        k = np.random.choice(range(1, len(relative_grid_weights)), p=P_k)
        chosen_grids = choose_k_grids_weighted(k, grid_ids, relative_grid_weights, method=method)

        sample_assignment = []
        for g in grid_ids:
            if g in chosen_grids:
                query_slices = conditioned_data[g]["query_slices"]
                query_shape = conditioned_data[g]["p_conditioned"].shape
                query_offset = [s.start for s in query_slices]
                cell_idx = sample_cell_from_conditioned(conditioned_data[g]["cdf"], query_shape, query_offset)
                sample_assignment.extend(cell_idx)
                stats[g] += 1
            else:
                dim_g = len(conditioned_data[g]["unconditioned_shape"])
                sample_assignment.extend([None] * dim_g)
        local_results.append(tuple(sample_assignment))
    return local_results, stats


def generate_benchmark_data(
        conditioned_data, T_rel, N,
        n_threads=5,
        start_idx=0,
        shuffle_afterward=True,
        random_subset_method=gumbel_top_k):
    """Generate synthetic tuples with controlled intersection and drag proportions."""
    N = int(N)
    grid_ids = list(conditioned_data.keys())
    grid_weights = np.array([conditioned_data[g]["w"] for g in grid_ids])
    min_w = min(grid_weights)
    T = T_rel * min_w
    if T > min_w:
        raise ValueError(f"T={T} > min w_g={min_w}, not feasible.")

    print("Generating benchmark data with N =", N)
    print("Maximum selectivity:", min_w)
    print("Configured Selectivity: T =", T)

    intersection_count = int(math.ceil(T * N))
    drag_count = N - intersection_count

    weights_for_grids_with_drag = grid_weights - T
    grid_positions_with_drag = np.nonzero(weights_for_grids_with_drag)
    weights_for_grids_with_drag = weights_for_grids_with_drag[grid_positions_with_drag]
    weights_for_grids_with_drag = weights_for_grids_with_drag / weights_for_grids_with_drag.sum()
    grid_ids_with_drag = [grid_ids[int(grid_pos)] for grid_pos in grid_positions_with_drag[0]]

    for g, meta_data in conditioned_data.items():
        dist = meta_data["p_conditioned"]
        dist = dist / dist.sum()
        conditioned_data[g]["cdf"] = np.cumsum(dist)

    def compute_I_matrix(w, n_grids):
        I = np.zeros((n_grids, n_grids - 1))
        for k in range(1, n_grids):
            probs = []
            for _ in range(10000):
                subset = np.random.choice(n_grids, size=k, replace=False, p=w / w.sum())
                inclusion = np.zeros(n_grids)
                inclusion[subset] = 1
                probs.append(inclusion)
            I[:, k - 1] = np.mean(probs, axis=0)
        return I

    print("Computing linear constraint matrix...")
    I_matrix = compute_I_matrix(weights_for_grids_with_drag, weights_for_grids_with_drag.size)
    print("Solving the linear equation for the correct subset size distribution...")
    res = lsq_linear(I_matrix, weights_for_grids_with_drag, bounds=(0, 1))
    P_k = res.x / res.x.sum()

    print("Subset size distribution P_k:", P_k)
    expected_k = np.sum((np.arange(1, len(P_k) + 1)) * P_k)
    drag_volume = (1 - T) * expected_k * N
    print("Intersection count:", intersection_count)
    print("Drag count:", drag_count)
    print("Drag volume:", drag_volume)

    tasks = distribute_tasks(n_threads, intersection_count, drag_count)

    def run_task(task):
        ttype, count = task
        print("Running task:", ttype, "with count", count)
        if ttype == "intersection":
            return _generate_intersection_chunk(count, conditioned_data)
        return _generate_drag_chunk(count, grid_ids_with_drag, conditioned_data, P_k, weights_for_grids_with_drag, method=random_subset_method)

    results = []
    stats_vec = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for fut in futures:
            data, stats = fut.result()
            print("Thread finished generating", len(data), "tuples.")
            results.extend(data)
            stats_vec.append(stats)

    if shuffle_afterward and len(results) > 1:
        np.random.shuffle(results)

    df = pd.DataFrame(results, columns=np.concatenate(list(conditioned_data.keys())), dtype=pd.Int8Dtype())
    df.index = range(start_idx, start_idx + len(df))

    print("Initial weights for grids with drag:", weights_for_grids_with_drag)
    rel_cards = np.array([stats[g] for g in grid_ids_with_drag])
    print("Relative sizes for grids with drag:", rel_cards / rel_cards.sum())
    return df, aggregate_dicts(stats_vec)


def compute_inverted_postings(df, grid_specs, start_id=0, n_jobs=1):
    """Turn a synthetic VA-file style table into per-grid inverted postings."""

    def _compute_inverted_postings_for_grid(df, grid_cols, shape):
        sub_df = df[list(grid_cols)]
        print("Creating postings for dataframe of shape:", sub_df.shape)
        valid_mask = sub_df.notna().all(axis=1)
        valid_ids = np.flatnonzero(valid_mask).astype(np.uint32)
        coords = sub_df[valid_mask].astype(np.int32).to_numpy()
        flat_idxs = np.ravel_multi_index(coords.T, dims=shape).astype(np.uint32)
        pairs = np.column_stack((flat_idxs, valid_ids))
        pairs.sort(axis=0)
        return pairs, shape

    df = df.reset_index(drop=True)
    inverted_postings = {}
    grid_cols_list = list(grid_specs.items())

    if n_jobs == 1:
        for cols, shape in grid_cols_list:
            pairs, shape_ = _compute_inverted_postings_for_grid(df, cols, shape)
            pairs[:, 1] += start_id
            inverted_postings[cols] = (pairs, shape_)
            print(inverted_postings[cols], ":", len(pairs), "postings created!")
        return inverted_postings

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {}
        for cols, shape in grid_cols_list:
            futures[executor.submit(_compute_inverted_postings_for_grid, df, cols, shape)] = (cols, shape)
        for fut in as_completed(futures):
            cols, shape = futures[fut]
            pairs, shape_ = fut.result()
            pairs[:, 1] += start_id
            inverted_postings[cols] = (pairs, shape_)
    return inverted_postings


def dump_inverted_postings(inverted_postings, output_dir, pagesize=4096, codec_id=1):
    """Dump inverted postings to TeamIndex-compatible files."""
    for grid_cols, (pairs, shape) in inverted_postings.items():
        base_name = "-".join(grid_cols)
        prefix = os.path.join(output_dir, base_name + ".copy")

        total_bins = np.prod(shape)
        cardinalities = np.zeros(total_bins, dtype=np.uint32)
        offsets = np.zeros(total_bins + 1, dtype=np.uint32)
        sizes = np.zeros(total_bins, dtype=np.uint64)
        codecs_ = np.full(total_bins, codec_id, dtype=np.uint8)

        page_offset = 0
        with open(prefix + ".lists", "wb") as list_file:
            idx_pos = 0
            n_pairs = pairs.shape[0]
            for bin_idx in range(total_bins):
                offsets[bin_idx] = page_offset
                if idx_pos >= n_pairs or pairs[idx_pos, 0] != bin_idx:
                    continue

                start_pos = idx_pos
                while idx_pos < n_pairs and pairs[idx_pos, 0] == bin_idx:
                    idx_pos += 1
                postings = pairs[start_pos:idx_pos, 1]
                cardinalities[bin_idx] = len(postings)

                postings_bytes = postings.tobytes()
                size_in_bytes = len(postings_bytes)
                pad_len = (pagesize - (size_in_bytes % pagesize)) % pagesize
                padded_bytes = postings_bytes + b"\x00" * pad_len
                list_file.write(padded_bytes)

                sizes[bin_idx] = size_in_bytes
                page_offset += len(padded_bytes) // pagesize

            offsets[-1] = page_offset

        cardinalities.tofile(prefix + ".cardinalities")
        offsets.tofile(prefix + ".offsets")
        sizes.tofile(prefix + ".sizes")
        codecs_.tofile(prefix + ".codecs")


def create_dummy_json_config(grid_specs, input_folder, output_path, quantiles, query):
    """Create a minimal TeamIndex config for the generated synthetic postings."""
    output_path = Path(output_path)
    input_folder = Path(input_folder)
    assert input_folder.exists(), f"Input folder {input_folder} does not exist."
    assert input_folder.is_dir(), f"Input folder {input_folder} is not a directory."
    assert output_path.parent.exists(), f"Output parent folder {output_path.parent} does not exist!"

    all_columns = set()
    for cols in grid_specs.keys():
        all_columns.update(cols)
    all_columns = sorted(all_columns)
    teams = [list(cols) for cols in grid_specs.keys()]

    quantiles_dict = {}
    for col, quantiles_list in quantiles.items():
        quantiles_dict[col] = []
        for q in quantiles_list:
            if q in (-np.inf, np.inf):
                continue
            quantiles_dict[col].append(float(q))

    config_data = {
        "compressions": ["copy"],
        "index_folder": str(input_folder.absolute()),
        "quantiles": quantiles_dict,
        "queries": [query],
        "source_table": None,
        "special_values": {},
        "teams": teams,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(config_data, fh, indent=2)


def generate_indices(N, T_rel_list, team_dists, team_queries, destination_folder, quantiles, query, n_jobs=4):
    """Generate one synthetic TeamIndex family for multiple T_rel values."""
    destination_folder = Path(destination_folder)
    assert destination_folder.exists(), f"Destination folder {destination_folder} does not exist!"

    N = int(N)
    grid_specs = {team: dist.shape for team, dist in team_dists.items()}
    config = preprocess_all_grids(team_queries, team_dists)

    for T_rel in T_rel_list:
        Trel_str = str(T_rel).replace(".", "")
        subfolder = destination_folder / f"selectivity_Trel{Trel_str}_N{int(N)}"
        if subfolder.exists():
            print(f"Folder {subfolder} already exists. Skipping.")
            continue

        print(f"Generating indices for T_rel = {T_rel}")
        benchmark_data, _stats = generate_benchmark_data(config, T_rel, N, n_threads=n_jobs)
        postings = compute_inverted_postings(benchmark_data, grid_specs, start_id=0, n_jobs=n_jobs)

        print("Creating folder to dump data:", subfolder.absolute())
        subfolder.mkdir(parents=True, exist_ok=False)
        dump_inverted_postings(postings, subfolder)

        cfg_file_path = subfolder / "index.json"
        create_dummy_json_config(grid_specs, subfolder, cfg_file_path, quantiles, query)
        del benchmark_data, postings


def generate_lhcb_benchmark_data(destination_folder="./indices/", N=1e8, T_rel_list=[0.0, 0.1, 0.5, 0.9, 1.0], n_jobs=13):
    """Optional helper that reuses an existing LHCb TeamIndex as source distribution."""
    repo_root = Path(__file__).resolve().parent.parent
    python_src = repo_root / "code" / "python"
    if str(python_src) not in sys.path:
        sys.path.insert(0, str(python_src))
    from TeamIndex.evaluation import TeamIndex

    query = """muplus_PIDmu > 0 and muplus_PT > 500 and muminus_PIDmu > 0 and muminus_PT > 500 and J_psi_1S_M < 3176.9 and J_psi_1S_M > 3016.9 and J_psi_1S_ENDVERTEX_CHI2 < 16 and Kst_892_0_M > 826 and Kst_892_0_M < 966 and Kst_892_0_PT > 1300 and Kst_892_0_ENDVERTEX_CHI2 < 25 and piminus_TRACK_CHI2_PER_NDOF < 5 and piminus_PIDK < 0 and Kplus_TRACK_CHI2_PER_NDOF < 5 and Kplus_PIDK > 0 and B0_M > 5150 and B0_M < 5450 and B0_ENDVERTEX_CHI2_PER_NDOF < 20 and B0_LOKI_DTF_CTAU > 0.0598"""
    ti = TeamIndex("lhcb_index.json", compression="roaring")
    team_dists = {tuple(ti.teams[team]): cards / ti.stats["number_of_tuples"] for team, cards in ti.cardinalities.items()}
    team_queries = {tuple(ti.teams[team]): ti._make_histogram_slicer(query, ti.teams[team]) for team in ti.teams}

    generate_indices(
        N,
        T_rel_list=T_rel_list,
        team_dists=team_dists,
        team_queries=team_queries,
        destination_folder=destination_folder,
        quantiles=ti.quantiles,
        query=query,
        n_jobs=n_jobs,
    )
