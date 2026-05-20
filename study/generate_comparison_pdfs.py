#!/usr/bin/env python3
"""
Erzeugt die vier wichtigsten Vergleichs-PDFs der mopts-Studie aus einer
bereits vorhandenen `study/results/results.csv`.

Dieses Skript fuehrt selbst keinen Benchmark aus. Es laedt nur die bereits
gespeicherten Ergebnisse und erstellt die zusammenfassenden Plots neu:

- runtime_comparison.pdf
- speedup_vs_baseline_runtime.pdf
- ids_per_second_comparison.pdf
- mib_per_second_comparison.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import mopts_study as ms


def main():
    # Dieses Hilfsskript ist bewusst leichtgewichtig: Es setzt voraus,
    # dass die Benchmark-Ergebnisse bereits vorliegen, und erzeugt nur
    # die wichtigen Vergleichs-PDFs neu.
    results_csv = ms.OUT_DIR / "results.csv"
    if not results_csv.exists():
        raise RuntimeError(f"Missing results file: {results_csv}")

    results_df = pd.read_csv(results_csv)
    plot_paths = ms.generate_all_summary_plots(results_df)

    print("Regenerated comparison PDFs from:")
    print(results_csv)
    print("\nCreated:")
    for plot_path in plot_paths:
        print(plot_path)


if __name__ == "__main__":
    main()
