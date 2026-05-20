#!/usr/bin/env python3
"""
Erzeugt einen neuen uniformen Parquet-Datensatz fuer die Studie und baut
anschliessend den TeamIndex auf dem dedizierten Datentraeger auf.

Wichtige Eigenschaften:
- `n` kann als Argument uebergeben oder interaktiv abgefragt werden
- vor der Erzeugung wird grob abgeschaetzt, wie viel Speicher benoetigt wird
- bei knapper Kapazitaet oder grossen Datenmengen wird vor dem Start gefragt
- die Parquet-Datei wird in Batches geschrieben, damit nicht alles auf einmal
  im RAM liegen muss
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from TeamIndex import creation as crt
from study_paths import DATA_PATH, INDEX_CONFIG, INDEX_DATA_DIR, SOURCE_INDEX_CONFIG, DATA_ROOT


COLUMNS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
DEFAULT_BATCH_ROWS = 5_000_000

# Grobe Erfahrungswerte aus dem bisherigen Datensatz.
EST_PARQUET_BYTES_PER_ROW = 77
EST_INDEX_BYTES_PER_ROW = 31


def parse_args():
    parser = argparse.ArgumentParser(description="Erzeuge Studiendaten und baue den TeamIndex.")
    parser.add_argument("--n", type=int, default=None, help="Anzahl Tupel fuer den neuen Datensatz.")
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help="Wie viele Tupel pro Schreib-Batch erzeugt werden sollen.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed fuer den Zufallszahlengenerator.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Rueckfrage vor der Erzeugung ueberspringen.",
    )
    parser.add_argument(
        "--overwrite-parquet",
        action="store_true",
        help="Vorhandene Parquet-Datei loeschen und neu erzeugen.",
    )
    parser.add_argument(
        "--overwrite-index",
        action="store_true",
        help="Vorhandenen Indexordner loeschen und den Index komplett neu bauen.",
    )
    return parser.parse_args()


def format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if value < 1024.0:
            return f"{value:.2f} {unit}"
    return f"{value:.2f} PiB"


def estimate_sizes(n: int) -> dict:
    parquet_bytes = n * EST_PARQUET_BYTES_PER_ROW
    index_bytes = n * EST_INDEX_BYTES_PER_ROW
    return {
        "parquet_bytes": parquet_bytes,
        "index_bytes": index_bytes,
        "total_bytes": parquet_bytes + index_bytes,
    }


def prompt_for_n() -> int:
    raw_value = input("Wie gross soll n sein? ").strip()
    if not raw_value:
        raise RuntimeError("Kein Wert fuer n eingegeben.")
    return int(raw_value.replace("_", ""))


def ask_for_confirmation(n: int, estimates: dict, free_bytes: int):
    print("\nGeplanter Datensatz:")
    print(f"- n: {n:,}".replace(",", "_"))
    print(f"- Ziel-Parquet: {DATA_PATH}")
    print(f"- Ziel-Index:   {INDEX_DATA_DIR}")
    print(f"- geschaetzte Parquet-Groesse: {format_bytes(estimates['parquet_bytes'])}")
    print(f"- geschaetzte Index-Groesse:   {format_bytes(estimates['index_bytes'])}")
    print(f"- geschaetzter Gesamtbedarf:   {format_bytes(estimates['total_bytes'])}")
    print(f"- freier Platz auf {DATA_ROOT.parent}: {format_bytes(free_bytes)}")

    if estimates["total_bytes"] >= free_bytes:
        print("\nWARNUNG: Der geschaetzte Gesamtbedarf liegt ueber dem freien Speicher.")
    elif estimates["total_bytes"] >= int(free_bytes * 0.8):
        print("\nWARNUNG: Der geschaetzte Gesamtbedarf nutzt mehr als 80% des freien Speichers.")

    answer = input("\nWirklich fortfahren? [y/N] ").strip().lower()
    if answer not in {"y", "yes", "j", "ja"}:
        raise RuntimeError("Abgebrochen durch Benutzer.")


def write_index_config():
    with SOURCE_INDEX_CONFIG.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg = copy.deepcopy(cfg)
    cfg["index_folder"] = str(INDEX_DATA_DIR.resolve())
    cfg["source_table"] = str(DATA_PATH.resolve())

    INDEX_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with INDEX_CONFIG.open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=4)


def generate_uniform_parquet(n: int, batch_rows: int, seed: int):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    writer = None

    try:
        for batch_start in range(0, n, batch_rows):
            rows = min(batch_rows, n - batch_start)
            data = {
                column: rng.uniform(0.0, 100.0, rows).astype(np.float64)
                for column in COLUMNS
            }
            table = pa.Table.from_pydict(data)
            if writer is None:
                writer = pq.ParquetWriter(DATA_PATH, table.schema)
            writer.write_table(table)
            batch_end = batch_start + rows
            print(
                f"Parquet-Fortschritt: {batch_end:,}/{n:,} Tupel".replace(",", "_"),
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()


def main():
    args = parse_args()
    n = args.n if args.n is not None else prompt_for_n()
    if n <= 0:
        raise RuntimeError("n muss groesser als 0 sein.")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(DATA_ROOT.parent).free
    estimates = estimate_sizes(n)

    if not args.yes:
        ask_for_confirmation(n, estimates, free_bytes)

    if args.overwrite_parquet and DATA_PATH.exists():
        DATA_PATH.unlink()
    if args.overwrite_index and INDEX_DATA_DIR.exists():
        shutil.rmtree(INDEX_DATA_DIR)

    write_index_config()

    if DATA_PATH.exists():
        print(f"Parquet existiert bereits, ueberspringe Neuerzeugung: {DATA_PATH}")
    else:
        print(f"Erzeuge Parquet-Datei: {DATA_PATH}")
        generate_uniform_parquet(n=n, batch_rows=args.batch_rows, seed=args.seed)

    INDEX_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Baue TeamIndex in: {INDEX_DATA_DIR}")
    crt.index_table(INDEX_CONFIG, table=None, overwrite_existing=args.overwrite_index)
    print("Fertig.")


if __name__ == "__main__":
    main()
