#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_my_example.py

Vollständiges Beispiel für die Erstellung eines eigenen TeamIndex-Datensatzes,
Ausführung einer Query mit Benchmarking-Konfiguration und anschließender Visualisierung.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess

sys.path.append(str(Path(__file__).resolve().parent.parent / "code" / "python" / "TeamIndex"))

from TeamIndex import creation as crt
from TeamIndex import evaluation as eva
from benchmark import import_benchmark_data, plot_worker_tasks





##################################################
# 1. Datensatz erstellen oder laden
##################################################

n = 500_000  # Anzahl Zeilen – nach Bedarf anpassen
columns = ["A", "B", "C", "D", "E"]

dataset_path = Path("./my_uniform_data.parquet")

if not dataset_path.exists():
    print(f"Erstelle neuen Datensatz unter {dataset_path} ...")
    table = pd.DataFrame(
        np.random.uniform(0, 100, n * len(columns)).reshape(n, len(columns)),
        columns=columns,
    )
    table.to_parquet(dataset_path)
else:
    print(f"Lade vorhandenen Datensatz aus {dataset_path} ...")
    table = pd.read_parquet(dataset_path)

##################################################
# 2. Index erstellen (falls noch nicht vorhanden)
##################################################

index_def_folder = Path("./my_index_data/")
index_def_folder.mkdir(exist_ok=True)

if not index_def_folder.exists():
    print("Erstelle neue Indexdefinition ...")

# Schritt 1: Setze globale Tabelle für Quantilberechnung
crt.table = table

# Schritt 2: Quantile bestimmen – steuert Bin‑Grenzen pro Spalte
quantiles = crt.determine_quantiles(5, target_folder=True)

# Schritt 3: Teams definieren – welche Spalten gehören zusammen?
teams_definition = [["A", "B", "C"]]

# Schritt 4: Beispielhafte Queries und Kompressionen festlegen
queries = [
    "A < 50 and B < 50",
    "A < 50 and B < 50 and C < 50"
]
compressions = ["roaring"]
special_values = None

# Schritt 5: JSON‑Konfigurationen schreiben – jetzt mit allen Parametern!
crt.create_configs(
    input_path=dataset_path,          # dein Parquet‑Datensatz
    target_path=index_def_folder,     # Zielordner für Team‑Configs
    quantiles=quantiles,
    team_compositions=teams_definition,
    queries=queries,
    compressions=compressions,
    special_values=special_values
)
print(f"Neue Konfigurationsdateien erstellt unter: {index_def_folder}")

# Schritt 6: Jetzt kann der eigentliche Index gebaut werden.
print("Erstelle neue Indizes basierend auf dem Datensatz ...")

for cfg_file in index_def_folder.glob("team_*/*.json"):
    print(f"→ Erstelle Index aus {cfg_file}")
    crt.index_table(str(cfg_file), table=table)

print("✅ Alle Team-Indizes erstellt.")

##################################################
# 3. Index öffnen & Query definieren
##################################################

# Lade alle erzeugten Team-Indices und führe Querys aus
index_files = sorted(index_def_folder.glob("team_*/*.json"))
print("\nLade folgende Indizes:")
for f in index_files:
    print(" →", f)

query = "A < 50 and B < 50 and C < 50"
print("\nStarte Query:", query)

result_sets = []
for idx_file in index_files:
    print(f"Führe Query auf {idx_file} aus ...")
    idx = eva.TeamIndex(str(idx_file))
    res_ids, *_ = idx.run_query(query)
    result_sets.append(set(res_ids))

# Schnittmenge aller Ergebnisse bilden
final_result_ids = set.intersection(*result_sets)
print("\nGesamtergebnisgröße:", len(final_result_ids))

##################################################
# 4. Benchmark-Konfiguration festlegen
##################################################

# Erstelle ein einzelnes TeamIndex-Objekt für Benchmarking:
index_json = next(index_def_folder.glob("team_*/*.json"))
index = eva.TeamIndex(str(index_json))

results_folder = Path("./my_results/")
results_folder.mkdir(exist_ok=True)

execution_config = eva.get_new_default_runtime_config()
execution_config["backend"] = "liburing"     # nutze DRAM-basiertes Backend
execution_config["verbose_runtime"] = True   # detaillierte Ausgabe aktivieren

execution_config["print_task_stats"]   = results_folder / "task_stats.json"
execution_config["print_result_stats"] = results_folder / "result_stats.json"
execution_config["print_execution_plan"] = results_folder / "execution_plan.dot"

##################################################
# 5. Query ausführen + Ergebnisse prüfen
##################################################

result_ids, runtime_stats, request_info, global_info = index.run_query(query, config=execution_config)

print("\nQuery abgeschlossen.")
print("Resultgröße:", len(result_ids))
print("Laufzeit [ms]:", round(runtime_stats.executor_runtime / 1e6, 2))

ref_result_ids = set(table.query(query).index)
correct_subset_check = ref_result_ids.issubset(result_ids)
print("Sind Referenz-Ergebnisse enthalten?:", correct_subset_check)

##################################################
# Optional: Plan exportieren (für Standalone-Ausführung)
##################################################

plan_export_file = results_folder / "exported_plan.json"
index.run_query(query, config=execution_config, dry_run=str(plan_export_file))
print("\nExportierter Plan gespeichert unter:", plan_export_file)

##################################################
# Optional: Worker-Task-Visualisierung erzeugen
##################################################

try:
    # Finde die neueste Task-Stats-Datei im Ordner
    task_files = sorted(results_folder.glob("task_stats-*.json"))
    if not task_files:
        raise FileNotFoundError("Keine Task-Stats-Dateien gefunden.")
    latest_task_file = task_files[-1]
    print(f"Verwende {latest_task_file} für Worker-Plot ...")

    task_data_df, meta_info = import_benchmark_data(latest_task_file)
    plot_worker_tasks(task_data_df, results_folder / "worker_tasks.pdf")
    print("\nWorker-Aufgabenplot gespeichert unter:", results_folder / "worker_tasks.pdf")
except Exception as e:
    print("\nFehler beim Erstellen des Plots:", e)

print("\n Fertig! Alle Dateien liegen in:", results_folder.resolve())

# Konvertiere alle DOT-Dateien im results_folder nach PDF:
for dot_file in sorted(results_folder.glob("execution_plan-*.dot")):
    pdf_file = dot_file.with_suffix(".pdf")
    print(f"Konvertiere {dot_file} → {pdf_file}")
    subprocess.run(["dot", "-Tpdf", str(dot_file), "-o", str(pdf_file)], check=True)

print("✅ Alle Execution-Pläne wurden erfolgreich nach PDF konvertiert.")