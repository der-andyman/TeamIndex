# Projektstand TeamIndex / Bachelorarbeit

## 1. 2D-Team-Bug

- Bug bei 2D-Teams wie `B-I` und `F-H` wurde untersucht und gefixt.
- Wichtiger Fix: in [creation.py](/home/duman/TeamIndex/code/python/TeamIndex/creation.py) wird `team_data` jetzt als Fortran-/column-major an den alten `BatchConverter` übergeben.
- Danach Indizes neu gebaut und Ergebnisse geprüft.
- Ergebnis: fehlende echte Treffer bei 2D-Teams sind weg.

## 2. Runtime / liburing

- Für die eigentliche Optimizer-Arbeit wird vor allem `DRAM` genutzt.
- `o_direct` wurde als Default auf `False` gesetzt, weil der `liburing + O_DIRECT`-Pfad inkorrekte Ergebnisse liefern konnte.
- Für die `mopts`-Arbeit ist das zweitrangig, da im `DRAM`-Modus gearbeitet wird.

## 3. mopts-Studie

- Es gibt ein neues Skript: [mopts_study.py](/home/duman/TeamIndex/minimal_example/mopts_study.py)
- Varianten aktuell:
  - `baseline_union_first`
  - `baseline_minimal_intersection`
  - `current_handcrafted`
  - `dynamic_selective_expansion`
  - `expand_all_adaptive_grouping`
- Ziel:
  - `baseline_union_first` als einfache theoretische Referenz ohne Expansion
  - `baseline_minimal_intersection` als Hauptbaseline mit genau einem expandierten Team
  - `current_handcrafted` als bestehende handgeschriebene Heuristik aus `run_example.py`
  - `dynamic_selective_expansion` als neue query-form-basierte Variante: erst Query-Struktur analysieren, dann nur guenstige Teams selektiv expandieren
  - `expand_all_adaptive_grouping` als dynamische Variante: alle Teams expandieren, aber grosse/unselektive Teams bei hoher ISE-Komplexitaet staerker gruppieren
  - Vergleich von Laufzeit, Trefferzahl, `missing_true_hits`, `extra_hits`
  - Export für `teamindexstandalone`
  - `execution plans` / `taskflow`-Artefakte / CSV-Auswertung
  - systematische Strukturmetriken pro Query/Variante

## 4. Outputs von mopts_study.py

- CSVs:
  - [results.csv](/home/duman/TeamIndex/minimal_example/mopts_study/results.csv)
  - [mopts_per_team.csv](/home/duman/TeamIndex/minimal_example/mopts_study/mopts_per_team.csv)
  - [comparison_vs_baseline.csv](/home/duman/TeamIndex/minimal_example/mopts_study/comparison_vs_baseline.csv)
- zusätzliche Strukturmetriken in `results.csv`:
  - `team_count`
  - `expanded_team_count_manual`
  - `sum_max_group_count`
  - `sum_group_count`
  - `imbalance_group_count`
  - `sum_union_cardinality`
  - `imbalance_union_cardinality`
  - `query_domain`
- neue Laufzeit-/Skalierungsmetriken:
  - `worker_count`
  - `queue_pair_count`
  - `ids_per_second`
  - `million_ids_per_second`
  - `read_mib_per_second`
- neue Query-Strukturmetriken:
  - `total_selected_bin_cells`
  - `total_selected_attribute_bins`
- in `mopts_per_team.csv` zusaetzlich:
  - `team_dimension_count`
  - `selected_bin_count_product`
  - `selected_bin_counts_per_attribute`
- `plan_runtime_ms` wird nicht mehr in den Haupt-CSVs getrackt, weil die Python-Planungszeit fuer die aktuelle Fragestellung bewusst ignoriert wird.
- Standalone-Pläne:
  - [plans](/home/duman/TeamIndex/minimal_example/mopts_study/plans)
- Graph-Artefakte:
  - [graphs](/home/duman/TeamIndex/minimal_example/mopts_study/graphs)
  - dort liegen `*execution_plan-*.dot` und `*task_graph.json`
- Plot-PDFs:
  - [runtime_comparison.pdf](/home/duman/TeamIndex/minimal_example/mopts_study/runtime_comparison.pdf)
  - [speedup_vs_baseline_runtime.pdf](/home/duman/TeamIndex/minimal_example/mopts_study/speedup_vs_baseline_runtime.pdf)
  - [speedup_vs_baseline_ids_per_second.pdf](/home/duman/TeamIndex/minimal_example/mopts_study/speedup_vs_baseline_ids_per_second.pdf)
  - [speedup_vs_baseline_mib_per_second.pdf](/home/duman/TeamIndex/minimal_example/mopts_study/speedup_vs_baseline_mib_per_second.pdf)

## 5. Wichtige Fixes in mopts_study / evaluation

- `exported_plan.json` nutzt absolute Pfade
- `dry_run`-Export in [evaluation.py](/home/duman/TeamIndex/code/python/TeamIndex/evaluation.py) wurde gegen `numpy.int64` / `numpy`-Skalare robuster gemacht
- [mopts_study.py](/home/duman/TeamIndex/minimal_example/mopts_study.py) hat jetzt auch `--convert-only` für die PDF-Konvertierung vorhandener `execution_plan`-DOT-Dateien
- [mopts_study.py](/home/duman/TeamIndex/minimal_example/mopts_study.py) hat jetzt auch `--no-reference`, damit der große Parquet-Datensatz nicht geladen werden muss, wenn nur Performance/Planstruktur untersucht werden soll
- [mopts_study.py](/home/duman/TeamIndex/minimal_example/mopts_study.py) hat jetzt auch:
  - `--worker-count`
  - `--queue-pair-count`
  - `--verbose-runtime`
  - `--query-filter`
  - `--skip-dangerous`
  - `--plots-only`
- es gibt eingebaute Stress-Warnungen fuer:
  - sehr viele relevante Blaetter
  - sehr hohe geschaetzte ISE Counts
  - grosse expandierte Gesamtvolumina
- `expand_all_unbounded` wurde wieder entfernt, weil die Strategie fuer breite Queries zu gefaehrlich und methodisch wenig nuetzlich war
- neue Artefakt-Logik in [mopts_study.py](/home/duman/TeamIndex/minimal_example/mopts_study.py):
  - vor jedem neuen Lauf werden alte Dateien in datumsbenannte Unterordner archiviert
  - `plans/`, `graphs/`, `stats/` und `plots/` bekommen je einen Unterordner wie `12-05-2026_14-03`
  - die CSVs und Summary-PDFs werden gesammelt unter `mopts_study/archives/<zeitstempel>/` wegarchiviert
- die zwei Summary-PDFs liegen nicht mehr in `plots/`, sondern direkt neben den CSVs im Hauptordner `mopts_study/`

## 6. Aktueller Blocker

- `Graphviz` / `dot` ist inzwischen installiert
- `execution_plan-*.dot` werden erfolgreich in PDFs umgewandelt
- zusätzlich gibt es jetzt [mopts_standalone_study.py](/home/duman/TeamIndex/minimal_example/mopts_standalone_study.py) für Batch-Ausführung exportierter Pläne mit `teamindexstandalone`

## 7. Nächste sinnvolle Schritte

- Thread-Scaling fuer ausgewaehlte Queries (`worker_count` 4 / 8 / 16 / ...) systematisch vergleichen
- `mopts`-Varianten auf großem Datensatz (`n = 1_000_000_000`) vorsichtig mit Warn-/Skip-Logik testen
- Query-Sets / Schwierigkeit / Benchmark-Design für die Bachelorarbeit weiter schärfen
- untersuchen, in welchen Query-Domänen Optimierer überhaupt relevant werden
- Python- und Standalone-Messungen gezielt vergleichen
- die neue `dynamic_selective_expansion` zuerst klein gegen `q01`, `q02`, `q03`, `q06` pruefen, bevor breite Stressqueries wieder voll durchlaufen

## 8. Letzter realer Lauf

- Am `2026-04-16` wurde [mopts_study.py](/home/duman/TeamIndex/minimal_example/mopts_study.py) erfolgreich mit den fünf Varianten durchgelaufen.
- Am `2026-04-23` wurden die Varianten konzeptionell umgebaut:
  - `baseline_union_first`
  - `baseline_minimal_intersection`
  - `current_handcrafted`
  - `dynamic_selective_expansion`
  - `expand_all_adaptive_grouping`
- Die neuen Builder wurden gegen die vorhandenen Beispielqueries getestet und erzeugen gueltige `mopts`.
- Ein kompletter Lauf mit den umgebauten Varianten wurde mit `--no-reference` erfolgreich ausgefuehrt.
- Die lokale Ergebnisnotiz liegt in [MOPTS_STRATEGY_RESULT_NOTES.md](/home/duman/TeamIndex/MOPTS_STRATEGY_RESULT_NOTES.md) und wird per `.gitignore` nicht committed.
- Erzeugt wurden:
  - [results.csv](/home/duman/TeamIndex/minimal_example/mopts_study/results.csv)
  - [comparison_vs_baseline.csv](/home/duman/TeamIndex/minimal_example/mopts_study/comparison_vs_baseline.csv)
  - `execution_plan`-PDFs für alle Query-/Varianten-Kombinationen in [plots](/home/duman/TeamIndex/minimal_example/mopts_study/plots)
- Erste Tendenz aus dem Lauf:
  - bei sehr kleinen Single-Team-Queries bringen Optimierer oft wenig oder verschlechtern leicht
  - bei unausgeglichenen Mehr-Team-Queries bringt `current_handcrafted` deutlich etwas
  - `expand_all_adaptive_grouping` half in einigen groesseren Mehr-Team-Faellen, war aber kein universeller Gewinner
  - breite Queries wie `q07` bis `q09` koennen den Server stark belasten; deshalb wurden Warnungen und `--skip-dangerous` eingebaut
  - die Wirkung von Optimierern hängt also klar von Query-Struktur, Leaf-Hits, Team-Imbalance und ISE-Komplexitaet ab

## 9. Speicherort Index / Daten

- Der grosse Parquet-Datensatz liegt weiterhin auf Scratch:
  - `/media/scratch/duman/teamindex/uniform_toy_data.parquet`
- Der eigentliche TeamIndex liegt jetzt wieder lokal auf der SSD im Projektordner:
  - [minimal_example/index_data](/home/duman/TeamIndex/minimal_example/index_data)
- [example_paths.py](/home/duman/TeamIndex/minimal_example/example_paths.py) wurde entsprechend vereinfacht:
  - `INDEX_CONFIG` zeigt immer auf `toy_index_local.json`
  - `INDEX_DATA_DIR` zeigt immer auf `minimal_example/index_data`
  - kein Scratch-Fallback mehr fuer den Index
- die lokale Konfigurationsdatei [toy_index_local.json](/home/duman/TeamIndex/minimal_example/toy_index_local.json) ist generiert und wird per `.gitignore` nicht committed
- der zwischenzeitliche Scratch-Index wurde wieder geloescht, damit auf Scratch nur noch die grosse Parquet-Datei liegt

## 10. ToDo für später

- DRAM-Cache über mehrere Runs prüfen:
  - aktuell lädt der DRAM-Backend pro Run nur die query-relevanten Team-Dateien, aber bei wiederholten Läufen werden diese Teams erneut in den RAM geladen
  - spätere Optimierungsidee: persistenter Team-Cache für den DRAM-Backend, damit identische oder ähnliche Queries bereits geladene Teams wiederverwenden können
