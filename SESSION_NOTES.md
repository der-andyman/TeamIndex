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

- Aktiver Arbeitsordner ist jetzt [study](/home/duman/TeamIndex/study)
- Aktives Hauptskript: [mopts_study.py](/home/duman/TeamIndex/study/mopts_study.py)
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
- `minimal_example/` ist jetzt nur noch der Legacy-/Demo-Bereich:
  - [run_example.py](/home/duman/TeamIndex/minimal_example/run_example.py)
  - [run_example2.py](/home/duman/TeamIndex/minimal_example/run_example2.py)
  - [example_paths.py](/home/duman/TeamIndex/minimal_example/example_paths.py)
  - [toy_index.json](/home/duman/TeamIndex/minimal_example/toy_index.json)
- Generierte Altdateien in `minimal_example/` werden nicht mehr mitgefuehrt:
  - `toy_index_data.json`
  - `toy_index_local.json`
  - `toy_index_scratch.json`
  - `__pycache__/`
  - diese Dateien entstehen bei Bedarf wieder durch die alten Demo-Skripte

## 4. Outputs von mopts_study.py

- CSVs:
  - [results.csv](/home/duman/TeamIndex/study/results/results.csv)
  - [mopts_per_team.csv](/home/duman/TeamIndex/study/results/mopts_per_team.csv)
  - [comparison_vs_baseline.csv](/home/duman/TeamIndex/study/results/comparison_vs_baseline.csv)
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
  - [plans](/home/duman/TeamIndex/study/results/plans)
- Graph-Artefakte:
  - [graphs](/home/duman/TeamIndex/study/results/graphs)
  - dort liegen `*execution_plan-*.dot` und `*task_graph.json`
- Plot-PDFs:
  - [runtime_comparison.pdf](/home/duman/TeamIndex/study/results/runtime_comparison.pdf)
  - [speedup_vs_baseline_runtime.pdf](/home/duman/TeamIndex/study/results/speedup_vs_baseline_runtime.pdf)
  - [ids_per_second_comparison.pdf](/home/duman/TeamIndex/study/results/ids_per_second_comparison.pdf)
  - [mib_per_second_comparison.pdf](/home/duman/TeamIndex/study/results/mib_per_second_comparison.pdf)

## 5. Wichtige Fixes in mopts_study / evaluation

- `exported_plan.json` nutzt absolute Pfade
- `dry_run`-Export in [evaluation.py](/home/duman/TeamIndex/code/python/TeamIndex/evaluation.py) wurde gegen `numpy.int64` / `numpy`-Skalare robuster gemacht
- [mopts_study.py](/home/duman/TeamIndex/study/mopts_study.py) hat jetzt auch `--convert-only` für die PDF-Konvertierung vorhandener `execution_plan`-DOT-Dateien
- [mopts_study.py](/home/duman/TeamIndex/study/mopts_study.py) hat jetzt auch `--no-reference`, damit der große Parquet-Datensatz nicht geladen werden muss, wenn nur Performance/Planstruktur untersucht werden soll
- [mopts_study.py](/home/duman/TeamIndex/study/mopts_study.py) hat jetzt auch:
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
- neue Artefakt-Logik in [mopts_study.py](/home/duman/TeamIndex/study/mopts_study.py):
  - vor jedem neuen Lauf werden alte Dateien in datumsbenannte Unterordner archiviert
  - `plans/`, `graphs/`, `stats/` und `plots/` bekommen je einen Unterordner wie `12-05-2026_14-03`
  - die CSVs und Summary-PDFs werden gesammelt unter `results/archives/<zeitstempel>/` wegarchiviert
- die Summary-PDFs liegen nicht mehr in `plots/`, sondern direkt neben den CSVs im Hauptordner `study/results/`
- Es gibt jetzt ein separates Hilfsskript:
  - [generate_comparison_pdfs.py](/home/duman/TeamIndex/study/generate_comparison_pdfs.py)
  - erzeugt die vier Vergleichs-PDFs nur aus `results.csv`, ohne neuen Benchmark-Lauf
- Es gibt jetzt ein separates Datenerzeugungs-/Indexbauskript:
  - [create_study_data.py](/home/duman/TeamIndex/study/create_study_data.py)
  - kann `n` per Argument setzen
  - schaetzt benoetigten Speicher grob ab
  - fragt vor grossen Builds bestaetigend nach
  - schreibt Parquet in Batches statt alles auf einmal im RAM zu halten

## 6. Aktueller Blocker

- `Graphviz` / `dot` ist inzwischen installiert
- `execution_plan-*.dot` werden erfolgreich in PDFs umgewandelt
- zusätzlich gibt es jetzt [mopts_standalone_study.py](/home/duman/TeamIndex/study/mopts_standalone_study.py) für Batch-Ausführung exportierter Pläne mit `teamindexstandalone`

## 7. Nächste sinnvolle Schritte

- Thread-Scaling fuer ausgewaehlte Queries (`worker_count` 4 / 8 / 16 / ...) systematisch vergleichen
- `mopts`-Varianten auf großem Datensatz (`n = 1_000_000_000`) vorsichtig mit Warn-/Skip-Logik testen
- Query-Sets / Schwierigkeit / Benchmark-Design für die Bachelorarbeit weiter schärfen
- untersuchen, in welchen Query-Domänen Optimierer überhaupt relevant werden
- Python- und Standalone-Messungen gezielt vergleichen
- die neue `dynamic_selective_expansion` zuerst klein gegen `q01`, `q02`, `q03`, `q06` pruefen, bevor breite Stressqueries wieder voll durchlaufen
- Daten- und Indexerzeugung kuenftig ueber `study/create_study_data.py` statt ueber das alte Demo-Skript `run_example.py`

## 8. Letzter realer Lauf

- Am `2026-04-16` wurde [mopts_study.py](/home/duman/TeamIndex/study/mopts_study.py) erfolgreich mit den fünf Varianten durchgelaufen.
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
  - [results.csv](/home/duman/TeamIndex/study/results/results.csv)
  - [comparison_vs_baseline.csv](/home/duman/TeamIndex/study/results/comparison_vs_baseline.csv)
  - `execution_plan`-PDFs für alle Query-/Varianten-Kombinationen in [plots](/home/duman/TeamIndex/study/results/plots)
- Erste Tendenz aus dem Lauf:
  - bei sehr kleinen Single-Team-Queries bringen Optimierer oft wenig oder verschlechtern leicht
  - bei unausgeglichenen Mehr-Team-Queries bringt `current_handcrafted` deutlich etwas
  - `expand_all_adaptive_grouping` half in einigen groesseren Mehr-Team-Faellen, war aber kein universeller Gewinner
  - breite Queries wie `q07` bis `q09` koennen den Server stark belasten; deshalb wurden Warnungen und `--skip-dangerous` eingebaut
  - die Wirkung von Optimierern hängt also klar von Query-Struktur, Leaf-Hits, Team-Imbalance und ISE-Komplexitaet ab

## 9. Speicherort Index / Daten

- Aktiver Datenpfad ist jetzt die neue grosse SSD:
  - `/media/data/duman/teamindex/`
- Dort liegen jetzt sowohl:
  - der TeamIndex unter `/media/data/duman/teamindex/index_data`
  - als auch die grosse Parquet-Datei unter `/media/data/duman/teamindex/uniform_toy_data.parquet`
- Aktiver Pfadhelfer ist jetzt:
  - [study_paths.py](/home/duman/TeamIndex/study/study_paths.py)
- Die aktive Index-Konfiguration fuer die Studie ist:
  - [study_index_data.json](/home/duman/TeamIndex/study/study_index_data.json)
- Diese lokale Pfaddatei wird per `.gitignore` nicht committed
- Der alte lokale Index im Projektordner wurde geloescht
- Die alte Scratch-Parquet-Datei wurde geloescht

## 10. ToDo für später

- DRAM-Cache über mehrere Runs prüfen:
  - aktuell lädt der DRAM-Backend pro Run nur die query-relevanten Team-Dateien, aber bei wiederholten Läufen werden diese Teams erneut in den RAM geladen
  - spätere Optimierungsidee: persistenter Team-Cache für den DRAM-Backend, damit identische oder ähnliche Queries bereits geladene Teams wiederverwenden können
