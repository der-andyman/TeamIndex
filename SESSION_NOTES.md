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
  - `baseline_naive`
  - `current_handcrafted`
  - `overhead_aware`
  - `imbalance_aware`
  - `leaf_count_aware`
- Ziel:
  - saubere, naive Baseline
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
- Standalone-Pläne:
  - [plans](/home/duman/TeamIndex/minimal_example/mopts_study/plans)
- Graph-Artefakte:
  - [graphs](/home/duman/TeamIndex/minimal_example/mopts_study/graphs)
  - dort liegen `*execution_plan-*.dot` und `*task_graph.json`
- Plot-PDFs:
  - [runtime_comparison.pdf](/home/duman/TeamIndex/minimal_example/mopts_study/plots/runtime_comparison.pdf)
  - [speedup_vs_baseline.pdf](/home/duman/TeamIndex/minimal_example/mopts_study/plots/speedup_vs_baseline.pdf)

## 5. Wichtige Fixes in mopts_study / evaluation

- `exported_plan.json` nutzt absolute Pfade
- `dry_run`-Export in [evaluation.py](/home/duman/TeamIndex/code/python/TeamIndex/evaluation.py) wurde gegen `numpy.int64` / `numpy`-Skalare robuster gemacht
- [mopts_study.py](/home/duman/TeamIndex/minimal_example/mopts_study.py) hat jetzt auch `--convert-only` für die PDF-Konvertierung vorhandener `execution_plan`-DOT-Dateien
- [mopts_study.py](/home/duman/TeamIndex/minimal_example/mopts_study.py) hat jetzt auch `--no-reference`, damit der große Parquet-Datensatz nicht geladen werden muss, wenn nur Performance/Planstruktur untersucht werden soll

## 6. Aktueller Blocker

- `Graphviz` / `dot` ist inzwischen installiert
- `execution_plan-*.dot` werden erfolgreich in PDFs umgewandelt
- zusätzlich gibt es jetzt [mopts_standalone_study.py](/home/duman/TeamIndex/minimal_example/mopts_standalone_study.py) für Batch-Ausführung exportierter Pläne mit `teamindexstandalone`

## 7. Nächste sinnvolle Schritte

- `mopts`-Varianten auf großem Datensatz (`n = 1_000_000_000`) systematisch testen
- Query-Sets / Schwierigkeit / Benchmark-Design für die Bachelorarbeit weiter schärfen
- untersuchen, in welchen Query-Domänen Optimierer überhaupt relevant werden
- Python- und Standalone-Messungen gezielt vergleichen

## 8. Letzter realer Lauf

- Am `2026-04-16` wurde [mopts_study.py](/home/duman/TeamIndex/minimal_example/mopts_study.py) erfolgreich mit den fünf Varianten durchgelaufen.
- Erzeugt wurden:
  - [results.csv](/home/duman/TeamIndex/minimal_example/mopts_study/results.csv)
  - [comparison_vs_baseline.csv](/home/duman/TeamIndex/minimal_example/mopts_study/comparison_vs_baseline.csv)
  - `execution_plan`-PDFs für alle Query-/Varianten-Kombinationen in [plots](/home/duman/TeamIndex/minimal_example/mopts_study/plots)
- Erste Tendenz aus dem Lauf:
  - bei sehr kleinen Single-Team-Queries bringen Optimierer oft wenig oder verschlechtern leicht
  - bei unausgeglichenen Mehr-Team-Queries bringt `current_handcrafted` deutlich etwas
  - bei manchen Zwei-3D-Team-Queries war `leaf_count_aware` sogar am schnellsten
  - die Wirkung von Optimierern hängt also klar von Query-Struktur, Leaf-Hits und Team-Imbalance ab

## 9. ToDo für später

- DRAM-Cache über mehrere Runs prüfen:
  - aktuell lädt der DRAM-Backend pro Run nur die query-relevanten Team-Dateien, aber bei wiederholten Läufen werden diese Teams erneut in den RAM geladen
  - spätere Optimierungsidee: persistenter Team-Cache für den DRAM-Backend, damit identische oder ähnliche Queries bereits geladene Teams wiederverwenden können
