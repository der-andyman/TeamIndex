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

## 11. team_bench / Strategiewahl statt perfekter Optimierer

- Aktueller Fokus wurde nach den Max-Notizen verschoben:
  - nicht mehr "einen Optimierer finden, der immer gewinnt"
  - sondern systematisch herausfinden, welche Strategie in welcher Situation sinnvoll ist
  - wichtig ist die Trennung zwischen konkreter Strategie und der späteren Wahl einer Strategie anhand vorab verfügbarer Query-Metriken
- Relevante Situationen im aktuellen Scope:
  - `2D` als Kontroll- und Overhead-/Rauschbereich
  - `3D` und `4D` als Hauptbereich
  - `5D` als kontrollierter Stressbereich
  - `2` und `3` Teams
  - `T_rel` als Regler für Intersection-Anteil und Drag
  - `worker_count` als eigener Skalierungsfaktor
- [TEAM_BENCH_SCOPE_NOTES.md](/home/duman/TeamIndex/study/TEAM_BENCH_SCOPE_NOTES.md) wurde angelegt:
  - kurze Methodik-/Scope-Notiz für `team_bench`
  - beschreibt Situationsraum, Strategieraum, Messgrößen und erste Interpretation
  - ist kein Ergebnisartefakt wie CSV/PDF, sondern eine versionierte Orientierung für die weitere Arbeit
- Die Plot-/Summary-Erzeugung wurde erweitert:
  - Standardabweichungen werden in den Comparison-PDFs als Fehlerbalken verwendet
  - Heatmaps wurden so angepasst, dass die Y-Achse nur eine Dimension enthält und die Darstellung besser lesbar ist
  - `summary_by_variant.csv` enthält zusätzliche Metadaten und relative Standardabweichungen
- [run_team_bench_experiments.py](/home/duman/TeamIndex/study/run_team_bench_experiments.py) wurde robuster:
  - schreibt `skipped_variants.csv` nur noch, wenn wirklich Varianten übersprungen wurden
  - kann Varianten mit zu hoher geschätzter ISE Count vor der Ausführung überspringen
  - setzt den effektiven `worker_count` auch im `index.default_runtime_config`, damit dynamische Strategien den Run korrekt sehen
- Neue Analyseskripte:
  - [analyze_handcrafted_behavior.py](/home/duman/TeamIndex/study/analyze_handcrafted_behavior.py) untersucht, warum `current_handcrafted` gut abschneidet
  - [analyze_strategy_selection.py](/home/duman/TeamIndex/study/analyze_strategy_selection.py) fasst mehrere `team_bench`-Runs zusammen und erzeugt Reports für Strategiewahl, Gewinner, Varianz, Planmerkmale und Feature-Tabellen
  - `study/export_thesis_tables.py` bleibt ein lokales persönliches Hilfsskript und wird per `.gitignore` ignoriert
- Erste Auswertung aktueller Kernläufe:
  - `2D` ist stark varianz- und overheadgeprägt; dort sind No-Expansion-Strategien oft sinnvoll
  - `3D` bis `5D` profitieren häufig von gebremster Ein-Team-Expansion
  - `current_handcrafted` gewinnt oft, weil es Expansion und Gruppierung kombiniert: frühes Pruning ja, aber ISE Count begrenzt
  - `dynamic_selective_expansion` ist in den team_bench-Szenarien oft zu konservativ und bleibt Union-First-ähnlich
  - `baseline_minimal_intersection` zeigt den Preis ungebremster Expansion und wird in höheren Dimensionen schnell sehr teuer oder sicherheitshalber übersprungen
- Neuer Strategiekandidat:
  - [mopts_strategies.py](/home/duman/TeamIndex/study/mopts_strategies.py) enthält jetzt `bounded_selective_expansion`
  - Idee: in 2D bewusst nicht expandieren, ab 3D große Team-Ergebnisse gruppieren und nur ein selektives Team expandieren
  - Pilotläufe zeigen, dass diese Variante mehrere 3D/4D/5D-Szenarien gewinnt und sonst meist nah an `current_handcrafted` bleibt
  - damit ist sie ein sinnvoller zusätzlicher Punkt im Strategieraum, aber noch keine endgültige neue Default-Strategie

## 12. Union First Parallel

- Neuer Strategiekandidat in [mopts_strategies.py](/home/duman/TeamIndex/study/mopts_strategies.py):
  - `union_first_parallel`
  - expandiert kein Team
  - bleibt logisch Union-First-ähnlich
  - reduziert aber die `group_count` pro Team auf eine worker-orientierte, cardinality-gewichtete Gruppenzahl
- Wichtiges Detail aus [evaluation.py](/home/duman/TeamIndex/code/python/TeamIndex/evaluation.py):
  - `_determine_groups(...)` verteilt Blätter innerhalb einer gesetzten Gruppenzahl bereits greedily nach Listengröße/Cardinality
  - die neue Strategie muss daher nicht selbst einzelne Blätter zuweisen
  - ihr Hebel ist die Frage, wie viele Gruppen pro Team sinnvoll sind
- Motivation:
  - bisher kam Parallelität vor allem durch Expansion und damit durch mehr ISEs
  - diese Variante testet physische Parallelität ohne zusätzliche logische Expansion
  - dadurch kann die ISE Count bei `0` bleiben, während große Team-Unions trotzdem in weniger, aber besser balancierte Arbeitseinheiten zerlegt werden
- Pilotläufe:
  - 12-Szenario-Pilot (`3T-2D`, `2T-3D`, `2T-4D`) mit w16:
    - `union_first_parallel` gewinnt 4 von 12 Szenarien
    - in 2T-3D ist es im Mittel die schnellste Variante
    - in 2T-4D liegt es im Mittel praktisch gleichauf mit den besten expandierenden Strategien
  - 5D-Pilot (`2T-5D`, `3T-5D`) mit w16:
    - `union_first_parallel` gewinnt 5 von 8 Szenarien
    - bleibt dabei bei `ISE=0`, während `current_handcrafted` und `bounded_selective_expansion` typischerweise `ISE=128` erzeugen
- Neue Arbeitshypothese:
  - Es gibt mindestens drei getrennte Optimierungsachsen:
    - ob expandiert wird
    - wie stark gruppiert wird
    - ob Union-Arbeit physisch passend zur Workerzahl balanciert wird
  - Für einige hohe Dimensionen kann physische Union-Parallelisierung besser sein als frühe Intersection durch Expansion.

## 13. Full/Close Runs mit Union First Parallel

- Am `2026-06-07` wurden die Kernexperimente mit `union_first_parallel` wiederholt:
  - Full `dims234`, w16, 3 Wiederholungen:
    - `/home/duman/TeamIndex/study/team_bench_results/team_bench_bins20_hit8_dims234_n50k/team_bench_bins20_hit8_dims234_n50k_w16_2026-06-07_22-03-40`
  - Full `dims345`, w16, 3 Wiederholungen:
    - `/home/duman/TeamIndex/study/team_bench_results/team_bench_bins20_hit8_dims345_n50k/team_bench_bins20_hit8_dims345_n50k_w16_2026-06-07_22-06-35`
  - Close Cases `dims234`, w16, 10 Wiederholungen:
    - `/home/duman/TeamIndex/study/team_bench_results/team_bench_bins20_hit8_dims234_n50k/team_bench_bins20_hit8_dims234_n50k_w16_2026-06-07_22-20-07`
  - Close Cases `dims345`, w16, 10 Wiederholungen:
    - `/home/duman/TeamIndex/study/team_bench_results/team_bench_bins20_hit8_dims345_n50k/team_bench_bins20_hit8_dims345_n50k_w16_2026-06-07_22-26-08`
- Zusammenfassende Reports:
  - Full:
    - `/home/duman/TeamIndex/study/team_bench_results/strategy_selection_analysis/full_union_first_parallel_core/strategy_selection_report.md`
  - Close Cases:
    - `/home/duman/TeamIndex/study/team_bench_results/strategy_selection_analysis/close_cases_w16_reps10/strategy_selection_report.md`
- Ergebnis aus den 48 Full-Szenarien:
  - `union_first_parallel` ist gegenueber Union First in `48/48` Faellen mindestens 10 Prozent schneller
  - mittlerer Faktor gegenueber Union First: `1.383`
  - in `33/48` Faellen ist `union_first_parallel` mindestens 2 Prozent schneller als `current_handcrafted`
  - `dynamic_selective_expansion` ist in `40` Faellen mehr als 10 Prozent langsamer als `current_handcrafted`
- Ergebnis aus den 25 Close Cases mit 10 Wiederholungen:
  - `union_first_parallel` ist gegenueber Union First in `25/25` Faellen mindestens 10 Prozent schneller
  - mittlerer Faktor gegenueber Union First: `1.346`
  - in `18/25` Faellen ist `union_first_parallel` mindestens 2 Prozent schneller als `current_handcrafted`
  - alle `25/25` Close Cases sind weiterhin knapp oder varianzempfindlich, d.h. die Standardabweichungen ueberlappen
- Interpretation:
  - Die neuen Ergebnisse bestaetigen Max' Hinweis zur Varianz: einzelne Siegerlabels sind in knappen Faellen nicht robust genug
  - Trotzdem ist das Muster stabil genug, um `union_first_parallel` als eigene Planfamilie zu behandeln
  - `4D` wirkt aktuell wie ein Grenzbereich zwischen gebremster Expansion und physischer Union-Balancierung
  - `5D` spricht stark fuer physische Union-Balancierung, weil Expansion dort typischerweise `ISE=128` erzeugt, waehrend `union_first_parallel` bei `ISE=0` bleibt
- Aktueller Regelentwurf:
  - `2D`: Kontroll-/Overheadbereich; No-Expansion ernst nehmen
  - `3D`: Union First Parallel ist meist stark, gebremste Expansion bleibt nahe dran
  - `4D`: Entscheidungsgrenze; beide Familien vergleichen und Varianz berichten
  - `5D`: Union First Parallel ist im aktuellen Scope der staerkste Kandidat
  - hohe ISE Counts vor Ausfuehrung pruefen und ggf. verwerfen oder staerker gruppieren

## 14. Strategy Selection Visuals

- Neues Visualisierungsskript:
  - [strategy_selection_to_plots.py](/home/duman/TeamIndex/study/strategy_selection_to_plots.py)
  - liest vorhandene `strategy_selection_analysis`-CSVs
  - fuehrt keine Benchmarks aus
  - erzeugt einzelne PDFs und ein mehrseitiges `strategy_selection_dashboard.pdf`
- Erzeugte Visuals fuer:
  - `/home/duman/TeamIndex/study/team_bench_results/strategy_selection_analysis/full_union_first_parallel_core/plots`
  - `/home/duman/TeamIndex/study/team_bench_results/strategy_selection_analysis/close_cases_w16_reps10/plots`
- Wichtige Plot-Typen:
  - `key_findings.pdf`: zentrale Zahlen, Gewinnerverteilung, Speedup-Histogramm, Margin-Histogramm
  - `winner_map.pdf`: Gewinner je `T_rel`, Teamanzahl und Dimension
  - `union_first_parallel_speedup_heatmap.pdf`: Speedup von Union First Parallel gegen Union First
  - `relative_runtime_by_family.pdf`: wie weit jede Strategie von der besten Strategie der Familie entfernt ist
  - `ise_vs_speedup.pdf`: zeigt, dass Union First Parallel bei `ISE=0` trotzdem hohe Speedups erreicht
  - `workload_vs_speedup.pdf`: zeigt Speedup in Abhaengigkeit der getroffenen Zellen
  - `winner_margin_variance.pdf`: zeigt, welche Siege varianzempfindlich sind
- Interpretation fuer die Arbeit:
  - Die Markdown-Reports bleiben als genaue Quelle erhalten
  - Die PDFs sind besser geeignet fuer Thesis/Meeting, weil sie Muster und Unsicherheit direkt sichtbar machen
  - Besonders wichtig sind `winner_map`, `union_first_parallel_speedup_heatmap`, `ise_vs_speedup` und `winner_margin_variance`

## 15. Nicht-uniforme team_bench-Profile

- Max' `team_bench`-Generator ist nicht auf uniforme Daten beschraenkt:
  - `generate_indices(...)` akzeptiert pro Team eine beliebige Wahrscheinlichkeitsmatrix
  - bisher war nur unser Wrapper [generate_team_bench_data.py](/home/duman/TeamIndex/study/generate_team_bench_data.py) uniform, weil dort immer `np.ones(...)` genutzt wurde
- Der Wrapper kann jetzt kontrollierte nicht-uniforme Profile erzeugen:
  - `uniform`: bisheriger Referenzfall
  - `mixed_team_imbalance`: Teams haben deutlich unterschiedliche Query-Massen
  - `cell_skew`: gleiche Query-Region, aber sehr ungleich volle Zellen/Posting Lists
  - zusaetzlich vorbereitet: `query_hotspot` und `anti_query_hotspot`
- Der Workflow schreibt `distribution_profile` und `distribution_strength` in Szenarien, CSVs und Analysen.
  Alte Runs ohne diese Spalten werden in [analyze_strategy_selection.py](/home/duman/TeamIndex/study/analyze_strategy_selection.py) und
  [strategy_selection_to_plots.py](/home/duman/TeamIndex/study/strategy_selection_to_plots.py) automatisch als `uniform` gelesen.
- Neues lokales Pilot-Experiment:
  - `study/experiments/team_bench_bins10_hit4_dims356_profiles_n200k.json`
  - `N=200000`, `bins=10`, `hit=4`, Dimensionen `3D/5D/6D`, Teamanzahlen `2/3/4`
  - Profile `uniform`, `mixed_team_imbalance`, `cell_skew`
  - `81` Szenarien in `27` Familien
  - Konfigurationsdatei ist lokal ignoriert, weil `study/experiments/*` per `.gitignore` nicht committed wird
- Sicherheits-/Methodikpunkt:
  - Bei 6D und kleinen `N` koennen sehr wenige Query-Treffer entstehen
  - [generate_team_bench_data.py](/home/duman/TeamIndex/study/generate_team_bench_data.py) gibt deshalb eine Warnung aus, wenn die erwarteten Treffer im kleinsten Team unter der Schwelle liegen
- Wichtiger Fix in [evaluation.py](/home/duman/TeamIndex/code/python/TeamIndex/evaluation.py):
  - `_determine_groups(..., group_count == 1)` meldete bisher `min_group_size=max_group_size=1`
  - korrekt ist die tatsaechliche Anzahl Blaetter in dieser einzigen Gruppe
  - der Bug wurde durch `mixed_team_imbalance` sichtbar, weil `union_first_parallel` bei stark unbalancierten Teams ein kleines Team auf eine einzige Union-Gruppe reduzieren kann
  - fuer aktuelle lokale Laeufe wurde derselbe Fix auch in der installierten venv-Kopie angewendet; langfristig sollte das Paket neu installiert werden
- Smoke-Test:
  - erzeugt wurde die Familie `teams2_dim3_mixed_team_imbalance` mit drei `T_rel`-Werten
  - Query-Massen: grosses Team ca. `0.589`, kleines Team ca. `0.044`
  - Ergebnisordner:
    - `/media/data/duman/teamindex/team_bench/team_bench_bins10_hit4_dims356_profiles_n200k/teams2_dim3_mixed_team_imbalance`
  - Mini-Benchmark mit einer Wiederholung:
    - `/home/duman/TeamIndex/study/team_bench_results/team_bench_bins10_hit4_dims356_profiles_n200k/team_bench_bins10_hit4_dims356_profiles_n200k_w16_2026-06-08_00-08-53`
    - `T_rel=0.10`: `union_first_parallel` gewinnt im Einzellauf
    - `T_rel=0.60`: `union_first_parallel` gewinnt im Einzellauf
    - `T_rel=0.85`: `dynamic_selective_expansion` gewinnt knapp im Einzellauf
- Interpretation:
  - nicht-uniforme Daten erzeugen tatsaechlich andere Situationen als die bisherigen uniformen Runs
  - das macht `dynamic_selective_expansion` nicht automatisch gut, aber es ist nicht mehr nur eine tote Strategie
  - fuer belastbare Aussagen brauchen diese Profilruns mehrere Wiederholungen und weitere Familien

## 16. Venv-Reinstall und Thesis-Heatmap

- Am `2026-06-08` wurde das virtuelle Environment neu erstellt:
  - altes Environment: `venv.old/`
  - neues Environment: `venv/`
  - [`.gitignore`](/home/duman/TeamIndex/.gitignore) ignoriert jetzt auch `venv.old/`
- Grund fuer den Reinstall:
  - die Python-Quellen liegen unter [code/python/TeamIndex](/home/duman/TeamIndex/code/python/TeamIndex)
  - die Study-Skripte importieren aber das installierte Paket aus `venv/lib/python3.12/site-packages/TeamIndex`
  - Aenderungen an [evaluation.py](/home/duman/TeamIndex/code/python/TeamIndex/evaluation.py) werden deshalb erst durch Neuinstallation des Pakets im `venv` wirksam, sofern keine editable Installation genutzt wird
- Geprueft:
  - `venv/bin/python --version`: Python `3.12.3`
  - `pip show TeamIndex`: Paket ist im neuen `venv` installiert
  - importierte Datei: `/home/duman/TeamIndex/venv/lib/python3.12/site-packages/TeamIndex/evaluation.py`
  - `_determine_groups(..., group_count == 1)` enthaelt dort den Fix mit `leaf_count`
- Thesis:
  - die Speedup-Heatmap aus `full_union_first_parallel_core` wurde nach `thesis/fig/ufp_speedup_heatmap.pdf` kopiert
  - [Abschlussarbeit.tex](/home/duman/TeamIndex/thesis/Abschlussarbeit.tex) enthaelt jetzt eine Figure mit erklaerendem Text zu Union First Parallel
  - `pdflatex` wurde zweimal erfolgreich ausgefuehrt
  - Ausgabe: `thesis/Abschlussarbeit.pdf` mit `24` Seiten, ohne LaTeX-Warnungen zu undefinierten Referenzen

## 17. Fokussierter 5D-Teamanzahl-Block

- Der finale Schwerpunkt wurde auf 5D-Teams gelegt:
  - Teamanzahl `2, 3, 4, 5`
  - `T_rel = 0.10, 0.35, 0.60, 0.85`
  - `N = 1,000,000`
  - `32` Worker
  - `5` Wiederholungen pro Strategie und Szenario
  - `20` Bins pro Dimension, davon `8` Query-Bins pro Dimension
- Fuer ein 5D-Team bedeutet das:
  - `20^5 = 3,200,000` Index-Zellen pro Team
  - `8^5 = 32,768` getroffene Zellen pro Team
- Im uniformen 5D-Teamanzahl-Block gewinnt `union_first_parallel` in allen `16` Szenarien.
- Interpretation:
  - In den uniformen/balancierten Faellen dominiert grosse Union-Arbeit pro Team.
  - `union_first_parallel` verbessert die physische Verteilung dieser Union-Arbeit, ohne die logische Query durch Expansion in mehr ISEs zu zerlegen.
  - Das stuetzt die These, dass Optimierung nicht nur "expandieren oder nicht expandieren" ist, sondern auch physische Parallelisierung der Union-Arbeit umfasst.
- Zentrale Outputs:
  - `study/team_bench_results/team_bench_bins20_hit8_5d_teams2345_n1m/combined_5d_teamcount/summary_5d_teamcount_latest.csv`
  - `team_bench_5d_teamcount_runtime.pdf`
  - `team_bench_5d_teamcount_ufp_speedup.pdf`

## 18. Mixed-Team-Imbalance mit 3, 4 und 5 Teams

- Das Profil `mixed_team_imbalance` wurde fuer 4 und 5 Teams nachgeneriert und gemessen.
- Bereits vorhanden war der 3-Team-Imbalance-Block; die neue Messung erweitert ihn auf mehr Teams.
- Voller Run:
  - `study/team_bench_results/team_bench_bins20_hit8_5d_teams2345_n1m/team_bench_bins20_hit8_5d_teams2345_n1m_w32_2026-06-24_21-52-47`
  - 4 und 5 Teams
  - 5 Strategien
  - 5 Wiederholungen
  - 32 Worker
- Ergebnisse:
  - 3 Teams, mixed imbalance:
    - `T_rel=0.10`: `bounded_selective_expansion`
    - `T_rel=0.35`: `bounded_selective_expansion`
    - `T_rel=0.60`: `current_handcrafted`
    - `T_rel=0.85`: `union_first_parallel`
  - 4 Teams, mixed imbalance:
    - `union_first_parallel` gewinnt alle vier `T_rel`-Werte.
  - 5 Teams, mixed imbalance:
    - `T_rel=0.10`: `current_handcrafted`
    - `T_rel=0.35`: `current_handcrafted`
    - `T_rel=0.60`: `union_first_parallel`
    - `T_rel=0.85`: `union_first_parallel`
- Interpretation:
  - Imbalance macht fruehe Intersections konkurrenzfaehig, garantiert aber keinen Sieg gegen `union_first_parallel`.
  - Mehr Teams erhoehen sowohl Pruning-Potenzial als auch Union-Arbeit und Abhaengigkeiten.
  - Die relevante Frage ist daher nicht nur "liegt Imbalance vor?", sondern ob der zusaetzliche Pruning-Nutzen die zusaetzliche Plan-, Scheduling- und ISE-Arbeit uebersteigt.
- Zentrale Outputs:
  - `summary_5d_mixed_teamcount_latest.csv`
  - `team_bench_5d_mixed_teamcount_runtime.pdf`
  - `team_bench_5d_mixed_teamcount_ufp_speedup.pdf`
  - `team_bench_5d_profile_teamcount_winners.pdf`

## 19. Bin-Geometrie in CSVs und PDFs

- Max hatte explizit nach der Anzahl der Bins gefragt.
- Die Run-Skripte schreiben inzwischen Bin-/Zell-Metriken in die Ergebnis-CSVs:
  - `bins_per_dimension`
  - `selected_bins_per_dimension`
  - `index_bin_cells_per_team`
  - `selected_bin_cells_per_team`
  - `total_index_bin_cells`
  - `total_selected_bin_cells_nominal`
  - `selected_bin_cell_fraction`
- Fuer neuere Runs sind diese Spalten direkt in `summary_by_variant.csv` vorhanden.
- Fuer aeltere 5D-Runs, die vor dieser Erweiterung erzeugt wurden, rekonstruiert [analyze_5d_teamcount_results.py](/home/duman/TeamIndex/study/analyze_5d_teamcount_results.py) die Bin-Geometrie aus der bekannten Experimentkonfiguration:
  - `20` Bins pro Dimension
  - `8` Query-Bins pro Dimension
  - `5D`
  - Teamanzahl aus dem Szenario
- Die kombinierten CSVs enthalten dadurch keine `NaN`-Werte mehr bei Bin-Geometrie.
- Die wichtigsten 5D-PDFs enthalten unten eine sichtbare Konfigurationszeile:
  - `5D-Teams, N=1M, 20 Bins/Dimension, 8 Query-Bins/Dimension, 32 Worker`
- Offener Thesis-Schritt:
  - Falls mit `thesis_reviewed/` weitergearbeitet wird, sollen nur gezielt die aktualisierten PDFs in `thesis_reviewed/fig/` ersetzt werden.
  - Keine allgemeine Thesis-Bereinigung ohne explizite Anweisung.

## 20. CPU-/Idle-Zeit und Taskflow-Fallstudien

- Max hatte angemerkt, dass Laufzeiten nicht nur ueber Gewinnerstrategien erklaert werden sollen.
- Relevante physische Metriken:
  - Busy-CPU-Zeit
  - Idle-Zeit
  - effektive Parallelitaet
  - Worker-Load-Imbalance
  - laengste Tasks
- Diese Metriken werden aktuell ueber Taskflow/Gantt-Fallstudien berechnet, nicht als Standardspalten in allen grossen Benchmark-Summaries.
- Bereits vorhandene Fallstudien:
  - balanced 5D: `tb_3t_5d_t010_uniform`
  - unbalanced 5D: `tb_3t_5d_t010_mixed_team_imbalance`
  - jeweils mit `baseline_union_first`, `union_first_parallel`, `current_handcrafted`, `bounded_selective_expansion`
- Zentrale Befunde:
  - Balanced: `union_first_parallel` ist schnell, weil es die Union-Arbeit besser verteilt; Baseline Union First hat hohe Idle-Anteile und schlechte effektive Parallelitaet.
  - Unbalanced: Handcrafted/Bounded koennen trotz mehr Busy-CPU-Zeit gewinnen, weil fruehe Intersections Pruning ermoeglichen und die Parallelitaet anders verteilt wird.
- Offener Schritt:
  - Fuer den Call beziehungsweise die Thesis sollten 2-3 repraesentative Faelle kurz zusammengefasst werden:
    - uniform, `union_first_parallel` gewinnt
    - mixed, `current_handcrafted` gewinnt
    - mixed, `union_first_parallel` gewinnt wieder

## 21. Thesis-Status und Arbeitsregeln

- Aktuelle Thesis-Version auf dem Server:
  - `thesis_reviewed/`
- Diese Version enthaelt bereits:
  - Kapitel "Zentrale Ergebnisse und Beitrag der Arbeit"
  - den 5D-Teamanzahl-Block
  - den nicht-uniformen 5D-Imbalance-Block
  - den Abschnitt "Einfluss zusaetzlicher Teams bei Imbalance"
  - die neuen mixed-Teamcount-PDFs
- Wichtige Arbeitsregel:
  - `thesis_reviewed/` ist die vom Nutzer hochgeladene, bereits bereinigte Thesis-Version.
  - Nicht proaktiv bereinigen.
  - Nur gezielt aendern, wenn der Nutzer es explizit anfordert.
  - Keine allgemeinen Umformulierungen oder Strukturveraenderungen ohne Rueckfrage.

## 22. Session Notes wiederhergestellt

- `SESSION_NOTES.md` wurde versehentlich als "alte lokale Notiz" geloescht.
- Die Datei wurde aus der Git-History wiederhergestellt:
  - letzter Commit mit Inhalt: `fdcded1`
  - Commit `55a023b` hatte die Datei geloescht
- Entscheidung vom `2026-06-25`:
  - `SESSION_NOTES.md` soll wieder gepflegt und gepusht werden.
  - Sie wurde daher aus `.gitignore` entfernt.
  - Alte kurzfristige Max-Notizen bleiben ignoriert beziehungsweise lokal.

