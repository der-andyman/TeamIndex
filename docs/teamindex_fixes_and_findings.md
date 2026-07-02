# TeamIndex Fixes und technische Befunde

Diese Datei fasst die wichtigsten technischen Probleme zusammen, die waehrend der Bachelorarbeits-Experimente im TeamIndex-Code oder in der Studien-Infrastruktur gefunden und behoben wurden. Sie ist als kurze Uebersicht fuer Ruecksprache mit Max gedacht, nicht als vollstaendige Commit-Historie.

## 1. Falsche Speicherordnung bei 2D-Team-Erzeugung

**Ort:** `code/python/TeamIndex/creation.py`

**Problem:** Bei 2D-Teams wie `B-I` und `F-H` traten auffaellige beziehungsweise falsche Ergebnisse auf. Ursache war die Uebergabe der Teamdaten an den alten `BatchConverter`: Die Daten mussten fuer den bestehenden C++/Converter-Pfad in Fortran-/Column-Major-Layout uebergeben werden.

**Fix:** `team_data` wird vor der Uebergabe an den alten `BatchConverter` passend als Fortran-/Column-Major-Array bereitgestellt.

**Relevanz:** Ohne diesen Fix konnten 2D-Team-Indizes inkonsistent erzeugt oder ausgewertet werden. Das war ein echter Korrektheitsfix, kein reiner Performance-Tweak.

## 2. Dry-Run-/Planexport nicht robust gegen NumPy-Skalare

**Ort:** `code/python/TeamIndex/evaluation.py`

**Problem:** Beim Export von Plaenen beziehungsweise Dry-Run-Informationen konnten `numpy.int64` oder andere NumPy-Skalare in Metadaten auftauchen. Diese Werte sind nicht automatisch JSON-serialisierbar.

**Fix:** Die Dry-Run-/Planexport-Logik wurde robuster gemacht, indem NumPy-Skalare und aehnliche Werte in normale Python-Typen konvertiert werden.

**Relevanz:** Dadurch lassen sich Plaene und Statistikdaten stabiler exportieren und spaeter fuer PDF-/CSV-Auswertungen weiterverwenden.

## 3. Importierter Paketstand im venv war nicht automatisch der Quellcode-Stand

**Ort:** Entwicklungsumgebung / `venv`

**Problem:** Aenderungen unter `code/python/TeamIndex/` wurden nicht automatisch vom laufenden Python-Code verwendet, weil das installierte Paket aus `venv/lib/python3.12/site-packages/TeamIndex` importiert wurde. Dadurch konnte scheinbar korrigierter Code in Experimenten noch nicht wirksam sein.

**Fix / Vorgehen:** Nach Aenderungen am Paket muss TeamIndex im `venv` neu installiert werden, sofern keine editable Installation genutzt wird. Bei der Fehlersuche wurde explizit geprueft, welche Datei Python tatsaechlich importiert.

**Relevanz:** Das erklaert, warum ein Fix im Repository nicht automatisch in Messlaeufen sichtbar war. Es ist ein wichtiger Reproduzierbarkeits- und Entwicklungsbefund.

## 4. Experimentlaeufe brauchten Schutz vor zu grossen Plaenen

**Ort:** `study/run_team_bench_experiments.py`, Strategielogik in `study/mopts_strategies.py`

**Problem:** Manche Plaene, insbesondere ungebremste Expansion in hoeheren Dimensionen, konnten extrem hohe ISE Counts erzeugen. Das fuehrte zu sehr langen Laeufen oder potenziell instabilem Verhalten.

**Fix:** Es wurden Sicherheitsgrenzen und Warnungen eingefuehrt, insbesondere fuer geschaetzte ISE Counts. Plaene mit zu hoher geschaetzter Komplexitaet koennen vor der nativen Ausfuehrung uebersprungen werden.

**Relevanz:** Das schuetzt Server und Messlaeufe und macht klar, dass unkontrollierte Plan-Enumeration praktisch nicht robust ist.

## 5. Bin-/Zellmetriken fehlten in Ergebnis-CSVs

**Ort:** `study/run_team_bench_experiments.py`, `study/analyze_5d_teamcount_results.py`

**Problem:** Max fragte explizit nach der Anzahl der Bins beziehungsweise Index-Zellen. Fruehe Ergebnisdateien enthielten diese Metriken nicht direkt oder nicht konsistent.

**Fix:** Neue Runs schreiben Bin-/Zellmetriken in die Ergebnis-CSVs, unter anderem `bins_per_dimension`, `selected_bins_per_dimension`, `index_bin_cells_per_team`, `selected_bin_cells_per_team`, `total_index_bin_cells`, `total_selected_bin_cells_nominal` und `selected_bin_cell_fraction`. Fuer aeltere 5D-Runs rekonstruiert die kombinierte Analyse diese Geometrie aus der bekannten Experimentkonfiguration.

**Relevanz:** Dadurch koennen Laufzeiten besser mit Blatt-/Listenoverhead erklaert werden, statt nur Gewinnerstrategien zu berichten.

## 6. Analyse musste Smoke-/Partial-Runs ignorieren koennen

**Ort:** `study/analyze_5d_teamcount_results.py`

**Problem:** Kurze Verifikations- oder Smoke-Runs konnten in kombinierte Auswertungen rutschen und dadurch Thesis-Plots verfälschen.

**Fix:** Die konsolidierte 5D-Auswertung ignoriert Partial-/Smoke-Runs beziehungsweise waehlt gezielter die relevanten vollstaendigen Runs aus.

**Relevanz:** Verhindert, dass Testlaeufe unbeabsichtigt in finale Ergebnisplots eingehen.

## 7. Falsche Gruppengroessen bei `group_count == 1`

**Ort:** `code/python/TeamIndex/evaluation.py`

**Problem:** Bei `_determine_groups(..., group_count == 1)` wurden `min_group_size` und `max_group_size` als `1` berichtet, obwohl die einzige Gruppe tatsaechlich mehrere Blaetter enthalten kann. Dadurch konnten Planstatistiken zur Gruppierung beziehungsweise Blattverteilung falsch interpretiert werden.

**Fix:** Fuer den Spezialfall `group_count == 1` wird nun die tatsaechliche Anzahl der Blaetter in dieser Gruppe als Gruppengroesse verwendet.

**Relevanz:** Der Fix betrifft nicht die logische Query-Ausfuehrung, aber die Korrektheit der ausgegebenen Plan- und Diagnosemetriken. Besonders bei unbalancierten Szenarien war das wichtig, weil `union_first_parallel` kleine Teams teilweise auf eine einzige Gruppe reduziert.

## 8. Nicht verwendeter liburing-/O_DIRECT-Pfad fuer die Experimente

**Ort:** Runtime-Konfiguration / Backend-Ausfuehrung

**Problem:** Der Pfad mit `liburing` und `O_DIRECT` konnte in bestimmten Konstellationen inkorrekte Ergebnisse liefern beziehungsweise war fuer die Optimizer-Experimente nicht die verlaesslichste Grundlage.

**Fix / Vorgehen:** Fuer die Bachelorarbeits-Experimente wurde der DRAM-Modus verwendet. `o_direct` wurde nicht als Default fuer die relevanten Messungen genutzt.

**Relevanz:** Die Arbeit bewertet Planstrategien und nicht den I/O-Pfad. Durch die Nutzung des DRAM-Modus wurden die Strategieexperimente von diesem Runtime-/I/O-Problem entkoppelt.

## 9. Resultat-PDFs mussten reproduzierbar aus CSVs erzeugbar sein

**Ort:** `study/generate_comparison_pdfs.py`, spaeter Study-Plot-Skripte

**Problem:** PDF-Vergleiche sollten nicht nur direkt am Ende eines langen Experiments entstehen. Sonst muesste man fuer reine Plot-Aenderungen die Experimente erneut ausfuehren.

**Fix:** Es wurden Skripte bereitgestellt, die aus vorhandenen CSV-Ergebnissen nachtraeglich Vergleichs-PDFs erzeugen.

**Relevanz:** Trennt Messung und Visualisierung und spart Serverzeit.

## 10. Taskflow-/Gantt-Diagnose fuer Parallelitaet

**Ort:** `study/export_team_bench_taskflow_trace.py`, `study/task_stats_to_gantt_pdf.py`

**Problem:** Laufzeitunterschiede liessen sich nicht allein durch Gewinnerlabels erklaeren. Max schlug vor, Gantt-/Taskflow-Diagramme beziehungsweise Busy-/Idle-Zeiten zu betrachten.

**Fix:** Es wurden Hilfsskripte genutzt/ergaenzt, um Taskflow-Traces, Worker-Auslastung, Busy-/Idle-Zeiten und effektive Parallelitaet fuer repraesentative Faelle auszuwerten.

**Relevanz:** Ermoeglicht qualitative Diagnose, warum eine Strategie schneller oder langsamer ist, ohne daraus eine neue grosse Experimentdimension zu machen.

## 11. Pfad- und Artefaktstruktur fuer Study-Workflow

**Ort:** `study/study_paths.py`, `minimal_example/example_paths.py`, Study-Skripte

**Problem:** Der urspruengliche `minimal_example`-Ordner wurde im Verlauf der Arbeit fuer nicht-minimale Experimente zweckentfremdet. Zudem sollten grosse Indexdaten ausserhalb des Repositories auf einem lokalen Datenpfad liegen.

**Fix:** Arbeits- und Auswertungsskripte wurden in `study/` organisiert. Pfade wurden so angepasst, dass grosse Index- und Studienartefakte ausserhalb des Repositories liegen und lokal erzeugte Artefakte ignoriert bleiben.

**Relevanz:** Erhoeht Nachvollziehbarkeit und reduziert Risiko, versehentlich grosse oder temporaere Daten ins Repository zu nehmen.

## Einordnung

Die meisten Punkte sind keine Aenderungen am theoretischen TeamIndex-Ansatz, sondern Korrektheits-, Robustheits- und Reproduzierbarkeitsfixes in der Implementierung und Experiment-Infrastruktur. Besonders wichtig fuer Max sind vermutlich:

- der Fortran-/Column-Major-Fix in `creation.py`,
- der NumPy-Serialisierungsfix in `evaluation.py`,
- die ISE-Sicherheitsgrenzen,
- der `group_count == 1`-Fix fuer Gruppierungsmetriken,
- der Befund zum nicht verwendeten liburing-/O_DIRECT-Pfad,
- die nachgetragenen Bin-/Zellmetriken,
- und die Trennung von Messung, Analyse und Visualisierung.
