# TeamIndex TODO

Interne Arbeitsliste fuer uns beide. Nicht als Meeting-Protokoll gedacht,
sondern als praktische Liste zum Abarbeiten bis zur Abgabe.

Wichtige Scope-Regel: Keine kuenstliche harte Begrenzung auf 1-2 Experimente,
aber jedes weitere Experiment muss eine klare Frage beantworten. Smoke-Tests,
Plot-Regeneration und Taskflow-Fallstudien zaehlen als Verifikation/Auswertung,
nicht als neue grosse Experimentfamilien.

## P0 - Muss als Naechstes sitzen

- [x] **Bin-/Blattanzahl einmal sauber testen**
  - Ziel: Sicherstellen, dass neue Bin-Spalten in `results.csv`, `summary_by_variant.csv` und Analyse-Outputs korrekt gefuellt werden.
  - Erledigt: Smoke-Test `tb_3t_5d_t010_uniform` mit `baseline_union_first` und `union_first_parallel`, 1 Wiederholung, 32 Worker.
  - Ergebnis: Terminal-Ausgabe und CSV-Spalten sind vorhanden.
  - Beispiel: 20 Bins/Dimension, 8 Query-Bins/Dimension, `20^5 = 3,200,000` Index-Zellen pro Team, `8^5 = 32,768` getroffene Zellen pro Team, `98,304` getroffene Zellen gesamt bei 3 Teams.

- [x] **Busy/Idle/Parallelitaet gezielt fuer Fallstudien auswerten**
  - Ziel: Laufzeiten nicht nur berichten, sondern begruenden.
  - Erledigt: balanced 5D-Trace `tb_3t_5d_t010_uniform` und unbalanced 5D-Trace `tb_3t_5d_t010_mixed_team_imbalance` mit vier Strategien und 32 Workern erzeugt.
  - Output: jeweils `gantt_overview.pdf` und `worker_utilization_summary.csv`.
  - Befund balanced: UF Parallel ist am schnellsten und hat weniger Busy-CPU-Zeit als Expansionsstrategien; Baseline UF hat deutlich schlechtere effektive Parallelitaet und hohes Lastungleichgewicht.
  - Befund unbalanced: Handcrafted/Bounded gewinnen knapp; UF Parallel hat weniger Busy-CPU-Zeit, aber deutlich mehr Idle und schlechtere effektive Parallelitaet.

- [x] **Balanced vs. unbalanced final visualisieren**
  - Ziel: Zeigen, dass UF Parallel im balanced/uniformen Fall stark ist, aber Imbalance die Strategiewahl verschiebt.
  - Erledigt: `team_bench_5d_strategy_shift_balanced_vs_unbalanced.pdf` erzeugt und in die Thesis-Figuren kopiert.
  - Zusatzfix: Konsolidierte 5D-Auswertung ignoriert jetzt partial/smoke runs, damit kurze Verifikationstests keine Thesis-Plots verfälschen.

- [x] **Finales neues Experiment: viele Teams mit Imbalance untersuchen**
  - Ziel: Klaeren, ob "Imbalance bevorzugt fruehe Intersection" auch bei vielen Teams gilt.
  - Erledigt: `mixed_team_imbalance` fuer 4 und 5 Teams erzeugt und mit 32 Workern, 5 Wiederholungen und allen fuenf Strategien gemessen.
  - Output: `team_bench_bins20_hit8_5d_teams2345_n1m_w32_2026-06-24_21-52-47` plus neue kombinierte Plots im `combined_5d_teamcount`-Ordner.
  - Befund: Bei 4 Teams gewinnt UF Parallel in allen vier T_rel-Stufen. Bei 5 Teams gewinnt Current Handcrafted fuer T_rel=0.10 und 0.35, UF Parallel fuer T_rel=0.60 und 0.85.
  - Interpretation: Imbalance macht fruehe Intersections konkurrenzfaehig, garantiert aber keinen Sieg gegen UF Parallel. Entscheidend ist die Kombination aus Pruning-Nutzen, Union-Arbeit, ISE Count und Parallelitaetsausnutzung.

## P1 - Wichtig fuer die Thesis

- [ ] **Finalen Experiment-Scope festziehen**
  - Vorschlag: Bestehende 5D balanced/uniform und 5D mixed-team-imbalance als Kern behalten.
  - Neue quantitative Experimente nur ergaenzen, wenn sie eine konkrete offene Frage beantworten.
  - Taskflow/Gantt als qualitative Fallstudie nutzen, nicht als eigene grosse Experimentfamilie aufblasen.
  - Ziel: Wenige Experimentfamilien, aber tief und gut begruendet.

- [ ] **Thesis-Begriffe konsequent saeubern**
  - Begriffe trennen: Bins, Blaetter, Listen, ISEs, Requests, Union Cardinality, Input Cardinality, CPU-Zeit, Wall-Time, Idle-Zeit.
  - Ziel: Keine unsauberen Formulierungen wie tautologische ISE-Erklaerungen.

- [ ] **Abbildungen ausduennen und zuspitzen**
  - Ziel: Jede Abbildung beantwortet genau eine Frage.
  - Entfernen oder ersetzen, wenn nur "sieht halt nett aus" oder "steht ueberall dasselbe" gilt.

- [x] **Uni-LaTeX-Vorlage uebernehmen**
  - Ziel: Die finale Bachelorarbeit kann sich an `UDO_Thesis_2026/main.tex` orientieren.
  - Erledigt: UDO-orientierte Parallelversion unter `thesis/main.tex` angelegt; alte `thesis/Abschlussarbeit.tex` bleibt erhalten.
  - Hinweis: Die Vorlage ist Orientierung, kein Zwang. Finale Kompilierung erfolgt auf dem PC mit vollständiger TeX-Installation.

- [ ] **Ergebnisorientierte Thesis-Struktur herstellen**
  - Ziel: Ergebnisse nicht bis zum Ende verstecken. Direkt nach Einleitung/Motivation frueh sagen, was die Arbeit herausgefunden hat.
  - Schreibprinzip: Erst Kernergebnis nennen, danach erlaeutern, wie Experimente, Metriken und Strategien zu diesem Ergebnis fuehren.
  - Konkreter Umbau: Nach der Einleitung eine kurze Ergebnis-/Beitragssektion einfuegen, z.B. "Zentrale Ergebnisse" oder "Beitrag der Arbeit".
  - Dort knapp nennen:
    - Es gibt keine universell beste Strategie.
    - UF Parallel ist stark bei grossen, balancierten Team-Unions.
    - Imbalance kann fruehe Intersection wieder attraktiv machen.
    - Bin-/Blattanzahl, Datenvolumen und Parallelitaetsausnutzung erklaeren Laufzeitunterschiede besser als reine Gewinnerlisten.

- [ ] **Ergebnistext nach Ursache statt nur Gewinner strukturieren**
  - Nicht nur: Strategie X gewinnt.
  - Sondern: Strategie X gewinnt, weil Datenvolumen, Bin-Anzahl, Parallelitaet, Imbalance oder Pruning-Potenzial in diesem Fall dominieren.

## P2 - Optional, wenn noch Zeit bleibt

- [ ] **SDSS-Daten nur noch als bewusst optionale Validierung behandeln**
  - Einschaetzung: Vermutlich nicht mehr als Kernexperiment geeignet, weil Vorbereitung, Datenverstaendnis, Debugging und Interpretation zu viel Restzeit binden koennen.
  - Risiko: Ein ganzer Tag Setup ist nur der Anfang; danach muessen Query-Auswahl, Datenverteilung, Korrelationen und Ergebnisse sauber erklaert werden.
  - Sinnvoll nur, wenn bereits fertige, leicht nutzbare Skripte existieren und ein sehr kleiner Smoke-Test schnell gelingt.
  - Default-Entscheidung: Nicht in den Kern der Arbeit aufnehmen, sondern als Ausblick oder externe Validierung erwaehnen.

- [ ] **Weitere nicht-uniforme Datenfamilie pruefen**
  - Nur falls sie eine klare neue Frage beantwortet.
  - Nicht mehr einfach "mehr Experimente" ohne These.

- [ ] **Alte Result-Ordner systematisch archivieren**
  - Ziel: Uebersichtlichkeit.
  - Nicht loeschen, bevor klar ist, welche Runs fuer Thesis/Plots noch gebraucht werden.

- [ ] **Kleine Entscheidungs-/Regelmatrix erstellen**
  - Beispiel:
    - grosse, balancierte Team-Unions -> UF Parallel
    - starke Team-Imbalance -> gebremste Expansion / Handcrafted
    - viele Bins/Listen -> Gruppierung/Planbegrenzung wichtig
    - wenig Arbeit -> konservativ bleiben, Overhead vermeiden

## Aktueller Arbeitsmodus

- Erst messen, wenn klar ist, welche Frage beantwortet wird.
- Keine harte Experimentgrenze, aber lieber Tiefe als Breite und keine Messung ohne klare Fragestellung.
- Ergebnisse immer mit Ursache interpretieren.
- Keine Suche nach perfektem Optimierer.
- Fokus: Situationen erkennen und passende Strategie/Stellschraube begruenden.
