# team_bench Scope und Methodik

## Unterliegende Frage

Die zentrale Frage ist nicht mehr:

- "Gibt es einen universell besten Optimierer?"

Sondern:

- "Welche sinnvollen Query-/Plansituationen gibt es?"
- "Welche Strategie ist in welcher Situation stark?"
- "Wo zeigt sich weiterer Optimierungsbedarf?"

Damit trennen wir zwei Raeume:

- `Situationsraum`: Welche Faelle koennen auftreten?
- `Strategieraum`: Welche Optimierungsstrategie waehlen wir dafuer?

## Warum team_bench?

`team_bench` erzeugt gezielt nur die fuer eine Query interessanten Teile des
Datensatzes. Dadurch koennen wir kontrollierte Szenarien untersuchen, ohne
wirklich riesige Tabellen materialisieren zu muessen.

Wichtig:

- Das ist gut fuer systematische Szenario-Exploration.
- Fuer Laufzeitmessungen muessen wir trotzdem auf Varianz und zu kleine
  Tupelzahlen achten.

## Aktuell betrachtete Nische

Im Moment betrachten wir bewusst keinen beliebigen Raum, sondern eine
eingeschraenkte und begruendbare Nische:

- `2` und `3` Teams
- Teamdimensionalitaet `3D` und `4D` als aktueller Hauptfokus
- `5D` als naechster Stressbereich, um hoehere Dimensionalitaet kontrolliert zu pruefen
- `2D` nur als Kontroll- und Randfall
- `T_rel` als Regler fuer den Intersection-/Drag-Anteil
- verschiedene `worker_count`-Werte fuer Thread-Skalierung

Diese Einschraenkung ist sinnvoll, weil:

- `1D`-Teams fachlich wenig interessant sind
- reine `2D`-Faelle oft so klein sind, dass Messrauschen und Scheduling-Overhead dominieren
- sehr hohe Dimensionalitaeten den Raum schnell unbeherrschbar machen
- `3D` bis `5D` einen Bereich bilden, in dem ISE Count, Gruppierung und Parallelisierung
  sichtbar relevant werden, ohne sofort in unrealistische Hochdimensionalitaet zu kippen

## Welche Parameter wir variieren

### Situationsraum

- Teamanzahl
- Teamdimensionalitaet
- `T_rel`
- Grid-Aufloesung (`bins_per_dimension`)
- Query-Ausdehnung pro Dimension (`hit`-Bins)
- Datenprofil:
  `uniform`, `mixed_team_imbalance`, `cell_skew`
- `worker_count`

### Strategieraum

- `baseline_union_first`
- `baseline_minimal_intersection`
- `union_first_parallel`
- `current_handcrafted`
- `bounded_selective_expansion`
- `dynamic_selective_expansion`

## Welche Messgroessen wir festhalten sollten

- mittlere Laufzeit
- Standardabweichung der Laufzeit
- IDs pro Sekunde
- Standardabweichung der IDs pro Sekunde
- MiB pro Sekunde
- Standardabweichung der MiB pro Sekunde
- ISE Count
- Request Count
- gewaehlte Bins / relevante Blaetter / Zellanzahl
- tatsaechlich gewaehlte Planform (`expanded teams`, `group_count`, ...)

## Welche Visualisierungen wir nutzen sollten

Die CSVs und Markdown-Reports sind gut als genaue Datenquelle, aber fuer die
Arbeit und fuer Meetings brauchen wir vor allem Darstellungen, die eine konkrete
Frage beantworten:

- Gewinnerkarte: Welche Strategie gewinnt in welcher Situation?
- Speedup-Heatmap: Wie stark verbessert Union First Parallel die Union-First-
  Baseline?
- Relative Laufzeiten: Welche Strategien sind nahe am Optimum und welche kann
  man praktisch ausschliessen?
- ISE-vs-Speedup: Entsteht der Gewinn durch mehr logische Teilausdruecke oder
  durch bessere physische Arbeitsteilung?
- Workload-vs-Speedup: Ab welchem Arbeitsvolumen lohnt sich die Optimierung?
- Margin/Varianz: Welche Siegerlabels sind belastbar und welche sind durch
  Messvarianz unsicher?

Dafuer gibt es [strategy_selection_to_plots.py](/home/duman/TeamIndex/study/strategy_selection_to_plots.py).
Das Skript arbeitet nur auf bereits vorhandenen Analyse-CSVs und erzeugt pro
Analyseordner einen `plots`-Unterordner sowie ein mehrseitiges
`strategy_selection_dashboard.pdf`.

## Warum `current_handcrafted` jetzt wichtig ist

`current_handcrafted` ist nicht nur eine weitere Vergleichsstrategie. Wenn diese
Strategie oft gut abschneidet, obwohl sie per Trial-and-Error entstanden ist,
dann ist das selbst ein Ergebnis:

- Vielleicht kodiert sie bereits robuste halb-adaptive Regeln.
- Vielleicht ist sie fuer bestimmte Situationen sogar naeher an einer guten
  Strategiewahl als komplexere heuristische Ansaetze.

Deshalb sollten wir nicht nur "wer gewinnt?" messen, sondern auch:

- welche Teams `current_handcrafted` expandiert,
- wie stark gruppiert wird,
- in welchen Situationen genau dieses Verhalten gut funktioniert.

## Erste Erkenntnisse zur Strategiewahl

Die aktuellen Laeufe zeigen ein recht klares Muster:

- `2D` ist vor allem ein Overhead-/Rauschbereich. Dort liegen Union First,
  Dynamic und Varianten ohne Expansion oft sehr nah beieinander. Eine
  expandierende Strategie kann dort leicht verlieren, obwohl sie in groesseren
  Faellen sinnvoll ist.
- `3D` bis `5D` profitieren deutlich von besserer Arbeitsteilung. Diese
  Arbeitsteilung muss aber nicht zwingend durch logische Expansion entstehen.
  Der wichtige Punkt ist nicht maximale Parallelitaet, sondern passende
  Parallelitaet mit begrenztem Overhead.
- `current_handcrafted` gewinnt haeufig, weil es zwei Dinge kombiniert:
  ein selektives Team expandieren und grosse Team-Ergebnisse vorher auf wenige
  Gruppen reduzieren.
- `dynamic_selective_expansion` ist in den bisherigen team_bench-Faellen oft zu
  konservativ. Es bleibt Union-First-aehnlich und verpasst dadurch das
  guenstige Pruning durch eine gebremste Expansion.
- `baseline_minimal_intersection` zeigt, warum ungebremste Expansion keine
  robuste Strategie ist: in hoeheren Dimensionen werden die ISE Counts sehr
  gross und die Laufzeiten steigen stark.
- `union_first_parallel` zeigt eine dritte Achse: Es wird kein Team
  expandiert, aber grosse Team-Unions werden in eine worker-orientierte Anzahl
  balancierter Gruppen zerlegt. Die aktuellen Full- und Close-Case-Laeufe
  zeigen, dass dies Union First in 3D bis 5D deutlich verbessern kann, ohne die
  ISE Count zu erhoehen.

Daraus folgt als erste Trennung fuer eine spaetere Strategiewahl:

- Wenn die Query nur einen kleinen 2D-Arbeitsbereich erzeugt, sollte ein
  No-Expansion-Zweig ernsthaft in Betracht gezogen werden.
- Wenn die Query 3D bis 5D Teams mit vielen getroffenen Zellen beruehrt, ist
  eine gebremste Ein-Team-Expansion oder Union First Parallel
  ein starker Kandidat.
- Wenn die geschaetzte ISE Count vor der Ausfuehrung zu gross ist, sollte der
  Plan verworfen oder staerker gruppiert werden.

Als erster neuer Strategiekandidat wurde deshalb
`bounded_selective_expansion` eingefuehrt. Die Strategie bleibt in 2D bewusst
Union-First-aehnlich, gruppiert ab 3D grosse Team-Ergebnisse und expandiert nur
ein selektives Team. Erste Pilotlaeufe zeigen, dass sie die Luecke zwischen
`dynamic_selective_expansion` und `current_handcrafted` tatsaechlich trifft:
sie gewinnt mehrere 3D/4D/5D-Szenarien und bleibt in den uebrigen Faellen meist
nah an `current_handcrafted`.

`union_first_parallel` wurde danach als zweite Folgestrategie
eingefuehrt. In den 48 Full-Szenarien mit w16 ist diese Variante gegenueber
Union First in allen Faellen mindestens 10 Prozent schneller. In den 25
Close-Cases mit 10 Wiederholungen bleibt dieses Muster bestehen. Gleichzeitig
ueberlappen in allen Close-Cases die Standardabweichungen von Gewinner und
Runner-up. Fuer die Arbeit ist deshalb die robuste Aussage nicht, dass
`union_first_parallel` jeden Einzelfall eindeutig dominiert, sondern:

- physische Parallelisierung von Union-Arbeit ist eine eigene Planfamilie
- `4D` ist ein Grenzbereich zwischen gebremster Expansion und Union First Parallel
- `5D` spricht im aktuellen Scope stark fuer Union First Parallel,
  weil Expansion dort hohe ISE Counts erzeugt

## Vorlaeufige Auswahlregeln

Die Regeln sind noch kein endgueltiges Kostenmodell, aber ein sinnvoller
Zwischenstand fuer die weitere Evaluation:

- Kleine `2D`-Workloads sollten nicht aggressiv expandiert werden. Sie dienen
  eher als Overhead- und Varianz-Kontrolle.
- Bei `3D`-Teams ist Union First Parallel ein starker Default-Kandidat;
  gebremste Ein-Team-Expansion sollte als nahe Alternative mitlaufen.
- Bei `4D`-Teams muss gezielt entschieden werden: Die Ergebnisse liegen oft
  nah beieinander, und Varianz kann einzelne Siegerlabels kippen.
- Bei `5D`-Teams ist im aktuellen Scope `union_first_parallel` der
  staerkste Kandidat, weil es die Arbeit verteilt, ohne `ISE=128` zu erzeugen.
- Plaene mit hoher geschaetzter ISE Count sollten vor der Ausfuehrung verworfen
  oder staerker gruppiert werden.
- Wenn mehrere Strategien innerhalb der Standardabweichung liegen, sollte nicht
  der einzelne Sieger, sondern die Planfamilie und ihr Risiko interpretiert
  werden.

## Sinnvolle naechste Experimentbloecke

1. Einfluss der Teamdimensionalitaet (`3D` vs `4D` vs `5D`)
2. Einfluss der Teamanzahl (`2` vs `3` Teams)
3. Einfluss von `T_rel`
4. Einfluss von `worker_count`
5. Einfluss nicht-uniformer Datenprofile
6. gezielte Analyse von `current_handcrafted`

Das Ziel ist dabei nicht ein einziges "Super-Experiment", sondern mehrere
sauber interpretierbare Teil-Experimente.

## Rolle von 2D

`2D` bleibt als Kontrollfall nuetzlich, aber sollte nicht die Hauptargumentation
tragen. Die bisherigen Laeufe zeigen, dass dort viele Strategien nur wenige
Millisekunden auseinanderliegen. Dadurch koennen einzelne Siegerlabels durch
Varianz kippen, obwohl die eigentliche Planlogik kaum unterschiedlich relevant
ist.

Fuer die Arbeit ist daher sauberer:

- `2D` als Hinweis auf Messrauschen, Overhead und kleine Workloads verwenden
- `3D/4D/5D` fuer die eigentliche Strategiediskussion verwenden

## Nicht-uniforme Datenprofile

Die bisherigen Kernlaeufe waren uniform. Das war fuer den Einstieg sinnvoll,
weil dadurch Dimension, Teamanzahl, `T_rel` und Gruppierung isolierter
betrachtet werden konnten. Fuer die eigentliche Strategiewahl reicht das aber
nicht aus, weil reale Daten selten gleichmaessig ueber alle Zellen verteilt
sind.

Deshalb werden jetzt kontrollierte nicht-uniforme Profile eingefuehrt:

- `uniform`: Referenzfall; alle Zellen eines Teams haben gleiche Masse.
- `mixed_team_imbalance`: Teams haben stark unterschiedliche Query-Massen.
  Dadurch entstehen Situationen mit einem sehr grossen und einem deutlich
  kleineren Team-Ergebnis.
- `cell_skew`: Die Query-Masse ist insgesamt nicht extrem unbalanciert, aber
  einzelne Zellen/Posting Lists innerhalb der Query sind deutlich voller als
  andere. Das testet, ob Gruppierung wirklich cardinality-balanciert genug ist.

Diese Profile sollen nicht "Realitaet simulieren" im Sinne eines echten
physikalischen Datensatzes. Sie sind kontrollierte Stoerungen, mit denen
sichtbar werden soll, welche Laufzeitfaktoren auf welche Strategie wirken.
Echte LHCb-/Hadronenbeschleuniger-Daten waeren danach ein sehr guter
Realitaetscheck.

Wichtig fuer die Interpretation:

- Zu kleine erwartete Query-Treffer machen Laufzeiten varianzlastig.
- Bei hoeheren Dimensionen muss `N` deshalb groesser sein als in kleinen
  Uniform-Piloten.
- Der Generator gibt eine Warnung aus, wenn die erwarteten Treffer im kleinsten
  Team unter einer konfigurierbaren Schwelle liegen.

Erster Smoke-Test:

- Profil: `2T-3D`, `mixed_team_imbalance`, `N=200000`
- Query-Massen: grosses Team ca. `0.589`, kleines Team ca. `0.044`
- Ergebnis:
  - `T_rel=0.10` und `T_rel=0.60`: `union_first_parallel` gewinnt im Einzellauf
  - `T_rel=0.85`: `dynamic_selective_expansion` gewinnt im Einzellauf knapp

Das ist noch kein belastbares Ergebnis, aber es zeigt, dass nicht-uniforme
Daten tatsaechlich neue Gewinnerkonstellationen erzeugen koennen. Damit wird
der naechste Schritt klarer: Wir sollten nicht nur "mehr Dimensionen" testen,
sondern gezielt `distribution_profile` als eigenen Situationsfaktor behandeln.
