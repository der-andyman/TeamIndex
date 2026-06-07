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
- `worker_count`

### Strategieraum

- `baseline_union_first`
- `baseline_minimal_intersection`
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
- `3D` bis `5D` profitieren deutlich von gebremster Expansion. Der wichtige
  Punkt ist nicht maximale Parallelitaet, sondern eine kleine, kontrollierte
  ISE Count.
- `current_handcrafted` gewinnt haeufig, weil es zwei Dinge kombiniert:
  ein selektives Team expandieren und grosse Team-Ergebnisse vorher auf wenige
  Gruppen reduzieren.
- `dynamic_selective_expansion` ist in den bisherigen team_bench-Faellen oft zu
  konservativ. Es bleibt Union-First-aehnlich und verpasst dadurch das
  guenstige Pruning durch eine gebremste Expansion.
- `baseline_minimal_intersection` zeigt, warum ungebremste Expansion keine
  robuste Strategie ist: in hoeheren Dimensionen werden die ISE Counts sehr
  gross und die Laufzeiten steigen stark.

Daraus folgt als erste Trennung fuer eine spaetere Strategiewahl:

- Wenn die Query nur einen kleinen 2D-Arbeitsbereich erzeugt, sollte ein
  No-Expansion-Zweig ernsthaft in Betracht gezogen werden.
- Wenn die Query 3D bis 5D Teams mit vielen getroffenen Zellen beruehrt, ist
  eine gebremste Ein-Team-Expansion ein starker Kandidat.
- Wenn die geschaetzte ISE Count vor der Ausfuehrung zu gross ist, sollte der
  Plan verworfen oder staerker gruppiert werden.

Als erster neuer Strategiekandidat wurde deshalb
`bounded_selective_expansion` eingefuehrt. Die Strategie bleibt in 2D bewusst
Union-First-aehnlich, gruppiert ab 3D grosse Team-Ergebnisse und expandiert nur
ein selektives Team. Erste Pilotlaeufe zeigen, dass sie die Luecke zwischen
`dynamic_selective_expansion` und `current_handcrafted` tatsaechlich trifft:
sie gewinnt mehrere 3D/4D/5D-Szenarien und bleibt in den uebrigen Faellen meist
nah an `current_handcrafted`.

## Sinnvolle naechste Experimentbloecke

1. Einfluss der Teamdimensionalitaet (`3D` vs `4D` vs `5D`)
2. Einfluss der Teamanzahl (`2` vs `3` Teams)
3. Einfluss von `T_rel`
4. Einfluss von `worker_count`
5. gezielte Analyse von `current_handcrafted`

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
