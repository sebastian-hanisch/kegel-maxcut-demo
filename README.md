# SDP-Relaxation: MaxCut und Rundung – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kegel-maxcut-demo.streamlit.app/)** -->

Stück 12 von 16 der **Konzepte-Reihe „Konvexe und konische Optimierung“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Wie jede Konzepte-Demo dreht sich alles um **ein** Verfahren mit wachsendem Beispiel: ein Löser für Kegelprogramme, dessen Kegel von Stück zu Stück wächst. Mit diesem Stück beginnt der Strang **Relaxation**: der PSD-Kegel, mit dem Stück 2 bis 7 Kugeln rechneten und Stück 11 Beweise, löst jetzt eine **NP-schwere** Aufgabe näherungsweise.

## Kernfrage

**Wie nah liegen die SDP-Schranke, die Rundung und der beste Schnitt, und wann ist die Relaxation exakt?** MaxCut verlangt Vorzeichen `sᵢ ∈ {±1}` mit größtem `Σ wᵢⱼ (1 − sᵢ sⱼ)/2`. Ersetzt man `sᵢ sⱼ` durch `Xᵢⱼ` einer Matrix `X ⪰ 0` mit `Xᵢᵢ = 1` (die Gram-Matrix von Einheitsvektoren), entsteht ein **SDP** im Format des Kerns (PSD-Kegel; Variablen: die Außerdiagonale); sein Wert ist eine **obere Schranke** für jeden Schnitt. Zerlegt man `X = V Vᵀ`, wirft eine Zufallshyperebene `r` und setzt `sᵢ = sign(vᵢ · r)`, wird die Kante `{i, j}` mit Wahrscheinlichkeit `arccos(Xᵢⱼ)/π` geschnitten: der **erwartete Schnitt ist exakt `Σ wᵢⱼ arccos(Xᵢⱼ)/π`** und mindestens **0,87856** mal die Schranke (Goemans, Williamson 1995). Gemessen mit dem primal-dualen Löser aus Stück 6: auf Zufallsgraphen liegt die erwartete Rundung bei **0,91 bis 0,99** der Schranke, der exakte Schnitt bei 0,91 bis 0,995, und der beste von 10 bis 100 Hyperebenen trifft den exakten Schnitt meist; die Garantie ist nur bei **Petersen-Graph (0,8787)** und **Fünfeck (exakt ÷ Schranke 0,8845)** nahezu scharf. Bei **bipartiten Graphen, Bäumen und Gittern** ist die Relaxation **exakt** (Rang 1). Das **Dual** ist ein Zertifikat: `Z = Diag(d) + W/2 ⪰ 0` liefert die Schranke `(Σ w + tr Z)/2` ohne Kenntnis des Schnitts.

Das Beispiel ist hier der **Graph**: ein Zufallsgraph `G(n, p)` (Gewichte 1 oder zufällig), dazu feste Familien.

## Modell

- **Kern** (`kegel_kern.py`, wortgleich mit Stück 2 bis 11, unverändert): `ConeProblem` (PSD-Kegel), `svec`/`smat`.
- **Löser** (`kmc_solver.py`, `kmc_cones.py`, aus Stück 6 übernommen): primal-dualer innerer Punkt (Mehrotra, Nesterov–Todd) mit Dualvariablen, ohne strikt zulässigen Start.
- **Graphen** (`kmc_graphs.py`): Gewichtsmatrix `W`, Zufallsgraph `G(n, p)`, Kreis, vollständiger, vollständig bipartiter Graph, Baum, Gitter, Rad, Petersen-Graph; Schnittwert `¼(Σ W − sᵀ W s)`; eine einfache Aufzählung als Gegenprobe.
- **SDP** (`kmc_sdp.py`): das Kegelprogramm (`min Σ w_ij X_ij` mit `s = svec(X) = b − A x`, `x` = Außerdiagonale, `A` mit `−√2` je Variable, `b = svec(I)`), die Lösung `X`, die Dualmatrix `Z`, die Schranke `(Σ w + tr Z)/2`, der Rang; `vectors` zerlegt `X = V Vᵀ`.
- **Rundung** (`kmc_rounding.py`): der exakte Erwartungswert `Σ w arccos(X)/π`, `k` Zufallshyperebenen (vektorisiert), die Kurve „bester von k“, der exakte Schnitt durch Aufzählen aller `2^(n−1)` Schnitte (n ≤ 20), die Konstante `α_GW = min_θ (2/π) θ/(1 − cos θ)`.
- **Kein Eingriff in den Kern:** alles Neue steht in `kmc_*.py`; BLAS-Threads laufen mit einem Thread (`threadpoolctl`).

## Methodik

Fünf Experimente mit live berechnetem Verdikt: (1) der Kreis `Cₙ` (Voreinstellung das **Fünfeck**): die Einheitsvektoren in der Ebene, eine Zufallshyperebene, die Randwahrscheinlichkeit `θ/π`, Schranke gegen Schnitt; (2) **Zufallsgraphen**: der Graph der Seitenleiste mit dem besten gerundeten Schnitt, darunter die Verteilung der Verhältnisse über 30 Graphen gegen die Garantie; (3) **die Rundung**: Histogramm der Schnittwerte von 2 000 Hyperebenen, Mittelwert gegen die Formel, „bester von `k`“ gegen `k`; (4) **Familien**: 14 Graphen mit Schranke, exaktem Schnitt, erwarteter Rundung und Rang; (5) **das Dual und der Rang**: die Dualmatrix `Z` als Zertifikat und der Rang der Lösung gegen `n`. Regler: Knoten `n` (4 bis 30; exakter Schnitt nur bis 20), Kantenwahrscheinlichkeit `p` (0,1 bis 0,8), Gewichte, Seed, Zahl der Hyperebenen `k`, Kreislänge; ein Schnellstart lädt sechs Szenarien, die Adresszeile spiegelt die Konfiguration.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (wo dort nur ein Band geprüft wird, wegen Rundung und Rechner, steht die gemessene Zahl trotzdem hier). Zeiten, der Status des Lösers und der **Rang der Mittelpunktslösung** werden nur in weiten Bändern getestet. Standardgraph: `G(12, ½)`, Seed 3.

| Frage | Befund |
|---|---|
| Der Standardgraph (27 Kanten) | SDP-Schranke **20,8082**, exakter größter Schnitt **20** (0,9612 der Schranke), erwartete Rundung **19,2717** (0,9262), der beste von 50 Hyperebenen ist der exakte Schnitt; Rang der Lösung 3; die Schranke aus dem Dual ist gleich dem SDP-Wert (Unterschied 2·10⁻⁹) |
| Das Fünfeck | Fünf Einheitsvektoren im Winkel 144° (`X_{k,k+1} = cos(4π/5)`): SDP-Schranke **4,5225** (`(5/2)(1 + (1 + √5)/4)`), exakter Schnitt **4**, Verhältnis **0,88446**; jede Kante wird mit Wahrscheinlichkeit 4/5 geschnitten, **die erwartete Rundung ist 4**, also exakt. Der Schnitt ist ganzzahlig, die Relaxation nicht |
| Kreise `Cₙ` | Ungerade `n`: Schranke `(n/2)(1 + cos(π/n))`, Schnitt `n − 1`, Rang 2; `C₃` 2,25 / 2, `C₅` 4,5225 / 4 (0,8845), `C₇` 6,6534 / 6 (0,9018), `C₉` 8,7286 / 8 (0,9165); gerade `n`: Schranke = Schnitt = `n`, Rang 1; die erwartete Rundung ist bei jedem Kreis der exakte Schnitt |
| Zufallsgraphen `G(12, p)`, je 30 Graphen: erwartete Rundung ÷ Schranke | p = 0,1 / 0,2 / 0,3 / 0,5 / 0,8: Mittel **0,993 / 0,962 / 0,941 / 0,936 / 0,952**, kleinster Wert 0,938 / 0,904 / 0,909 / 0,915 / 0,929; exakter Schnitt ÷ Schranke im Mittel 0,995 / 0,976 / 0,970 / 0,970 / 0,983. **Kein Graph unter 0,87857**; am schwächsten bei mittlerer Dichte |
| Die Rundung (Standardgraph, 2 000 Hyperebenen) | Mittelwert 19,2955 ± 0,0175 gegen die Formel 19,2717 (innerhalb von 4 Standardfehlern); der Erwartungswert des besten von `k`: 19,296 (k = 1), **20,000 (k = 10 und 100)**: wenige Hyperebenen genügen |
| Familien (Schranke / exakter Schnitt / exakt ÷ Schranke / erwartete Rundung ÷ Schranke / Rang) | Baum (n = 12): 11 / 11 / 1 / 1 / **1**; K₆,₆: 36 / 36 / 1 / 1 / **1**; Gitter 4×4: 24 / 24 / 1 / 1 / **1**; C₅: 4,5225 / 4 / **0,8845** / 0,8845 / 2; C₇: 6,6534 / 6 / 0,9018 / 0,9018 / 2; K₃: 2,25 / 2 / 0,8889 / 0,8889 / 2; K₄: 4 / 4 / 1 / 0,9123 / **3**; K₅: 6,25 / 6 / 0,96 / 0,9287 / 4; K₆: 9 / 9 / 1 / 0,9402 / **5**; Petersen: 12,5 / 12 / 0,96 / **0,8787** / 4; Rad W₉: 12,5 / 12 / 0,96 / 0,9085 / 2 |
| Wann ist die Schranke exakt? | Bei Bäumen, bipartiten Graphen, Gittern und geraden Kreisen (Rang 1) **und** bei K₄ und K₆ (7 von 14 Familien). **Die Umkehrung „exakt ⇒ Rang 1“ gilt nicht:** bei K₄ und K₆ liefert der innere Punkt Rang 3 und 5, weil er den **Mittelpunkt** der optimalen Menge liefert, nicht die Ecke mit kleinstem Rang (dieselbe Beobachtung wie beim OGM in Stück 11) |
| Garantie und Wirklichkeit | Die kleinste erwartete Rundung ÷ Schranke aller Familien hat der **Petersen-Graph mit 0,8787** (Garantie 0,87857); sonst liegen alle Familien und alle 80 Zufallsgraphen des Tests darüber. Schlechte Fälle brauchen sehr große, speziell gebaute Graphen (hier nicht gemessen) |
| Rang der Lösung auf `G(n, ½)` | n = 8 / 12 / 16 / 20 / 24: Rang **2 / 3 / 3 / 3 / 3**, `√(2n)` = 4,0 / 4,9 / 5,7 / 6,3 / 6,9: weit darunter (die Phase-0-Messung bis n = 30: 4). Bei dünnen Graphen mit isolierten Knoten höher: `G(16, 0,1)`, Seed 3, hat 9 Kanten und 6 isolierte Knoten, Rang 9 (drei für den verbundenen Teil, sechs für die isolierten Knoten, deren Vektoren der Mittelpunkt orthogonal setzt) |
| Das Dual | `Z = Diag(d) + W/2` ist positiv semidefinit (kleinster Eigenwert 5·10⁻¹¹), die Außerdiagonale ist gleich `wᵢⱼ/2` auf 4·10⁻¹³, die Schranke `(Σ w + tr Z)/2` gleich dem SDP-Wert auf 2·10⁻⁹ (Standardgraph) |
| Löser (Vormessung, nicht live) | primal-dual: immer 11 Iterationen, n = 10 / 20 / 30 / 40: 29 / 185 / 988 / 4 181 ms; die Barriere des Kerns: 27 / 3 522 / 13 368 / 41 134 ms (37 bis 49 Newton-Schritte), gleicher SDP-Wert auf 6 Stellen: ab n = 20 zehnmal langsamer |
| Gegen cvxpy | Ein **eigener** Aufbau des SDP in cvxpy (`cp.Variable((n, n), PSD=True)`, `cp.diag(X) == 1`) gibt mit Clarabel denselben SDP-Wert (10⁻⁶) für C₅, Petersen, K₄, `G(10, ½)`, `G(16, ½)` und einen gewichteten Graphen, mit SCS für C₅ (10⁻⁵); die Schranke aus unserer Dualmatrix ist gleich dem cvxpy-Optimum; der exakte Schnitt gleich einer einfachen Aufzählung |

## Befunde und Korrekturen gegenüber dem Plan

- **Die Verteilung rechnet mit `n = 12` und fortlaufenden Seeds**, nicht mit den 40 Graphen und `n = 16` der Vormessung; die Mittelwerte weichen deshalb leicht ab (0,936 gegen 0,938 bei `p = ½`). Die Tabelle nach `p` ist neu gemessen (die Vormessung hatte `n = 16`).
- **Ein Textfehler eines Verdikts fand sich im Test:** bei Graphen unter der Garantie hätte der Text „unterschreitet kein Graph (1 von 1)“ gelautet; jetzt „unterschreiten 1 von 1 Graphen“.
- **Der Rang hängt an isolierten Knoten** (nicht im Plan): der Mittelpunkt setzt die Vektoren isolierter Knoten orthogonal, der Rang wächst um die Zahl der isolierten Knoten; die Rang-Reihe nimmt deshalb `p = ½`.
- **Ein Fehler der Abbildung fand erst der Blick in den Browser:** die Einheitsvektoren des Fünfecks lagen im Diagramm bei null, weil `numpy.linalg.eigh` die Eigenwerte aufsteigend sortiert und die Ebene aus den zwei **kleinsten** Spalten gebaut wurde; jetzt die zwei größten, und der Test `test_odd_cycles_have_a_gap_and_even_cycles_none` prüft, dass die Punkte der Ebene die ganze Lösung tragen (`P Pᵀ = X`).
- **Der Port ist 9336** (Plan: 9310 gehört `aufzaehlen-rekursion-demo`; die Ports 9300 bis 9335 sind alle vergeben).
- **Die Rangschranke von Barvinok und Pataki steht nur als Verweis.** Die gespeicherte PDF der Arbeit ist ein Scan ohne Text; die Aussage `r(r+1)/2 ≤ n` ist nicht am Original geprüft und wird nicht gemessen.
- **Quellen:** geprüft am 2026-11-03: Goemans und Williamson, *Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming*, Journal of the ACM 42 (6), 1995, S. 1115–1145 (der Abstract nennt die Garantie 0,87856 und die Zufallshyperebene; geprüft über das MIT-Exemplar der Arbeit). Die Formel der Randwahrscheinlichkeit `arccos(X)/π` und die Konstante `α_GW` sind hier **nachgerechnet** (Monte Carlo innerhalb von 4 Standardfehlern; Minimum der Verhältnisse auf einem feinen Gitter: 0,8785672), nicht am Text geprüft. Pataki, Mathematics of Operations Research 23 (2), 1998, S. 339–358: Bibliografie geprüft, Inhalt nicht. Beweise der Arbeiten nicht nachgelesen; einen Beleg für die NP-Härte von MaxCut zitiert die Demo nicht.
- **Abgrenzung:** `lokale-optima-demo` (Komplexitätsklassen-Linie) behandelt MaxCut mit **Flip-Nachbarschaft** (lokale Suche, PLS); hier die SDP-Relaxation mit Garantie. Die Schnitte-Linie der Website behandelt Minimierung (globaler Schnitt) und spektrale Heuristiken; hier die Maximierung mit PSD-Relaxation. Keine Zahlen werden übernommen.

## Ehrliche Grenzen

- **Die Garantie 0,87856 ist ein Worst-Case-Wert.** Die Demo zeigt, dass sie auf Zufallsgraphen weit unterschritten wird; schlechte Fälle (bekannte Gegenbeispiele brauchen sehr große, speziell gebaute Graphen) sind hier **nicht** gemessen.
- **Nur ungerichtete Graphen mit nichtnegativen Gewichten;** kein MAX-2-SAT, kein MAX-DICUT, keine negativen Gewichte.
- **Nur die Zufallshyperebene** (keine Ableitung, keine lokale Verbesserung nach der Rundung, keine Verbesserung durch Rundungsvarianten).
- **Der Rang der Lösung ist eine Eigenschaft des Lösers,** nicht des Graphen (Mittelpunkt der optimalen Menge); die Aussage „Rang 1 ⇒ exakt“ gilt, die Umkehrung nicht.
- **Der exakte Schnitt nur bis `n = 20`** (Aufzählung aller `2^(n−1)` Schnitte, rund 0,6 s und 84 MB bei 20 Knoten); darüber zeigt die App Schranke und Rundung ohne Vergleichswert.
- **Der Löser skaliert nicht beliebig:** das dichte KKT-System hat `n(n−1)/2` Variablen (1,3 s bei `n = 30`, 4 s bei 40 laut Vormessung); große Graphen brauchen Niedrigrang-Verfahren (Stück 13).
- **Zeiten und der Status des Lösers hängen vom Rechner ab** und werden nicht festgeschrieben (Lehre aus Stück 9 bis 11).

## Verifikation

- **Graphen** (`tests/test_graphs.py`): Größen und Grade der Familien von Hand, Schnittwerte an Handbeispielen (Fünfeck mit `s = (+, −, +, −, +)` gibt 4), einfache Aufzählung, Zufallsgraphen (deterministisch, Dichte, Gewichte), Kreislayout, abgelehnte Eingaben.
- **SDP** (`tests/test_sdp.py`): das **Fünfeck von Hand** (fünf Einheitsvektoren im Winkel 144°: zulässig, Wert 4,5225), K₃ (`X = (3I − J)/2`, Wert 9/4), K₄ (`X = (4I − J)/3`, Wert 4), bipartite Graphen (Rang 1), Zulässigkeit und Schranke über dem exakten Schnitt an Zufallsgraphen, das Dual als Zertifikat (PSD, Außerdiagonale `wᵢⱼ/2`, Schranke gleich Primalwert), die Barriere des Kerns gleicher Wert, Randfälle (keine Kanten, eine Kante, Skala der Gewichte, abgelehnte Eingaben).
- **Rundung** (`tests/test_rounding.py`): der Erwartungswert von Hand (Fünfeck 4, K₃ 2, K₄ 3,649, bipartit: Zahl der Kanten), **die Garantie an 80 Zufallsgraphen und allen Familien**, `α_GW` als Minimum, Monte Carlo gegen die Formel, Schnittwerte gegen die einzelnen Vorzeichenvektoren, „bester von `k`“ monoton, die vektorisierte Aufzählung gleich der einfachen.
- **Orakel** (`tests/test_oracle_kegel_maxcut.py`): ein **eigener** Aufbau des SDP in cvxpy gegen unseren (Clarabel, SCS), starke Dualität, die Aufzählung gegen die einfache.
- **Löser** (`tests/test_solver.py`): die Hand-Programme des Kerns (Orthant, Lorentz, PSD) mit Primal- und Dualwert.
- **Randfälle** (`tests/test_edge_cases.py`): zwei und drei Knoten, Graph ohne Kanten, nicht zusammenhängende Graphen (Summe der Teile), Gewicht 0 auf einer Kante, Skala der Gewichte, größter Graph mit und ohne exakten Schnitt, alle Reglerecken, Reproduzierbarkeit, **Serialisierbarkeit der zwischengespeicherten Ergebnisse (pickle)**, jede Kreislänge.
- **Verdikte** (`tests/test_verdicts.py`, Werte von Hand gesetzt), **Claims** (`tests/test_claims.py`: jede Zahl dieser README und der Hilfetexte), **Oberfläche** (AppTest: jeder Button, jeder Regler an Minimum und Maximum, jede Dichte, jede Kreislänge, Gewichte, `k`, PDF), Presets, Permalink, Layout, Abbildungen, Footer, Laufzeit-Importe, Kern (Prüfsumme gegen das Schwester-Repo).
- **Umfang:** 147 Tests (8 davon, die cvxpy-Orakel gegen Clarabel und SCS, überspringen sich ohne cvxpy; mit cvxpy laufen alle 9 Orakel-Tests grün), rund 2 Minuten (2026-11-03, Windows); kein Mutationstest (seit 2026-10-28 nicht mehr Teil des Ablaufs). cvxpy ist Gegenprobe im dev-Satz; ohne es überspringen sich die Orakel-Tests. Linux-CI: noch nicht gelaufen (Repo noch nicht veröffentlicht).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `kegel_kern.py` | Gemeinsamer Kern der Reihe, wortgleich mit Stück 2 bis 11 |
| `kmc_graphs.py` | Graphen, Familien, Schnittwert, einfache Aufzählung |
| `kmc_sdp.py` | Die SDP-Relaxation als Kegelprogramm, Lösung, Dual, Zerlegung `X = V Vᵀ` |
| `kmc_rounding.py` | Erwartungswert der Rundung, Zufallshyperebenen, exakter Schnitt, `α_GW` |
| `kmc_solver.py`, `kmc_cones.py` | Primal-dualer innerer Punkt (Mehrotra, Nesterov–Todd) mit Dualvariablen, aus Stück 6 |
| `kmc_constants.py` | Regler-Grenzen, feste Reihen, Farben, Presets |
| `kmc_evaluation.py` | Experimente, Familien, Verdikte, Formatierung |
| `kmc_layout.py` | Fester Kopf (Stylesheet) |
| `kmc_pdf_export.py` | PDF-Export (fpdf2) |
| `kmc_presets.py` | Permalink, Presets (Schnellstart, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kmc_visualization.py` | Plotly-Abbildungen (Graph auf dem Kreis, Einheitsvektoren mit Hyperebene, Histogramme; 2D-Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und cvxpy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Graphen, SDP, Rundung, Löser, Orakel, Randfälle, Verdikte, Claims, Presets, Layout, Abbildungen, AppTest, Footer, Laufzeit-Importe, Kern |

## Bewusst nicht umgesetzt

- **Rundung mit lokaler Verbesserung** (Flip-Suche nach der Hyperebene): das ist die Verbindung zur `lokale-optima-demo`; hier nicht gebaut.
- **Harte Instanzen** (die bekannten Gegenbeispiele für die Garantie brauchen sehr große Graphen) und **Varianten** (MAX-2-SAT, MAX-DICUT, negative Gewichte): nicht gebaut.
- **Niedrigrang-Löser** (Burer–Monteiro) und speichersparende SDP-Verfahren: Stück 13.
- **Der Beweis der Garantie und der Rangschranke:** nicht nachgelesen und nicht geführt; die Demo zeigt die Formel und misst.
- **Mutationstest:** nicht Teil des Ablaufs.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, fpdf2 und threadpoolctl.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
