"""SDP-Relaxation: MaxCut und Rundung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 12 der Reihe „Konvexe und konische Optimierung“: eine NP-schwere Aufgabe als Kegelprogramm mit einem PSD-Kegel. Die SDP-Relaxation von MaxCut (Goemans, Williamson) liefert eine obere Schranke; eine Zufallshyperebene rundet
die Lösung zu einem Schnitt, dessen Erwartungswert mindestens 0,87856 mal die Schranke ist. Die Demo zeigt, wie nah Schranke, Schnitt und Rundung auf Zufallsgraphen liegen, wann die Relaxation exakt ist und was das Dual beweist.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import kmc_constants as C
import kmc_rounding as R
from kmc_evaluation import (
    Settings,
    cycle_view,
    de,
    family_rows,
    graph_name,
    main_view,
    num,
    p_text,
    rank_rows,
    rounding_view,
    sci,
    stat_rows,
    stat_summary,
    verdict_cycle,
    verdict_dual,
    verdict_families,
    verdict_main,
    verdict_rounding,
    verdict_stat,
)
from kmc_layout import dunkles_design, kopf_css
from kmc_pdf_export import build_pdf
from kmc_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from kmc_visualization import build_curve, build_families, build_graph, build_hist, build_ratios, build_rank, build_vectors

st.set_page_config(page_title="SDP-Relaxation: MaxCut – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _cycle(n_cycle, seed):
    return cycle_view(n_cycle, seed)


@st.cache_data(show_spinner=False)
def _main(n, p, seed, weights, k):
    return main_view(Settings(n=n, p=p, seed=seed, weights=weights, k=k))


@st.cache_data(show_spinner=False)
def _stat(n, p, seed, weights):
    return stat_rows(Settings(n=n, p=p, seed=seed, weights=weights))


@st.cache_data(show_spinner=False)
def _rounding(n, p, seed, weights, k):
    return rounding_view(_main(n, p, seed, weights, k), seed)


@st.cache_data(show_spinner=False)
def _families():
    return family_rows()


@st.cache_data(show_spinner=False)
def _ranks(seed):
    return rank_rows(seed)


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kmc_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("✂️ SDP-Relaxation: MaxCut und Rundung")
st.markdown(
    """
**Eine NP-schwere Aufgabe wird zu einem Kegelprogramm mit einem PSD-Kegel, und eine Zufallshyperebene macht aus der Lösung einen Schnitt.** MaxCut verlangt Vorzeichen `sᵢ ∈ {±1}` mit größtem `Σ wᵢⱼ (1 − sᵢ sⱼ)/2`. Ersetzt man `sᵢ sⱼ` durch `Xᵢⱼ`
einer Matrix `X ⪰ 0` mit `Xᵢᵢ = 1` (die Gram-Matrix von Einheitsvektoren), erhält man ein **SDP** mit einer **oberen Schranke** für den Schnitt. Zerlegt man `X = V Vᵀ`, wirft eine Zufallshyperebene und setzt `sᵢ = sign(vᵢ · r)`, ist der
**erwartete Schnitt exakt `Σ wᵢⱼ arccos(Xᵢⱼ)/π`** und mindestens **0,87856** mal die Schranke (Goemans, Williamson 1995). Auf Zufallsgraphen liegt er viel näher an der Schranke; nahezu scharf ist die Garantie nur beim Petersen-Graph,
Fünfeck und K₃ liegen ebenfalls nahe dran. Das **Dual** ist ein Zertifikat: es beweist die Schranke, ohne den Schnitt zu kennen.
Stück 12 der Reihe „Konvexe und konische Optimierung“; der Strang Relaxation beginnt.

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Die Aufgabe:** ein Graph mit Kantengewichten `wᵢⱼ`; ein Schnitt teilt die Knoten in zwei Seiten, sein Wert ist die Summe der Gewichte der Kanten zwischen den Seiten. Den größten Schnitt zu finden ist NP-schwer.
2. **Die Relaxation:** statt Vorzeichen `sᵢ` Einheitsvektoren `vᵢ` (`‖vᵢ‖ = 1`), statt `sᵢ sⱼ` das Skalarprodukt `Xᵢⱼ = ⟨vᵢ, vⱼ⟩`. Die Matrix `X` ist positiv semidefinit mit Diagonale 1; `max Σ wᵢⱼ (1 − Xᵢⱼ)/2` ist ein **Kegelprogramm** im Format des Kerns (PSD-Kegel, Variablen: die Außerdiagonale).
3. **Der Löser:** der primal-duale innere Punkt aus Stück 6 (10 bis 15 Iterationen bei 10 bis 40 Knoten, wachsend mit n); die Barriere des Kerns ist mit einem BLAS-Thread bei 20 Knoten etwa 1,5-mal, bei 30 etwa 2-mal und bei 40 rund 7-mal langsamer (Größenordnung, rechnerabhängig).
4. **Die Rundung:** `X = V Vᵀ`, ein Zufallsvektor `r`, `sᵢ = sign(vᵢ · r)`. Eine Kante wird mit Wahrscheinlichkeit `θᵢⱼ/π` geschnitten (`θᵢⱼ = arccos Xᵢⱼ`): der Erwartungswert ist eine Formel, kein Experiment.
5. **Das Dual:** `Z = Diag(d) + W/2 ⪰ 0` liefert die Schranke `(Σ w + tr Z)/2`: ein bewiesener oberer Wert für jeden Schnitt.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Der Zufallsgraph (Ergebnis, Experiment 2 bis 5)**")
    n = st.slider("Knoten n", *bounds("n_slider"), key="n_slider", help=f"Bis n = {C.EXACT_MAX} gibt es den exakten größten Schnitt (Aufzählung aller 2^(n−1) Schnitte); darüber nur Schranke und Rundung.")
    p = st.select_slider("Kantenwahrscheinlichkeit p", options=C.P_CHOICES, key="p_select", format_func=p_text, help="Jede der n(n−1)/2 möglichen Kanten ist mit dieser Wahrscheinlichkeit vorhanden.")
    weights = st.selectbox("Gewichte", C.WEIGHTS, key="weights_select", format_func=C.WEIGHT_NAMES.get, help="Alle Kanten mit Gewicht 1 oder gleichverteilte Gewichte in (0, 1].")
    seed = st.number_input("Zufalls-Seed des Graphen", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuer Graph", width="stretch", on_click=randomize_seed)
    k = st.select_slider("Hyperebenen k", options=C.K_CHOICES, key="k_select", help="Wie viele Zufallshyperebenen die Rundung probiert; der beste Schnitt zählt.")
    st.markdown("**Experiment 1**")
    cycle = st.select_slider("Kreislänge", options=C.CYCLE_CHOICES, key="cycle_select", help="Der Kreis Cₙ: ungerade Längen zeigen die Lücke der Relaxation, gerade nicht.")

sync_query_params({"n_slider": int(n), "p_select": float(p), "seed_input": int(seed), "weights_select": weights, "k_select": int(k), "cycle_select": int(cycle)})

settings = Settings(int(n), float(p), int(seed), weights, int(k), int(cycle))
main = _main(settings.n, settings.p, settings.seed, settings.weights, settings.k)
res = main["res"]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3 = st.columns(3)
m4, m5 = st.columns(2)
m1.metric("SDP-Schranke", num(res.value), help="Obere Schranke für jeden Schnitt: der Wert der Relaxation.")
m2.metric("Exakter größter Schnitt", num(main["exact"]) if main["exact"] is not None else "–", help=f"Aufzählung aller Schnitte, nur bis n = {C.EXACT_MAX}.")
m3.metric("Erwartete Rundung", num(main["expected"]), help="Σ wᵢⱼ arccos(Xᵢⱼ)/π: der erwartete Schnitt einer Zufallshyperebene (exakt, kein Experiment).")
m4.metric(f"Bester von {settings.k}", num(main["best"]), help="Der beste Schnitt der gewürfelten Hyperebenen.")
m5.metric("Erwartung / Schranke", de(main["expected"] / res.value, 4), help=f"Die Garantie von Goemans und Williamson ist {de(R.ALPHA_GW, 5)}.")
st.plotly_chart(build_graph(main["W"], main["best_s"], f"{graph_name(settings)}: bester gefundener Schnitt ({num(main['best'])} von {num(res.value)})"), width="stretch", key="main_chart")
st.caption("Der Graph auf einem Kreis; rote Kanten liegen im Schnitt, die Farbe der Knoten zeigt die Seite.")

if st.button("📄 PDF des aktuellen Ergebnisses erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, main))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="maxcut-sdp.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie nah liegen Schranke, Rundung und bester Schnitt?")
verdict(verdict_main(settings, main))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Das Fünfeck (oder ein anderer Kreis)")
st.caption("Die Einheitsvektoren der SDP-Lösung für den Kreis Cₙ (in der Ebene; ungerade Kreise haben Rang 2) und eine Zufallshyperebene: sie trennt die Vektoren, die Kante `{i, i+1}` wird mit Wahrscheinlichkeit `θ/π` geschnitten.")
cyc = _cycle(settings.cycle, settings.seed)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_vectors(cyc), width="stretch", key="vectors_chart")
with c2:
    st.dataframe(pd.DataFrame([{"Kennzahl": "SDP-Schranke", "Wert": num(cyc["res"].value)}, {"Kennzahl": "exakter größter Schnitt", "Wert": num(cyc["exact"])}, {"Kennzahl": "exakt / Schranke", "Wert": de(cyc["ratio"], 5)},
                                {"Kennzahl": "Winkel benachbarter Vektoren", "Wert": f"{de(cyc['angles'][0], 1)}°"}, {"Kennzahl": "Kante im Schnitt mit Wahrscheinlichkeit", "Wert": de(cyc["edge_probability"], 3)},
                                {"Kennzahl": "erwartete Rundung", "Wert": num(cyc["expected"])}, {"Kennzahl": "Rang von X", "Wert": str(cyc["res"].rank)}]), hide_index=True, width="stretch")
verdict(verdict_cycle(cyc))

st.markdown("---")

st.subheader("🔬 Experiment 2: Zufallsgraphen")
stat = _stat(settings.n, settings.p, settings.seed, settings.weights)
st.caption(f"{len(stat)} Zufallsgraphen G({min(settings.n, C.STAT_N_MAX)}, {p_text(settings.p)}) (Gewichte der Seitenleiste): wie nah die erwartete Rundung und der exakte Schnitt an der SDP-Schranke liegen; die rote Linie ist die Garantie.")
st.plotly_chart(build_ratios(stat_summary(stat)), width="stretch", key="ratios_chart")
verdict(verdict_stat(stat, settings))

st.markdown("---")

st.subheader("🔬 Experiment 3: Die Rundung")
st.caption(f"Die Schnittwerte von {C.HYPER_SAMPLES} Zufallshyperebenen auf dem Graphen der Seitenleiste, ihr Mittelwert gegen die Formel, und der beste von `k` gegen `k`.")
rnd = _rounding(settings.n, settings.p, settings.seed, settings.weights, settings.k)
c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(build_hist(rnd, main), width="stretch", key="hist_chart")
with c4:
    st.plotly_chart(build_curve(rnd, main), width="stretch", key="curve_chart")
verdict(verdict_rounding(rnd, settings, main))

st.markdown("---")

st.subheader("🔬 Experiment 4: Wann ist die Relaxation exakt?")
st.caption("Vierzehn Graphen: Bäume, bipartite Graphen und Gitter (Rang 1, exakt), gerade und ungerade Kreise, vollständige Graphen, der Petersen-Graph, ein Rad und der Zufallsgraph der Voreinstellung.")
fam = _families()
st.plotly_chart(build_families(fam), width="stretch", key="families_chart")
st.dataframe(pd.DataFrame([{"Graph": r["name"], "Knoten": r["n"], "Kanten": r["edges"], "SDP-Schranke": num(r["sdp"]), "exakter Schnitt": num(r["exact"]), "exakt / Schranke": de(r["exact_ratio"], 4),
                            "erwartete Rundung / Schranke": de(r["expected_ratio"], 4), "Rang der Lösung": r["rank"], "Schranke exakt": "ja" if r["exact_sdp"] else "nein"} for r in fam]), hide_index=True, width="stretch")
verdict(verdict_families(fam))

st.markdown("---")

st.subheader("🔬 Experiment 5: Das Dual und der Rang")
st.caption("Die Dualmatrix `Z` des Graphen der Seitenleiste (Zertifikat der Schranke) und der Rang der Lösung des inneren Punkts für Zufallsgraphen G(n, ½) gegen `n`.")
ranks = _ranks(settings.seed)
c5, c6 = st.columns(2)
with c5:
    st.dataframe(pd.DataFrame([{"Kennzahl": "SDP-Wert (primal)", "Wert": num(res.value)}, {"Kennzahl": "Schranke aus dem Dual (Σ w + tr Z)/2", "Wert": num(res.dual_bound)}, {"Kennzahl": "Unterschied", "Wert": sci(abs(res.gap))},
                                {"Kennzahl": "kleinster Eigenwert von Z", "Wert": sci(res.z_min_eig)}, {"Kennzahl": "größte Abweichung der Außerdiagonale von wᵢⱼ/2", "Wert": sci(res.z_offdiag)},
                                {"Kennzahl": "exakter Schnitt", "Wert": num(main["exact"]) if main["exact"] is not None else "–"}]), hide_index=True, width="stretch")
with c6:
    st.plotly_chart(build_rank(ranks), width="stretch", key="rank_chart")
verdict(verdict_dual(main, settings, ranks))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Garantie 0,87856 beschreibt die Rundung** | Sie ist eine Worst-Case-Garantie. Auf Zufallsgraphen liegt die erwartete Rundung bei etwa 0,90 bis 0,99 der Schranke (Experiment 2); nahe an der Garantie liegen nur wenige Graphen (Petersen-Graph, Fünfeck, K₃). Dass sich die Garantie im Grenzwert nicht verbessern lässt, braucht nach der Literatur sehr große, speziell gebaute Graphen (hier nicht gemessen). | – |
| **Die Relaxation ist exakt, wenn die Lösung Rang 1 hat** | Das gilt in dieser Richtung (bipartite Graphen, Bäume, Gitter). Die Umkehrung nicht: bei K₄ und K₆ ist die Schranke exakt, die Lösung des Lösers hat aber Rang 3 und 5, weil der innere Punkt den Mittelpunkt der optimalen Menge liefert, nicht die Ecke (Experiment 4). | Stück 13 |
| **Die Lösung hat den Rang, den die Aufgabe braucht** | Der Rang der Lösung ist eine Eigenschaft des Lösers, nicht des Graphen; gemessen sind 2 bis 3 bei Graphen G(n, ½) mit n bis 24 (bei dem dünnen Graphen des Schnellstarts ist er 9). Ein Verfahren, das den Rang vorgibt (Burer–Monteiro), liefert andere Lösungen. | Stück 13 |
| **Die Schranke ist der Optimalwert** | Sie ist nur eine obere Schranke; der Unterschied zum Schnitt ist die Integralitätslücke (am größten unter den 14 Familien beim Fünfeck, 0,8845). Das Dual beweist die Schranke, nicht die Optimalität des Schnitts. | – |
| **Eine Hyperebene genügt** | Eine Hyperebene erreicht im Mittel die erwartete Rundung; der beste von 10 bis 100 liegt meist beim exakten Schnitt (Experiment 3). Das ist eine Messung kleiner Graphen, keine Garantie. | – |
| **Der Löser skaliert beliebig** | Das dichte KKT-System wächst mit `n²` Variablen: rund 1 s bei n = 30, einige Sekunden bei n = 40 (Größenordnung, nicht live). Für große Graphen braucht es Niedrigrang-Verfahren (Stück 13). Zeiten hängen vom Rechner ab und werden nicht getestet. | Stück 13 |
| **Der exakte Schnitt ist immer verfügbar** | Nur bis n = 20 (Aufzählung aller 2^(n−1) Schnitte); darüber zeigt die App Schranke und Rundung ohne Vergleichswert. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Die Aufgabe.** Für einen Graphen mit Gewichten $w_{ij} \ge 0$: $\max_{s \in \{\pm 1\}^n} \sum_{i<j} w_{ij}\,\tfrac{1 - s_i s_j}{2}$ (NP-schwer).

**Die Relaxation.** Ersetze $s_i s_j$ durch $X_{ij}$ mit $X \succeq 0$, $X_{ii} = 1$ (Gram-Matrix von Einheitsvektoren $v_i$): $\max \sum_{i<j} w_{ij}\tfrac{1 - X_{ij}}{2}$. Im Format des Kerns ($\min c^\top x$, $s = b - Ax \in \mathbb{S}^n_+$):
Variablen $x_{ij} = X_{ij}$ ($i < j$), $c_{ij} = w_{ij}$, $s = \mathrm{svec}(X)$ mit $b = \mathrm{svec}(I)$ und $A_{\mathrm{pos}(ij),\,ij} = -\sqrt2$. Der SDP-Wert ist $\tfrac12\big(\sum_{i<j} w_{ij} - \min c^\top x\big)$, eine obere Schranke des Schnitts, denn jeder Schnitt $s s^\top$ ist zulässig.

**Die Rundung.** Zerlege $X = V V^\top$, ziehe $r \sim \mathcal{N}(0, I)$, setze $s_i = \mathrm{sign}(v_i^\top r)$. Die Kante $\{i, j\}$ wird mit Wahrscheinlichkeit $\theta_{ij}/\pi$ getrennt, $\theta_{ij} = \arccos X_{ij}$; also
$$\mathbb{E}[\text{Schnitt}] = \sum_{i<j} w_{ij}\,\frac{\arccos X_{ij}}{\pi}.$$
Mit $\alpha_{GW} = \min_{0 < \theta \le \pi} \frac{2}{\pi}\frac{\theta}{1 - \cos\theta} \approx 0{,}87856$ gilt $\theta/\pi \ge \alpha_{GW}\,\tfrac{1 - \cos\theta}{2}$ für jede Kante, also $\mathbb{E}[\text{Schnitt}] \ge \alpha_{GW} \cdot \text{SDP-Wert}$.

**Das Dual.** $Z = \mathrm{Diag}(d) + \tfrac12 W \succeq 0$ ist dual zulässig; die Schranke $\tfrac12\big(\sum_{i<j} w_{ij} + \mathrm{tr}\,Z\big)$ gilt für jeden Schnitt und ist im Optimum gleich dem SDP-Wert.

**Das Fünfeck.** Fünf Einheitsvektoren im Winkel $4\pi/5$: $X_{k,k+1} = \cos(4\pi/5)$, SDP-Wert $\tfrac52\big(1 - \cos\tfrac{4\pi}{5}\big) = 4{,}5225$, exakter Schnitt 4, Verhältnis $\tfrac{8}{5(1 + \cos(\pi/5))} = 0{,}8845$; die erwartete Rundung ist $5 \cdot \tfrac45 = 4$.

**Rang.** Nach Barvinok und Pataki gibt es eine optimale Lösung vom Rang $r$ mit $r(r+1)/2 \le n$ (hier nur als Verweis, nicht am Text nachgelesen); die Lösung des inneren Punkts ist der Mittelpunkt der optimalen Menge.

**Literatur.** M. X. Goemans, D. P. Williamson: *Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming*, Journal of the ACM 42 (6), 1995, S. 1115–1145 (die Garantie 0,87856 und die Zufallshyperebene).
G. Pataki: *On the rank of extreme matrices in semidefinite programs and the multiplicity of optimal eigenvalues*, Mathematics of Operations Research 23 (2), 1998, S. 339–358 (Rang; die Aussage ist hier nicht am Text geprüft). Die NP-Härte von MaxCut ist ein Standardresultat (R. M. Karp 1972, Entscheidungsvariante NP-vollständig); hier nur als Verweis, nicht nachgelesen. Beweise der Arbeiten nicht nachgelesen; die Formel der Randwahrscheinlichkeit und die Konstante sind hier nachgerechnet.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Konvexe Optimierung: vom Beweis zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-konvexe-optimierung.html)."
)
