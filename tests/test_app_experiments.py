"""Pfade der App, die der Rauchtest nicht erreicht: die fünf Experimente im Seiteninhalt, PDF, jede Dichte, jede Kreislänge, Eckfälle (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kmc_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _widget(at, key):
    if key == "n_slider":
        return at.sidebar.slider(key=key)
    if key == "weights_select":
        return at.sidebar.selectbox(key=key)
    if key == "seed_input":
        return at.sidebar.number_input(key=key)
    return at.sidebar.select_slider(key=key)


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        _widget(at, key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_default_page_shows_the_metrics_and_all_experiment_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert values["SDP-Schranke"] == "20,8082" and values["Exakter größter Schnitt"] == "20" and values["Erwartete Rundung"] == "19,2717" and values["Bester von 50"] == "20" and values["Erwartung / Schranke"] == "0,9262"
    text = _texts(at)
    for part in ("G(12, 0,5) (Seed 3, 27 Kanten): die SDP-Schranke ist **20,8082**", "Kreis C₅: die SDP-Schranke ist 4,5225", "30 Zufallsgraphen G(12, 0,5): die erwartete Rundung liegt bei", "Über 2000 Hyperebenen: Mittelwert",
                 "Die Schranke ist exakt bei 7 von 14 Familien", "Die Dualmatrix Z = Diag(d) + W/2 ist positiv semidefinit"):
        assert part in text, part


def test_the_five_experiments_the_limits_table_and_the_formulation_are_on_the_page():
    at = _app()
    headings = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headings if h.startswith("🔬")] == [f"🔬 Experiment {i}" for i in range(1, 6)] and "🚧 Wo die Annahmen enden" in headings
    assert [e.label for e in at.expander] == ["So funktioniert es", "📐 Mathematische Formulierung"]
    assert len(at.dataframe) == 3 and not at.error


def test_the_tables_have_the_expected_rows():
    at = _app()
    frames = [d.value for d in at.dataframe]
    assert len(frames[0]) == 7 and len(frames[1]) == 14 and list(frames[1]["Graph"])[0] == "Baum (n = 12)" and len(frames[2]) == 6
    assert frames[1]["Schranke exakt"].tolist().count("ja") == 7


def test_pdf_button_produces_a_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert at.session_state["pdf_made"][1][:4] == b"%PDF"


@pytest.mark.parametrize("p", C.P_CHOICES)
def test_every_density_renders(p):
    at = _app(p_select=p)
    assert not at.error and any("die SDP-Schranke ist" in s.value for s in list(at.success) + list(at.warning))


@pytest.mark.parametrize("cycle", C.CYCLE_CHOICES)
def test_every_cycle_length_renders(cycle):
    at = _app(cycle_select=cycle)
    assert not at.error and any(f"Kreis C{'₀₁₂₃₄₅₆₇₈₉'[cycle]}" in s.value for s in at.info)


def test_the_weights_and_the_hyperplane_counts_render():
    at = _app(weights_select="zufall", k_select=1)
    assert not at.error and {m.label: m.value for m in at.metric}["Bester von 1"] != ""
    at = _app(k_select=100)
    assert not at.error and "Bester von 100" in {m.label for m in at.metric}


@pytest.mark.parametrize("n", [C.N_MIN, C.EXACT_MAX, C.N_MAX])
def test_the_extreme_numbers_of_nodes_render(n):
    at = _app(n_slider=n)
    assert not at.error and not at.exception
    values = {m.label: m.value for m in at.metric}
    assert (values["Exakter größter Schnitt"] == "–") == (n > C.EXACT_MAX)
