"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink verwirft ungültige Werte; die Aussagen der Hilfetexte."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import kmc_constants as C
import kmc_evaluation as ev
import kmc_presets as PR

APP = str(Path(__file__).resolve().parent.parent / "app.py")
DEFAULT = "Standardfall: G(12, ½), Gewichte 1"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(PR.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_bounds():
    for preset in C.PRESETS.values():
        assert C.N_MIN <= preset["n"] <= C.N_MAX and preset["p"] in C.P_CHOICES and preset["weights"] in C.WEIGHTS and preset["k"] in C.K_CHOICES and preset["cycle"] in C.CYCLE_CHOICES and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[DEFAULT]) == ev.Settings() and next(iter(C.PRESETS)) == DEFAULT


def test_setting_specs_cover_every_preset_key():
    assert set(PR.PRESET_KEYS.values()) == set(PR.SETTING_SPECS)


def test_the_preset_values_differ_from_the_default_where_the_help_says_so():
    d = C.PRESETS[DEFAULT]
    assert C.PRESETS["Dünner Graph (p = 0,1): fast exakt"]["p"] == 0.1 and C.PRESETS["Dünner Graph (p = 0,1): fast exakt"]["n"] == 16 and C.PRESETS["Dichter Graph (p = 0,8)"]["p"] == 0.8
    assert C.PRESETS["Zufallsgewichte"]["weights"] == "zufall" and C.PRESETS["Größter Graph mit exaktem Schnitt (n = 20)"]["n"] == C.EXACT_MAX and C.PRESETS["Ungerader Kreis C₇ (Experiment 1)"]["cycle"] == 7
    assert all(C.PRESETS[name][k] == d[k] for name in C.PRESETS for k in ("seed", "k") if name != DEFAULT)


def test_permalink_takes_valid_values_and_drops_invalid_ones():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "999"            # über der Grenze: wird begrenzt
    at.query_params["p"] = "0.3"
    at.query_params["weights"] = "zufall"
    at.query_params["k"] = "7"              # kein erlaubter Wert: bleibt Standard
    at.query_params["cycle"] = "7"
    at.run()
    assert not at.exception
    assert at.sidebar.slider(key="n_slider").value == C.N_MAX and at.session_state["p_select"] == 0.3 and at.session_state["weights_select"] == "zufall" and at.session_state["k_select"] == C.DEFAULT_K and at.session_state["cycle_select"] == 7
    bad = AppTest.from_file(APP, default_timeout=300)
    bad.query_params["weights"] = "viel"
    bad.query_params["seed"] = "abc"
    bad.run()
    assert not bad.exception and bad.session_state["weights_select"] == C.DEFAULT_WEIGHTS and bad.sidebar.number_input(key="seed_input").value == C.DEFAULT_SEED
