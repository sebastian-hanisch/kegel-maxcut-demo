"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

from types import SimpleNamespace

import kmc_constants as C
import kmc_evaluation as ev


def test_formatting_helpers():
    assert ev.de(2.3456, 3) == "2,346" and ev.de(-1.0, 0) == "−1" and ev.sci(1.7e-10) == "1,7·10⁻¹⁰" and ev.sci(2.5e5) == "2,5·10⁵" and ev.sci(float("inf")) == "unendlich"
    assert ev.p_text(0.5) == "0,5" and ev.num(20.0) == "20" and ev.num(20.8082) == "20,8082" and ev.num(4.5225425) == "4,5225" and ev.ratio(1.0, 2.0) == 0.5 and ev.ratio(1.0, 0.0) == 1.0
    assert ev.graph_name(ev.Settings()) == "G(12, 0,5)" and ev.graph_name(ev.Settings(n=8, p=0.1, weights="zufall")) == "G(8, 0,1) mit Zufallsgewichten" and ev._sub(10) == "₁₀"


def _main(exact=20.0, **kw):
    res = SimpleNamespace(value=20.8082, rank=3, iterations=11, dual_bound=20.8082)
    base = {"res": res, "exact": exact, "expected": 19.2717, "best": 20.0, "edges": 27}
    base.update(kw)
    return base


def test_main_verdict_names_bound_cut_expectation_best_and_rank():
    kind, text = ev.verdict_main(ev.Settings(), _main())
    assert kind == "success"
    for part in ("G(12, 0,5) (Seed 3, 27 Kanten): die SDP-Schranke ist **20,8082**, der exakte größte Schnitt 20 (Verhältnis 0,9612)", "die erwartete Rundung 19,2717 (0,9262 der Schranke; Garantie 0,87857)",
                 "der beste von 50 Hyperebenen 20 und damit **der exakte Schnitt**", "Rang der Lösung 3, 11 Iterationen, Schranke aus dem Dual 20,8082."):
        assert part in text, part
    assert "und damit 0,9500 des exakten Schnitts" in ev.verdict_main(ev.Settings(), _main(best=19.0))[1]
    kind, text = ev.verdict_main(ev.Settings(n=26), _main(exact=None))
    assert kind == "success" and "(0,9612 der Schranke; für n > 20 gibt es keinen exakten Wert)" in text
    assert ev.verdict_main(ev.Settings(), _main(expected=17.0))[0] == "warning"                       # unter der Garantie: Warnung


def _cycle(n):
    return {"n": n, "res": SimpleNamespace(value=4.5225, rank=2 if n % 2 else 1), "exact": 4.0, "ratio": 0.88446, "angles": [144.0], "edge_probability": 0.8, "expected": 4.0}


def test_cycle_verdict_distinguishes_odd_and_even_cycles():
    kind, text = ev.verdict_cycle(_cycle(5))
    assert kind == "info"
    for part in ("Kreis C₅: die SDP-Schranke ist 4,5225, der exakte größte Schnitt 4 (Verhältnis 0,88446)", "Benachbarte Einheitsvektoren haben den Winkel 144,0°", "schneidet jede Kante mit Wahrscheinlichkeit 0,800, der erwartete Schnitt ist 4",
                 "Der ungerade Kreis hat keinen Schnitt, der alle Kanten trifft: die Relaxation ist nicht exakt (Rang 2)."):
        assert part in text, part
    assert "Der gerade Kreis ist bipartit: die Relaxation ist exakt (Rang 1), die Vektoren liegen auf einer Geraden." in ev.verdict_cycle(_cycle(4))[1]


def test_stat_verdict_gives_means_minima_and_the_count_below_the_guarantee():
    rows = [{"sdp": 10.0, "exact": 9.7, "expected": 9.4, "rank": 2}, {"sdp": 20.0, "exact": 19.0, "expected": 18.4, "rank": 3}]
    kind, text = ev.verdict_stat(rows, ev.Settings())
    assert kind == "info"
    for part in ("2 Zufallsgraphen G(12, 0,5): die erwartete Rundung liegt bei 0,930 der Schranke im Mittel, am schwächsten bei 0,920", "unterschreitet kein Graph (0 von 2)", "Der exakte Schnitt liegt bei 0,960 der Schranke im Mittel, am weitesten darunter bei 0,950"):
        assert part in text, part
    assert "G(16, 0,5)" in ev.verdict_stat(rows, ev.Settings(n=26))[1]                                    # die Verteilung rechnet höchstens mit n = 16
    low = [{"sdp": 10.0, "exact": 9.0, "expected": 8.0, "rank": 2}]
    assert "unterschreiten 1 von 1 Graphen" in ev.verdict_stat(low, ev.Settings())[1] and "kein Graph" not in ev.verdict_stat(low, ev.Settings())[1]
    assert ev.stat_summary(rows)["count"] == 2


def test_rounding_verdict_compares_mean_and_formula_and_gives_the_curve():
    r = {"mean": 19.2955, "stderr": 0.0175, "expected": 19.2717, "max": 20.0, "curve": {1: 19.296, 10: 20.0, 100: 20.0}}
    kind, text = ev.verdict_rounding(r, ev.Settings(), {"exact": 20.0})
    assert kind == "info"
    for part in ("Über 2000 Hyperebenen: Mittelwert 19,2955 ± 0,0175, die Formel Σ w arccos(X)/π gibt 19,2717", "der beste Schnitt der Stichprobe ist 20 (exakt 20)", "wächst von 19,296 (k = 1) über 20,000 (k = 10) auf 20,000 (k = 100): wenige Hyperebenen genügen"):
        assert part in text, part
    assert "(exakt" not in ev.verdict_rounding(r, ev.Settings(), {"exact": None})[1]


def _fam():
    def row(name, exact_sdp, rank, er, xr):
        return {"name": name, "exact_sdp": exact_sdp, "rank": rank, "expected_ratio": er, "exact_ratio": xr}
    return [row("A", True, 1, 1.0, 1.0), row("B", True, 3, 0.9123, 1.0), row("C", False, 2, 0.8787, 0.96)]


def test_families_verdict_counts_exact_ones_and_names_the_converse_failure():
    kind, text = ev.verdict_families(_fam())
    assert kind == "info"
    for part in ("Die Schranke ist exakt bei 2 von 3 Familien; Rang 1 haben A.", "Die Umkehrung „exakt ⇒ Rang 1“ gilt nicht: B (Rang 3) haben eine exakte Schranke, aber eine Lösung mit höherem Rang", "Das kleinste Verhältnis von erwarteter Rundung zu Schranke hat C mit 0,8787",
                 "das kleinste Verhältnis exakt zu Schranke hat C mit 0,96000"):
        assert part in text, part
    assert "Umkehrung" not in ev.verdict_families([_fam()[0], _fam()[2]])[1]


def test_dual_verdict_states_the_certificate_and_the_rank_scale():
    res = SimpleNamespace(z_min_eig=5e-11, z_offdiag=3.7e-13, dual_bound=20.8082, value=20.8082, gap=1.8e-9)
    ranks = [{"n": 8, "rank": 2, "bound": 4.0}, {"n": 12, "rank": 3, "bound": 4.9}]
    kind, text = ev.verdict_dual({"res": res, "exact": 20.0}, ev.Settings(), ranks)
    assert kind == "success"
    for part in ("kleinster Eigenwert 5,0·10⁻¹¹, Außerdiagonale gleich wᵢⱼ/2 auf 3,7·10⁻¹³", "(Σ w + tr Z)/2 = 20,8082 gleich dem SDP-Wert 20,8082 (Unterschied 1,8·10⁻⁹)", "der exakte Schnitt ist 20",
                 "ist 2 (n = 8), 3 (n = 12) gegenüber √(2n) = 4,0, 4,9", "Stück 13 (Burer–Monteiro)"):
        assert part in text, part
    bad = SimpleNamespace(z_min_eig=-1e-3, z_offdiag=0.0, dual_bound=21.0, value=20.0, gap=1.0)
    assert ev.verdict_dual({"res": bad, "exact": None}, ev.Settings(), ranks)[0] == "warning"


def test_constants_are_consistent():
    assert C.EXACT_MAX == 20 and C.STAT_N_MAX <= C.EXACT_MAX and C.N_MAX > C.EXACT_MAX and 1 in C.K_CHOICES and 5 in C.CYCLE_CHOICES and 7 in C.CYCLE_CHOICES and 0.5 in C.P_CHOICES
