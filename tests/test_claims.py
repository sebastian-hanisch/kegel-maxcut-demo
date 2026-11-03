"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Die Werte sind deterministisch bis auf die Rundung der BLAS-Routinen (Toleranz des Lösers 10⁻⁹): Bänder; Zeiten, Iterationszahlen, der **Status des Lösers** und der **Rang der Mittelpunktslösung** nur in weiten Bändern."""

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

import kmc_constants as C
import kmc_evaluation as ev
import kmc_graphs as G
import kmc_rounding as R
import kmc_sdp as S

S0 = ev.Settings()


@pytest.fixture(scope="module")
def main():
    return ev.main_view(S0)


@pytest.fixture(scope="module")
def families():
    return {r["name"]: r for r in ev.family_rows()}


def test_the_default_graph(main):
    """README: G(12, ½), Seed 3, 27 Kanten: Schranke 20,8082, exakter Schnitt 20 (0,9612 der Schranke), erwartete Rundung 19,2717 (0,9262), der beste von 50 Hyperebenen ist der exakte Schnitt, Rang 3 (Band 2 bis 4)."""
    r = main["res"]
    assert main["edges"] == 27 and r.value == pytest.approx(20.8082, abs=5e-4) and main["exact"] == 20.0 and main["expected"] == pytest.approx(19.2717, abs=5e-4) and main["best"] == 20.0
    assert main["exact"] / r.value == pytest.approx(0.9612, abs=5e-4) and main["expected"] / r.value == pytest.approx(0.9262, abs=5e-4) and 2 <= r.rank <= 4 and abs(r.gap) < 1e-6 and r.iterations <= 25
    assert ev.verdict_main(S0, main)[0] == "success" and main["expected"] >= R.ALPHA_GW * r.value


def test_odd_cycles_have_a_gap_and_even_cycles_none():
    """README: Cₙ ungerade: SDP (n/2)(1 + cos(π/n)), exakter Schnitt n − 1; C₅: 4,5225 / 4 = 0,88446, C₇: 6,6534 / 6 = 0,9018; gerade: Schranke = Schnitt = n, Rang 1; die erwartete Rundung ist bei Kreisen exakt der Schnitt."""
    for n in (3, 4, 5, 6, 7, 9):
        c = ev.cycle_view(n)
        if n % 2:
            assert c["res"].value == pytest.approx(n / 2 * (1 + math.cos(math.pi / n)), abs=1e-6) and c["exact"] == n - 1 and c["res"].rank == 2
            assert c["angles"][0] == pytest.approx(180.0 - 180.0 / n, abs=1e-3) and c["edge_probability"] == pytest.approx(1.0 - 1.0 / n, abs=1e-5)
        else:
            assert c["res"].value == pytest.approx(n, abs=1e-6) and c["exact"] == n and c["res"].rank == 1
        assert c["expected"] == pytest.approx(c["exact"], abs=1e-3)
    c9 = ev.cycle_view(9)
    assert c9["res"].value == pytest.approx(8.7286, abs=5e-5) and c9["ratio"] == pytest.approx(0.9165, abs=5e-5)
    c5, c7 = ev.cycle_view(5), ev.cycle_view(7)
    assert c5["res"].value == pytest.approx(4.52254249, abs=1e-6) and c5["ratio"] == pytest.approx(0.88446, abs=1e-5) and c7["res"].value == pytest.approx(6.65339104, abs=1e-6) and c7["ratio"] == pytest.approx(0.90180, abs=1e-5)
    assert c5["points"].shape == (5, 2) and np.allclose(np.linalg.norm(c5["points"], axis=1), 1.0)
    assert np.allclose(c5["points"] @ c5["points"].T, c5["res"].X, atol=1e-6) and np.allclose(c7["points"] @ c7["points"].T, c7["res"].X, atol=1e-6)         # die Ebene trägt die ganze Lösung (Rang 2)
    c4 = ev.cycle_view(4)
    assert np.allclose(c4["points"] @ c4["points"].T, c4["res"].X, atol=1e-6) and np.allclose(np.abs(c4["points"][:, 0]) + np.abs(c4["points"][:, 1]), np.abs(c4["points"][0, 0]) + np.abs(c4["points"][0, 1]))


def test_random_graphs_stay_far_above_the_guarantee():
    """README (n = 12, je 30 Graphen): erwartete Rundung / Schranke im Mittel 0,9358 (p = 0,5; Band 0,93 bis 0,945), kleinster Wert 0,9146; exakt / Schranke 0,9702 (kleinster 0,9416); keiner unter 0,87857.
    Nach Dichte p = 0,1 / 0,2 / 0,3 / 0,5 / 0,8: Mittel 0,993 / 0,962 / 0,941 / 0,936 / 0,952 (Band ±0,02): am schwächsten bei mittlerer Dichte."""
    sm = ev.stat_summary(ev.stat_rows(S0))
    assert sm["count"] == 30 and sm["expected_mean"] == pytest.approx(0.9358, abs=0.008) and sm["expected_min"] == pytest.approx(0.9146, abs=0.015) and sm["exact_mean"] == pytest.approx(0.9702, abs=0.008)
    assert sm["exact_min"] == pytest.approx(0.9416, abs=0.015) and (sm["expected"] >= R.ALPHA_GW).all() and (sm["exact"] <= 1.0 + 1e-9).all() and (sm["exact"] >= sm["expected"] - 1e-9).all()
    sums = {p: ev.stat_summary(ev.stat_rows(ev.Settings(n=12, p=p))) for p in C.P_CHOICES}
    means = {p: s["expected_mean"] for p, s in sums.items()}
    for p, v, lo, xm in zip(C.P_CHOICES, (0.993, 0.962, 0.941, 0.936, 0.952), (0.938, 0.904, 0.909, 0.915, 0.929), (0.995, 0.976, 0.970, 0.970, 0.983)):
        assert means[p] == pytest.approx(v, abs=0.02), p
        assert sums[p]["expected_min"] == pytest.approx(lo, abs=0.03) and sums[p]["exact_mean"] == pytest.approx(xm, abs=0.015) and (sums[p]["expected"] >= R.ALPHA_GW).all(), p
    assert means[0.1] > means[0.3] and means[0.3] < means[0.8] and min(means.values()) > R.ALPHA_GW


def test_the_rounding_view(main):
    """README (Standardgraph, 2 000 Hyperebenen): Mittelwert 19,2955 ± 0,0175 gegen die Formel 19,2717 (innerhalb von 4 Standardfehlern); der Erwartungswert des besten von k: 19,296 (k = 1), 20,000 (k = 10 und 100)."""
    r = ev.rounding_view(main, S0.seed)
    assert abs(r["mean"] - r["expected"]) < 4 * r["stderr"] and r["stderr"] == pytest.approx(0.0175, abs=0.004) and r["max"] == 20.0
    assert r["curve"][1] == pytest.approx(r["mean"], abs=1e-9) and r["curve"][10] >= 19.95 and r["curve"][100] == pytest.approx(20.0, abs=1e-9) and list(r["curve"]) == list(C.CURVE_KS)


def test_the_families_table(families):
    """README (SDP / exakt / exakt÷SDP / E[Rundung]÷SDP / Rang): Baum 11 / 11 / 1 / 1 / 1; K₆,₆ 36 / 36 / 1 / 1 / 1; Gitter 4×4 24 / 24 / 1 / 1 / 1; C₅ 4,5225 / 4 / 0,8845 / 0,8845 / 2; C₇ 6,6534 / 6 / 0,9018;
    K₃ 2,25 / 2 / 0,8889; K₄ 4 / 4 / 1 / 0,9123 / 3; K₅ 6,25 / 6 / 0,96 / 0,9287 / 4; K₆ 9 / 9 / 1 / 0,9402 / 5; Petersen 12,5 / 12 / 0,96 / 0,8787 / 4; Rad W₉ 12,5 / 12 / 0,96 / 0,9085 / 2."""
    expect = {"Baum (n = 12)": (11, 11, 1, 1, 1), "vollständig bipartit K₆,₆": (36, 36, 1, 1, 1), "Gitter 4×4": (24, 24, 1, 1, 1), "Kreis C₅": (4.5225, 4, 0.8845, 0.8845, 2), "Kreis C₇": (6.6534, 6, 0.9018, 0.9018, 2),
              "vollständig K₃": (2.25, 2, 0.8889, 0.8889, 2), "vollständig K₄": (4, 4, 1, 0.9123, 3), "vollständig K₅": (6.25, 6, 0.96, 0.9287, 4), "vollständig K₆": (9, 9, 1, 0.9402, 5),
              "Petersen-Graph": (12.5, 12, 0.96, 0.8787, 4), "Rad W₉": (12.5, 12, 0.96, 0.9085, 2)}
    for name, (sdp, exact, er, xr, rank) in expect.items():
        r = families[name]
        assert r["sdp"] == pytest.approx(sdp, abs=1e-3) and r["exact"] == pytest.approx(exact) and r["exact_ratio"] == pytest.approx(er, abs=1e-4) and r["expected_ratio"] == pytest.approx(xr, abs=1e-4), name
        assert abs(r["rank"] - rank) <= 1 if rank > 1 else r["rank"] == 1, name                                    # der Rang der Mittelpunktslösung: Band ±1, Rang 1 genau
    assert all(r["expected_ratio"] >= R.ALPHA_GW - 1e-6 for r in families.values())
    assert min(r["expected_ratio"] for r in families.values()) == families["Petersen-Graph"]["expected_ratio"] and families["Petersen-Graph"]["expected_ratio"] == pytest.approx(0.8787, abs=1e-4)
    assert min(r["exact_ratio"] for r in families.values()) == families["Kreis C₅"]["exact_ratio"]


def test_exact_means_a_bound_equal_to_the_cut_but_not_rank_one(families):
    """README: die Schranke ist exakt bei Bäumen, bipartiten Graphen, Gittern, geraden Kreisen und bei K₄ und K₆ (Rang 3 und 5 statt 1): „exakt ⇒ Rang 1“ gilt nicht."""
    exact = {n for n, r in families.items() if r["exact_sdp"]}
    assert {"Baum (n = 12)", "vollständig bipartit K₆,₆", "Gitter 4×4", "Kreis C₄", "Kreis C₆", "vollständig K₄", "vollständig K₆"} <= exact and len(exact) == 7
    assert all(families[n]["rank"] == 1 for n in ("Baum (n = 12)", "vollständig bipartit K₆,₆", "Gitter 4×4", "Kreis C₄", "Kreis C₆"))
    assert families["vollständig K₄"]["rank"] > 1 and families["vollständig K₆"]["rank"] > 1


def test_the_rank_of_the_solution_is_far_below_the_pataki_scale():
    """README: Rang der Mittelpunktslösung auf G(n, ½) für n = 8 / 12 / 16 / 20 / 24: 2 / 3 / 3 / 3 / 3 (Band 2 bis 5), weit unter √(2n) = 4,0 / 4,9 / 5,7 / 6,3 / 6,9."""
    rows = ev.rank_rows()
    assert [r["n"] for r in rows] == list(C.RANK_NS) and all(2 <= r["rank"] <= 5 and r["rank"] < r["bound"] for r in rows) and rows[0]["bound"] == pytest.approx(4.0) and rows[-1]["bound"] == pytest.approx(math.sqrt(48))


def test_the_preset_help_texts_name_the_numbers():
    """Hilfetexte der Presets: Standard (27 Kanten, 20,8082 / 20 / 19,2717), p = 0,1 und n = 16 (9 Kanten, 8,25 / 8, 0,9697), p = 0,8 (53 Kanten, 0,9498), Zufallsgewichte (10,8628 / 10,6643 / 0,9258), n = 20 (88 Kanten, 61 / 63,1334), C₇."""
    def view(name):
        return ev.main_view(ev.Settings(**C.PRESETS[name]))
    v = view("Dünner Graph (p = 0,1): fast exakt")
    assert v["edges"] == 9 and v["res"].value == pytest.approx(8.25, abs=1e-4) and v["exact"] == 8.0 and v["expected"] / v["res"].value == pytest.approx(0.9697, abs=5e-4) and v["res"].rank >= 8
    assert int(((v["W"] > 0).sum(axis=1) == 0).sum()) == 6                                  # sechs isolierte Knoten: der Mittelpunkt setzt ihre Vektoren orthogonal (Rang 3 + 6)
    v = view("Dichter Graph (p = 0,8)")
    assert v["edges"] == 53 and v["expected"] / v["res"].value == pytest.approx(0.9498, abs=5e-4) and v["exact"] == 34.0
    v = view("Zufallsgewichte")
    assert v["res"].value == pytest.approx(10.8628, abs=5e-4) and v["exact"] == pytest.approx(10.6643, abs=5e-4) and v["expected"] / v["res"].value == pytest.approx(0.9258, abs=5e-4)
    v = view("Größter Graph mit exaktem Schnitt (n = 20)")
    assert v["edges"] == 88 and v["exact"] == 61.0 and v["res"].value == pytest.approx(63.1334, abs=5e-4)
    c = ev.cycle_view(7)
    assert c["res"].value == pytest.approx(6.6534, abs=5e-5) and c["exact"] == 6.0 and c["expected"] == pytest.approx(6.0, abs=1e-3)


def test_above_twenty_nodes_there_is_no_exact_value_but_a_bound_and_a_rounding():
    s = ev.Settings(n=26)
    v = ev.main_view(s)
    assert v["exact"] is None and v["expected"] >= R.ALPHA_GW * v["res"].value and v["best"] <= v["res"].value + 1e-9 and v["res"].z_min_eig > -1e-7 and "für n > 20" in ev.verdict_main(s, v)[1]


def test_the_kernel_is_byte_identical_in_the_series():
    """README: kegel_kern.py ist wortgleich mit Stück 2 bis 11: Prüfsumme wie im Schwester-Repo, sofern es nebenan liegt."""
    here = Path(__file__).resolve().parent.parent / "kegel_kern.py"
    sibling = here.parent.parent / "kegel-worstcase-demo" / "kegel_kern.py"
    if not sibling.exists():
        pytest.skip("Schwester-Repo liegt nicht nebenan")
    assert hashlib.sha256(here.read_bytes()).hexdigest() == hashlib.sha256(sibling.read_bytes()).hexdigest()
