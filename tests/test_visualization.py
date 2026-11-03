"""Abbildungen: Achsen gesperrt, Spuren stimmen mit den Zahlen überein."""

import numpy as np
import pytest

import kmc_constants as C
import kmc_evaluation as ev
import kmc_graphs as G
import kmc_visualization as V

S = ev.Settings(n=10)


def _locked(fig):
    axes = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
    return bool(axes) and all(ax.get("fixedrange") is True for ax in axes)


@pytest.fixture(scope="module")
def main():
    return ev.main_view(S)


def test_graph_chart_colors_the_edges_by_the_cut(main):
    W, s = main["W"], main["best_s"]
    fig = V.build_graph(W, s, "Titel")
    names = [t.name for t in fig.data]
    assert _locked(fig) and names == ["Kante nicht im Schnitt", "Kante im Schnitt", "Seite +", "Seite −"]
    cut = sum(1 for i, j, _ in G.edges(W) if s[i] != s[j])
    assert sum(1 for x in fig.data[1].x if x is None) == cut and sum(1 for x in fig.data[0].x if x is None) == len(G.edges(W)) - cut
    assert sum(len(t.x) for t in fig.data[2:]) == len(W) and fig.layout.title.text == "Titel"


def test_vector_chart_shows_the_points_the_hyperplane_and_the_arrows():
    c = ev.cycle_view(5)
    fig = V.build_vectors(c)
    assert _locked(fig) and [t.name for t in fig.data][1] == "Hyperebene (Gerade durch 0)" and len(fig.layout.annotations) == 5
    pts = sum(len(t.x) for t in fig.data[2:])
    assert pts == 5 and fig.layout.yaxis.scaleanchor == "x"
    side = c["points"] @ c["hyperplane"] >= 0
    assert len(fig.data[2].x) == int(side.sum()) and len(fig.data[3].x) == int((~side).sum())


def test_ratio_chart_has_two_histograms_and_the_guarantee_line():
    st = ev.stat_summary(ev.stat_rows(ev.Settings(n=8)))
    fig = V.build_ratios(st)
    assert _locked(fig) and [t.name for t in fig.data] == ["erwartete Rundung / Schranke", "exakter Schnitt / Schranke"] and len(fig.layout.shapes) == 1 and fig.layout.shapes[0].x0 == pytest.approx(0.878567, abs=1e-6)
    assert len(fig.data[0].x) == C.STAT_GRAPHS and fig.layout.barmode == "overlay"


def test_histogram_and_curve_charts(main):
    r = ev.rounding_view(main, S.seed)
    h = V.build_hist(r, main)
    assert _locked(h) and len(h.data[0].x) == C.HYPER_SAMPLES and len(h.layout.shapes) == 3                          # SDP, Erwartungswert, exakt
    no_exact = dict(main, exact=None)
    assert len(V.build_hist(r, no_exact).layout.shapes) == 2
    c = V.build_curve(r, main)
    assert _locked(c) and list(c.data[0].x) == list(C.CURVE_KS) and list(c.data[0].y) == pytest.approx([r["curve"][k] for k in C.CURVE_KS]) and c.layout.xaxis.type == "log" and len(c.layout.shapes) == 2


def test_family_and_rank_charts():
    rows = ev.family_rows()
    f = V.build_families(rows)
    assert _locked(f) and len(f.data) == 2 and list(f.data[0].x) == [r["name"] for r in rows] and list(f.data[1].y) == pytest.approx([r["expected_ratio"] for r in rows]) and len(f.layout.shapes) == 1
    ranks = ev.rank_rows()
    rk = V.build_rank(ranks)
    assert _locked(rk) and [t.name for t in rk.data] == ["Rang der Lösung", "√(2n)"] and list(rk.data[0].x) == list(C.RANK_NS) and np.all(np.array(rk.data[1].y) > np.array(rk.data[0].y))
