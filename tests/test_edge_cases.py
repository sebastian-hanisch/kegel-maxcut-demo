"""Randfälle, an denen die Relaxation erfahrungsgemäß bricht (DEMO-PLAYBOOK §12)."""

import pickle

import numpy as np
import pytest

import kmc_constants as C
import kmc_evaluation as ev
import kmc_graphs as G
import kmc_rounding as R
import kmc_sdp as S


def test_two_and_three_nodes():
    one = ev.main_view(ev.Settings(n=2, p=1.0))
    assert one["res"].value == pytest.approx(1.0, abs=1e-6) and one["exact"] == 1.0 and one["best"] == 1.0
    for n in (3, 4):
        v = ev.main_view(ev.Settings(n=n, p=0.8, seed=2))
        assert v["exact"] <= v["res"].value + 1e-6 and v["expected"] >= R.ALPHA_GW * v["res"].value - 1e-6


def test_a_graph_without_edges_gives_zero_everywhere():
    assert G.total_weight(G.gnp(6, 0.0, 0)) == 0.0
    r = S.solve(np.zeros((6, 6)))
    assert r.value == 0.0 and R.expected_rounding(np.zeros((6, 6)), r.X) == 0.0 and R.exact_maxcut(np.zeros((6, 6)))[0] == 0.0
    v = ev.main_view(ev.Settings(n=6, p=0.1, seed=0)) if G.total_weight(G.gnp(6, 0.1, 0)) == 0.0 else None
    assert v is None or (v["res"].value == 0.0 and v["best"] == 0.0)


def test_the_statistic_skips_graphs_without_edges():
    rows = ev.stat_rows(ev.Settings(n=4, p=0.1, seed=0))
    assert len(rows) == C.STAT_GRAPHS and all(r["sdp"] > 0 for r in rows)


def test_a_disconnected_graph_is_the_sum_of_its_parts():
    A = G.cycle(5)
    B = G.complete(3)
    W = np.zeros((8, 8))
    W[:5, :5] = A
    W[5:, 5:] = B
    r = S.solve(W)
    assert r.value == pytest.approx(S.solve(A).value + S.solve(B).value, abs=1e-6) and R.exact_maxcut(W)[0] == 6.0


def test_a_zero_weight_edge_changes_nothing():
    W = G.gnp(8, 0.6, 3)
    W2 = W.copy()
    W2[0, 1] = W2[1, 0] = 0.0
    assert np.isfinite(S.solve(W2).value) and G.brute_force_maxcut(W2) <= S.solve(W2).value + 1e-6


def test_the_scale_of_the_weights_scales_the_values():
    W = G.gnp(8, 0.6, 3)
    a, b = S.solve(W), S.solve(3.0 * W)
    assert b.value == pytest.approx(3.0 * a.value, rel=1e-6) and R.expected_rounding(3.0 * W, b.X) == pytest.approx(3.0 * R.expected_rounding(W, a.X), rel=1e-6)


def test_the_largest_graph_with_an_exact_cut_and_the_largest_without():
    v = ev.main_view(ev.Settings(n=C.EXACT_MAX))
    assert v["exact"] is not None and v["exact"] <= v["res"].value + 1e-6
    w = ev.main_view(ev.Settings(n=C.N_MAX, p=0.3))
    assert w["exact"] is None and w["expected"] >= R.ALPHA_GW * w["res"].value - 1e-6


def test_every_corner_of_the_controls_gives_a_finite_result_and_a_verdict():
    for n in (C.N_MIN, C.EXACT_MAX, C.N_MAX):
        for p in (C.P_CHOICES[0], C.P_CHOICES[-1]):
            for weights in C.WEIGHTS:
                for k in (C.K_CHOICES[0], C.K_CHOICES[-1]):
                    s = ev.Settings(n=n, p=p, seed=7, weights=weights, k=k)
                    v = ev.main_view(s)
                    assert np.isfinite(v["res"].value) and ev.verdict_main(s, v)[0] in {"success", "warning"} and len(v["values"]) == k, (n, p, weights, k)


def test_the_hyperplane_rounding_is_reproducible():
    s = ev.Settings()
    a, b = ev.main_view(s), ev.main_view(s)
    assert a["best"] == b["best"] and np.array_equal(a["values"], b["values"]) and np.array_equal(a["best_s"], b["best_s"])
    assert not np.array_equal(ev.main_view(ev.Settings(seed=4))["values"], a["values"])


def test_the_cached_results_can_be_pickled():
    """Streamlit speichert die Rückgabewerte mit pickle: Ansichten ohne lokale Funktionen."""
    s = ev.Settings(n=8)
    for obj in (ev.main_view(s), ev.cycle_view(5), ev.family_rows(), ev.rank_rows(), ev.stat_rows(ev.Settings(n=6))):
        assert pickle.loads(pickle.dumps(obj)) is not None
    m = ev.main_view(s)
    assert pickle.loads(pickle.dumps(ev.rounding_view(m, s.seed)))["max"] == ev.rounding_view(m, s.seed)["max"]


def test_the_view_covers_every_cycle_length():
    for n in C.CYCLE_CHOICES:
        c = ev.cycle_view(n)
        assert c["points"].shape == (n, 2) and ev.verdict_cycle(c)[0] == "info" and abs(np.linalg.norm(c["hyperplane"]) - 1.0) < 1e-12
