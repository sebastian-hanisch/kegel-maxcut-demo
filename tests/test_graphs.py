"""Graphen (kmc_graphs): Familien von Hand, Zufallsgraphen, Schnittwert, einfache Aufzählung."""

import numpy as np
import pytest

import kmc_graphs as G


def test_the_families_have_the_known_sizes_and_degrees():
    c5, k5, b = G.cycle(5), G.complete(5), G.bipartite(3, 4)
    assert len(G.edges(c5)) == 5 and (c5.sum(axis=1) == 2).all() and len(G.edges(k5)) == 10 and (k5.sum(axis=1) == 4).all() and len(G.edges(b)) == 12 and b.shape == (7, 7)
    t = G.tree(12, 0)
    assert len(G.edges(t)) == 11 and np.linalg.matrix_rank(np.diag(t.sum(axis=1)) - t) == 11                                # n − 1 Kanten, zusammenhängend (Rang der Laplace-Matrix n − 1)
    g = G.grid(4, 4)
    assert len(G.edges(g)) == 24 and g.sum(axis=1).max() == 4 and g.sum(axis=1).min() == 2
    w = G.wheel(9)
    assert len(G.edges(w)) == 16 and w[0].sum() == 8 and (w[1:].sum(axis=1) == 3).all()
    p = G.petersen()
    assert len(G.edges(p)) == 15 and (p.sum(axis=1) == 3).all()
    for W in (c5, k5, b, t, g, w, p):
        assert np.allclose(W, W.T) and np.diag(W).sum() == 0 and G.total_weight(W) == len(G.edges(W))


def test_the_cut_value_by_hand():
    """Fünfeck, s = (+, −, +, −, +): die Kanten (0,1), (1,2), (2,3), (3,4) liegen im Schnitt, (4,0) nicht: 4. Dreieck: jeder Schnitt trennt höchstens 2 Kanten. Alle Knoten auf einer Seite: 0."""
    assert G.cut_value(G.cycle(5), [1, -1, 1, -1, 1]) == 4.0 and G.cut_value(G.cycle(5), [1, 1, 1, 1, 1]) == 0.0
    assert G.cut_value(G.complete(3), [1, 1, -1]) == 2.0 and G.cut_value(G.bipartite(2, 3), [1, 1, -1, -1, -1]) == 6.0
    W = np.array([[0, 2.5], [2.5, 0]])
    assert G.cut_value(W, [1, -1]) == 2.5 and G.cut_value(W, [1, 1]) == 0.0


def test_brute_force_values():
    assert G.brute_force_maxcut(G.cycle(5)) == 4.0 and G.brute_force_maxcut(G.cycle(6)) == 6.0 and G.brute_force_maxcut(G.complete(4)) == 4.0 and G.brute_force_maxcut(G.complete(5)) == 6.0
    assert G.brute_force_maxcut(G.bipartite(3, 3)) == 9.0 and G.brute_force_maxcut(np.zeros((4, 4))) == 0.0
    with pytest.raises(ValueError):
        G.brute_force_maxcut(np.zeros((13, 13)))


def test_random_graphs_are_deterministic_and_respect_p_and_weights():
    a, b = G.gnp(12, 0.5, 3), G.gnp(12, 0.5, 3)
    assert np.array_equal(a, b) and not np.array_equal(a, G.gnp(12, 0.5, 4)) and len(G.edges(a)) == 27
    assert len(G.edges(G.gnp(10, 0.0, 1))) == 0 and len(G.edges(G.gnp(10, 1.0, 1))) == 45
    w = G.gnp(14, 0.5, 5, "zufall")
    assert np.allclose(w, w.T) and w.max() <= 1.0 and 0.0 < w[w > 0].min() and len(G.edges(w)) == len(G.edges(G.gnp(14, 0.5, 5)))
    big = G.gnp(60, 0.3, 0)
    assert abs(len(G.edges(big)) / (60 * 59 / 2) - 0.3) < 0.05


def test_the_circle_layout_puts_the_nodes_on_the_unit_circle():
    pos = G.circle_layout(7)
    assert pos.shape == (7, 2) and np.allclose(np.linalg.norm(pos, axis=1), 1.0) and np.allclose(pos.sum(axis=0), 0.0, atol=1e-12)


def test_bad_arguments_are_rejected():
    for bad in (lambda: G.gnp(1, 0.5, 0), lambda: G.gnp(5, 1.5, 0), lambda: G.gnp(5, 0.5, 0, "viel"), lambda: G.cycle(2), lambda: G.complete(1), lambda: G.bipartite(0, 3), lambda: G.tree(1), lambda: G.grid(1, 1), lambda: G.wheel(3)):
        with pytest.raises(ValueError):
            bad()
