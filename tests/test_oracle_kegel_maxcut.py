"""Orakel (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): die SDP-Relaxation gegen einen anderen Rechenweg.

- cvxpy formuliert das SDP **selbst** (`cp.Variable((n, n), PSD=True)`, `cp.diag(X) == 1`, Zielfunktion über die Kanten) und löst es mit Clarabel und SCS: gleicher SDP-Wert für C₅, den Petersen-Graphen und Zufallsgraphen
- der exakte Schnitt gegen eine unabhängige Rechnung (einfache Aufzählung in kmc_graphs; für n ≤ 12)

Fehlt cvxpy im Test-venv, überspringen sich diese Tests (pytest.importorskip), sie raten nie."""

import numpy as np
import pytest

import kmc_graphs as G
import kmc_rounding as R
import kmc_sdp as S


def _cp():
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel fehlt")
    return cp


def _cvxpy_value(cp, W, solver=None):
    n = len(W)
    X = cp.Variable((n, n), PSD=True)
    obj = sum(W[i, j] * (1 - X[i, j]) / 2 for i in range(n) for j in range(i + 1, n) if W[i, j] != 0)
    prob = cp.Problem(cp.Maximize(obj), [cp.diag(X) == 1])
    kw = dict(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10) if solver is None else dict(solver=solver, eps=1e-9, max_iters=200000)
    prob.solve(**kw)
    return float(prob.value)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
@pytest.mark.parametrize("name,W", [("C5", G.cycle(5)), ("Petersen", G.petersen()), ("K4", G.complete(4)), ("G(10, ½)", G.gnp(10, 0.5, 2)), ("G(16, ½)", G.gnp(16, 0.5, 3)), ("gewichtet", G.gnp(12, 0.6, 5, "zufall"))])
def test_cvxpy_finds_the_same_sdp_value_with_clarabel(name, W):
    cp = _cp()
    assert S.solve(W).value == pytest.approx(_cvxpy_value(cp, W), abs=1e-6), name


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_scs_the_admm_of_cvxpy_finds_the_same_value():
    cp = _cp()
    if "SCS" not in cp.installed_solvers():
        pytest.skip("SCS fehlt")
    assert S.solve(G.cycle(5)).value == pytest.approx(_cvxpy_value(cp, G.cycle(5), solver=cp.SCS), abs=1e-5)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_the_bound_from_our_dual_equals_the_cvxpy_optimum():
    """Starke Dualität: die Schranke (Σ w + tr Z)/2 aus unserer Dualmatrix ist gleich dem Optimalwert, den cvxpy für das Primalprogramm findet."""
    cp = _cp()
    for W in (G.cycle(5), G.gnp(10, 0.5, 2)):
        assert S.solve(W).dual_bound == pytest.approx(_cvxpy_value(cp, W), abs=1e-6)


def test_the_exact_cut_equals_an_independent_enumeration():
    for seed in range(5):
        W = G.gnp(10, 0.5, seed)
        assert R.exact_maxcut(W)[0] == pytest.approx(G.brute_force_maxcut(W))
