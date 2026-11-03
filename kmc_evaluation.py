"""Kennzahlen, Experimente und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

import kmc_constants as C
import kmc_graphs as G
import kmc_rounding as R
import kmc_sdp as S


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    p: float = C.DEFAULT_P
    seed: int = C.DEFAULT_SEED
    weights: str = C.DEFAULT_WEIGHTS
    k: int = C.DEFAULT_K
    cycle: int = C.DEFAULT_CYCLE


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,0·10⁻¹⁰."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def p_text(p: float) -> str:
    return de(p, 1)


def num(x: float) -> str:
    """Ein Schnittwert: ganze Zahlen ohne Nachkommastellen, sonst vier."""
    return de(x, 0) if abs(x - round(x)) < 5e-5 else de(x, 4)


def ratio(a: float, b: float) -> float:
    return a / b if b else 1.0


def graph_name(s: Settings) -> str:
    return f"G({s.n}, {p_text(s.p)})" + ("" if s.weights == "eins" else " mit Zufallsgewichten")


# ----------------------------------------------------------------------------- Experiment 1: das Fünfeck (oder ein anderer Kreis)

def cycle_view(n_cycle: int, seed: int = C.DEFAULT_SEED) -> dict:
    """Der Kreis Cₙ: SDP, exakter Schnitt, erwartete Rundung, die Einheitsvektoren in der Ebene (Rang ≤ 2) und die Winkel benachbarter Knoten; dazu eine Zufallshyperebene."""
    W = G.cycle(n_cycle)
    with threadpool_limits(1):
        res = S.solve(W, 1e-9)
    exact, s_exact = R.exact_maxcut(W)
    V = S.vectors(res.X)
    P = V[:, ::-1][:, :2]                                                         # die zwei größten Eigenwerte (eigh sortiert aufsteigend): Rang ≤ 2 hat dort alles
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed + C.ROUND_SEED_OFFSET)
    r = rng.normal(size=2)
    r = r / np.linalg.norm(r)
    angles = [math.degrees(math.acos(float(np.clip(res.X[i, (i + 1) % n_cycle], -1.0, 1.0)))) for i in range(n_cycle)]
    return {"W": W, "res": res, "exact": exact, "expected": R.expected_rounding(W, res.X), "points": P, "hyperplane": r, "angles": angles, "n": n_cycle, "edge_probability": angles[0] / 180.0,
            "ratio": exact / res.value}


# ----------------------------------------------------------------------------- Experiment 2: Zufallsgraphen

def main_view(s: Settings) -> dict:
    """Der Zufallsgraph der Seitenleiste: SDP, exakter Schnitt (bis EXACT_MAX), erwartete Rundung, bester von k Hyperebenen und der zugehörige Schnitt."""
    W = G.gnp(s.n, s.p, s.seed, s.weights)
    with threadpool_limits(1):
        res = S.solve(W, 1e-9)
    exact = R.exact_maxcut(W)[0] if s.n <= C.EXACT_MAX else None
    V = S.vectors(res.X)
    rng = np.random.default_rng(s.seed + C.ROUND_SEED_OFFSET)
    values, best_s = R.round_values(W, V, s.k, rng)
    return {"W": W, "res": res, "exact": exact, "expected": R.expected_rounding(W, res.X), "best": float(values.max()), "best_s": best_s, "values": values, "V": V, "edges": len(G.edges(W)),
            "total": G.total_weight(W)}


def stat_rows(s: Settings) -> list:
    """Zufallsgraphen G(n', p) (n' = min(n, STAT_N_MAX), Gewichte der Seitenleiste), STAT_GRAPHS Stück mit mindestens einer Kante: SDP, exakter Schnitt, erwartete Rundung."""
    n = min(s.n, C.STAT_N_MAX)
    rows, seed = [], s.seed
    with threadpool_limits(1):
        while len(rows) < C.STAT_GRAPHS and seed < s.seed + 10 * C.STAT_GRAPHS:
            W = G.gnp(n, s.p, seed, s.weights)
            seed += 1
            if G.total_weight(W) == 0.0:
                continue
            res = S.solve(W, 1e-9)
            exact = R.exact_maxcut(W)[0]
            rows.append({"sdp": res.value, "exact": exact, "expected": R.expected_rounding(W, res.X), "rank": res.rank})
    return rows


def stat_summary(rows: list) -> dict:
    e = np.array([r["expected"] / r["sdp"] for r in rows])
    x = np.array([r["exact"] / r["sdp"] for r in rows])
    return {"count": len(rows), "expected_mean": float(e.mean()), "expected_min": float(e.min()), "exact_mean": float(x.mean()), "exact_min": float(x.min()), "expected": e, "exact": x}


# ----------------------------------------------------------------------------- Experiment 3: die Rundung

def rounding_view(main: dict, seed: int) -> dict:
    """Die Schnittwerte von HYPER_SAMPLES Zufallshyperebenen auf dem Graphen der Seitenleiste, ihr Mittelwert gegen die Formel, der beste von k gegen k."""
    rng = np.random.default_rng(seed + 2 * C.ROUND_SEED_OFFSET)
    values, _ = R.round_values(main["W"], main["V"], C.HYPER_SAMPLES, rng)
    curve = R.best_of_k_curve(values, C.CURVE_KS)
    return {"values": values, "mean": float(values.mean()), "stderr": float(values.std() / math.sqrt(len(values))), "expected": main["expected"], "curve": dict(zip(C.CURVE_KS, curve)), "max": float(values.max())}


# ----------------------------------------------------------------------------- Experiment 4: Familien

def families() -> list:
    return [("Baum (n = 12)", G.tree(12, 0)), ("vollständig bipartit K₆,₆", G.bipartite(6, 6)), ("Gitter 4×4", G.grid(4, 4)), ("Kreis C₄", G.cycle(4)), ("Kreis C₅", G.cycle(5)), ("Kreis C₆", G.cycle(6)),
            ("Kreis C₇", G.cycle(7)), ("vollständig K₃", G.complete(3)), ("vollständig K₄", G.complete(4)), ("vollständig K₅", G.complete(5)), ("vollständig K₆", G.complete(6)), ("Petersen-Graph", G.petersen()),
            ("Rad W₉", G.wheel(9)), ("Zufallsgraph G(12, ½)", G.gnp(12, 0.5, C.DEFAULT_SEED))]


def family_rows() -> list:
    rows = []
    with threadpool_limits(1):
        for name, W in families():
            res = S.solve(W, 1e-9)
            exact = R.exact_maxcut(W)[0]
            rows.append({"name": name, "n": len(W), "edges": len(G.edges(W)), "sdp": res.value, "exact": exact, "exact_ratio": exact / res.value, "expected": R.expected_rounding(W, res.X),
                         "expected_ratio": R.expected_rounding(W, res.X) / res.value, "rank": res.rank, "exact_sdp": abs(exact - res.value) < 1e-6})
    return rows


# ----------------------------------------------------------------------------- Experiment 5: Dual und Rang

def rank_rows(seed: int = C.DEFAULT_SEED) -> list:
    rows = []
    with threadpool_limits(1):
        for n in C.RANK_NS:
            res = S.solve(G.gnp(n, 0.5, seed), 1e-9)
            rows.append({"n": n, "rank": res.rank, "bound": math.sqrt(2.0 * n), "seconds": res.seconds, "iterations": res.iterations})
    return rows


# ----------------------------------------------------------------------------- Verdikte (reine Funktionen: (Art, Text))

def verdict_main(s: Settings, v: dict) -> tuple:
    res = v["res"]
    head = f"{graph_name(s)} (Seed {s.seed}, {v['edges']} Kanten): die SDP-Schranke ist **{num(res.value)}**"
    if v["exact"] is not None:
        head += f", der exakte größte Schnitt {num(v['exact'])} (Verhältnis {de(v['exact'] / res.value, 4)})"
    head += f", die erwartete Rundung {num(v['expected'])} ({de(v['expected'] / res.value, 4)} der Schranke; Garantie {de(R.ALPHA_GW, 5)}), der beste von {s.k} Hyperebenen {num(v['best'])}"
    if v["exact"] is not None:
        head += " und damit " + ("**der exakte Schnitt**" if abs(v["best"] - v["exact"]) < 1e-9 else f"{de(v['best'] / v['exact'], 4)} des exakten Schnitts")
    else:
        head += f" ({de(v['best'] / res.value, 4)} der Schranke; für n > {C.EXACT_MAX} gibt es keinen exakten Wert)"
    tail = f". Rang der Lösung {res.rank}, {res.iterations} Iterationen, Schranke aus dem Dual {num(res.dual_bound)}."
    ok = v["expected"] >= R.ALPHA_GW * res.value - 1e-6 and (v["exact"] is None or v["exact"] <= res.value + 1e-6)
    return ("success" if ok else "warning"), head + tail


def verdict_cycle(c: dict) -> tuple:
    n = c["n"]
    odd = n % 2 == 1
    text = (f"Kreis C{_sub(n)}: die SDP-Schranke ist {num(c['res'].value)}, der exakte größte Schnitt {num(c['exact'])} (Verhältnis {de(c['ratio'], 5)}). Benachbarte Einheitsvektoren haben den Winkel {de(c['angles'][0], 1)}°, die Hyperebene "
            f"schneidet jede Kante mit Wahrscheinlichkeit {de(c['edge_probability'], 3)}, der erwartete Schnitt ist {num(c['expected'])}.")
    if odd:
        text += f" Der ungerade Kreis hat keinen Schnitt, der alle Kanten trifft: die Relaxation ist nicht exakt (Rang {c['res'].rank})."
    else:
        text += f" Der gerade Kreis ist bipartit: die Relaxation ist exakt (Rang {c['res'].rank}), die Vektoren liegen auf einer Geraden."
    return "info", text


def verdict_stat(rows: list, s: Settings) -> tuple:
    st = stat_summary(rows)
    n = min(s.n, C.STAT_N_MAX)
    below = int(np.sum(st["expected"] < R.ALPHA_GW))
    return "info", (f"{st['count']} Zufallsgraphen G({n}, {p_text(s.p)}): die erwartete Rundung liegt bei {de(st['expected_mean'], 3)} der Schranke im Mittel, am schwächsten bei {de(st['expected_min'], 3)}; die Garantie {de(R.ALPHA_GW, 5)} "
                    + (f"unterschreitet kein Graph (0 von {st['count']})" if below == 0 else f"unterschreiten {below} von {st['count']} Graphen") + f". Der exakte Schnitt liegt bei {de(st['exact_mean'], 3)} der Schranke im Mittel, am weitesten darunter bei {de(st['exact_min'], 3)}: "
                    f"die Schranke ist nah am Schnitt, die Garantie ist selten scharf.")


def verdict_rounding(r: dict, s: Settings, main: dict) -> tuple:
    c = r["curve"]
    text = (f"Über {C.HYPER_SAMPLES} Hyperebenen: Mittelwert {de(r['mean'], 4)} ± {de(r['stderr'], 4)}, die Formel Σ w arccos(X)/π gibt {de(r['expected'], 4)}; der beste Schnitt der Stichprobe ist {num(r['max'])}"
            + (f" (exakt {num(main['exact'])})" if main["exact"] is not None else "") + ". Der aus den Hyperebenen in Blöcken geschätzte Erwartungswert des besten von k wächst von "
            + f"{de(c[1], 3)} (k = 1) über {de(c[10], 3)} (k = 10) auf {de(c[100], 3)} (k = 100): wenige Hyperebenen genügen auf diesem Graphen.")
    return "info", text


def verdict_families(rows: list) -> tuple:
    exact = [r["name"] for r in rows if r["exact_sdp"]]
    gap = [r for r in rows if not r["exact_sdp"]]
    rank1 = [r["name"] for r in rows if r["rank"] == 1]
    high = [r for r in rows if r["exact_sdp"] and r["rank"] > 1]
    worst = min(rows, key=lambda r: r["expected_ratio"])
    text = (f"Die Schranke ist exakt bei {len(exact)} von {len(rows)} Familien; Rang 1 haben {', '.join(rank1)}. "
            + ("Die Umkehrung „exakt ⇒ Rang 1“ gilt nicht: " + ", ".join(f"{r['name']} (Rang {r['rank']})" for r in high) + " haben eine exakte Schranke, aber eine Lösung mit höherem Rang (der Löser liefert den Mittelpunkt der optimalen Menge). " if high else "")
            + f"Das kleinste Verhältnis von erwarteter Rundung zu Schranke hat {worst['name']} mit {de(worst['expected_ratio'], 4)}; das kleinste Verhältnis exakt zu Schranke hat "
            + f"{min(gap, key=lambda r: r['exact_ratio'])['name']} mit {de(min(r['exact_ratio'] for r in gap), 5)}.")
    return "info", text


def verdict_dual(main: dict, s: Settings, ranks: list) -> tuple:
    res = main["res"]
    text = (f"Die Dualmatrix Z = Diag(d) + W/2 ist positiv semidefinit (kleinster Eigenwert {sci(res.z_min_eig)}, Außerdiagonale gleich wᵢⱼ/2 auf {sci(res.z_offdiag)}); sie liefert die Schranke (Σ w + tr Z)/2 = {num(res.dual_bound)} "
            f"gleich dem SDP-Wert {num(res.value)} (Unterschied {sci(abs(res.gap))}): ein bewiesener oberer Wert für jeden Schnitt" + (f", der exakte Schnitt ist {num(main['exact'])}" if main["exact"] is not None else "") + ". "
            f"Der Rang der Lösung des inneren Punkts auf G(n, ½) ist " + ", ".join(f"{r['rank']} (n = {r['n']})" for r in ranks) + " gegenüber √(2n) = " + ", ".join(de(r["bound"], 1) for r in ranks)
            + ": weit unter der Schranke, die Stück 13 (Burer–Monteiro) braucht.")
    return ("success" if res.z_min_eig > -1e-7 and abs(res.gap) < 1e-6 else "warning"), text


_SUBS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _sub(n: int) -> str:
    return str(n).translate(_SUBS)
