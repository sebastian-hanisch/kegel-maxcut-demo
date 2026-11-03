"""Konstanten: Regler-Grenzen, feste Reihen der Experimente, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 4, 30, 12
P_CHOICES = (0.1, 0.2, 0.3, 0.5, 0.8)
DEFAULT_P = 0.5
WEIGHTS = ("eins", "zufall")
WEIGHT_NAMES = {"eins": "alle Gewichte 1", "zufall": "Zufallsgewichte in (0, 1]"}
DEFAULT_WEIGHTS = "eins"
K_CHOICES = (1, 5, 10, 50, 100)                          # Zahl der Hyperebenen der Rundung
DEFAULT_K = 50
CYCLE_CHOICES = (3, 4, 5, 6, 7, 9)                       # Kreislänge in Experiment 1
DEFAULT_CYCLE = 5
SEED_MAX, DEFAULT_SEED = 999999, 3
EXACT_MAX = 20                                           # größtes n mit exakter Aufzählung (kmc_rounding.EXACT_MAX)
STAT_GRAPHS = 30                                         # Zufallsgraphen der Verteilung in Experiment 2
STAT_N_MAX = 16                                          # größtes n dort (mit exaktem Schnitt)
HYPER_SAMPLES = 2000                                     # Hyperebenen in Experiment 3
CURVE_KS = (1, 2, 5, 10, 20, 50, 100)
RANK_NS = (8, 12, 16, 20, 24)                            # n der Rang-Reihe in Experiment 5
ROUND_SEED_OFFSET = 1000                                 # die Hyperebenen kommen aus Seed + Offset: dieselbe Wolke, dieselbe Rundung

COLORS = {"cut": "#c0504d", "uncut": "#bbbbbb", "side_a": "#2F6B65", "side_b": "#b08a2e", "sdp": "#4c78a8", "exact": "#2F6B65", "expected": "#b08a2e", "alpha": "#c0504d"}
COLOR_OK = "#2F6B65"
COLOR_WARN = "#b08a2e"
COLOR_BAD = "#b34a4a"

_BASE = {"n": DEFAULT_N, "p": DEFAULT_P, "seed": DEFAULT_SEED, "weights": DEFAULT_WEIGHTS, "k": DEFAULT_K, "cycle": DEFAULT_CYCLE}
PRESETS = {
    "Standardfall: G(12, ½), Gewichte 1": dict(_BASE),
    "Dünner Graph (p = 0,1): fast exakt": dict(_BASE, p=0.1, n=16),
    "Dichter Graph (p = 0,8)": dict(_BASE, p=0.8),
    "Zufallsgewichte": dict(_BASE, weights="zufall"),
    "Größter Graph mit exaktem Schnitt (n = 20)": dict(_BASE, n=20),
    "Ungerader Kreis C₇ (Experiment 1)": dict(_BASE, cycle=7),
}
PRESET_HELP = {
    "Standardfall: G(12, ½), Gewichte 1": "Ein Zufallsgraph mit zwölf Knoten und 27 Kanten: Schranke 20,8082, exakter Schnitt 20, erwartete Rundung 19,2717 (0,9262 der Schranke); der beste von 50 Hyperebenen trifft den exakten Schnitt.",
    "Dünner Graph (p = 0,1): fast exakt": "Wenige Kanten (9 bei n = 16): Schranke 8,25, exakter Schnitt 8, die erwartete Rundung ist 0,9697 der Schranke; bei dünnen Graphen ist die Relaxation fast exakt.",
    "Dichter Graph (p = 0,8)": "Viele Kanten (53 bei n = 12): die erwartete Rundung liegt bei 0,9498 der Schranke, näher als bei mittlerer Dichte (0,9262 bei p = 0,5).",
    "Zufallsgewichte": "Gewichte gleichverteilt in (0, 1]: Schranke 10,8628, exakter Schnitt 10,6643, erwartete Rundung 0,9258 der Schranke: die Verhältnisse ändern sich gegenüber den Gewichten 1 (0,9262) kaum.",
    "Größter Graph mit exaktem Schnitt (n = 20)": "Zwanzig Knoten und 88 Kanten: der größte Schnitt 61 wird durch Aufzählen aller 2¹⁹ Schnitte bestimmt, die Schranke ist 63,1334; darüber gibt es nur Schranke und Rundung.",
    "Ungerader Kreis C₇ (Experiment 1)": "Der Kreis mit sieben Knoten: Schranke 6,6534, bester Schnitt 6; die erwartete Rundung ist 6, also exakt (Experiment 1).",
}
assert set(PRESETS) == set(PRESET_HELP) and all(N_MIN <= p["n"] <= N_MAX and p["p"] in P_CHOICES and p["weights"] in WEIGHTS and p["k"] in K_CHOICES and p["cycle"] in CYCLE_CHOICES for p in PRESETS.values())
