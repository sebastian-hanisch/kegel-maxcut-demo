"""PDF-Export der aktuellen Relaxation (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import kmc_rounding as R
from kmc_evaluation import Settings, de, graph_name, num, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "ν": "nu", "λ": "lambda", "β": "beta", "α": "alpha", "ρ": "rho", "μ": "mu", "τ": "tau", "θ": "theta", "κ": "kappa",
            "€": "EUR", "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "„": '"', "“": '"', "ᵀ": "^T", "ℓ": "l", "∞": "inf", "√": "sqrt", "‖": "|",
            "°": "o", "⟨": "<", "⟩": ">", "½": "1/2", "⪰": ">="}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(s: Settings, v: dict) -> bytes:
    res = v["res"]
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("MaxCut: SDP-Relaxation und Rundung"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{graph_name(s)}, Seed {s.seed}, {v['edges']} Kanten, {s.k} Hyperebenen. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    rows = [("SDP-Schranke", num(res.value)), ("Schranke aus dem Dual", num(res.dual_bound)), ("exakter größter Schnitt", num(v["exact"]) if v["exact"] is not None else "nicht berechnet (n > 20)"),
            ("erwartete Rundung (Formel)", de(v["expected"], 4)), (f"bester von {s.k} Hyperebenen", num(v["best"])), ("erwartete Rundung / Schranke", de(v["expected"] / res.value, 4)),
            ("Garantie (Goemans, Williamson)", de(R.ALPHA_GW, 5)), ("Rang der Lösung", str(res.rank)), ("Iterationen des Lösers", str(res.iterations)), ("Dual: kleinster Eigenwert von Z", sci(res.z_min_eig))]
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(90, 6, "Kennzahl", border=1)
    pdf.cell(100, 6, "Wert", border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for k, val in rows:
        pdf.cell(90, 5.5, _s(k), border=1)
        pdf.cell(100, 5.5, _s(val), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: kegel-maxcut-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
