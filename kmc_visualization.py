"""Plotly-Abbildungen. Alle 2D-Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

import kmc_constants as C
import kmc_graphs as G
import kmc_rounding as R


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=True):
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _square(fig, rng_=1.35):
    fig.update_xaxes(range=[-rng_, rng_], visible=False)
    fig.update_yaxes(range=[-rng_, rng_], visible=False, scaleanchor="x", scaleratio=1)
    return fig


# ----------------------------------------------------------------------------- Experiment 1: die Einheitsvektoren

def build_vectors(c: dict) -> go.Figure:
    P_, r = c["points"], c["hyperplane"]
    side = P_ @ r >= 0
    fig = go.Figure()
    t = np.linspace(0, 2 * math.pi, 200)
    fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color="#dddddd", width=1), hoverinfo="skip", showlegend=False))
    d = np.array([-r[1], r[0]])
    fig.add_trace(go.Scatter(x=[-1.3 * d[0], 1.3 * d[0]], y=[-1.3 * d[1], 1.3 * d[1]], mode="lines", name="Hyperebene (Gerade durch 0)", line=dict(color=C.COLORS["cut"], width=3, dash="dash")))
    for flag, color, name in ((True, C.COLORS["side_a"], "Seite +"), (False, C.COLORS["side_b"], "Seite −")):
        idx = [i for i in range(len(P_)) if bool(side[i]) == flag]
        fig.add_trace(go.Scatter(x=[P_[i, 0] for i in idx], y=[P_[i, 1] for i in idx], mode="markers+text", text=[str(i) for i in idx], textposition="top center", name=name,
                                 marker=dict(size=13, color=color), hovertemplate="Knoten %{text}<extra></extra>"))
    for i in range(len(P_)):
        fig.add_annotation(x=P_[i, 0], y=P_[i, 1], ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#888888")
    return _base(_square(fig))


# ----------------------------------------------------------------------------- Experiment 2: der Graph mit dem Schnitt

def build_graph(W: np.ndarray, s: np.ndarray, title: str = "") -> go.Figure:
    n = len(W)
    pos = G.circle_layout(n)
    cut_x, cut_y, un_x, un_y = [], [], [], []
    for i, j, _ in G.edges(W):
        target = (cut_x, cut_y) if s[i] != s[j] else (un_x, un_y)
        target[0].extend([pos[i, 0], pos[j, 0], None])
        target[1].extend([pos[i, 1], pos[j, 1], None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=un_x, y=un_y, mode="lines", name="Kante nicht im Schnitt", line=dict(color=C.COLORS["uncut"], width=1.2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=cut_x, y=cut_y, mode="lines", name="Kante im Schnitt", line=dict(color=C.COLORS["cut"], width=2), hoverinfo="skip"))
    for flag, color, name in ((1.0, C.COLORS["side_a"], "Seite +"), (-1.0, C.COLORS["side_b"], "Seite −")):
        idx = [i for i in range(n) if s[i] == flag]
        fig.add_trace(go.Scatter(x=[pos[i, 0] for i in idx], y=[pos[i, 1] for i in idx], mode="markers+text", text=[str(i) for i in idx], textposition="middle center", name=name,
                                 marker=dict(size=22, color=color), textfont=dict(color="white", size=11), hovertemplate="Knoten %{text}<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(size=14)))
    return _base(_square(fig, 1.25), height=420)


def build_ratios(st: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=st["expected"], name="erwartete Rundung / Schranke", marker_color=C.COLORS["expected"], opacity=0.75, xbins=dict(size=0.01)))
    fig.add_trace(go.Histogram(x=st["exact"], name="exakter Schnitt / Schranke", marker_color=C.COLORS["exact"], opacity=0.75, xbins=dict(size=0.01)))
    fig.add_vline(x=R.ALPHA_GW, line=dict(color=C.COLORS["alpha"], dash="dash"), annotation_text="Garantie 0,87857", annotation_position="top left")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Verhältnis zur SDP-Schranke", range=[0.85, 1.005])
    fig.update_yaxes(title_text="Graphen")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 3: die Rundung

def build_hist(r: dict, main: dict) -> go.Figure:
    fig = go.Figure(go.Histogram(x=r["values"], name="Schnittwert einer Hyperebene", marker_color=C.COLORS["sdp"], opacity=0.8, nbinsx=40))
    fig.add_vline(x=main["res"].value, line=dict(color=C.COLORS["sdp"], dash="dash"), annotation_text="SDP-Schranke", annotation_position="top left")
    fig.add_vline(x=r["expected"], line=dict(color=C.COLORS["expected"], width=3), annotation_text="Erwartungswert (Formel)", annotation_position="top right")
    if main["exact"] is not None:
        fig.add_vline(x=main["exact"], line=dict(color=C.COLORS["exact"], dash="dot", width=3), annotation_text="exakt", annotation_position="bottom right")
    fig.update_xaxes(title_text="Schnittwert")
    fig.update_yaxes(title_text=f"Hyperebenen (von {C.HYPER_SAMPLES})")
    return _base(fig)


def build_curve(r: dict, main: dict) -> go.Figure:
    ks = list(r["curve"])
    fig = go.Figure(go.Scatter(x=ks, y=[r["curve"][k] for k in ks], mode="lines+markers", name="bester von k (Mittel)", line=dict(color=C.COLORS["expected"], width=3), hovertemplate="k = %{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=main["res"].value, line=dict(color=C.COLORS["sdp"], dash="dash"), annotation_text="SDP-Schranke")
    if main["exact"] is not None:
        fig.add_hline(y=main["exact"], line=dict(color=C.COLORS["exact"], dash="dot"), annotation_text="exakt")
    fig.update_xaxes(title_text="Zahl der Hyperebenen k", type="log", tickvals=ks, ticktext=[str(k) for k in ks])
    fig.update_yaxes(title_text="Schnittwert")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 4 und 5

def build_families(rows: list) -> go.Figure:
    names = [r["name"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[r["exact_ratio"] for r in rows], name="exakter Schnitt / Schranke", marker_color=C.COLORS["exact"]))
    fig.add_trace(go.Bar(x=names, y=[r["expected_ratio"] for r in rows], name="erwartete Rundung / Schranke", marker_color=C.COLORS["expected"]))
    fig.add_hline(y=R.ALPHA_GW, line=dict(color=C.COLORS["alpha"], dash="dash"), annotation_text="Garantie 0,87857", annotation_position="bottom right")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Verhältnis", range=[0.84, 1.01])
    fig.update_xaxes(tickangle=-40)
    return _base(fig, height=420)


def build_rank(rows: list) -> go.Figure:
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["rank"] for r in rows], mode="lines+markers", name="Rang der Lösung", line=dict(color=C.COLORS["sdp"], width=3)))
    fig.add_trace(go.Scatter(x=ns, y=[r["bound"] for r in rows], mode="lines", name="√(2n)", line=dict(color="#888888", dash="dot", width=2)))
    fig.update_xaxes(title_text="Knoten n (G(n, ½))", tickvals=ns)
    fig.update_yaxes(title_text="Rang", rangemode="tozero")
    return _base(fig)
