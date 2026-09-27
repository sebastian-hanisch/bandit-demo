"""Plotly-Abbildungen der Bandit-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go

import bd_algorithms as A
import bd_constants as C

METHOD_COLORS = {"random": "#9e9e9e", "greedy": "#d62728", "epsilon_greedy": "#e6550d", "ucb1": "#1f77b4", "thompson": "#2e7d32"}
TRUE_COLOR = "#14233B"
SHORT = A.METHOD_SHORT


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def build_arms(bandit, arms, rewards, step, method):
    """Balken je Station bis zur gewählten Runde: Anteil erfolgreicher Ziehungen (Balkenhöhe), Zahl der Ziehungen (Beschriftung),
    die wahre Erfolgswahrscheinlichkeit als Raute, die beste Station hervorgehoben."""
    K = bandit.K
    counts = np.zeros(K, dtype=int)
    sums = np.zeros(K)
    if step > 0:
        for a, r in zip(arms[:step], rewards[:step]):
            counts[a] += 1
            sums[a] += r
    mean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    colors = ["rgba(31,119,180,0.75)" if i != bandit.best else "rgba(46,125,50,0.8)" for i in range(K)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(K)), y=mean, marker=dict(color=colors), text=[f"n={c}" for c in counts], textposition="outside", cliponaxis=False, name="geschätzter Erfolgsanteil"))
    fig.add_trace(go.Scatter(x=list(range(K)), y=bandit.theta, mode="markers", marker=dict(symbol="diamond", size=11, color=TRUE_COLOR), name="wahre Erfolgswahrscheinlichkeit"))
    fig.update_xaxes(title_text="Station", tickmode="linear", dtick=1)
    fig.update_yaxes(title_text="Erfolgsanteil", range=[0, 1.05])
    return _base(fig, 320)


def build_regret_comparison(cum_regret, upto=None, highlight=None):
    """Kumuliertes Regret über die Runden, ein Verfahren je Linie."""
    fig = go.Figure()
    for m in A.METHODS:
        y = cum_regret[m]
        x = np.arange(1, len(y) + 1)
        if upto is not None:
            x, y = x[:upto], y[:upto]
        fig.add_trace(go.Scatter(x=x, y=y, name=SHORT[m], mode="lines", line=dict(color=METHOD_COLORS[m], width=2.4 if m == highlight else 1.6)))
    fig.update_xaxes(title_text="Runde")
    fig.update_yaxes(title_text="kumuliertes Regret", rangemode="tozero")
    return _base(fig, 340)


def build_growth(exp):
    """Regret an mehreren Zeitpunkten, doppelt-logarithmische Achsen: eine Gerade mit Steigung 1 heißt linear, eine flache Kurve
    heißt (an)logarithmisch."""
    cps = exp["checkpoints"]
    fig = go.Figure()
    for m in A.METHODS:
        r = exp["rows"][m]
        ys = [r["checkpoints"][t][0] for t in cps]
        se = [r["checkpoints"][t][1] for t in cps]
        fig.add_trace(go.Scatter(x=list(cps), y=ys, name=f"{SHORT[m]} (Steigung {de(r['slope'], 2)})", mode="lines+markers", line=dict(color=METHOD_COLORS[m], width=2), marker=dict(size=6),
                                 error_y=dict(type="data", array=se, visible=True, thickness=1, width=3)))
    fig.update_xaxes(title_text="Runde (log)", type="log")
    fig.update_yaxes(title_text="kumuliertes Regret (log)", type="log")
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.35))


def build_epsilon(exp):
    """Mittleres Regret je Epsilon-Stufe (Balken) und Anteil der Seeds, die auf einer schlechten Station hängen geblieben sind (Linie, zweite Achse)."""
    levels = exp["levels"]
    labels = [("0 (gierig)" if e == 0.0 else ("1 (zufällig)" if e == 1.0 else de(e, 2))) for e in levels]
    y = [exp["rows"][e]["regret"][0] for e in levels]
    se = [exp["rows"][e]["regret"][1] for e in levels]
    stuck = [100 * exp["rows"][e]["stuck_share"] for e in levels]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=y, name="Regret am Ende", marker=dict(color="#1f77b4"), error_y=dict(type="data", array=se, visible=True, thickness=1, width=3), yaxis="y1"))
    fig.add_trace(go.Scatter(x=labels, y=stuck, name="Anteil hängen geblieben", mode="lines+markers", line=dict(color="#d62728", width=2), marker=dict(size=7), yaxis="y2"))
    fig.update_layout(yaxis=dict(title="Regret am Ende (Runde " + str(exp["T"]) + ")", rangemode="tozero"), yaxis2=dict(title="Anteil hängen geblieben (%)", overlaying="y", side="right", range=[0, 100], showgrid=False))
    fig.update_xaxes(title_text="Epsilon")
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.3))


def build_gap(exp):
    """Regret je Abstand zwischen bester und zweitbester Station, UCB1 gegen Thompson Sampling."""
    levels = exp["levels"]
    fig = go.Figure()
    for m, color in (("ucb1", METHOD_COLORS["ucb1"]), ("thompson", METHOD_COLORS["thompson"])):
        y = [exp["rows"][(g, m)][0] for g in levels]
        se = [exp["rows"][(g, m)][1] for g in levels]
        fig.add_trace(go.Scatter(x=list(levels), y=y, name=SHORT[m], mode="lines+markers", line=dict(color=color, width=2.2), marker=dict(size=7),
                                 error_y=dict(type="data", array=se, visible=True, thickness=1, width=3)))
    fig.update_xaxes(title_text="Abstand beste – zweitbeste Station")
    fig.update_yaxes(title_text=f"Regret am Ende (Runde {exp['T']})", rangemode="tozero")
    return _base(fig, 340)
