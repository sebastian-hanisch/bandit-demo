"""Jede Zahl aus README.md und den Preset-Hilfen wird hier nachgerechnet. Mehrere feste Seeds -> Bänder, die plattformbedingte
Rundungsunterschiede des Trainings vertragen; Einzelläufe (Presets, Seed 3) nur mit großzügigen Bändern."""

import pytest

import bd_algorithms as A
import bd_evaluation as E
import bd_presets as P


def _settings(p):
    return E.Settings(p["K"], p["gap"], p["spread"], p["T"], p["eps"], p["seed"])


@pytest.fixture(scope="module")
def growth():
    return E.growth_experiment()


@pytest.fixture(scope="module")
def epsilon():
    return E.epsilon_experiment()


@pytest.fixture(scope="module")
def gap():
    return E.gap_experiment()


@pytest.fixture(scope="module")
def presets():
    return {name: E.analyse(_settings(p)) for name, p in P.PRESETS.items()}


def _m(a, m):
    return a.cum_regret[m][-1], a.best_share_window[m]


# --- Wachstum: Steigungen und der Gierig-gegen-UCB1-Überkreuzungspunkt ---------------------------------------------------------------------------

def test_growth_slopes_match_the_expected_order(growth):
    sl = {m: growth["rows"][m]["slope"] for m in A.METHODS}
    assert sl["random"] == pytest.approx(1.00, abs=0.03)
    assert sl["greedy"] == pytest.approx(0.92, abs=0.05)
    assert sl["epsilon_greedy"] == pytest.approx(0.72, abs=0.05)
    assert sl["ucb1"] == pytest.approx(0.50, abs=0.05)
    assert sl["thompson"] == pytest.approx(0.22, abs=0.05)
    assert sl["thompson"] < sl["ucb1"] < sl["epsilon_greedy"] < sl["greedy"] < sl["random"] + 0.01


def test_greedy_leads_ucb1_early_but_falls_behind_later(growth):
    g, u = growth["rows"]["greedy"]["checkpoints"], growth["rows"]["ucb1"]["checkpoints"]
    assert g[10000][0] == pytest.approx(445, abs=40) and u[10000][0] == pytest.approx(466, abs=15)
    assert g[10000][0] < u[10000][0]                                                        # bei 10 000 liegt Gierig im Mittel noch vorn
    assert g[15000][0] == pytest.approx(663, abs=60) and u[15000][0] == pytest.approx(526, abs=15)
    assert g[15000][0] > u[15000][0]                                                        # bei 15 000 hat UCB1 aufgeholt und überholt


# --- Erkundungsrate: Sweet Spot und Gierig-Festlegungsrisiko -------------------------------------------------------------------------------------

def test_epsilon_sweet_spot_and_greedy_lockin_risk(epsilon):
    r = epsilon["rows"]
    assert r[0.0]["stuck_share"] == pytest.approx(0.40, abs=0.12) and r[0.0]["regret"][0] == pytest.approx(1031, rel=0.3)
    assert r[0.0]["regret"][1] > 100                                                        # riesige Streuung bei reinem Ausnutzen
    best = min(epsilon["levels"], key=lambda e: r[e]["regret"][0])
    assert best == pytest.approx(0.05, abs=1e-9)
    assert r[0.05]["regret"][0] == pytest.approx(388, rel=0.3)
    assert r[0.1]["regret"][0] == pytest.approx(547, rel=0.2) and r[0.1]["stuck_share"] == pytest.approx(0.0, abs=0.05)
    assert r[0.1]["regret"][1] < r[0.0]["regret"][1] / 3                                     # deutlich zuverlässiger als Gierig


# --- Abstand: UCB1 und Thompson gegen die Schwierigkeit des Problems -------------------------------------------------------------------------

def test_gap_experiment_matches_the_theoretical_direction(gap):
    r = gap["rows"]
    g0, g1 = gap["levels"][0], gap["levels"][-1]
    assert r[(g0, "ucb1")][0] == pytest.approx(658, rel=0.15) and r[(g1, "ucb1")][0] == pytest.approx(415, rel=0.15)
    assert r[(g0, "thompson")][0] == pytest.approx(150, rel=0.2) and r[(g1, "thompson")][0] == pytest.approx(56, rel=0.2)
    assert r[(g0, "ucb1")][0] > r[(g1, "ucb1")][0] and r[(g0, "thompson")][0] > r[(g1, "thompson")][0]
    for lvl in gap["levels"]:
        assert r[(lvl, "thompson")][0] < r[(lvl, "ucb1")][0]                                 # Thompson bei jedem Abstand unter UCB1


# --- Presets (Seed 3, Einzelläufe) ------------------------------------------------------------------------------------------------------------

def test_standard_preset(presets):
    a = presets["Standardfall"]
    reg, share = _m(a, "greedy")
    assert reg == pytest.approx(11.3, abs=1.0) and share == pytest.approx(1.0)
    reg, share = _m(a, "thompson")
    assert reg == pytest.approx(72.5, abs=3.0) and share == pytest.approx(0.87, abs=0.03)
    reg, share = _m(a, "ucb1")
    assert reg == pytest.approx(142.9, abs=5.0) and share == pytest.approx(0.55, abs=0.03)
    reg, share = _m(a, "epsilon_greedy")
    assert reg == pytest.approx(32.1, abs=2.0) and share == pytest.approx(0.86, abs=0.03)
    reg, share = _m(a, "random")
    assert reg == pytest.approx(244.4, abs=5.0) and share == pytest.approx(0.09, abs=0.03)


def test_small_gap_preset(presets):
    a = presets["Kleiner Abstand (schwer zu unterscheiden)"]
    reg, share = _m(a, "ucb1")
    assert share == pytest.approx(0.37, abs=0.03) and reg == pytest.approx(106.9, abs=5.0)
    reg, share = _m(a, "thompson")
    assert share == pytest.approx(0.92, abs=0.03) and reg == pytest.approx(25.2, abs=3.0)


def test_large_gap_preset(presets):
    a = presets["Großer Abstand (leicht)"]
    reg, share = _m(a, "ucb1")
    assert share == pytest.approx(0.86, abs=0.03) and reg == pytest.approx(148.1, abs=5.0)
    reg, share = _m(a, "thompson")
    assert share == pytest.approx(0.99, abs=0.03) and reg == pytest.approx(21.8, abs=3.0)


def test_many_arms_preset(presets):
    a = presets["Viele Stationen"]
    reg, share = _m(a, "ucb1")
    assert share == pytest.approx(0.23, abs=0.03) and reg == pytest.approx(216.9, abs=8.0)


def test_greedy_epsilon_presets_reduce_correctly(presets):
    zero, wide = presets["Gierig, kein Erkunden"], presets["Viel Erkunden (fast zufällig)"]
    r0, s0 = _m(zero, "epsilon_greedy")
    rg, sg = _m(zero, "greedy")
    assert r0 == pytest.approx(rg) and s0 == pytest.approx(sg)                               # Epsilon 0 == Gierig, exakt
    rw, sw = _m(wide, "epsilon_greedy")
    assert sw == pytest.approx(0.57, abs=0.03) and rw == pytest.approx(126.6, abs=5.0)
