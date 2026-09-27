"""Analyse und die drei Experimente: Aufbau, Kennzahlen von Hand, die erwarteten Wachstumsmuster (nicht die genauen Zahlen - die stehen in test_claims.py)."""

import numpy as np
import pytest

import bd_algorithms as A
import bd_evaluation as E


@pytest.fixture(scope="module")
def analysis():
    return E.analyse(E.Settings(T=500))


def test_analysis_shapes_and_keys(analysis):
    a = analysis
    assert set(a.arms) == set(A.METHODS) == set(a.rewards) == set(a.regret) == set(a.cum_regret) == set(a.best_share_window)
    for m in A.METHODS:
        assert a.arms[m].shape == (500,) and a.rewards[m].shape == (500,)
        assert np.array_equal(a.cum_regret[m], np.cumsum(a.regret[m]))


def test_best_share_window_by_hand(analysis):
    a = analysis
    window = int(round(0.1 * 500))
    for m in A.METHODS:
        expected = np.mean(a.arms[m][-window:] == a.bandit.best)
        assert a.best_share_window[m] == pytest.approx(expected)


def test_analyse_is_deterministic_given_the_same_settings():
    s = E.Settings(K=6, gap=0.08, seed=9, T=300)
    a1, a2 = E.analyse(s), E.analyse(s)
    for m in A.METHODS:
        assert np.array_equal(a1.arms[m], a2.arms[m])


def test_bandit_cache_reuses_the_same_object_for_the_same_key():
    s = E.Settings(K=7, gap=0.1, seed=4)
    assert E._bandit(s.bandit_key) is E._bandit(s.bandit_key)


def test_growth_experiment_shapes_and_qualitative_order():
    exp = E.growth_experiment(checkpoints=(50, 100, 200), seeds=range(6))
    assert exp["n_seeds"] == 6 and set(exp["rows"]) == set(A.METHODS)
    for m in A.METHODS:
        assert set(exp["rows"][m]["checkpoints"]) == {50, 100, 200}
    # UCB1 und Thompson wachsen zwischen den Prüfpunkten langsamer (in absoluten Zuwächsen) als Zufall - qualitative Reihenfolge, keine genauen Zahlen
    rnd = exp["rows"]["random"]["checkpoints"]
    ucb = exp["rows"]["ucb1"]["checkpoints"]
    tsm = exp["rows"]["thompson"]["checkpoints"]
    assert (ucb[200][0] - ucb[100][0]) < (rnd[200][0] - rnd[100][0])
    assert (tsm[200][0] - tsm[100][0]) < (ucb[200][0] - ucb[100][0]) + 1e-6


def test_epsilon_experiment_greedy_has_more_spread_and_gets_stuck_more_often():
    exp = E.epsilon_experiment(levels=(0.0, 0.2), seeds=range(30), T=3000)
    assert set(exp["rows"]) == {0.0, 0.2}
    # Gierig (0.0) hat eine viel größere Streuung über die Seeds als eine moderate Erkundungsrate: manche Läufe finden die beste
    # Station schnell (fast kein Regret mehr), andere legen sich dauerhaft auf eine schlechte fest (bleiben "hängen").
    assert exp["rows"][0.0]["regret"][1] > exp["rows"][0.2]["regret"][1]
    assert exp["rows"][0.0]["stuck_share"] > exp["rows"][0.2]["stuck_share"]


def test_gap_experiment_regret_grows_as_the_problem_gets_harder():
    exp = E.gap_experiment(levels=(0.05, 0.2), seeds=range(20), T=5000)
    for m in ("ucb1", "thompson"):
        assert exp["rows"][(0.05, m)][0] > exp["rows"][(0.2, m)][0]                          # kleinerer Abstand = schwerer = mehr Regret
