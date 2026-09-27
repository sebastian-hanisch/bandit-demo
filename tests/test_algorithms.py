"""Die einzelnen Formeln von Hand nachgerechnet (Bewertung, UCB1, Thompson-Posterior, Gleichstand), dazu `run()` auf einem Banditen ohne
Rauschen (Erfolgswahrscheinlichkeit 0 oder 1): dort ist die Belohnung nie zufällig, nur der Gleichstand zu Beginn - das macht die
gesamte Spur bis auf diesen einen Zufallszug exakt nachrechenbar."""

import math

import numpy as np
import pytest

import bd_algorithms as A
import bd_bandit as B


def test_estimate_greedy_by_hand():
    counts = np.array([0, 3, 2])
    sums = np.array([0.0, 2.0, 1.0])
    est = A.estimate_greedy(counts, sums)
    assert est[0] == np.inf and est[1] == pytest.approx(2 / 3) and est[2] == pytest.approx(0.5)


def test_estimate_ucb1_by_hand():
    # zwei Stationen je einmal gezogen (Erfolg, Misserfolg), t = 2 Runden bereits gespielt.
    counts = np.array([1, 1])
    sums = np.array([1.0, 0.0])
    est = A.estimate_ucb1(counts, sums, t=2)
    bonus = math.sqrt(2 * math.log(2) / 1)
    assert est[0] == pytest.approx(1.0 + bonus) and est[1] == pytest.approx(0.0 + bonus)


def test_estimate_ucb1_never_pulled_is_infinite_regardless_of_t():
    est = A.estimate_ucb1(np.array([0, 5]), np.array([0.0, 3.0]), t=100)
    assert est[0] == np.inf and np.isfinite(est[1])


def test_ucb1_bonus_shrinks_with_more_pulls_at_fixed_t():
    est = A.estimate_ucb1(np.array([1, 4]), np.array([0.5, 2.0]), t=50)
    bonus0 = est[0] - 0.5
    bonus1 = est[1] - 0.5
    assert bonus0 > bonus1 > 0                                                              # weniger gezogen -> größerer Vertrauensbonus, beide Mittelwerte gleich (0.5)


def test_thompson_sample_posterior_by_hand():
    rng = np.random.default_rng(0)
    counts = np.array([10, 10])
    sums = np.array([9.0, 1.0])                                                             # Station 0: 9/10 Erfolge, Station 1: 1/10
    wins = np.array([np.argmax(A.thompson_sample(counts, sums, rng)) for _ in range(500)])
    assert (wins == 0).mean() > 0.95                                                        # Beta(10,2) liegt weit über Beta(2,10)


def test_thompson_sample_with_no_data_is_uniform_beta_1_1():
    rng = np.random.default_rng(1)
    draws = A.thompson_sample(np.array([0, 0]), np.array([0.0, 0.0]), rng)
    assert ((draws >= 0) & (draws <= 1)).all()
    many = np.concatenate([A.thompson_sample(np.array([0, 0]), np.array([0.0, 0.0]), rng) for _ in range(2000)])
    assert many.mean() == pytest.approx(0.5, abs=0.03)                                      # Beta(1,1) = Gleichverteilung


def test_choose_breaks_ties_uniformly_and_deterministically_otherwise():
    rng = np.random.default_rng(0)
    assert A.choose(np.array([1.0, 5.0, 2.0]), rng) == 1                                    # eindeutiges Maximum, kein Zufall
    picks = np.array([A.choose(np.array([5.0, 5.0, 1.0]), np.random.default_rng(sd)) for sd in range(400)])
    assert (picks == 2).sum() == 0
    assert 0.35 < (picks == 0).mean() < 0.65                                                # ungefähr hälftig zwischen den beiden Bestplatzierten


def _noiseless(theta_best_first):
    """Ein Bandit mit Erfolgswahrscheinlichkeit exakt 0 oder 1 je Station: keine Zufallsbelohnung, nur der Anfangs-Gleichstand ist zufällig."""
    return B.Bandit(theta=np.array(theta_best_first, dtype=float), seed=0)


def test_noiseless_bandit_greedy_locks_onto_the_true_best_arm_forever_after_warmup():
    # Erfolgswahrscheinlichkeit exakt 0 oder 1: keine Rausch-Belohnung, nur der Anfangs-Gleichstand ist zufällig.
    bd = _noiseless([1.0, 0.0, 0.0])
    for seed in range(10):
        arms, rewards = A.run("greedy", bd, T=50, seed=seed)
        assert np.array_equal(rewards, bd.theta[arms])                                      # Belohnung == wahrer Wert, kein Rauschen
        assert (arms[3:] == 0).all()                                                        # nach der Aufwärmphase (3 Stationen) nur noch die beste - Gierig erkundet nie wieder
        assert arms[:3].tolist().count(0) == 1 and set(arms[:3].tolist()) == {0, 1, 2}       # jede Station genau einmal in den ersten 3 Runden


def test_noiseless_bandit_ucb1_mostly_exploits_but_provably_keeps_revisiting():
    # Dieselbe rauschfreie Konstruktion; UCB1 darf (muss laut Konstruktion) die schlechten Stationen gelegentlich erneut
    # ziehen (der Vertrauensbonus einer lange nicht gezogenen Station wächst unbeschränkt) - anders als Gierig.
    bd = _noiseless([1.0, 0.0, 0.0])
    for seed in range(10):
        arms, rewards = A.run("ucb1", bd, T=50, seed=seed)
        assert np.array_equal(rewards, bd.theta[arms])
        assert (arms[3:] == 0).mean() > 0.8                                                 # weit überwiegend die beste Station
        assert (arms[3:] != 0).any()                                                        # aber mindestens ein nachweisbarer Rückfall auf eine schlechte Station


def test_run_reproduces_estimate_ucb1_and_choose_step_by_step():
    # Unabhängige Gegenprobe: derselbe Ablauf, von Hand mit den einzeln getesteten Bausteinen `estimate_ucb1`/`choose`
    # nachgebaut, muss `run()`s Ergebnis bei gleichem Seed exakt reproduzieren.
    bd = B.generate(K=4, seed=7)
    T, seed = 60, 3
    rng = np.random.default_rng(seed)
    K = bd.K
    counts, sums = np.zeros(K, dtype=int), np.zeros(K)
    arms_expected, rewards_expected = np.empty(T, dtype=int), np.empty(T)
    for t in range(T):
        a = A.choose(A.estimate_ucb1(counts, sums, t), rng)
        r = float(rng.random() < bd.theta[a])
        counts[a] += 1
        sums[a] += r
        arms_expected[t], rewards_expected[t] = a, r
    arms, rewards = A.run("ucb1", bd, T, seed=seed)
    assert np.array_equal(arms, arms_expected) and np.array_equal(rewards, rewards_expected)


def test_epsilon_greedy_reduces_to_greedy_exactly_at_eps_zero():
    # eps=0.0 darf nicht einmal den Erkundungs-Münzwurf ziehen (sonst verschiebt sich der Zufallsstrom) - exakt dieselbe Spur wie Gierig.
    bd = B.generate(K=5, seed=2)
    for seed in range(5):
        a1, r1 = A.run("epsilon_greedy", bd, T=40, seed=seed, eps=0.0)
        a2, r2 = A.run("greedy", bd, T=40, seed=seed, eps=0.0)
        assert np.array_equal(a1, a2) and np.array_equal(r1, r2)


def test_epsilon_greedy_behaves_like_random_at_eps_one():
    # Bei eps=1.0 wird immer erkundet, aber ein zusätzlicher Münzwurf je Runde verschiebt den Zufallsstrom gegenüber "random" -
    # deshalb nur die GLEICHVERTEILUNG prüfen, nicht die exakte Spur.
    bd = B.generate(K=5, seed=2)
    arms, _ = A.run("epsilon_greedy", bd, T=5000, seed=0, eps=1.0)
    counts = np.bincount(arms, minlength=bd.K)
    assert (np.abs(counts / 5000 - 1 / bd.K) < 0.03).all()


def test_run_is_deterministic_given_a_seed_and_uses_every_arm_at_least_once_with_enough_rounds():
    bd = B.generate(K=6, seed=1)
    for m in A.METHODS:
        a1, r1 = A.run(m, bd, T=200, seed=5, eps=0.1)
        a2, r2 = A.run(m, bd, T=200, seed=5, eps=0.1)
        assert np.array_equal(a1, a2) and np.array_equal(r1, r2)
        assert set(a1.tolist()) == set(range(bd.K))
        assert set(np.unique(a1)) <= set(range(bd.K)) and len(a1) == 200


def test_regret_is_zero_only_when_the_best_arm_is_pulled():
    bd = B.Bandit(theta=np.array([0.2, 0.9, 0.5]), seed=0)
    g = A.regret(bd, np.array([1, 0, 2, 1]))
    assert np.allclose(g, [0.0, 0.7, 0.4, 0.0])
