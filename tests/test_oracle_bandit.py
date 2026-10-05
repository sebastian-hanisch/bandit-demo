"""Orakel-Tests mit anderen Rechenwegen: Bewertungsformeln per Schleife, `run()` gegen eine Neuschreibung in reinem Python mit demselben Zufallsstrom, Thompson-Ziehung gegen die Beta-Verteilung
(Kolmogorov-Smirnov), erwartetes Regret von "Zufällig" und die Schranke von Auer et al. für UCB1, der Abstand des Vehikels über den ganzen Reglerbereich, die Experimente von Hand nachgerechnet."""

import math

import numpy as np
import pytest

import bd_algorithms as A
import bd_bandit as B
import bd_constants as C
import bd_evaluation as E


def _my_run(method, theta, T, seed, eps):
    r = np.random.default_rng(seed)
    K = len(theta)
    n, s, arms = [0] * K, [0.0] * K, []

    def pick(vals):
        top = [i for i, v in enumerate(vals) if v == max(vals)]
        return top[0] if len(top) == 1 else int(r.choice(np.array(top)))

    for t in range(T):
        mean = [s[k] / n[k] if n[k] else math.inf for k in range(K)]
        if method == "random":
            a = int(r.integers(K))
        elif method == "greedy":
            a = pick(mean)
        elif method == "epsilon_greedy":
            a = int(r.integers(K)) if eps > 0.0 and r.random() < eps else pick(mean)
        elif method == "ucb1":
            a = pick([math.inf if n[k] == 0 else mean[k] + math.sqrt(2 * math.log(max(t, 1)) / n[k]) for k in range(K)])
        else:
            a = pick(list(r.beta(np.array(s) + 1.0, np.array(n) - np.array(s) + 1.0)))
        x = float(r.random() < theta[a])
        n[a] += 1
        s[a] += x
        arms.append(a)
    return np.array(arms)


def test_estimates_match_the_formulas_by_loop():
    rng = np.random.default_rng(0)
    for _ in range(100):
        K = int(rng.integers(1, 8))
        counts = rng.integers(0, 20, K)
        sums = np.array([rng.integers(0, c + 1) for c in counts], float)
        t = int(rng.integers(0, 500))
        g, u = A.estimate_greedy(counts, sums), A.estimate_ucb1(counts, sums, t)
        for k in range(K):
            if counts[k] == 0:
                assert math.isinf(g[k]) and math.isinf(u[k])
            else:
                assert g[k] == pytest.approx(sums[k] / counts[k])
                assert u[k] == pytest.approx(sums[k] / counts[k] + math.sqrt(2 * math.log(max(t, 1)) / counts[k]))


def test_thompson_draws_follow_the_beta_posterior():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    for c, s in ((0, 0.0), (5, 1.0), (20, 15.0), (30, 30.0)):
        draws = np.array([A.thompson_sample(np.array([c]), np.array([s]), rng)[0] for _ in range(2000)])
        assert stats.kstest(draws, stats.beta(s + 1, c - s + 1).cdf).pvalue > 1e-3


def test_run_equals_an_independent_rewrite_with_the_same_random_stream():
    rng = np.random.default_rng(2)
    for i in range(12):
        b = B.generate(K=int(rng.integers(2, 8)), gap=float(rng.choice([0.02, 0.1, 0.3])), spread=float(rng.choice([0.0, 0.05, 0.15])), seed=i)
        for m in A.METHODS:
            eps = float(rng.choice([0.0, 0.05, 0.3, 1.0]))
            assert np.array_equal(A.run(m, b, 200, 7 + i, eps)[0], _my_run(m, b.theta, 200, 7 + i, eps))


def test_regret_of_the_random_policy_and_the_ucb1_bound_of_auer_et_al():
    b = B.generate(K=8, gap=0.1, spread=0.05, seed=2)
    delta = b.theta_star - b.theta
    reg = np.array([A.regret(b, A.run("random", b, 1000, s)[0]).sum() for s in range(40)])
    assert abs(reg.mean() - 1000 * delta.mean()) < 4 * reg.std(ddof=1) / math.sqrt(40)                  # E[Regret] = T * mittlerer Abstand
    T = 3000
    bound = 8 * sum(math.log(T) / d for d in delta if d > 0) + (1 + math.pi ** 2 / 3) * delta.sum()     # Theorem 1, Auer, Cesa-Bianchi & Fischer (2002)
    assert np.mean([A.regret(b, A.run("ucb1", b, T, s)[0]).sum() for s in range(6)]) < bound
    assert np.allclose(A.regret(B.Bandit(np.array([0.2, 0.9, 0.5]), 0), np.array([0, 1, 1, 2])), [0.7, 0.0, 0.0, 0.4])


def test_the_gap_is_exact_over_the_whole_slider_range():
    """Regression: bei großer Streuung und großem Abstand wurde die beste Station auf 0.98 geklemmt und der Abstand war kleiner als eingestellt (bis zu 90 % der Seeds bei K=20, Streuung 0.15, Abstand 0.30)."""
    for K in (2, 5, 20):
        for spread in (0.0, 0.1, 0.15):
            for gap in (0.02, 0.2, 0.3):
                for seed in range(15):
                    b = B.generate(K=K, gap=gap, spread=spread, seed=seed)
                    assert b.gap == pytest.approx(gap, abs=1e-9) and b.theta.min() >= 0.02 - 1e-12 and b.theta.max() <= 0.98 + 1e-12


def test_experiments_equal_a_hand_computation():
    stats = pytest.importorskip("scipy.stats")
    base = E.Settings()
    bandit = E._bandit(base.bandit_key)
    checkpoints, seeds = (50, 100, 200), (0, 1, 2, 3)
    gr = E.growth_experiment(checkpoints=checkpoints, seeds=seeds, base=base)
    for m in A.METHODS:
        vals = np.array([np.cumsum(bandit.theta_star - bandit.theta[_my_run(m, bandit.theta, 200, sd, base.eps)])[[49, 99, 199]] for sd in seeds])
        X, Y = np.log(checkpoints), np.log(np.maximum(vals.mean(axis=0), 1e-9))
        assert gr["rows"][m]["slope"] == pytest.approx(((X - X.mean()) * (Y - Y.mean())).sum() / ((X - X.mean()) ** 2).sum())
        for j, t in enumerate(checkpoints):
            mu, se = gr["rows"][m]["checkpoints"][t]
            assert mu == pytest.approx(vals[:, j].mean()) and se == pytest.approx(stats.sem(vals[:, j]))
    ee = E.epsilon_experiment(levels=(0.0, 0.1), seeds=range(6), base=base, T=400)
    for eps in (0.0, 0.1):
        arms = [_my_run("greedy" if eps == 0.0 else "epsilon_greedy", bandit.theta, 400, sd, eps) for sd in range(6)]
        assert ee["rows"][eps]["regret"][0] == pytest.approx(np.mean([(bandit.theta_star - bandit.theta[a]).sum() for a in arms]))
        assert ee["rows"][eps]["stuck_share"] == pytest.approx(np.mean([(a[-40:] == bandit.best).mean() < 0.10 for a in arms]))
