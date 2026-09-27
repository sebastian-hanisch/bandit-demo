"""Das Vehikel von Hand nachgerechnet: der Abstand ist exakt, die Streuung ist deterministisch beim Seed."""

import numpy as np
import pytest

import bd_bandit as B


def test_generate_is_deterministic_and_seeded():
    a = B.generate(K=10, gap=0.1, seed=1)
    b = B.generate(K=10, gap=0.1, seed=1)
    c = B.generate(K=10, gap=0.1, seed=2)
    assert np.array_equal(a.theta, b.theta) and not np.array_equal(a.theta, c.theta)


def test_the_gap_is_exact_unless_clamped():
    for seed in range(30):
        bd = B.generate(K=8, base=0.5, gap=0.1, spread=0.05, seed=seed)
        s = np.sort(bd.theta)[::-1]
        if s[0] < 0.98 - 1e-9:                                                              # nicht an der Obergrenze geklemmt
            assert s[0] - s[1] == pytest.approx(0.1, abs=1e-9)
        assert bd.gap == pytest.approx(s[0] - s[1])


def test_best_and_theta_star_by_hand():
    bd = B.Bandit(theta=np.array([0.3, 0.7, 0.5]), seed=0)
    assert bd.K == 3 and bd.best == 1 and bd.theta_star == pytest.approx(0.7) and bd.gap == pytest.approx(0.2)


def test_all_values_are_clamped_to_the_open_unit_interval():
    for seed in range(20):
        bd = B.generate(K=5, base=0.9, gap=0.3, spread=0.2, seed=seed)
        assert (bd.theta >= 0.02).all() and (bd.theta <= 0.98).all()


def test_pull_returns_zero_or_one_and_matches_the_probability():
    bd = B.Bandit(theta=np.array([0.2, 0.8]), seed=0)
    rng = np.random.default_rng(0)
    draws = np.array([B.pull(bd, 1, rng) for _ in range(20000)])
    assert set(np.unique(draws)) <= {0.0, 1.0}
    assert draws.mean() == pytest.approx(0.8, abs=0.02)


def test_a_single_arm_bandit_has_zero_gap():
    bd = B.generate(K=1, seed=0)
    assert bd.gap == pytest.approx(0.0) and bd.best == 0
