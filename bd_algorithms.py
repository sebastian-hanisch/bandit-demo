"""Fünf Verfahren für das Banditenproblem: zufällig, gierig, Epsilon-gierig, UCB1 (Auer, Cesa-Bianchi & Fischer 2002) und Thompson Sampling (Thompson 1933).
Jede Station, die noch nie gezogen wurde, gilt als unendlich gut (+inf) - so werden alle Stationen einmal ausprobiert, bevor eine Regel greift; das ist kein Sonderfall, sondern
dieselbe Formel mit `counts=0`. Ein Gleichstand (auch der Gleichstand aller +inf zu Beginn) wird zufällig unter den Bestplatzierten aufgelöst."""

import numpy as np

METHODS = ("random", "greedy", "epsilon_greedy", "ucb1", "thompson")
METHOD_NAMES = {"random": "Zufällig", "greedy": "Gierig (kein Erkunden)", "epsilon_greedy": "Epsilon-gierig", "ucb1": "UCB1", "thompson": "Thompson Sampling"}
METHOD_SHORT = {"random": "Zufällig", "greedy": "Gierig", "epsilon_greedy": "ε-gierig", "ucb1": "UCB1", "thompson": "Thompson"}


def estimate_greedy(counts, sums):
    """Geschätzter Wert je Station: Mittelwert der Ziehungen, +inf für eine noch nie gezogene Station."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.inf)


def estimate_ucb1(counts, sums, t):
    """UCB1 (Auer et al. 2002): Mittelwert + sqrt(2 ln(t) / n); t = Zahl der bisher gespielten Runden (>= 1 ab der ersten
    Runde nach der Aufwärmphase). +inf für eine noch nie gezogene Station."""
    mean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    bonus = np.where(counts > 0, np.sqrt(2.0 * np.log(max(t, 1)) / np.maximum(counts, 1)), np.inf)
    return mean + bonus


def thompson_sample(counts, sums, rng):
    """Eine Ziehung aus der Beta-Posterior je Station (Beta(1,1) = Gleichverteilung als Vorwissen, Bernoulli-Rückmeldung:
    Beta(Erfolge + 1, Misserfolge + 1))."""
    alpha = sums + 1.0
    beta = (counts - sums) + 1.0
    return rng.beta(alpha, beta)


def choose(values, rng):
    """Index des größten Werts; bei Gleichstand zufällig unter den Bestplatzierten."""
    top = np.flatnonzero(values == values.max())
    return int(top[0]) if len(top) == 1 else int(rng.choice(top))


def run(method, bandit, T, seed, eps=0.1):
    """T Runden des gewählten Verfahrens auf `bandit`. Rückgabe: gezogene Stationen (T,), Rewards (T,)."""
    rng = np.random.default_rng(seed)
    K = bandit.K
    counts = np.zeros(K, dtype=int)
    sums = np.zeros(K)
    arms = np.empty(T, dtype=int)
    rewards = np.empty(T)
    for t in range(T):
        if method == "random":
            a = int(rng.integers(K))
        elif method == "greedy":
            a = choose(estimate_greedy(counts, sums), rng)
        elif method == "epsilon_greedy":
            a = int(rng.integers(K)) if eps > 0.0 and rng.random() < eps else choose(estimate_greedy(counts, sums), rng)
        elif method == "ucb1":
            a = choose(estimate_ucb1(counts, sums, t), rng)
        elif method == "thompson":
            a = choose(thompson_sample(counts, sums, rng), rng)
        else:
            raise ValueError(method)
        r = float(rng.random() < bandit.theta[a])
        counts[a] += 1
        sums[a] += r
        arms[t] = a
        rewards[t] = r
    return arms, rewards


def regret(bandit, arms):
    """Erwartetes (Pseudo-)Regret je Runde: theta* - theta[gezogene Station]; kumuliert = Summe. Die Definition aus
    Auer et al. (2002): der Erwartungswert des entgangenen Ertrags, nicht der verrauschte tatsächliche."""
    return bandit.theta_star - bandit.theta[arms]
