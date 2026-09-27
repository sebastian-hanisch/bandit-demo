"""Das Vehikel: K Ladestationen mit unbekannter, aber fester Erfolgswahrscheinlichkeit. Der Roboter wählt in jeder Runde eine Station; sie liefert mit Wahrscheinlichkeit theta_k eine kurze Ladezeit
(Belohnung 1) oder eine lange (Belohnung 0) - ein Bernoulli-Banditenproblem. Keine Folgen für künftige Runden (nicht-assoziativ): das ist der einfachste Fall, bevor in den Nachfolgern ein Zustand dazukommt."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bandit:
    theta: np.ndarray   # (K,) wahre Erfolgswahrscheinlichkeit je Station
    seed: int

    @property
    def K(self):
        return len(self.theta)

    @property
    def best(self):
        return int(np.argmax(self.theta))

    @property
    def theta_star(self):
        return float(self.theta.max())

    @property
    def gap(self):
        """Abstand der besten zur zweitbesten Station (die Schwierigkeit des Problems); 0, wenn es nur eine Station gibt."""
        if self.K < 2:
            return 0.0
        s = np.sort(self.theta)[::-1]
        return float(s[0] - s[1])


def generate(K=10, base=0.5, gap=0.1, spread=0.05, seed=0):
    """K Stationen: die anderen K-1 um `base` gestreut (Standardabweichung `spread`), die beste genau `gap` über deren Maximum -
    das erzeugt den gewünschten Abstand unabhängig von der Streuung. Alle Werte auf [0.02, 0.98] geklemmt."""
    rng = np.random.default_rng(seed)
    theta = np.clip(base + spread * rng.normal(size=K), 0.02, 0.98)
    best = int(rng.integers(K))
    others_max = np.delete(theta, best).max() if K > 1 else 0.0
    theta[best] = float(np.clip(others_max + gap, 0.02, 0.98))
    return Bandit(theta, int(seed))


def pull(bandit, arm, rng):
    """Eine Ziehung an Station `arm`: 1 mit Wahrscheinlichkeit theta[arm], sonst 0."""
    return float(rng.random() < bandit.theta[arm])
