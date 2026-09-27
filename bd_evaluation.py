"""Auswertung: alle fünf Verfahren auf demselben Banditen, dazu drei Experimente (Regret-Wachstum über den Horizont, Wirkung von Epsilon
inklusive Gierig-Aussetzern, Wirkung des Abstands zwischen bester und zweitbester Station)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import bd_algorithms as A
import bd_bandit as B
import bd_constants as C


@dataclass(frozen=True)
class Settings:
    K: int = C.DEFAULT_K
    gap: float = C.DEFAULT_GAP
    spread: float = C.DEFAULT_SPREAD
    T: int = C.DEFAULT_T
    eps: float = C.DEFAULT_EPS
    seed: int = 3

    @property
    def bandit_key(self):
        return (self.K, self.gap, self.spread, self.seed)


@lru_cache(maxsize=32)
def _bandit(key):
    K, gap, spread, seed = key
    return B.generate(K=K, gap=gap, spread=spread, seed=seed)


@dataclass
class Analysis:
    settings: Settings
    bandit: B.Bandit
    arms: dict            # Verfahren -> (T,) gezogene Stationen
    rewards: dict          # Verfahren -> (T,) Rewards
    regret: dict           # Verfahren -> (T,) Pseudo-Regret je Runde
    cum_regret: dict       # Verfahren -> (T,) kumuliertes Regret
    best_share_window: dict  # Verfahren -> Anteil der Ziehungen der besten Station in den letzten WINDOW-Runden


def analyse(s):
    bandit = _bandit(s.bandit_key)
    arms, rewards, regret, cum_regret, share = {}, {}, {}, {}, {}
    window = max(1, int(round(C.WINDOW * s.T)))
    for m in A.METHODS:
        a, r = A.run(m, bandit, s.T, seed=s.seed, eps=s.eps)
        g = A.regret(bandit, a)
        arms[m], rewards[m], regret[m], cum_regret[m] = a, r, g, np.cumsum(g)
        share[m] = float(np.mean(a[-window:] == bandit.best))
    return Analysis(s, bandit, arms, rewards, regret, cum_regret, share)


def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


# --- Experiment 1: Wächst das Regret linear oder logarithmisch? -------------------------------------------------------------------------------

def growth_experiment(checkpoints=None, seeds=None, base=None):
    """Kumuliertes Regret je Verfahren an mehreren Zeitpunkten (Mittel über Seeds); dazu die Steigung einer Ausgleichsgeraden
    von log(Regret) gegen log(Zeitpunkt) über alle Zeitpunkte (1 = linear wachsend, 0 = konstant/flach)."""
    checkpoints = C.EXP_CHECKPOINTS if checkpoints is None else checkpoints
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    bandit = _bandit(base.bandit_key)
    T = max(checkpoints)
    rows = {}
    for m in A.METHODS:
        vals = np.zeros((len(seeds), len(checkpoints)))
        for i, sd in enumerate(seeds):
            a, _ = A.run(m, bandit, T, seed=sd, eps=base.eps)
            cum = np.cumsum(A.regret(bandit, a))
            vals[i] = cum[np.array(checkpoints) - 1]
        mean = vals.mean(axis=0)
        slope = float(np.polyfit(np.log(checkpoints), np.log(np.maximum(mean, C.EPS)), 1)[0])
        rows[m] = {"checkpoints": {t: _mean_se(vals[:, j]) for j, t in enumerate(checkpoints)}, "slope": slope, "final": vals[:, -1]}
    return {"n_seeds": len(seeds), "checkpoints": tuple(checkpoints), "rows": rows, "bandit": bandit}


# --- Experiment 2: Wie stark soll man erkunden? -----------------------------------------------------------------------------------------------

def epsilon_experiment(levels=None, seeds=None, base=None, T=None):
    """Für Epsilon-gierig (Epsilon = 0 heißt: gierig) über mehrere Epsilon-Stufen: mittleres Regret am Ende, Streuung über die Seeds
    (Gierig ohne Erkunden kann sich dauerhaft auf eine schlechte Station festlegen - das zeigt sich als hohe Streuung), und der Anteil
    der Seeds, die in den letzten WINDOW-Runden die beste Station in weniger als 10 % der Fälle ziehen ("hängen geblieben")."""
    levels = C.EXP_EPS_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS_EPS if seeds is None else seeds
    base = Settings() if base is None else base
    T = C.EXP_T if T is None else T
    bandit = _bandit(base.bandit_key)
    window = max(1, int(round(C.WINDOW * T)))
    rows = {}
    for eps in levels:
        finals, stuck = [], []
        for sd in seeds:
            method = "greedy" if eps == 0.0 else "epsilon_greedy"
            a, _ = A.run(method, bandit, T, seed=sd, eps=eps)
            finals.append(np.cumsum(A.regret(bandit, a))[-1])
            stuck.append(np.mean(a[-window:] == bandit.best) < 0.10)
        rows[eps] = {"regret": _mean_se(finals), "stuck_share": float(np.mean(stuck))}
    return {"n_seeds": len(seeds), "T": T, "levels": tuple(levels), "rows": rows, "bandit": bandit}


# --- Experiment 3: Wie schwer ist das Problem? ------------------------------------------------------------------------------------------------

def gap_experiment(levels=None, seeds=None, K=None, spread=None, seed_bandit=None, T=None):
    """UCB1 und Thompson Sampling über mehrere Abstände zwischen bester und zweitbester Station: kleinerer Abstand = schwerer zu
    unterscheiden = mehr Regret (Mittel über Seeds, ein neuer Bandit je Abstand mit demselben seed_bandit)."""
    levels = C.EXP_GAP_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    K = C.DEFAULT_K if K is None else K
    spread = C.DEFAULT_SPREAD if spread is None else spread
    seed_bandit = 3 if seed_bandit is None else seed_bandit
    T = C.EXP_T if T is None else T
    rows = {}
    for gap in levels:
        bandit = B.generate(K=K, gap=gap, spread=spread, seed=seed_bandit)
        for m in ("ucb1", "thompson"):
            finals = []
            for sd in seeds:
                a, _ = A.run(m, bandit, T, seed=sd)
                finals.append(np.cumsum(A.regret(bandit, a))[-1])
            rows[(gap, m)] = _mean_se(finals)
    return {"n_seeds": len(seeds), "T": T, "levels": tuple(levels), "rows": rows}
