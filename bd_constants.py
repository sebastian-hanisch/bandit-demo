"""Konstanten der Bandit-Demo: Vehikel "Ladestationswahl" (Stück 1, Wurzel A, der Reinforcement-Learning-Linie), Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999

# --- Regler und Voreinstellungen --------------------------------------------------------------------------------------------------------------
K_MIN, K_MAX, DEFAULT_K = 2, 20, 10
GAP_MIN, GAP_MAX, GAP_STEP, DEFAULT_GAP = 0.02, 0.30, 0.02, 0.10
SPREAD_MIN, SPREAD_MAX, SPREAD_STEP, DEFAULT_SPREAD = 0.0, 0.15, 0.01, 0.05
T_MIN, T_MAX, T_STEP, DEFAULT_T = 50, 3000, 50, 1000
EPS_MIN, EPS_MAX, EPS_STEP, DEFAULT_EPS = 0.0, 1.0, 0.01, 0.10
WINDOW = 0.1        # Anteil der letzten Runden für den Kernfrage-Anteil "in den letzten ... % Runden"

# --- Experimente (feste Seeds) -----------------------------------------------------------------------------------------------------------------
EXP_SEEDS = tuple(range(30))
EXP_SEEDS_EPS = tuple(range(60))          # das Epsilon-Experiment braucht mehr Seeds gegen das hohe Rauschen der Gierig-Variante
EXP_T = 20000
EXP_CHECKPOINTS = (500, 1000, 2000, 5000, 10000, 15000, 20000)
EXP_EPS_LEVELS = (0.0, 0.01, 0.05, 0.1, 0.3, 1.0)
EXP_GAP_LEVELS = (0.02, 0.05, 0.1, 0.2)
