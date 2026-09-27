"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import bd_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "gap_slider": SettingSpec("gap", float, C.DEFAULT_GAP, C.GAP_MIN, C.GAP_MAX),
    "spread_slider": SettingSpec("spread", float, C.DEFAULT_SPREAD, C.SPREAD_MIN, C.SPREAD_MAX),
    "t_slider": SettingSpec("t", int, C.DEFAULT_T, C.T_MIN, C.T_MAX),
    "eps_slider": SettingSpec("eps", float, C.DEFAULT_EPS, C.EPS_MIN, C.EPS_MAX),
    "seed_input": SettingSpec("seed", int, 3, 0, C.SEED_MAX),
}
PRESET_KEYS = {"K": "k_slider", "gap": "gap_slider", "spread": "spread_slider", "T": "t_slider", "eps": "eps_slider", "seed": "seed_input"}
STEPS = {"gap_slider": C.GAP_STEP, "spread_slider": C.SPREAD_STEP, "t_slider": C.T_STEP, "eps_slider": C.EPS_STEP}


def _p(**kw):
    base = {"K": C.DEFAULT_K, "gap": C.DEFAULT_GAP, "spread": C.DEFAULT_SPREAD, "T": C.DEFAULT_T, "eps": C.DEFAULT_EPS, "seed": 3}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall": _p(),
    "Kleiner Abstand (schwer zu unterscheiden)": _p(gap=0.02),
    "Großer Abstand (leicht)": _p(gap=0.2),
    "Viele Stationen": _p(K=20),
    "Gierig, kein Erkunden": _p(eps=0.0),
    "Viel Erkunden (fast zufällig)": _p(eps=0.5),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 3)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall": "K=10, Abstand 0,10, 1000 Runden, Seed 3: Gierig gewinnt hier zufällig (Regret 11,3, 100 % beste Station) - ein einzelner glücklicher Lauf, siehe Experiment 1. Thompson Sampling ist zuverlässiger (Regret 72,5, 87 %), UCB1 liegt bei diesem Seed noch zurück (Regret 142,9, 55 %), Zufällig bei 244,4 (9 %).",
    "Kleiner Abstand (schwer zu unterscheiden)": "Abstand 0,02: UCB1 fällt auf 37 % beste Station (Regret 106,9), Thompson Sampling bleibt bei 92 % (Regret 25,2) - ein schwer zu unterscheidendes Problem trifft UCB1 stärker als Thompson.",
    "Großer Abstand (leicht)": "Abstand 0,20: alle lernenden Verfahren treffen die beste Station fast immer (UCB1 86 %, Thompson 99 %) - ein leichtes Problem verwischt die Unterschiede zwischen den Verfahren.",
    "Viele Stationen": "K=20 Stationen: UCB1 trifft nur noch 23 % (Regret 216,9) - mehr Stationen heißt mehr Zeit für die anfängliche Aufwärmphase (jede Station einmal), bevor irgendein Lernen greift.",
    "Gierig, kein Erkunden": "Epsilon 0 heißt: Epsilon-gierig ist identisch mit Gierig (Regret 11,3 für beide) - keine eigene Erkundung mehr übrig.",
    "Viel Erkunden (fast zufällig)": "Epsilon 0,5: Epsilon-gierig verliert die Hälfte seiner Runden an reines Raten (Regret 126,6, nur 57 % beste Station) - deutlich schlechter als bei Epsilon 0,1.",
}
