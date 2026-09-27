"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Regler und Schritt-Slider, Würfel-Knopf, Permalink-Grenzen/-Raster, Extremwerte, drei Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import bd_algorithms as A
import bd_constants as C
import bd_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value, el.value[:120]


def test_default_run_shows_metrics_charts_and_a_verdict():
    at = _run()
    _ok(at)
    assert len(at.metric) == 5 and len(at.get("plotly_chart")) >= 2 and len(at.info) + len(at.success) + len(at.warning) >= 1


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == pytest.approx(p[key]) if isinstance(p[key], float) else at.session_state[state_key] == p[key]


def test_method_and_step_selection_survive_a_smaller_horizon():
    at = _run(bd_step=900)
    _ok(at)
    at.slider(key="t_slider").set_value(200).run()
    _ok(at)
    assert at.session_state["bd_step"] <= 200


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Stationen generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["k"] = "999"
    at.query_params["gap"] = "-5"
    at.query_params["spread"] = "10"
    at.query_params["t"] = "50000"
    at.query_params["eps"] = "abc"
    at.query_params["seed"] = "-1"
    at.run()
    _ok(at)
    s = at.session_state
    assert s["k_slider"] == C.K_MAX and s["gap_slider"] == C.GAP_MIN and s["spread_slider"] == C.SPREAD_MAX and s["t_slider"] == C.T_MAX
    assert s["eps_slider"] == C.DEFAULT_EPS and s["seed_input"] == 0


@pytest.mark.parametrize("kw", [dict(k_slider=C.K_MIN, gap_slider=C.GAP_MIN, t_slider=C.T_MIN), dict(k_slider=C.K_MAX, gap_slider=C.GAP_MAX, spread_slider=C.SPREAD_MAX),
                                dict(eps_slider=0.0, t_slider=C.T_MAX), dict(eps_slider=1.0, spread_slider=C.SPREAD_MIN)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


@pytest.mark.parametrize("m", list(A.METHODS))
def test_every_method_selectable_in_the_growing_example(m):
    at = _run(method_select=m)
    _ok(at)
    assert at.session_state["method_select"] == m


def _small(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))
    monkeypatch.setattr(C, "EXP_SEEDS_EPS", (0, 1))
    monkeypatch.setattr(C, "EXP_CHECKPOINTS", (20, 40, 80))
    monkeypatch.setattr(C, "EXP_EPS_LEVELS", (0.0, 0.1, 1.0))
    monkeypatch.setattr(C, "EXP_GAP_LEVELS", (0.05, 0.2))
    monkeypatch.setattr(C, "EXP_T", 200)


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()
    _ok(at)


def test_growth_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "growth_start")
    assert at.session_state["growth_on"] and any("Befund" in w.value for w in at.warning)


def test_epsilon_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "eps_start")
    assert at.session_state["eps_on"] and any("hängen" in w.value for w in at.warning)


def test_gap_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "gap_start")
    assert at.session_state["gap_on"] and any("Abstand" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
