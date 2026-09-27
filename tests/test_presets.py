"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import bd_constants as C
import bd_evaluation as E
import bd_presets as P


def _settings(p):
    return E.Settings(p["K"], p["gap"], p["spread"], p["T"], p["eps"], p["seed"])


def test_every_preset_has_all_keys():
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS), name


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi
        for key, state_key in (("gap", "gap_slider"), ("spread", "spread_slider"), ("T", "t_slider"), ("eps", "eps_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-6, (key, p[key])


def test_standard_preset_equals_the_default_settings():
    assert _settings(P.PRESETS["Standardfall"]) == E.Settings()


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("gap_slider") == (C.GAP_MIN, C.GAP_MAX) and P.bounds("k_slider") == (C.K_MIN, C.K_MAX)
    assert set(P.STEPS) == {"gap_slider", "spread_slider", "t_slider", "eps_slider"}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
