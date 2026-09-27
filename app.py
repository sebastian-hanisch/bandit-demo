"""Bandit - Exploration gegen Exploitation - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Erstes Stück (Wurzel A) der Reinforcement-Learning-Linie der "Konzepte"-Reihe: ein Lagerroboter muss sich in jeder Runde für eine von K Ladestationen mit unbekannter, aber fester
Erfolgswahrscheinlichkeit entscheiden - der einfachste Fall von Verstärkungslernen, ganz ohne State. Fünf Verfahren im Vergleich: zufällig, gierig, Epsilon-gierig, UCB1 und Thompson Sampling.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import bd_algorithms as A
import bd_constants as C
from bd_evaluation import Settings, analyse, epsilon_experiment, gap_experiment, growth_experiment
from bd_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from bd_visualization import SHORT, build_arms, build_epsilon, build_gap, build_growth, build_regret_comparison

st.set_page_config(page_title="Bandit – Sebastian Hanisch", layout="wide")


def de(x, digits=2):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=1):
    return f"{de(100 * x, digits)} %"


@st.cache_data(show_spinner=False)
def _growth(checkpoints, seeds):
    return growth_experiment(checkpoints=checkpoints, seeds=seeds)


@st.cache_data(show_spinner=False)
def _epsilon(levels, seeds):
    return epsilon_experiment(levels=levels, seeds=seeds)


@st.cache_data(show_spinner=False)
def _gap(levels, seeds):
    return gap_experiment(levels=levels, seeds=seeds)


st.title("🎰 Bandit – Exploration gegen Exploitation")
st.markdown(
    """
Ein Lagerroboter muss sich in jeder Runde für eine von **K Ladestationen** entscheiden, deren Erfolgswahrscheinlichkeit (kurze statt lange Ladezeit) er nicht kennt - fest, aber unbekannt, und ohne Folgen für künftige Runden.
Das ist der einfachste Fall von **Reinforcement Learning**: kein State, keine Übergänge, nur die Frage, wie man aus wiederholtem Ausprobieren lernt. Wer nur **exploitiert** (die bisher beste Station immer wieder wählt), riskiert, sich auf eine
schlechte Station festzulegen; wer nur **erkundet** (zufällig wählt), lernt zwar viel, nutzt es aber nie. Die Demo vergleicht fünf Verfahren: zufällig, gierig (kein Erkunden), Epsilon-gierig, **UCB1** (ein Vertrauensbonus für selten
gezogene Stationen) und **Thompson Sampling** (eine Bayes'sche Stichprobe aus dem, was bisher gelernt wurde). Alle Daten sind erzeugt; die Rechnung ist in numpy geschrieben.
"""
)
st.caption(
    "Erstes Stück (Wurzel A) der **Reinforcement-Learning-Linie** der \"Konzepte\"-Reihe: kein State, die einfachste Form des Exploration-gegen-Exploitation-Kompromisses. **Bezug zu OR:** dieselbe Frage stellt sich bei jeder wiederholten "
    "Entscheidung unter unbekannter Verteilung (Preistests, A/B-Tests, Auswahl eines Lieferanten) - eine Vorstufe zu den Markov-Entscheidungsprozessen der Nachfolger."
)

with st.expander("So funktionieren Exploration und Exploitation", expanded=True):
    st.markdown(
        r"""
1. **Das Problem.** $K$ Stationen, Station $k$ liefert mit unbekannter, fester Wahrscheinlichkeit $\theta_k$ eine kurze Ladezeit (Reward 1), sonst eine lange (Reward 0). Keine Station verändert sich, keine Runde beeinflusst eine spätere - nur die Reihenfolge der Ziehungen liegt in der Hand des Roboters.
2. **Das Regret.** Der Maßstab ist nicht der Ertrag selbst, sondern der entgangene: $\text{Regret} = \sum_t (\theta^* - \theta_{a_t})$, wobei $\theta^*$ die beste Station ist. Null Regret heißt: immer die beste Station gewählt.
3. **Fünf Verfahren.** *Zufällig* und *Gierig* sind die beiden Extreme (nur Exploration, nur Exploitation). *Epsilon-gierig* mischt: mit Wahrscheinlichkeit $\varepsilon$ zufällig, sonst die bisher beste. *UCB1* addiert auf den Mittelwert einen Vertrauensbonus $\sqrt{2\ln t / n_k}$, der für selten gezogene Stationen wächst. *Thompson Sampling* zieht aus einer Bayes'schen Verteilung (Beta) über jede Station und wählt die höchste Stichprobe.
4. **Der Vergleich** läuft als wachsendes Beispiel über die Runden; die drei Experimente darunter messen, wie das Regret mit der Zeit, mit der Erkundungsrate und mit der Schwierigkeit des Problems wächst.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Das Problem**")
    K = st.slider("Zahl der Stationen K", *bounds("k_slider"), key="k_slider", help="Wie viele Ladestationen zur Wahl stehen.")
    gap = st.slider("Abstand beste – zweitbeste Station", *bounds("gap_slider"), key="gap_slider", step=C.GAP_STEP, format="%.2f", help="Je kleiner, desto schwerer zu unterscheiden.")
    spread = st.slider("Streuung der übrigen Stationen", *bounds("spread_slider"), key="spread_slider", step=C.SPREAD_STEP, format="%.2f", help="Wie unterschiedlich die K-1 übrigen Stationen sind.")
    st.markdown("**Der Lauf**")
    T = st.slider("Zahl der Runden", *bounds("t_slider"), key="t_slider", step=C.T_STEP, help="Wie viele Runden im wachsenden Beispiel und im Vergleich gespielt werden.")
    eps = st.slider("Epsilon (für Epsilon-gierig)", *bounds("eps_slider"), key="eps_slider", step=C.EPS_STEP, format="%.2f", help="Explorationsrate von Epsilon-gierig; 0 = Gierig, 1 = Zufällig.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt die wahren Erfolgswahrscheinlichkeiten fest.")
    st.button("🎲 Neue Stationen generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"k_slider": int(K), "gap_slider": round(float(gap), 3), "spread_slider": round(float(spread), 3), "t_slider": int(T), "eps_slider": round(float(eps), 3), "seed_input": int(seed)})

settings = Settings(int(K), round(float(gap), 3), round(float(spread), 3), int(T), round(float(eps), 3), int(seed))
with st.spinner("Alle fünf Verfahren spielen die Runden durch ..."):
    a = analyse(settings)
bandit = a.bandit

# --- Wachsendes Beispiel -----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Eine Station nach der anderen")
method = st.selectbox("Verfahren", list(A.METHODS), index=list(A.METHODS).index("thompson"), format_func=lambda m: A.METHOD_NAMES[m], key="method_select", help="Welches Verfahren im wachsenden Beispiel gezeigt wird.")
data_key = (settings, method)
if "bd_step" not in st.session_state or st.session_state.get("bd_step_owner") != data_key:
    st.session_state["bd_step"] = settings.T
    st.session_state["bd_step_owner"] = data_key
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.slider("Runde", 0, settings.T, key="bd_step", help="0 = noch keine Station gezogen.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()
arms, rewards = a.arms[method], a.rewards[method]


def _render(s):
    with view_slot.container():
        c1, c2 = st.columns([3, 2])
        head = "Vor der ersten Runde" if s == 0 else f"Nach Runde {s} von {settings.T}"
        c1.markdown(f"**{head} ({A.METHOD_NAMES[method]})**")
        c1.plotly_chart(build_arms(bandit, arms, rewards, s, method), width="stretch", key=f"bd_arms_{s}")
        c2.markdown("**Kumuliertes Regret**")
        c2.plotly_chart(build_regret_comparison(a.cum_regret, upto=max(s, 1), highlight=method), width="stretch", key=f"bd_regret_{s}")
        if s > 0:
            arm, r = int(arms[s - 1]), rewards[s - 1]
            c1.caption(f"Runde {s}: Station {arm} gezogen ({'Erfolg' if r == 1 else 'Misserfolg'}). Wahre beste Station: {bandit.best} (θ={de(bandit.theta_star, 2)}).")


def _frames(T):
    return sorted({int(round(x)) for x in np.linspace(0, T, min(T + 1, 40))})


if auto_play:
    import time as _time
    for f in _frames(settings.T):
        _render(f)
        _time.sleep(0.12)
else:
    _render(step)

st.markdown("---")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Konvergiert das Verfahren auf die beste Station?")
mcols = st.columns(5)
window_pct = int(round(100 * C.WINDOW))
for col, m in zip(mcols, A.METHODS):
    col.metric(SHORT[m], pct(a.best_share_window[m]), delta=f"Regret {de(a.cum_regret[m][-1], 1)}", delta_color="off", help=f"{A.METHOD_NAMES[m]} - Anteil der Ziehungen der besten Station in den letzten {window_pct} % der Runden.")
best_of = max(A.METHODS, key=lambda m: a.best_share_window[m])
worst_of = min(A.METHODS, key=lambda m: a.best_share_window[m])
if a.best_share_window[method] > 0.8:
    st.success(f"✅ {A.METHOD_NAMES[method]} zieht in den letzten {window_pct} % der Runden zu {pct(a.best_share_window[method])} die beste Station (Regret {de(a.cum_regret[method][-1], 1)}). Am besten hier: {A.METHOD_NAMES[best_of]} ({pct(a.best_share_window[best_of])}), am schlechtesten: {A.METHOD_NAMES[worst_of]} ({pct(a.best_share_window[worst_of])}).")
else:
    st.warning(f"⚠️ {A.METHOD_NAMES[method]} zieht in den letzten {window_pct} % der Runden nur zu {pct(a.best_share_window[method])} die beste Station (Regret {de(a.cum_regret[method][-1], 1)}). Am besten hier: {A.METHOD_NAMES[best_of]} ({pct(a.best_share_window[best_of])}).")
st.plotly_chart(build_regret_comparison(a.cum_regret), width="stretch", key="bd_regret_all")
st.caption(
    f"Kumuliertes Regret aller fünf Verfahren über {settings.T} Runden auf demselben Banditen (K={settings.K}, Abstand {de(settings.gap, 2)}) - **ein einzelner Lauf**, kein Mittelwert. Null Regret heißt: immer die beste Station. "
    "Gierig kann hier durch schlichtes Glück gut oder schlecht abschneiden (siehe die Streuung im ersten Experiment); erst über viele Seeds gemittelt zeigt sich das verlässliche Bild."
)

st.markdown("---")

# --- Experimente -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wächst das Regret linear oder logarithmisch?")
st.caption(f"Derselbe Bandit, {len(C.EXP_SEEDS)} feste Seeds, Runden {', '.join(str(c) for c in C.EXP_CHECKPOINTS)}. Gezeigt: kumuliertes Regret auf doppelt-logarithmischen Achsen - eine Gerade mit Steigung 1 heißt linear wachsend, eine flache Kurve heißt näherungsweise logarithmisch. Dauer etwa eine halbe Minute.")
if st.button("Wachstum durchrechnen", key="growth_start"):
    st.session_state["growth_on"] = True
if st.session_state.get("growth_on"):
    rg = _growth(C.EXP_CHECKPOINTS, C.EXP_SEEDS)
    st.plotly_chart(build_growth(rg), width="stretch", key="growth_chart")
    sl = {m: rg["rows"][m]["slope"] for m in A.METHODS}
    cp = {m: rg["rows"][m]["checkpoints"] for m in A.METHODS}
    t10, t15 = C.EXP_CHECKPOINTS[-3], C.EXP_CHECKPOINTS[-2]
    st.warning(
        f"**Befund:** Zufällig wächst mit Steigung {de(sl['random'], 2)} - fast exakt linear, wie die Theorie es für ein Verfahren ohne jedes Lernen vorhersagt. **UCB1 ({de(sl['ucb1'], 2)}) und Thompson Sampling ({de(sl['thompson'], 2)}) wachsen deutlich flacher als linear** - "
        f"näherungsweise logarithmisch, wie Auer, Cesa-Bianchi und Fischer (2002) es für UCB1 beweisen. Thompson Sampling liegt schon ab wenigen hundert Runden vorn und bleibt es (Chapelle & Li 2011: Thompson Sampling schlägt UCB1 in der Praxis oft). "
        f"**Überraschung:** UCB1s eigene, spürbare Anfangs-Exploration lässt Gierig (Steigung {de(sl['greedy'], 2)}, im Mittel) bis Runde {de(t10, 0)} sogar davonziehen ({de(cp['greedy'][t10][0], 0)} gegen {de(cp['ucb1'][t10][0], 0)}) - erst danach kehrt sich das um "
        f"({de(cp['greedy'][t15][0], 0)} gegen {de(cp['ucb1'][t15][0], 0)} bei Runde {de(t15, 0)}), weil Gierigs Mittelwert eine riesige Streuung verdeckt (manche Läufe legen sich sofort auf die beste Station fest, andere für immer auf eine schlechte - das nächste Experiment zeigt das direkt). "
        f"Epsilon-gierig ({de(sl['epsilon_greedy'], 2)}) bleibt asymptotisch linear, aber mit kleinerer Steigung und ohne Gierigs Anfangsvorteil zu brauchen."
    )

st.markdown("---")

st.subheader("🔬 Wie stark soll man erkunden?")
st.caption(f"Derselbe Bandit, {C.EXP_T} Runden, {len(C.EXP_SEEDS_EPS)} feste Seeds. Epsilon-gierig über mehrere Explorationsraten (0 = Gierig, 1 = Zufällig); zusätzlich der Anteil der Läufe, die in den letzten {window_pct} % der Runden die beste Station in weniger als 10 % der Fälle ziehen (\"hängen geblieben\"). Dauer etwa eine Minute.")
if st.button("Explorationsraten durchrechnen", key="eps_start"):
    st.session_state["eps_on"] = True
if st.session_state.get("eps_on"):
    re_ = _epsilon(C.EXP_EPS_LEVELS, C.EXP_SEEDS_EPS)
    st.plotly_chart(build_epsilon(re_), width="stretch", key="eps_chart")
    r0, rbest = re_["rows"][0.0], min(re_["rows"].items(), key=lambda kv: kv[1]["regret"][0])
    st.warning(
        f"**Befund:** Gierig (Epsilon 0) bleibt bei {pct(r0['stuck_share'])} der Seeds auf einer schlechten Station hängen (Regret {de(r0['regret'][0], 0)} ± {de(r0['regret'][1], 0)}) - die Streuung ist riesig, weil manche Läufe die beste Station früh finden und andere nie. Das beste mittlere Regret in dieser Messung "
        f"liefert Epsilon {rbest[0]} ({de(rbest[1]['regret'][0], 0)} ± {de(rbest[1]['regret'][1], 0)}), aber mit sinkendem Epsilon steigt zugleich das Risiko: eine niedrige, aber positive Erkundungsrate ist zuverlässiger als gar keine, auch wenn ihr Mittel nicht immer das kleinste ist."
    )

st.markdown("---")

st.subheader("🔬 Wie schwer ist das Problem?")
st.caption(f"UCB1 und Thompson Sampling über mehrere Abstände zwischen bester und zweitbester Station ({', '.join(de(g, 2) for g in C.EXP_GAP_LEVELS)}), {len(C.EXP_SEEDS)} feste Seeds, {C.EXP_T} Runden. Dauer etwa eine Minute.")
if st.button("Abstände durchrechnen", key="gap_start"):
    st.session_state["gap_on"] = True
if st.session_state.get("gap_on"):
    rgap = _gap(C.EXP_GAP_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_gap(rgap), width="stretch", key="gap_chart")
    g0, g1 = C.EXP_GAP_LEVELS[0], C.EXP_GAP_LEVELS[-1]
    st.warning(
        f"**Befund:** Bei Abstand {de(g0, 2)} (schwer zu unterscheiden) liegt UCB1 bei {de(rgap['rows'][(g0, 'ucb1')][0], 0)} Regret, bei Abstand {de(g1, 2)} (leicht) nur bei {de(rgap['rows'][(g1, 'ucb1')][0], 0)} - genau wie die theoretische Schranke von Auer et al. (2002) es vorhersagt: "
        f"das Regret wächst näherungsweise mit 1/Abstand. Thompson Sampling zeigt dasselbe Muster ({de(rgap['rows'][(g0, 'thompson')][0], 0)} gegen {de(rgap['rows'][(g1, 'thompson')][0], 0)}) und bleibt bei jedem Abstand deutlich unter UCB1."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Erfolgswahrscheinlichkeit ist fest** | Ändert sich eine Station mit der Zeit (Abnutzung, Tageszeit), lernen alle fünf Verfahren einen veralteten Wert; UCB1/Thompson haben kein Vergessen eingebaut. | Gleitendes Fenster, diskontierte Zähler, gleitender UCB |
| **Keine Folgen für künftige Runden** | Sobald eine Wahl den State verändert (z. B. wie viele Aufträge als Nächstes anstehen), ist es kein Bandit mehr, sondern ein Markov-Entscheidungsprozess - der nächste Schritt dieser Linie. | Value Iteration und Policy Iteration (Stück 2) |
| **Ein Entscheider, keine Gegenspieler** | Bei mehreren Lkw, die sich gegenseitig beeinflussen (dasselbe Tor, dieselbe Ressource), ist die Umgebung nicht mehr stationär - das ist die Situation von `noregret-demo` (Spieltheorie-Linie), nicht diese hier. | Adversariales Online-Lernen (Hedge/EXP3) |
| **Bernoulli-Reward (0 oder 1)** | Bei kontinuierlichen Rewards (z. B. echte Ladezeit in Minuten) ändert sich UCB1s Bonusformel und Thompson Samplings Beta-Verteilung passt nicht mehr direkt. | Gauß'sches UCB, Gauß'sche Thompson-Variante |
| **Gierig hat kein Sicherheitsnetz** | Ein einzelner unglücklicher erster Zug kann eine schlechte Station für immer festschreiben - kein Verfahren dieser Demo erkennt das im Nachhinein. | Erkundung mit abklingendem Epsilon, optimistische Startwerte |
| **Erzeugte Stationen, feste Seeds** | Die Zahlen gelten für dieses Vehikel; reale Auswahlprobleme haben oft mehr Stationen mit ähnlicheren Werten (kleinerer Abstand) als hier gezeigt. | - |
"""
)
st.caption("Die Linie: Bandit (dieses Stück) → Value Iteration und Policy Iteration → Q-Learning → SARSA / Dyna-Q / Funktionsapproximation (DQN) / Policy Gradient → Actor-Critic.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Das Problem.** $K$ Stationen mit fester, unbekannter Erfolgswahrscheinlichkeit $\theta_k \in (0,1)$. In Runde $t$ wird Station $a_t$ gezogen, der Reward $r_t \sim \text{Bernoulli}(\theta_{a_t})$ beobachtet. $n_k(t)$ = Zahl der Ziehungen von Station $k$ bis Runde $t$, $\hat\theta_k(t) = \frac{1}{n_k(t)}\sum r$ ihr Mittelwert.

**Regret.** $R(T) = \sum_{t=1}^T (\theta^* - \theta_{a_t})$ mit $\theta^* = \max_k \theta_k$ (Auer, Cesa-Bianchi & Fischer 2002) - der Erwartungswert des entgangenen Ertrags, nicht der verrauschte tatsächliche.

**Verfahren.** *Zufällig:* $a_t$ gleichverteilt. *Gierig:* $a_t = \arg\max_k \hat\theta_k(t)$ (eine nie gezogene Station gilt als $+\infty$). *Epsilon-gierig:* mit Wahrscheinlichkeit $\varepsilon$ zufällig, sonst gierig.
**UCB1:** $a_t = \arg\max_k \left(\hat\theta_k(t) + \sqrt{\dfrac{2\ln t}{n_k(t)}}\right)$. **Thompson Sampling:** Vorwissen $\text{Beta}(1,1)$ je Station, nach $n_k$ Ziehungen mit $s_k$ Erfolgen Posterior $\text{Beta}(s_k+1,\, n_k-s_k+1)$; $a_t = \arg\max_k \tilde\theta_k$ mit $\tilde\theta_k \sim \text{Beta}(s_k+1, n_k-s_k+1)$.

**Theoretische Schranken.** UCB1 erreicht $R(T) = O\!\left(\sum_{k \ne k^*} \dfrac{\ln T}{\Delta_k}\right)$ mit $\Delta_k = \theta^* - \theta_k$ (Auer et al. 2002) - logarithmisch in $T$, linear in $1/\Delta$. Ohne Erkundung (Gierig) und bei fester Erkundungsrate (Epsilon-gierig) ist das Regret asymptotisch **linear** in $T$.

Implementiert in `bd_bandit.py` (das Vehikel), `bd_algorithms.py` (alle fünf Verfahren, Regret), `bd_evaluation.py` (Analyse, drei Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
