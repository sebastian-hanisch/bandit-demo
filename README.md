# 🎰 Bandit – Erkunden gegen Ausnutzen

Erstes Stück (Wurzel A) der **Reinforcement-Learning-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Ein Lagerroboter muss sich in jeder Runde für eine von **K Ladestationen** mit unbekannter, aber fester Erfolgswahrscheinlichkeit entscheiden – der einfachste Fall von Reinforcement Learning: kein Zustand, keine Übergänge, nur die Frage, wie man aus wiederholtem Ausprobieren lernt. Fünf Verfahren im Vergleich: zufällig, gierig, Epsilon-gierig, **UCB1** und **Thompson Sampling**.

## Kernfrage

**Wer nur ausnutzt, riskiert, sich auf eine schlechte Station festzulegen; wer nur erkundet, lernt viel, nutzt es aber nie.** Wächst das entgangene Regret mit der Zeit **linear** (kein echtes Lernen) oder **logarithmisch** (die Theorie von Auer, Cesa-Bianchi und Fischer 2002 für UCB1)? Wie stark man erkunden sollte, ist keine triviale Frage – und "gierig" hat ein reales, messbares Risiko, sich für immer auf die falsche Station festzulegen.

## Modell

- **Vehikel** (`bd_bandit.py`): $K$ Ladestationen, Station $k$ liefert mit fester, unbekannter Wahrscheinlichkeit $\theta_k$ eine kurze Ladezeit (Belohnung 1), sonst eine lange (0). Die beste Station liegt um einen wählbaren **Abstand** über dem Maximum der übrigen $K-1$ Stationen (die selbst um eine Grundrate mit wählbarer Streuung schwanken) – so ist die Schwierigkeit des Problems ein eigener, unabhängig einstellbarer Regler.
- **Regret** (`bd_algorithms.py`): $R(T) = \sum_{t=1}^T (\theta^* - \theta_{a_t})$ – der Erwartungswert des entgangenen Ertrags (Auer et al. 2002), nicht der verrauschte tatsächliche.
- **Fünf Verfahren:** Zufällig, Gierig ($\arg\max$ des Mittelwerts, nie gezogene Stationen zählen als $+\infty$), Epsilon-gierig (Mischung), **UCB1** (Mittelwert + Vertrauensbonus $\sqrt{2\ln t/n_k}$, Auer, Cesa-Bianchi & Fischer 2002), **Thompson Sampling** (Beta-Posterior je Station, Beta(1,1)-Vorwissen, Thompson 1933).

## Methodik

Alle fünf Verfahren laufen auf demselben erzeugten Banditen (Rolling-Ziehung über $T$ Runden); Kennzahl ist das kumulierte Regret und der Anteil der Ziehungen der besten Station in den letzten 10 % der Runden. Drei Experimente: wie das Regret mit der Zeit wächst (linear gegen logarithmisch), wie die Erkundungsrate Epsilon das Regret und das Risiko des dauerhaften Festlegens beeinflusst, und wie der Abstand zwischen bester und zweitbester Station UCB1 und Thompson Sampling unterschiedlich trifft.

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Wächst das Regret linear oder logarithmisch?** (30 feste Seeds, Runden 500 bis 20 000) | Steigung einer Ausgleichsgeraden log(Regret) gegen log(Runde): Zufällig **1,00** (exakt linear), Gierig 0,92, Epsilon-gierig (0,1) 0,72, **UCB1 0,50, Thompson Sampling 0,22** (deutlich flacher als linear). | `test_growth_slopes_match_the_expected_order` |
| **Überraschung: Gierig zieht anfangs vor UCB1 davon** | Bei Runde 10 000 liegt Gierig im Mittel noch vorn (445 gegen 466 bei UCB1); erst bei Runde 15 000 dreht sich das um (663 gegen 526) – UCB1s Anfangs-Erkundung kostet zunächst mehr, als sie bringt. | `test_greedy_leads_ucb1_early_but_falls_behind_later` |
| **Wie stark soll man erkunden?** (20 000 Runden, 60 feste Seeds) | Gierig (Epsilon 0) bleibt bei **40,0 %** der Seeds auf einer schlechten Station hängen (Regret 1030,7 ± 179,4 – riesige Streuung); das beste mittlere Regret liefert Epsilon 0,05 (388,4 ± 41,7); Epsilon 0,1 ist mit 547,4 ± 20,0 schlechter im Mittel, aber deutlich zuverlässiger (kein hängen gebliebener Lauf). | `test_epsilon_sweet_spot_and_greedy_lockin_risk` |
| **Wie schwer ist das Problem?** (30 feste Seeds, 20 000 Runden) | Abstand 0,02 (schwer): UCB1 657,7 ± 9,7, Thompson 150,2 ± 7,2. Abstand 0,20 (leicht): UCB1 415,3 ± 6,4, Thompson 56,5 ± 2,0 – beide wachsen mit sinkendem Abstand, wie die Schranke $O(\ln T/\Delta)$ es vorhersagt; Thompson Sampling bleibt bei jedem Abstand deutlich unter UCB1 (Chapelle & Li 2011). | `test_gap_experiment_matches_the_theoretical_direction` |
| Standardfall (Preset, K=10, Abstand 0,10, 1000 Runden, Seed 3) | Gierig gewinnt hier **zufällig** (Regret 11,3, 100 % beste Station – ein einzelner glücklicher Lauf); Thompson Sampling 72,5 (87 %), UCB1 142,9 (55 %), Epsilon-gierig 32,1 (86 %), Zufällig 244,4 (9 %). | `test_standard_preset` |
| Kleiner Abstand (Preset, 0,02) | UCB1 fällt auf 37 % beste Station (Regret 106,9), Thompson Sampling hält 92 % (Regret 25,2). | `test_small_gap_preset` |
| Großer Abstand (Preset, 0,20) | UCB1 86 % (Regret 148,1), Thompson Sampling 99 % (Regret 21,8) – ein leichtes Problem verwischt die Unterschiede. | `test_large_gap_preset` |
| Viele Stationen (Preset, K=20) | UCB1 trifft nur 23 % (Regret 216,9) – mehr Stationen heißt eine längere Aufwärmphase, bevor irgendein Lernen greift. | `test_many_arms_preset` |

Die Preset-Zeilen sind **Einzelläufe** (Seed 3, 1000 Runden); belastbar sind die Zeilen über mehrere Seeds (die drei Experimente).

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Erfolgswahrscheinlichkeit ist fest** | Ändert sich eine Station mit der Zeit, lernen alle fünf Verfahren einen veralteten Wert; UCB1/Thompson haben kein Vergessen eingebaut. | Gleitendes Fenster, diskontierte Zähler, gleitender UCB |
| **Keine Folgen für künftige Runden** | Sobald eine Wahl den Zustand verändert, ist es kein Bandit mehr, sondern ein Markov-Entscheidungsprozess. | Wert- und Politikiteration (Stück 2) |
| **Ein Entscheider, keine Gegenspieler** | Bei mehreren sich gegenseitig beeinflussenden Entscheidern ist die Umgebung nicht mehr stationär – das ist die Situation von `noregret-demo` (Spieltheorie-Linie), nicht diese hier. | Adversariales Online-Lernen (Hedge/EXP3) |
| **Bernoulli-Belohnung (0 oder 1)** | Bei kontinuierlichen Belohnungen ändern sich UCB1s Bonusformel und Thompson Samplings Beta-Verteilung. | Gauß'sches UCB, Gauß'sche Thompson-Variante |
| **Gierig hat kein Sicherheitsnetz** | Ein einzelner unglücklicher erster Zug kann eine schlechte Station für immer festschreiben – gemessen: 40 % der Seeds bei Epsilon 0. | Abklingendes Epsilon, optimistische Startwerte |
| **Erzeugte Stationen, feste Seeds** | Die Zahlen gelten für dieses Vehikel; reale Auswahlprobleme haben oft mehr Stationen mit ähnlicheren Werten. | – |

## Tests

`tests/` prüft das Vehikel (`bd_bandit.py`: Abstand exakt, Determinismus, Klemmen), die fünf Verfahren (`bd_algorithms.py`: jede Bewertungsformel von Hand nachgerechnet – Gierig, UCB1, Thompson-Posterior –, Gleichstand-Auflösung, ein rauschfreier Bandit mit $\theta \in \{0,1\}$ macht `run()` bis auf den Anfangs-Gleichstand exakt nachrechenbar, dazu eine unabhängige Gegenprobe von `run()` gegen die einzeln getesteten Bausteine), die Auswertung und drei Experimente (Aufbau, Kennzahlen von Hand, die erwarteten Wachstumsmuster), die Presets und Permalinks, die App (AppTest: Standard, jedes Preset, Schritt-Slider, Permalink-Klemmen/-Einrasten, Extremwerte, jedes Verfahren, drei Experimente auf Abruf) und jede Zahl dieses READMEs (`test_claims.py`). 63 Tests, Laufzeit gut zwei Minuten (die drei Experimente in `test_claims.py` messen mit vollen Seed-Zahlen); die CI läuft bei jedem Push und wöchentlich (die Abhängigkeiten sind nicht gepinnt).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche: wachsendes Beispiel mit Schritt-Slider, Kernfrage, drei Experimente auf Abruf, Grenzen, Formeln |
| `bd_constants.py` | Regler-Grenzen, Experimentkonstanten |
| `bd_bandit.py` | Das Vehikel: K Ladestationen mit fester, unbekannter Erfolgswahrscheinlichkeit |
| `bd_algorithms.py` | Alle fünf Verfahren, Regret |
| `bd_evaluation.py` | Analyse, drei Experimente |
| `bd_visualization.py` | Plotly-Abbildungen (alle Achsen gesperrt) |
| `bd_presets.py` | Presets, Permalink |
| `tests/` | Tests (siehe oben) |

## Bewusst nicht umgesetzt

- **Kontextuelle Bandits** (Merkmale je Runde, z. B. Tageszeit) – ein natürlicher nächster Schritt, aber kein eigenes Stück dieser Linie.
- **Nicht-stationäre Banditen** (sich ändernde Erfolgswahrscheinlichkeiten, gleitendes UCB) – die Annahme "fest" wird bewusst nicht verletzt, um Erkunden-gegen-Ausnutzen isoliert zu zeigen.
- **Adversariale Bandits** (EXP3, Hedge) – das ist die Situation von `noregret-demo` (Spieltheorie-Linie): mehrere Entscheider, die sich gegenseitig beeinflussen, keine stationäre Umgebung.
- **Gauß'sche statt Bernoulli-Belohnung.**

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
python -m pytest tests/ -q
```

Gebaut mit Streamlit, Plotly und numpy.
