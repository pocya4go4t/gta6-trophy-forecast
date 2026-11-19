# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/pocya4go4t/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/pocya4go4t/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

**Dilution-power forecaster** for GTA VI trophy earn rates on Exophase at 2026-11-26 00:00 UTC, as five quantiles per trophy (5/25/50/75/95%), scored by pinball loss on the logit scale.

The rate a trophy shows is earners / players-tracked, and in a launch week both move: tracked explodes as new players link accounts (638 on launch night, 3,868 at the first refresh hours later, ~13,400 that afternoon) while earners grow only as players actually reach the trophy — so rates can fall fast even though nobody un-earns anything. The program models the two parts separately, from the case's own history: it collapses identical reads into refresh states, fits tracked growth against ln(1 + hours since release), fits each trophy's earners as a power law in the tracked base (slopes shrunk toward tier medians), and combines them into a median rate with an honest spread from the fit residuals and the refresh-to-refresh jump size. A trophy at 100% stays at 100%. Backtest on a synthetic case (flat launch-night history, live page ~10 h later as the answer): pinball loss **0.369** vs **2.387** for the house persistence baseline.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=d9dj2dyr&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
