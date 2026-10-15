# Methodology

This document explains how the predictor is built and evaluated, and why it is built that way.
For usage, see the [README](../README.md); for intended use and limitations, see the
[model card](../MODEL_CARD.md).

- [1. Problem](#1-problem)
- [2. Data](#2-data)
- [3. Leakage prevention](#3-leakage-prevention)
- [4. Elo ratings](#4-elo-ratings)
- [5. Features](#5-features)
- [6. Models](#6-models)
- [7. Evaluation](#7-evaluation)
- [8. Serving predictions](#8-serving-predictions)
- [9. The prototype audit](#the-prototype-audit)
- [10. Limitations and future work](#10-limitations-and-future-work)

## 1. Problem

Given two players and the context of a match (surface, format, tournament level, round, date),
estimate the probability that player A wins, **using only information available before the
match**.

The output is a probability, not a pick, so the primary metric is **log-loss**, with the
**Brier score** and calibration as secondary metrics. Accuracy is reported because it is
intuitive, but it ignores confidence: a model that says 51% and one that says 95% get the same
credit when right.

## 2. Data

All data comes from Jeff Sackmann's [tennis_atp](https://github.com/JeffSackmann/tennis_atp) and
[tennis_wta](https://github.com/JeffSackmann/tennis_wta) repositories (CC BY-NC-SA 4.0), vendored
in [`data/raw/`](../data/raw/README.md).

| Tour | Files loaded | Matches after cleaning | Training / evaluation rows |
|---|---|---:|---|
| ATP | main tour, Challengers and tour qualifying, Futures (1968-2024) | 916,628 | tour-level main draws (G, M, A, F, O), 1995-2024 |
| WTA | main tour, ITF and qualifying (1968-2024) | 740,537 | tour-level main draws (G, PM, P, I, F, O, W), 2009-2024 |

**Why load lower tiers.** Most professionals spend years on the Challenger/ITF circuits before
reaching the tour. Feeding those matches into the ratings gives every player a meaningful Elo on
debut. With the current pipeline the Elo baseline's 2023-24 log-loss is 0.634 with main-tour
matches only and 0.604 with all tiers (WTA: 0.621 → 0.599).

**Cleaning** ([`data/loader.py`](../src/tennis_predictor/data/loader.py),
[`data/scores.py`](../src/tennis_predictor/data/scores.py)):

- Files are read as text and numeric columns are coerced. This survives upstream values such as
  seeds `"1F"` or draw size `"exho"`.
- **Outcomes** are parsed from the score: completed, retired, walkover, default, abandoned,
  unknown. Variants such as `RET`, `RE`, `Ret'd`, `W/O`, `W.0`, `Def.` and `Played and
  unfinished` are recognized.
  - Scores without a status marker whose last set is unfinished, or whose winner won fewer sets
    than the loser (`2-6 3-0`), are retirements.
  - A retirement before any game is a walkover.
- **Statistics** are used only for completed matches whose counts are internally consistent:
  - first serves in ≤ serve points;
  - points won on second serve ≤ second-serve points minus double faults;
  - break points saved ≤ break points faced;
  - total serve points between 3 and 10 per game played.

  Walkovers carry fake zero statistics upstream, and a few rows carry impossible volumes.
- **Level codes** that drift between seasons are normalized:
  - the 2023 ATP Finals is coded `A`;
  - WTA Indian Wells, Miami, Madrid and Beijing are `P` in some years and `PM` in others;
  - the 2015 WTA Finals and the 2021 WTA 250s are `W`;
  - the Olympics are `A` before 2024.

  WTA 125 events are not treated as tour level.
- Carpet is merged into Hard. Heights below 140 cm are discarded. Self-matches and exact
  duplicates (11k upstream WTA duplicates) are dropped. Exhibitions (Laver Cup, Next Gen
  Finals) are excluded everywhere.
- **Ordering.** Upstream files are not chronological: recent events are stored final-first, and
  team events are appended at the end of the year. Rows are sorted by
  `(tournament Monday, tourney_id, round order, match number)`, with the explicit round order
  `Q1 < … < Q5 < ER < R128 < … < R16 < RR < QF < SF < BR < F`. Never test qualifying with
  `startswith("Q")`, which also matches quarter-finals.

## 3. Leakage prevention

Leakage is the main way tennis prediction projects go wrong (see [§9](#the-prototype-audit)).
The design makes it structurally hard:

1. **Snapshot before update.** [`FeatureTracker`](../src/tennis_predictor/features/tracker.py)
   holds all player state. For each match the dataset builder calls `snapshot(a, b, context)`
   and only afterwards `update(winner, loser, context, result)`. `snapshot` never receives the
   outcome, so a match cannot influence its own features.
2. **Same-round matches are simultaneous.** Inside a round-robin stage the upstream match
   numbers are not chronological: the Tour Finals and team cups are numbered player by player.
   The builder therefore snapshots every match of a `(week, event, round)` block before applying
   any of its results. Knockout rounds are unaffected (one match per player per round).
3. **Ranks are as of the tournament Monday.** `winner_rank` / `loser_rank` in the match files
   equal the official ranking published before the event (verified against the ranking files
   for ~99-100% of rows).
4. **Chronological split.**
   - Train: up to 2020.
   - Validation: 2021-2022, used for early stopping, grid selection and calibration.
   - Test: 2023-2024, used once.

   No shuffling, no k-fold across time.
5. **Random orientation, mirrored only in training.** Each row shows the match from a random
   player's side. The side is a hash of the match, so it is stable and labels are balanced.
   XGBoost also trains on the player-swapped copy of each training row; both copies stay in the
   training split.
6. **Tests** ([`tests/test_leakage.py`](../tests/test_leakage.py)):
   - flipping the result of a match never changes its features (checked on many rows,
     including rows with previous meetings and retirements);
   - rank and head-to-head features match independent recomputations;
   - truncating history at any point, including mid-tournament, reproduces earlier rows
     bit for bit;
   - shuffling a round-robin stage changes nothing;
   - splits are strictly ordered in time.

   The round-robin test fails on the earlier version of the builder, so it would catch a
   regression.

## 4. Elo ratings

[`features/elo.py`](../src/tennis_predictor/features/elo.py) implements the FiveThirtyEight
tennis Elo described by [Kovalchik (2016)](https://doi.org/10.1515/jqas-2015-0059):

- expected score `1 / (1 + 10^(-Δ/400))` for a best-of-three match;
- dynamic K-factor `K = 250 / (n + 5)^0.4`, where `n` counts the player's rated matches, so
  ratings move fast for newcomers and settle for veterans;
- K × 1.1 at Grand Slams and K × 0.5 for retirements; walkovers, defaults and abandoned
  matches are skipped;
- **surface ratings** for hard (including carpet), clay and grass. The prediction uses the
  **50/50 blend** of overall and surface rating differences, the best combination in
  [Tennis Abstract's analysis](https://www.tennisabstract.com/blog/2019/12/03/an-introduction-to-tennis-elo/);
- **best-of-five adjustment.** The best-of-three probability `P3` is converted to a per-set
  probability `p` by inverting `P3 = p²(3 − 2p)` in closed form,
  `p = ½ + cos((arccos(1 − 2·P3) − 2π)/3)`, and then `P5 = p³(10 − 15p + 6p²)`.
  This reproduces Tennis Abstract's example of 81.4% → 86.7% (86.77% here). The same expectation is used in
  the rating update.

The parameters are the published ones. They were not tuned, to avoid spending validation data
on them.

## 5. Features

Each of the 37 features is either **antisymmetric** (it changes sign when the players are
swapped, e.g. A's Elo minus B's) or **symmetric** (match context). This contract is declared in
[`features/specs.py`](../src/tennis_predictor/features/specs.py) and tested for every feature.

- **Serve and return rates** are ratios of time-decayed sums (half-life 365 days), shrunk
  towards the running tour average with a prior worth two average matches:
  `(Σ decayed numerators + k·μ) / (Σ decayed denominators + k)`.
  - New or long-inactive players fall back to the tour mean instead of a noisy single match.
  - Points-weighted sums avoid the bias of averaging per-match ratios.
- **Form** is a decayed win rate (half-life 120 days) with a prior of 3 matches at 50%.
- **Head-to-head** is `(wins A − wins B) / (meetings + 2)`, so one past meeting is weak evidence.
- **Activity:**
  - rest days (capped at a year);
  - matches in the last 28 days;
  - games already played earlier in the same event (fatigue: upstream data has no match day,
    only the tournament Monday);
  - a flag for returning from a retirement.
- **Ranking:** log-rank ratio, log ranking points, and an unranked flag (unranked players get
  rank 2000 and 0 points).
- **Missing values:** only age, height and ranking points can be missing. XGBoost handles NaN
  natively; the logistic regression imputes 0, meaning "no difference", which keeps it symmetric.

| Feature | Group | Swap | Description |
|---|---|---|---|
| `elo_diff` | elo | antisymmetric | Overall Elo of A minus B |
| `surface_elo_diff` | elo | antisymmetric | Surface-specific Elo of A minus B |
| `elo_blend_logit` | elo | antisymmetric | Logit of the Elo win probability from the 50/50 overall/surface blend, adjusted for best-of-5 |
| `elo_matches_log_diff` | elo | antisymmetric | log1p(rated matches) of A minus B |
| `surface_matches_log_diff` | elo | antisymmetric | log1p(rated matches on this surface) of A minus B |
| `elo_matches_log_min` | elo | symmetric | log1p(rated matches) of the less experienced player |
| `log_rank_diff` | ranking | antisymmetric | ln(rank A) - ln(rank B); unranked players get a large rank |
| `log_points_diff` | ranking | antisymmetric | log1p(ranking points) of A minus B |
| `unranked_diff` | ranking | antisymmetric | 1 if only A is unranked, -1 if only B is |
| `spw_diff` | serve_return | antisymmetric | Serve points won rate, A minus B |
| `rpw_diff` | serve_return | antisymmetric | Return points won rate, A minus B |
| `ace_rate_diff` | serve_return | antisymmetric | Aces per serve point, A minus B |
| `df_rate_diff` | serve_return | antisymmetric | Double faults per serve point, A minus B |
| `first_in_diff` | serve_return | antisymmetric | First serves in per serve point, A minus B |
| `first_won_diff` | serve_return | antisymmetric | First-serve points won rate, A minus B |
| `second_won_diff` | serve_return | antisymmetric | Second-serve points won rate, A minus B |
| `bp_saved_diff` | serve_return | antisymmetric | Break points saved / faced, A minus B |
| `bp_converted_diff` | serve_return | antisymmetric | Break points converted / opportunities, A minus B |
| `spw_plus_rpw_diff` | serve_return | antisymmetric | (spw + rpw) of A minus B: overall point-winning edge |
| `stats_mass_log_min` | serve_return | symmetric | log1p(decayed serve points observed), minimum of A and B |
| `form_diff` | form | antisymmetric | Time-decayed, shrunk win rate of A minus B |
| `h2h_score` | head_to_head | antisymmetric | (wins A over B - wins B over A) / (meetings + prior) |
| `h2h_matches_log` | head_to_head | symmetric | log1p(previous meetings) |
| `rest_days_log_diff` | activity | antisymmetric | log1p(days since previous match, capped) of A minus B |
| `recent_matches_diff` | activity | antisymmetric | Matches in the trailing window, A minus B |
| `tourney_games_diff` | activity | antisymmetric | Games already played in this event, A minus B |
| `after_retirement_diff` | activity | antisymmetric | 1 if A retired in their previous match (minus same for B) |
| `age_diff` | player | antisymmetric | Age in years, A minus B |
| `height_diff` | player | antisymmetric | Height in cm, A minus B |
| `left_handed_diff` | player | antisymmetric | Left-handedness indicator, A minus B |
| `qualifier_diff` | player | antisymmetric | Qualifier / lucky loser indicator, A minus B |
| `wildcard_diff` | player | antisymmetric | Wild card indicator, A minus B |
| `best_of_5` | context | symmetric | 1 for best-of-five matches |
| `surface_clay` | context | symmetric | 1 on clay |
| `surface_grass` | context | symmetric | 1 on grass |
| `level_ordinal` | context | symmetric | Tournament importance (Grand Slam highest) |
| `round_ordinal` | context | symmetric | Round of the event (qualifying lowest, final highest) |

## 6. Models

| Model | Details |
|---|---|
| Rank baseline | Logistic regression on log-rank ratio and unranked flag, no intercept |
| Elo baseline | `sigmoid(elo_blend_logit)`, no fitting |
| Logistic regression | L2 (C = 1) on the 29 antisymmetric features, imputed with 0, scaled without centering, **no intercept**: exactly symmetric by construction |
| XGBoost | `hist` trees, learning rate 0.03, early stopping (100 rounds) on validation log-loss; a 3-candidate grid over depth / min child weight, selected on validation; swap-augmented training data; predictions averaged over both player orders |
| Calibration | Temperature scaling (Platt scaling without intercept) fitted on validation. Unlike full Platt scaling it preserves `p(A,B) = 1 − p(B,A)`. Fitted slopes: ATP 1.006, WTA 1.104 |

The XGBoost grid selected depth 5 / min child weight 50 for both tours, with early stopping at
iteration 466 (ATP) and 123 (WTA).

## 7. Evaluation

Test years 2023-2024. The rows are completed tour-level main-draw matches; retirements and
walkovers are excluded, as in Kovalchik (2016).

| Model | ATP log-loss | ATP Brier | ATP acc. | WTA log-loss | WTA Brier | WTA acc. |
|---|---:|---:|---:|---:|---:|---:|
| Rank baseline | 0.6282 | 0.2196 | 63.8% | 0.6312 | 0.2209 | 63.8% |
| Elo baseline | 0.6045 | 0.2098 | 65.6% | 0.5991 | 0.2072 | 66.3% |
| Logistic regression | **0.5915** | **0.2044** | **66.6%** | 0.5918 | **0.2040** | **67.7%** |
| XGBoost (calibrated) | 0.5949 | 0.2058 | 65.8% | **0.5916** | **0.2040** | 67.4% |

Paired bootstrap (2,000 resamples of test matches), log-loss difference with 95% CI:

| Comparison | ATP | WTA |
|---|---|---|
| Logistic − Elo | −0.0130 [−0.0174, −0.0091] | −0.0073 [−0.0114, −0.0033] |
| XGBoost − Elo | −0.0096 [−0.0136, −0.0057] | −0.0074 [−0.0109, −0.0039] |
| XGBoost − Logistic | **+0.0034 [+0.0016, +0.0054]** | −0.0001 [−0.0023, +0.0020] |

Observations:

- **Learned models beat Elo, modestly but significantly.** The lift comes mostly from the
  experience counts, ranking points, serve/return edge, activity and age. Elo remains the
  backbone: the three Elo features carry ~80% (ATP) and ~84% (WTA) of XGBoost's split gain.
- **A linear model is enough.** On ATP the logistic regression is significantly better than
  XGBoost; on WTA they are tied. Its coefficients are interpretable. The largest standardized
  ATP coefficients:
  - `elo_blend_logit` (+1.26);
  - `surface_elo_diff` (−0.38), which partly offsets the blend;
  - `surface_matches_log_diff` (+0.27) and `elo_matches_log_diff` (−0.20);
  - `log_points_diff` (+0.18);
  - `qualifier_diff` (+0.17), because qualifiers arrive match-tight and under-rated;
  - `age_diff` (−0.16);
  - `tourney_games_diff` (−0.16), i.e. fatigue;
  - `form_diff` (−0.14), recent form beyond what Elo already captures regresses.
- **Calibration** is good for all models (ECE ≤ 0.032). Temperature scaling barely changes ATP
  (slope 1.006) and sharpens WTA slightly (1.104).
- **Breakdowns.** Best-of-five Grand Slam matches are far more predictable: ATP logistic 73.3%
  accuracy, log-loss 0.505. ATP 250/500 events are the hardest (64.7%).
- **Context.** Kovalchik (2016) reports 70% for FiveThirtyEight Elo and 72% for bookmakers on ATP
  2014. Later studies on 2018-2025 report 64-66% for Elo and 69-74% for bookmakers. Our Elo
  (65.6% ATP, 66.3% WTA) sits in that range, and the learned models gain 0.3-1.4 points of accuracy.

Reliability diagrams and feature importances for both tours are in [`reports/`](../reports).

## 8. Serving predictions

`tennis train` saves:

- `model.ubj`, the XGBoost model in its native format;
- the tracker state as parquet (players, head-to-head) plus JSON;
- `metadata.json`: settings, library versions, selected parameters, calibration slope, logistic
  coefficients and a feature schema version;
- the validation and test predictions.

Before saving, every player's rank and points are replaced by the **official ranking list at
the data cutoff**, so that serving uses as-of ranks exactly like training.

`Predictor.predict` resolves names, builds the context (surface, format, the tour-specific level
of the event, round, date) and calls the same `snapshot` used in training. It returns:

- the symmetric, calibrated probability;
- the Elo-only probability;
- the five largest TreeSHAP contributions, computed over the trees kept by early stopping,
  averaged over both player orders and expressed in calibrated log-odds;
- warnings for players inactive for over a year.

A test checks that serving a past final from the state just before it reproduces that match's
training row, apart from the in-event features (games already played, qualifier and wild-card
flags), and that the served probability equals the calibrated symmetric model output.

## The prototype audit

The repository started as `super_ia.py`, an interactive script reporting **66.7% (ATP) and
67.5% (WTA)** accuracy. Re-running its training code on the same data reproduces those numbers.
A careful audit found that they measured leakage rather than skill:

| # | Defect | Consequence |
|---|---|---|
| 1 | The current match's statistics were appended to each player's rolling history *before* the average used as that match's feature was computed | Each feature contained the result being predicted |
| 2 | `train_test_split` with shuffling over 2000-2023, applied after mirroring every match as `(X, 1)` and `(−X, 0)` | Trained on the future; ~80% of test rows had their mirror image in training |
| 3 | Yearly files used in file order, which is not chronological | About 10% of consecutive player appearances went backwards in time |
| 4 | `bp_saved_pct = bpSaved / max(bpSaved, 1)` and a constant `bp_converted_pct = 0` | One feature was a 0/1 indicator, another was dead |
| 5 | Players keyed by display name | Namesakes merged, renamed players split |
| 6 | No baseline, accuracy only, 2024 data ignored, retraining on every launch (~1.5 min) | No way to tell whether the model was any good |

The same seven features were then recomputed honestly:
- completed matches with statistics only (no retirements or walkovers);
- sorted chronologically;
- each player's 30-match average taken *before* adding the match;
- one randomly oriented row per match;
- the prototype's XGBoost settings, trained on 2000-2018 and tested on 2019-2023.

They score **60.4% (ATP) and 62.8% (WTA)**. That is below always picking the better-ranked
player on the same test matches (63.2% and 64.6%).
The rewrite keeps the good ideas: per-serve-point rates, player differences, rolling form, fuzzy
name lookup and separate ATP/WTA models. It rebuilds everything else around the guarantees in §3.

## 10. Limitations and future work

- **No match dates.** Upstream rows carry the tournament Monday only. Rest and fatigue are
  approximated with tournament and game counts.
- **Missing context.** There is no injury information, no betting odds and no in-match data.
  Bookmaker odds would be the obvious next feature, and the honest benchmark.
- **Match Charting Project.** Its shot-level data is vendored but unused: it covers only 2-21% of
  tour matches, heavily biased towards finals and big names.
- **Possible extensions:**
  - Glicko-2 or margin-of-victory Elo ([Kovalchik 2020](https://doi.org/10.1016/j.ijforecast.2020.01.006));
  - common-opponent features ([Sipko & Knottenbelt 2015](https://www.doc.ic.ac.uk/teaching/distinguished-projects/2015/m.sipko.pdf));
  - a walk-forward backtest by season;
  - a scheduled refresh of the data.
