"""Sequential, leakage-free computation of pre-match features.

Matches are processed in chronological order. For every match the caller first asks for a
:meth:`FeatureTracker.snapshot` of the two players and only then calls
:meth:`FeatureTracker.update` with the result. ``snapshot`` never receives the outcome, so a
match can never leak into its own features, and the exact same code produces the features
used for training and for serving predictions.
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pandas as pd

from tennis_predictor.config import EloConfig, FeatureConfig, Tour
from tennis_predictor.data.schema import PLAYED_OUTCOMES, SURFACES, Outcome
from tennis_predictor.features.elo import Rating, match_win_probability, update_ratings

#: Rates tracked per player, computed from serve statistics.
STAT_NAMES: Final = (
    "spw",
    "rpw",
    "ace_rate",
    "df_rate",
    "first_in",
    "first_won",
    "second_won",
    "bp_saved",
    "bp_converted",
)
#: Typical tour values, only used before any statistic has been observed.
DEFAULT_RATES: Final = (0.62, 0.38, 0.07, 0.04, 0.60, 0.72, 0.51, 0.60, 0.40)
_N_STATS: Final = len(STAT_NAMES)
_SURFACE_INDEX: Final = {surface: index for index, surface in enumerate(SURFACES)}
_LOG2: Final = math.log(2.0)
_P_CLIP: Final = 1e-6

# Positions inside a per-match stat tuple (same order as schema.STAT_FIELDS).
_ACE, _DF, _SVPT, _1IN, _1WON, _2WON, _SVGMS, _BPS, _BPF = range(9)


@dataclass(frozen=True, slots=True)
class PlayerInfo:
    """Facts known about a player before a match (from the match row or the latest state)."""

    player_id: int
    name: str = ""
    rank: float = math.nan
    rank_points: float = math.nan
    age: float = math.nan
    height: float = math.nan
    hand: str = "U"
    entry: str = ""
    ioc: str = ""


@dataclass(frozen=True, slots=True)
class MatchContext:
    """Match-level facts known before the first point."""

    day: int
    tourney_id: str
    surface: str
    best_of: int
    level_ordinal: int
    round_order: int
    is_slam: bool


@dataclass(frozen=True, slots=True)
class MatchResult:
    """What happened in a match: only passed to :meth:`FeatureTracker.update`."""

    outcome: Outcome
    winner_stats: tuple[float, ...] | None = None
    loser_stats: tuple[float, ...] | None = None
    winner_games: int = 0
    loser_games: int = 0


@dataclass(slots=True)
class PlayerState:
    """Everything the tracker remembers about one player."""

    elo: Rating
    surface_elo: list[Rating]
    stat_num: list[float] = field(default_factory=lambda: [0.0] * _N_STATS)
    stat_den: list[float] = field(default_factory=lambda: [0.0] * _N_STATS)
    stat_day: int = 0
    form_wins: float = 0.0
    form_matches: float = 0.0
    form_day: int = 0
    last_day: int = -1
    recent_days: deque[int] = field(default_factory=deque)
    current_tourney: str = ""
    tourney_games: int = 0
    retired_last: bool = False
    name: str = ""
    ioc: str = ""
    hand: str = "U"
    height: float = math.nan
    rank: float = math.nan
    rank_points: float = math.nan
    age: float = math.nan
    info_day: int = 0


def _decay(days: int, half_life: float) -> float:
    return math.exp(-_LOG2 * max(days, 0) / half_life)


def _stat_pairs(own: Sequence[float], opp: Sequence[float]) -> tuple[tuple[float, float], ...]:
    """(numerator, denominator) of every tracked rate for one player in one match."""
    opp_won = opp[_1WON] + opp[_2WON]
    return (
        (own[_1WON] + own[_2WON], own[_SVPT]),
        (opp[_SVPT] - opp_won, opp[_SVPT]),
        (own[_ACE], own[_SVPT]),
        (own[_DF], own[_SVPT]),
        (own[_1IN], own[_SVPT]),
        (own[_1WON], own[_1IN]),
        (own[_2WON], own[_SVPT] - own[_1IN]),
        (own[_BPS], own[_BPF]),
        (opp[_BPF] - opp[_BPS], opp[_BPF]),
    )


def _finite_or_nan(value: float) -> float:
    return value if math.isfinite(value) else math.nan


class FeatureTracker:
    """Online state of ratings, statistics, form, head-to-head and activity for one tour."""

    def __init__(self, elo: EloConfig, features: FeatureConfig, tour: Tour) -> None:
        self.elo_cfg = elo
        self.feature_cfg = features
        self.tour: Tour = tour
        self.players: dict[int, PlayerState] = {}
        self.h2h: dict[int, list[int]] = {}
        self.global_num = [0.0] * _N_STATS
        self.global_den = [0.0] * _N_STATS
        self.global_count = 0
        self.last_day = 0

    # ------------------------------------------------------------------ state access
    def _new_state(self) -> PlayerState:
        initial = self.elo_cfg.initial
        return PlayerState(elo=Rating(initial), surface_elo=[Rating(initial) for _ in SURFACES])

    def _state(self, player_id: int) -> PlayerState:
        state = self.players.get(player_id)
        if state is None:
            state = self._new_state()
            self.players[player_id] = state
        return state

    def has_player(self, player_id: int) -> bool:
        """Whether the player has appeared in the processed history."""
        return player_id in self.players

    def player_info(self, player_id: int, *, as_of_day: int) -> PlayerInfo:
        """Latest known facts about a player, with age advanced to ``as_of_day``."""
        state = self.players[player_id]
        age = (
            state.age + (as_of_day - state.info_day) / 365.25
            if math.isfinite(state.age)
            else math.nan
        )
        return PlayerInfo(
            player_id=player_id,
            name=state.name,
            rank=state.rank,
            rank_points=state.rank_points,
            age=age,
            height=state.height,
            hand=state.hand,
            entry="",
            ioc=state.ioc,
        )

    # ------------------------------------------------------------------ feature snapshot
    def _rates(self, state: PlayerState | None, day: int) -> tuple[list[float], float]:
        """Shrunk, time-decayed rates of a player plus the decayed serve-point mass."""
        cfg = self.feature_cfg
        count = self.global_count
        rates: list[float] = []
        decay = _decay(day - state.stat_day, cfg.stats_half_life_days) if state else 0.0
        for index in range(_N_STATS):
            global_den = self.global_den[index]
            if count == 0 or global_den <= 0.0:
                rates.append(DEFAULT_RATES[index])
                continue
            prior_mean = self.global_num[index] / global_den
            prior_weight = cfg.stats_prior_matches * global_den / count
            if state is None:
                rates.append(prior_mean)
                continue
            num = state.stat_num[index] * decay + prior_weight * prior_mean
            den = state.stat_den[index] * decay + prior_weight
            rates.append(num / den)
        mass = state.stat_den[0] * decay if state else 0.0
        return rates, mass

    def _form(self, state: PlayerState | None, day: int) -> float:
        prior = self.feature_cfg.form_prior_matches
        if state is None:
            return 0.5
        decay = _decay(day - state.form_day, self.feature_cfg.form_half_life_days)
        return (state.form_wins * decay + 0.5 * prior) / (state.form_matches * decay + prior)

    def _activity(
        self, state: PlayerState | None, ctx: MatchContext
    ) -> tuple[float, int, int, int]:
        cfg = self.feature_cfg
        if state is None or state.last_day < 0:
            return math.log1p(cfg.rest_cap_days), 0, 0, 0
        rest = math.log1p(min(max(ctx.day - state.last_day, 0), cfg.rest_cap_days))
        horizon = ctx.day - cfg.recent_window_days
        recent = sum(1 for played in state.recent_days if played > horizon)
        games = state.tourney_games if state.current_tourney == ctx.tourney_id else 0
        return rest, recent, games, int(state.retired_last)

    def _rank(self, info: PlayerInfo) -> tuple[float, float, int]:
        """(log rank, log1p points, unranked flag)."""
        if math.isfinite(info.rank) and info.rank > 0:
            points = math.log1p(info.rank_points) if math.isfinite(info.rank_points) else math.nan
            return math.log(info.rank), points, 0
        return math.log(self.feature_cfg.unranked_rank), 0.0, 1

    def snapshot(self, a: PlayerInfo, b: PlayerInfo, ctx: MatchContext) -> list[float]:
        """Features of ``a`` versus ``b`` before the match, in ``specs.FEATURE_NAMES`` order.

        This method only reads state; it must be called before :meth:`update` for the
        same match.
        """
        state_a = self.players.get(a.player_id)
        state_b = self.players.get(b.player_id)
        initial = self.elo_cfg.initial
        surface_index = _SURFACE_INDEX.get(ctx.surface, -1)

        elo_a = state_a.elo.value if state_a else initial
        elo_b = state_b.elo.value if state_b else initial
        n_a = state_a.elo.matches if state_a else 0
        n_b = state_b.elo.matches if state_b else 0
        if surface_index >= 0:
            surf_a = state_a.surface_elo[surface_index] if state_a else Rating(initial)
            surf_b = state_b.surface_elo[surface_index] if state_b else Rating(initial)
            surface_diff = surf_a.value - surf_b.value
            surface_n_a, surface_n_b = surf_a.matches, surf_b.matches
        else:
            surface_diff, surface_n_a, surface_n_b = elo_a - elo_b, n_a, n_b
        weight = self.elo_cfg.surface_weight
        blend = (1.0 - weight) * (elo_a - elo_b) + weight * surface_diff
        p_elo = min(max(match_win_probability(blend, ctx.best_of), _P_CLIP), 1.0 - _P_CLIP)

        log_rank_a, points_a, unranked_a = self._rank(a)
        log_rank_b, points_b, unranked_b = self._rank(b)

        rates_a, mass_a = self._rates(state_a, ctx.day)
        rates_b, mass_b = self._rates(state_b, ctx.day)
        rate_diffs = [rate_a - rate_b for rate_a, rate_b in zip(rates_a, rates_b, strict=True)]

        lo, hi = (
            (a.player_id, b.player_id) if a.player_id < b.player_id else (b.player_id, a.player_id)
        )
        meetings = self.h2h.get((lo << 32) | hi)
        if meetings is None:
            h2h_score, h2h_n = 0.0, 0
        else:
            wins_a, wins_b = (
                (meetings[0], meetings[1]) if a.player_id == lo else (meetings[1], meetings[0])
            )
            h2h_n = wins_a + wins_b
            h2h_score = (wins_a - wins_b) / (h2h_n + self.feature_cfg.h2h_prior)

        rest_a, recent_a, games_a, retired_a = self._activity(state_a, ctx)
        rest_b, recent_b, games_b, retired_b = self._activity(state_b, ctx)

        log_n_a, log_n_b = math.log1p(n_a), math.log1p(n_b)
        return [
            elo_a - elo_b,
            surface_diff,
            math.log(p_elo / (1.0 - p_elo)),
            log_n_a - log_n_b,
            math.log1p(surface_n_a) - math.log1p(surface_n_b),
            min(log_n_a, log_n_b),
            log_rank_a - log_rank_b,
            points_a - points_b,
            float(unranked_a - unranked_b),
            *rate_diffs,
            rate_diffs[0] + rate_diffs[1],
            min(math.log1p(mass_a), math.log1p(mass_b)),
            self._form(state_a, ctx.day) - self._form(state_b, ctx.day),
            h2h_score,
            math.log1p(h2h_n),
            rest_a - rest_b,
            float(recent_a - recent_b),
            float(games_a - games_b),
            float(retired_a - retired_b),
            _finite_or_nan(a.age - b.age),
            _finite_or_nan(a.height - b.height),
            float((a.hand == "L") - (b.hand == "L")),
            float((a.entry in {"Q", "LL"}) - (b.entry in {"Q", "LL"})),
            float((a.entry == "WC") - (b.entry == "WC")),
            float(ctx.best_of == 5),
            float(ctx.surface == "Clay"),
            float(ctx.surface == "Grass"),
            float(ctx.level_ordinal),
            float(ctx.round_order),
        ]

    # ------------------------------------------------------------------ update
    def _refresh_info(self, state: PlayerState, info: PlayerInfo, day: int) -> None:
        if day < state.info_day:
            return
        state.info_day = day
        state.name = info.name or state.name
        state.ioc = info.ioc or state.ioc
        state.hand = info.hand if info.hand in {"L", "R"} else state.hand
        if math.isfinite(info.height):
            state.height = info.height
        state.rank = info.rank
        state.rank_points = info.rank_points
        if math.isfinite(info.age):
            state.age = info.age

    def _update_stats(
        self, state: PlayerState, pairs: tuple[tuple[float, float], ...], day: int
    ) -> None:
        decay = _decay(day - state.stat_day, self.feature_cfg.stats_half_life_days)
        for index, (num, den) in enumerate(pairs):
            state.stat_num[index] = state.stat_num[index] * decay + num
            state.stat_den[index] = state.stat_den[index] * decay + den
            self.global_num[index] += num
            self.global_den[index] += den
        state.stat_day = day
        self.global_count += 1

    def _update_activity(
        self, state: PlayerState, ctx: MatchContext, won: bool, games: int
    ) -> None:
        cfg = self.feature_cfg
        decay = _decay(ctx.day - state.form_day, cfg.form_half_life_days)
        state.form_wins = state.form_wins * decay + float(won)
        state.form_matches = state.form_matches * decay + 1.0
        state.form_day = ctx.day
        state.last_day = ctx.day
        state.recent_days.append(ctx.day)
        horizon = ctx.day - cfg.recent_window_days
        while state.recent_days and state.recent_days[0] <= horizon:
            state.recent_days.popleft()
        if state.current_tourney != ctx.tourney_id:
            state.current_tourney = ctx.tourney_id
            state.tourney_games = 0
        state.tourney_games += games

    def update(
        self, winner: PlayerInfo, loser: PlayerInfo, ctx: MatchContext, result: MatchResult
    ) -> None:
        """Fold a finished match into the state.

        Walkovers, defaults, abandoned and unknown results only refresh biographical facts.
        Retirements count for ratings (with a reduced K), form, head-to-head and activity, but
        their partial serve statistics are ignored.
        """
        w_state = self._state(winner.player_id)
        l_state = self._state(loser.player_id)
        self._refresh_info(w_state, winner, ctx.day)
        self._refresh_info(l_state, loser, ctx.day)
        if result.outcome not in PLAYED_OUTCOMES:
            return

        cfg = self.elo_cfg
        multiplier = cfg.slam_multiplier if ctx.is_slam else 1.0
        if result.outcome is Outcome.RETIRED:
            multiplier *= cfg.retirement_multiplier
        update_ratings(
            w_state.elo, l_state.elo, best_of=ctx.best_of, k_multiplier=multiplier, cfg=cfg
        )
        surface_index = _SURFACE_INDEX.get(ctx.surface, -1)
        if surface_index >= 0:
            update_ratings(
                w_state.surface_elo[surface_index],
                l_state.surface_elo[surface_index],
                best_of=ctx.best_of,
                k_multiplier=multiplier,
                cfg=cfg,
            )

        if result.outcome is Outcome.COMPLETED and result.winner_stats and result.loser_stats:
            self._update_stats(
                w_state, _stat_pairs(result.winner_stats, result.loser_stats), ctx.day
            )
            self._update_stats(
                l_state, _stat_pairs(result.loser_stats, result.winner_stats), ctx.day
            )

        games = result.winner_games + result.loser_games
        self._update_activity(w_state, ctx, won=True, games=games)
        self._update_activity(l_state, ctx, won=False, games=games)
        w_state.retired_last = False
        l_state.retired_last = result.outcome is Outcome.RETIRED

        lo, hi = sorted((winner.player_id, loser.player_id))
        record = self.h2h.setdefault((lo << 32) | hi, [0, 0])
        record[0 if winner.player_id == lo else 1] += 1
        self.last_day = max(self.last_day, ctx.day)

    def apply_rankings(self, ranks: dict[int, tuple[float, float]]) -> None:
        """Replace every player's rank and points with an official ranking list.

        Players missing from the list are unranked. Only used once the whole history has
        been replayed, to serve predictions with the ranks in force at the data cutoff.
        """
        for player_id, state in self.players.items():
            state.rank, state.rank_points = ranks.get(player_id, (math.nan, math.nan))

    # ------------------------------------------------------------------ introspection
    def players_frame(self) -> pd.DataFrame:
        """One row per player with identity, latest facts, ratings and activity."""
        rows: list[dict[str, Any]] = []
        for player_id, state in self.players.items():
            row: dict[str, Any] = {
                "player_id": player_id,
                "name": state.name,
                "ioc": state.ioc,
                "hand": state.hand,
                "height": state.height,
                "rank": state.rank,
                "rank_points": state.rank_points,
                "age": state.age,
                "info_day": state.info_day,
                "elo": state.elo.value,
                "elo_matches": state.elo.matches,
                "stat_day": state.stat_day,
                "form_wins": state.form_wins,
                "form_matches": state.form_matches,
                "form_day": state.form_day,
                "last_day": state.last_day,
                "recent_days": list(state.recent_days),
                "current_tourney": state.current_tourney,
                "tourney_games": state.tourney_games,
                "retired_last": state.retired_last,
            }
            for surface, rating in zip(SURFACES, state.surface_elo, strict=True):
                row[f"elo_{surface.lower()}"] = rating.value
                row[f"elo_{surface.lower()}_matches"] = rating.matches
            for index, name in enumerate(STAT_NAMES):
                row[f"num_{name}"] = state.stat_num[index]
                row[f"den_{name}"] = state.stat_den[index]
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        """Persist the full state as parquet + JSON (no pickle)."""
        directory.mkdir(parents=True, exist_ok=True)
        self.players_frame().to_parquet(directory / "players.parquet", index=False)
        h2h = pd.DataFrame(
            [(key, wins[0], wins[1]) for key, wins in self.h2h.items()],
            columns=["key", "wins_lo", "wins_hi"],
        )
        h2h.to_parquet(directory / "h2h.parquet", index=False)
        meta = {
            "tour": self.tour,
            "last_day": self.last_day,
            "global_num": self.global_num,
            "global_den": self.global_den,
            "global_count": self.global_count,
            "elo": self.elo_cfg.model_dump(mode="json"),
            "features": self.feature_cfg.model_dump(mode="json"),
        }
        (directory / "tracker.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> FeatureTracker:
        """Restore a tracker written by :meth:`save`."""
        meta = json.loads((directory / "tracker.json").read_text(encoding="utf-8"))
        tracker = cls(
            EloConfig.model_validate(meta["elo"]),
            FeatureConfig.model_validate(meta["features"]),
            meta["tour"],
        )
        tracker.last_day = int(meta["last_day"])
        tracker.global_num = [float(value) for value in meta["global_num"]]
        tracker.global_den = [float(value) for value in meta["global_den"]]
        tracker.global_count = int(meta["global_count"])

        players = pd.read_parquet(directory / "players.parquet")
        for row in players.to_dict(orient="records"):
            state = PlayerState(
                elo=Rating(float(row["elo"]), int(row["elo_matches"])),
                surface_elo=[
                    Rating(float(row[f"elo_{s.lower()}"]), int(row[f"elo_{s.lower()}_matches"]))
                    for s in SURFACES
                ],
                stat_num=[float(row[f"num_{name}"]) for name in STAT_NAMES],
                stat_den=[float(row[f"den_{name}"]) for name in STAT_NAMES],
                stat_day=int(row["stat_day"]),
                form_wins=float(row["form_wins"]),
                form_matches=float(row["form_matches"]),
                form_day=int(row["form_day"]),
                last_day=int(row["last_day"]),
                recent_days=deque(int(day) for day in row["recent_days"]),
                current_tourney=str(row["current_tourney"]),
                tourney_games=int(row["tourney_games"]),
                retired_last=bool(row["retired_last"]),
                name=str(row["name"]),
                ioc=str(row["ioc"]),
                hand=str(row["hand"]),
                height=float(row["height"]),
                rank=float(row["rank"]),
                rank_points=float(row["rank_points"]),
                age=float(row["age"]),
                info_day=int(row["info_day"]),
            )
            tracker.players[int(row["player_id"])] = state

        h2h = pd.read_parquet(directory / "h2h.parquet")
        for key, wins_lo, wins_hi in h2h.itertuples(index=False, name=None):
            tracker.h2h[int(key)] = [int(wins_lo), int(wins_hi)]
        return tracker
