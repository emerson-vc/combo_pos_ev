"""Price thin rungs from the liquid rungs of the same market.

- Count props (2+ HR, 3+ TD...): treat the stat as Poisson, solve λ from the liquid lower rung,
  read off the thin rung, then apply a per-kind correction (empirical / Poisson tail ratio fit on
  history) for over/under-dispersion.
- Line ladders (team totals, spreads, game/half/quarter totals, yardage, attempts): fit
  logit P(over X) linearly in X across the liquid rungs and read thin rungs off the curve, clamped
  inside the rung's own bid/ask. A 0¢/6¢ "over 42.5" is priced from the 24.5–35.5 rungs that trade.
"""

import dataclasses
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import poisson

from app.legs import Leg, market_key

# Total bases / H+R+RBI jump in chunks (a homer is 4 TB), so a Poisson ladder doesn't fit them.
COUNT_KINDS = {"hr", "hit", "rbi", "ks", "td", "pass_td", "pass_int", "rec"}
THIN_REL_SPREAD = 0.6  # bid/ask gap larger than 60% of the midpoint -> midpoint is noise
# Line ladders priced by a logit-linear curve across their liquid rungs.
CURVE_KINDS = {"team_total", "team_spread", "game_total", "f5_total", "f5_spread", "h1_total", "h1_spread",
               "h1_team_total", "q1_total", "q1_spread", "rec_yds", "rush_yds", "pass_yds", "rr_yds",
               "pass_att", "pass_cmp", "rush_att", "tb", "hrr", "outs"}
MAX_K = 6
BASE_RUNGS = (1, 2, 3)  # liquid rungs we price long shots from
MIN_GAMES = {"MLB": 20, "MLB_SP": 10, "NFL": 6}  # player-seasons with fewer games are too noisy to calibrate on
_LAM_GRID = np.concatenate([np.geomspace(1e-6, 1, 400), np.linspace(1.0025, 30, 4000)])


def is_thin(leg: Leg) -> bool:
    return leg.yes_bid <= 0 or leg.spread > THIN_REL_SPREAD * max(leg.mid, 1e-9)


def tail(lam: float, k: int) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    return float(poisson.sf(k - 1, lam))


def solve_lambda(p: float, k: int) -> float:
    return brentq(lambda lam: tail(lam, k) - p, 1e-9, 50.0)


def lambdas_for(p: np.ndarray, j: int) -> np.ndarray:
    """Vectorized inverse of tail(., j) via a monotone grid (P(X>=j) increases with λ)."""
    return np.interp(p, poisson.sf(j - 1, _LAM_GRID), _LAM_GRID)


def fit_corrections(players: pd.DataFrame, kinds: list[str], group_cols: list[str], min_games: int) -> dict:
    """{(kind, j, k): empirical P(>=k) / Poisson P(>=k) given P(>=j)}, pooled over player-seasons."""
    out = {}
    for kind in (k for k in kinds if k in players):
        n = players.groupby(group_cols).size()
        keep = n[n >= min_games].index
        if len(keep) == 0:
            continue
        weights = n.loc[keep].to_numpy(dtype=float)
        rungs = {k: (players[kind] >= k).groupby([players[c] for c in group_cols]).mean().loc[keep].to_numpy()
                 for k in range(1, MAX_K + 1)}
        for j in BASE_RUNGS:
            pj = rungs[j]
            ok = (pj > 0.005) & (pj < 0.995)
            if ok.sum() < 30:
                continue
            lams = lambdas_for(pj[ok], j)
            for k in range(j + 1, MAX_K + 1):
                emp = float((rungs[k][ok] * weights[ok]).sum())
                pois = float((poisson.sf(k - 1, lams) * weights[ok]).sum())
                if emp > 0 and pois > 0:
                    out[(kind, j, k)] = emp / pois
    return out


class Ladder:
    def __init__(self, corrections: dict | None = None):
        self.corrections = corrections or {}

    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame]) -> "Ladder":
        corr = {}
        if "MLB_player" in frames:
            corr.update(fit_corrections(frames["MLB_player"], ["hr", "hit", "rbi"], ["season", "player_id"],
                                        MIN_GAMES["MLB"]))
        if "MLB_pitcher" in frames:
            corr.update(fit_corrections(frames["MLB_pitcher"], ["ks"], ["season", "player_id"], MIN_GAMES["MLB_SP"]))
        if "NFL_player" in frames:
            corr.update(fit_corrections(frames["NFL_player"], ["td", "pass_td", "pass_int", "rec"], ["season", "player_id"],
                                        MIN_GAMES["NFL"]))
        return cls(corr)

    def implied(self, kind: str, base_p: float, j: int, k: int) -> float:
        lam = solve_lambda(base_p, j)
        p = tail(lam, k) * self.corrections.get((kind, j, k), 1.0)
        return float(min(p, base_p))

    def apply(self, legs: list[Leg]) -> list[Leg]:
        """Return legs with thin rungs re-priced: count props by Poisson ladder, lines by curve."""
        return curve_fit(self._apply_counts(legs))

    def _apply_counts(self, legs: list[Leg]) -> list[Leg]:
        ladders: dict[tuple, list[Leg]] = defaultdict(list)
        for leg in legs:
            if leg.kind in COUNT_KINDS and leg.player and leg.count_k:
                ladders[(leg.player, leg.kind)].append(leg)
        replaced: dict[str, Leg] = {}
        for (_, kind), rungs in ladders.items():
            rungs.sort(key=lambda leg: leg.count_k)
            for i, leg in enumerate(rungs):
                if not is_thin(leg):
                    continue
                base = next((r for r in reversed(rungs[:i]) if not is_thin(r) and 0 < r.mid < 1), None)
                if base is None:
                    continue
                p = self.implied(kind, base.mid, base.count_k, leg.count_k)
                replaced[leg.ticker] = dataclasses.replace(leg, p_model=p, p_source="ladder")
        return [replaced.get(leg.ticker, leg) for leg in legs]



def curve_fit(legs: list[Leg]) -> list[Leg]:
    """Re-price thin rungs of line ladders from a logit-linear fit of their liquid rungs."""
    ladders: dict[str, list[Leg]] = defaultdict(list)
    for leg in legs:
        if leg.kind in CURVE_KINDS and leg.threshold is not None and leg.p_source == "market":
            ladders[market_key(leg.event_ticker, leg.ticker)].append(leg)
    replaced: dict[str, Leg] = {}
    for rungs in ladders.values():
        liquid = [r for r in rungs if not is_thin(r) and 0.02 < r.mid < 0.98]
        if len(liquid) < 2 or len({r.threshold for r in liquid}) < 2:
            continue
        x = np.array([r.threshold for r in liquid])
        y = logit(np.array([r.mid for r in liquid]))
        w = 1 / np.array([max(r.spread, 0.01) for r in liquid])  # tighter markets count more
        slope, icpt = np.polyfit(x, y, 1, w=w)
        if slope >= 0:  # P(over X) must fall as X rises
            continue
        for leg in rungs:
            if is_thin(leg):
                p = float(expit(icpt + slope * leg.threshold))
                hi = leg.yes_ask if leg.yes_ask > 0 else 1.0
                p = min(max(p, leg.yes_bid), hi)  # never outside the rung's own market
                replaced[leg.ticker] = dataclasses.replace(leg, p_model=p, p_source="curve")
    return [replaced.get(leg.ticker, leg) for leg in legs]
