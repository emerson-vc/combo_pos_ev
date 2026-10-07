"""Estimate the true probability that both legs of a same-team combo hit.

For each pair we rebuild the two legs on historical games (same role / lineup slot, same
count threshold, or a quantile-matched threshold for yardage and team scoring), fit the
tetrachoric ρ on that 2x2 table, and plug it into the copula with today's Kalshi marginals.
"""

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app import config
from app.legs import PLAYER_KINDS, Leg
from app.model.copula import joint_prob, tetrachoric
from app.model.ladder import Ladder

YARD_KINDS = {"rec_yds", "rush_yds", "pass_yds"}
TEAM_COL = {
    # Team outcomes are residuals vs the pre-game line, so "over X" means "beat expectation".
    "NFL": {"team_win": "won", "team_spread": "margin_resid", "team_total": "pts_resid"},
    "MLB": {"team_win": "won", "team_spread": "margin_resid", "team_total": "runs_resid"},
}
# Without data/mlb_lines.parquet (Kalshi pre-game prices), MLB falls back to raw runs/margins.
MLB_RAW_TEAM_COL = {"team_win": "won", "team_spread": "margin", "team_total": "runs"}
QUANTILE_TEAM_KINDS = {"team_spread", "team_total"}
MIN_N, MIN_N11, MIN_N11_LAST = 300, 8, 3
# History columns are named after leg kinds.
PLAYER_STAT_COLS = ["td", "rec_yds", "rush_yds", "rec", "pass_yds", "pass_td"]


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(s.split())


def mlb_slot_bucket(slot: int | None) -> str:
    if slot is None:
        return "ANY"
    return "TOP" if slot <= 2 else "MID" if slot <= 5 else "BOT"


@dataclass
class Estimate:
    q: float
    q_lo: float
    q_hi: float
    rho: float | None
    rho_lo: float | None
    rho_hi: float | None
    n: int
    n11: int
    role_a: str
    role_b: str
    method: str  # "copula" | "subset"


class Model:
    def __init__(self, data_dir=config.DATA_DIR):
        self.frames: dict[str, pd.DataFrame] = {}
        self._cache: dict[tuple, dict | None] = {}
        self._pools: dict[tuple, object] = {}
        self.lineup_slots: dict[str, int] = {}  # norm name -> today's batting slot
        self.team_cols = dict(TEAM_COL)
        nfl_p, nfl_t = data_dir / "nfl_players.parquet", data_dir / "nfl_teams.parquet"
        mlb_b, mlb_t = data_dir / "mlb_batters.parquet", data_dir / "mlb_teams.parquet"
        if nfl_p.exists() and nfl_t.exists():
            players = pd.read_parquet(nfl_p)
            teams = pd.read_parquet(nfl_t)
            players["norm"] = players["name"].fillna("").map(norm_name)
            self.frames["NFL_team"] = teams
            self.frames["NFL_player"] = players.merge(
                teams[["game_id", "team", "won", "margin_resid", "pts_resid"]], on=["game_id", "team"])
            self.frames["NFL_roles"] = (players.sort_values(["season", "week"])
                                        .drop_duplicates("norm", keep="last").set_index("norm")["role"])
        if mlb_b.exists() and mlb_t.exists():
            bat = pd.read_parquet(mlb_b)
            teams = pd.read_parquet(mlb_t)
            lines_path = data_dir / "mlb_lines.parquet"
            if lines_path.exists():
                lines = pd.read_parquet(lines_path)
                teams = teams.merge(lines[["game_pk", "team", "exp_margin", "implied_runs"]],
                                    on=["game_pk", "team"], how="left")
                teams["runs_resid"] = teams["runs"] - teams["implied_runs"]
                teams["margin_resid"] = teams["margin"] - teams["exp_margin"]
            else:
                teams["runs_resid"] = teams["margin_resid"] = np.nan
                self.team_cols["MLB"] = MLB_RAW_TEAM_COL
            bat["norm"] = bat["name"].fillna("").map(norm_name)
            bat["bucket"] = bat["slot"].map(mlb_slot_bucket)
            self.frames["MLB_team"] = teams
            self.frames["MLB_player"] = bat.merge(
                teams[["game_pk", "team", "won", "margin", "runs", "runs_resid", "margin_resid"]]
                .rename(columns={"runs": "team_runs"}),
                on=["game_pk", "team"])
            latest = bat[bat["season"] == bat["season"].max()]
            self.frames["MLB_slots"] = latest.groupby("norm")["slot"].agg(lambda s: int(s.mode().iloc[0]))
        self.ladder = Ladder.from_frames(self.frames)

    def has(self, sport: str) -> bool:
        return f"{sport}_team" in self.frames

    # ---- roles -------------------------------------------------------------------------------

    def role(self, leg: Leg) -> str:
        if leg.kind not in PLAYER_KINDS:
            return "TEAM"
        name = norm_name(leg.player or "")
        if leg.sport == "NFL":
            if leg.kind in ("pass_yds", "pass_td"):
                return "QB1"
            return self.frames["NFL_roles"].get(name, "SKILL")
        slot = self.lineup_slots.get(name) or self.frames["MLB_slots"].get(name)
        return mlb_slot_bucket(int(slot) if slot is not None else None)

    def _player_pool(self, sport: str, role: str) -> list[tuple[str, pd.DataFrame]]:
        """Candidate historical pools from most to least specific (memoized)."""
        key = (sport, role)
        if key not in self._pools:
            df = self.frames[f"{sport}_player"]
            pools = []
            if sport == "MLB":
                if role != "ANY":
                    pools.append((role, df[df["bucket"] == role]))
                pools.append(("ANY", df))
            elif role.startswith("QB"):
                pools.append(("QB1", df[df["role"] == "QB1"]))  # passing stats only make sense for starters
            else:
                if role[:2] in ("WR", "TE", "RB") and role[-1].isdigit():
                    pools.append((role, df[df["role"] == role]))
                    pools.append((role[:2], df[df["role"].str.startswith(role[:2])]))
                pools.append(("SKILL", df[df["role"].str[:2].isin(["WR", "TE", "RB"])]))
            self._pools[key] = pools
        return self._pools[key]

    def _pair_pool(self, ra: str, pool_a: pd.DataFrame, rb: str, pool_b: pd.DataFrame) -> pd.DataFrame:
        """Same-team, same-game rows for two different NFL players (memoized)."""
        key = ("pair", ra, rb)
        if key not in self._pools:
            cols = ["game_id", "team", "player_id", *PLAYER_STAT_COLS]
            m = pool_a[cols].merge(pool_b[cols], on=["game_id", "team"], suffixes=("_a", "_b"))
            self._pools[key] = m[m["player_id_a"] != m["player_id_b"]]
        return self._pools[key]

    def _complete_rows(self, frame: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
        """Rows where both columns exist (MLB residuals only exist for games with Kalshi lines); memoized."""
        key = ("complete", id(frame), col_a, col_b)
        if key not in self._pools:
            self._pools[key] = frame[frame[col_a].notna() & frame[col_b].notna()][[col_a, col_b]]
        return self._pools[key]

    # ---- binarizing a leg on history -----------------------------------------------------------

    @staticmethod
    def _leg_key(leg: Leg, p: float) -> tuple:
        if leg.kind == "team_win":
            return ("win",)
        if leg.kind in YARD_KINDS or leg.kind in QUANTILE_TEAM_KINDS:
            return ("q", float(f"{p:.2g}"))  # 2 significant figures keeps tail thresholds distinct
        return ("k", leg.count_k)

    def _binarize(self, frame: pd.DataFrame, col: str, key: tuple) -> np.ndarray:
        if key[0] == "win":
            return frame[col].to_numpy() == 1
        x = frame[col].to_numpy(dtype=float)
        if key[0] == "q":
            return x > self._quantile_threshold(frame, col, key[1])
        return x >= key[1]

    def _quantile_threshold(self, frame: pd.DataFrame, col: str, p: float) -> float:
        """Threshold t (an observed value) so that mean(col > t) is as close to p as possible.

        The column's sorted exceedance table is memoized: it's reused across thousands of pairs.
        """
        key = ("exceed", id(frame), col)
        if key not in self._pools:
            xs = np.sort(frame[col].to_numpy(dtype=float))
            vals = np.unique(xs)
            self._pools[key] = (vals, 1 - np.searchsorted(xs, vals, side="right") / len(xs))
        vals, exceed = self._pools[key]
        return float(vals[np.argmin(np.abs(exceed - p))])

    # ---- main entry ---------------------------------------------------------------------------

    def estimate(self, a: Leg, b: Leg) -> Estimate | None:
        """a/b must be same game; returns None when the pair has no historical analogue."""
        sport = a.sport
        if not self.has(sport):
            return None
        pa, pb = a.p, b.p
        if b.kind in PLAYER_KINDS and a.kind not in PLAYER_KINDS:
            a, b, pa, pb = b, a, pb, pa  # player leg first

        kinds = {a.kind, b.kind}
        if kinds == {"team_win", "team_spread"} and a.team == b.team:
            # "wins by over X" implies "wins": the combo is just the spread leg.
            q = min(pa, pb)
            return Estimate(q, q, q, None, None, None, 0, 0, "TEAM", "TEAM", "subset")

        role_a, role_b = self.role(a), self.role(b)
        stats = self._fit(sport, a, b, role_a, role_b, pa, pb)
        if stats is None:
            return None
        q = joint_prob(pa, pb, stats["rho"])
        q_lo = joint_prob(pa, pb, stats["rho_lo"])
        q_hi = joint_prob(pa, pb, stats["rho_hi"])
        return Estimate(q, q_lo, q_hi, stats["rho"], stats["rho_lo"], stats["rho_hi"], stats["n"], stats["n11"],
                        stats["role_a"], stats["role_b"], "copula")

    def _fit(self, sport, a: Leg, b: Leg, role_a, role_b, pa, pb) -> dict | None:
        """Tetrachoric fit on the most specific historical pool with enough joint hits."""
        a_player, b_player = a.kind in PLAYER_KINDS, b.kind in PLAYER_KINDS
        team_cols = self.team_cols[sport]
        team_col_b = team_cols.get(b.kind)
        if team_col_b == "runs":
            team_col_b = "team_runs"  # batter rows have their own "runs" column

        if not a_player and not b_player:
            candidates = [("TEAM", self.frames[f"{sport}_team"], team_cols[a.kind], "TEAM", team_cols[b.kind])]
        elif a_player and not b_player:
            candidates = [(ra, pool, a.kind, "TEAM", team_col_b) for ra, pool in self._player_pool(sport, role_a)]
        elif sport == "NFL":
            candidates = [
                (ra, self._pair_pool(ra, pool_a, rb, pool_b), f"{a.kind}_a", rb, f"{b.kind}_b")
                for ra, pool_a in self._player_pool(sport, role_a)
                for rb, pool_b in self._player_pool(sport, role_b)
                if not (ra == rb == "SKILL")
            ]
        else:
            return None

        key_a, key_b = self._leg_key(a, pa), self._leg_key(b, pb)
        for i, (ra, frame, col_a, rb, col_b) in enumerate(candidates):
            cache_key = (sport, a.kind, ra, key_a, b.kind, rb, key_b)
            if cache_key not in self._cache:
                result = None
                frame = self._complete_rows(frame, col_a, col_b)
                if len(frame) >= MIN_N:
                    xa, xb = self._binarize(frame, col_a, key_a), self._binarize(frame, col_b, key_b)
                    n11 = int(np.sum(xa & xb))
                    if n11 >= MIN_N11 or (i == len(candidates) - 1 and n11 >= MIN_N11_LAST):
                        result = {**tetrachoric(xa, xb), "role_a": ra, "role_b": rb}
                self._cache[cache_key] = result
            if self._cache[cache_key] is not None:
                return self._cache[cache_key]
        return None
