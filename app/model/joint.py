"""Estimate the true probability that both legs of a same-game combo hit.

For each pair we rebuild the two legs on historical games (same role / lineup slot, same
count threshold, or a quantile-matched threshold for yardage and team scoring), fit the
tetrachoric ρ on that 2x2 table, and plug it into the copula with today's Kalshi marginals.

Supported relationships (see app/pairs.py): same player (two stats), teammates, player × own
team or game outcome, team × team (same team), game × team, game × game, plus the one opponent
case (pitcher × opponent team total).
"""

import dataclasses
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app import config
from app.legs import GAME_KINDS, MODEL_LEG_HAIRCUT, PITCHER_KINDS, PLAYER_KINDS, Leg
from app.model.copula import joint_prob, joint_prob_n, tetrachoric
from app.model.ladder import Ladder

# Player kinds with wide ranges: thresholds are quantile-matched instead of exact counts.
YARD_KINDS = {"rec_yds", "rush_yds", "pass_yds", "rr_yds", "pass_att", "pass_cmp", "rush_att"}
BINARY_KINDS = {"team_win", "f5_win", "f3_win", "h1_win", "q1_win", "first_td_team", "first_td", "team_first_td"}
QUANTILE_TEAM_KINDS = {"team_spread", "team_total", "f5_spread", "game_total", "f5_total", "h1_spread", "h1_total",
                       "h1_team_total", "q1_spread", "q1_total"}
QB_KINDS = {"pass_yds", "pass_td", "pass_att", "pass_cmp", "pass_int"}
TEAM_COL = {
    # Full-game team outcomes are residuals vs the pre-game line ("over X" = beat expectation);
    # half/quarter/F5 outcomes have no historical lines, so they're raw.
    "NFL": {"team_win": "won", "team_spread": "margin_resid", "team_total": "pts_resid",
            "game_total": "total_resid", "first_td_team": "first_td_team",
            "h1_win": "h1_won", "h1_spread": "h1_margin", "h1_team_total": "h1_pts", "h1_total": "h1_total",
            "q1_win": "q1_won", "q1_spread": "q1_margin", "q1_total": "q1_total"},
    "MLB": {"team_win": "won", "team_spread": "margin_resid", "team_total": "runs_resid",
            "f5_win": "f5_won", "f3_win": "f3_won", "f5_spread": "f5_margin",
            "game_total": "total_resid", "f5_total": "f5_total", "rfi": "first_inning_runs"},
}
# Without data/mlb_lines.parquet (Kalshi pre-game prices), MLB falls back to raw runs/margins.
MLB_RAW_TEAM_COL = {**TEAM_COL["MLB"], "team_spread": "margin", "team_total": "runs", "game_total": "game_runs"}
# Team/game outcome columns copied onto every player row so any player × own team/game pair works.
OUTCOME_COLS = {
    "NFL": ["won", "margin_resid", "pts_resid", "total_resid", "first_td_team", "h1_won", "h1_margin", "h1_pts",
            "h1_total", "q1_won", "q1_margin", "q1_total"],
    "MLB": ["won", "margin", "runs", "runs_resid", "margin_resid", "game_runs", "total_resid", "f5_won", "f3_won",
            "f5_margin", "f5_total", "first_inning_runs"],
}
GAME_ID = {"NFL": "game_id", "MLB": "game_pk"}
# Columns describing the OTHER team from a player's row (pitcher Ks vs opponent's runs).
OPP_COL = {"team_runs": "opp_runs", "runs_resid": "opp_runs_resid"}
MIN_N, MIN_N11, MIN_N11_LAST = 300, 8, 3


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


def add_mlb_game_columns(teams: pd.DataFrame) -> pd.DataFrame:
    """Inning-split and whole-game outcome columns (present once data/mlb_teams.parquet has line scores)."""
    t = teams.copy()
    t["game_runs"] = t["runs"] + t["opp_runs"]
    if "implied_runs" in t:
        opp_implied = t[["game_pk", "team", "implied_runs"]].rename(
            columns={"team": "opp", "implied_runs": "opp_implied_runs"})
        t = t.merge(opp_implied, on=["game_pk", "opp"], how="left")
        t["total_resid"] = t["game_runs"] - t["implied_runs"] - t["opp_implied_runs"]
    if "f5_runs" in t:
        t["f5_won"] = (t["f5_runs"] > t["f5_opp_runs"]).astype(int)
        t["f3_won"] = (t["f3_runs"] > t["f3_opp_runs"]).astype(int)
        t["f5_margin"] = t["f5_runs"] - t["f5_opp_runs"]
        t["f5_total"] = t["f5_runs"] + t["f5_opp_runs"]
    return t


def add_nfl_game_columns(teams: pd.DataFrame) -> pd.DataFrame:
    """Game total vs the closing line, and 1st-half / 1st-quarter splits (from play-by-play)."""
    t = teams.copy()
    if {"pts", "opp_pts", "total_line"} <= set(t.columns):
        t["total_resid"] = t["pts"] + t["opp_pts"] - t["total_line"]
    for p in ("h1", "q1"):
        if f"{p}_pts" in t:
            t[f"{p}_won"] = (t[f"{p}_pts"] > t[f"{p}_opp_pts"]).astype(int)
            t[f"{p}_margin"] = t[f"{p}_pts"] - t[f"{p}_opp_pts"]
            t[f"{p}_total"] = t[f"{p}_pts"] + t[f"{p}_opp_pts"]
    return t


class Model:
    def __init__(self, data_dir=config.DATA_DIR):
        self.frames: dict[str, pd.DataFrame] = {}
        self._cache: dict[tuple, dict | None] = {}
        self._atlas: dict[tuple, float | None] = {}
        self._pools: dict[tuple, object] = {}
        self.lineup_slots: dict[str, int] = {}  # norm name -> today's batting slot
        self.team_cols = dict(TEAM_COL)
        nfl_p, nfl_t = data_dir / "nfl_players.parquet", data_dir / "nfl_teams.parquet"
        mlb_b, mlb_t = data_dir / "mlb_batters.parquet", data_dir / "mlb_teams.parquet"
        if nfl_p.exists() and nfl_t.exists():
            players = pd.read_parquet(nfl_p)
            teams = add_nfl_game_columns(pd.read_parquet(nfl_t))
            players["norm"] = players["name"].fillna("").map(norm_name)
            self.frames["NFL_team"] = teams
            self.frames["NFL_game"] = teams.drop_duplicates("game_id")  # one row per game
            self.frames["NFL_player"] = players.merge(self._outcomes("NFL", teams), on=["game_id", "team"])
            self.frames["NFL_roles"] = (players.sort_values(["season", "week"])
                                        .drop_duplicates("norm", keep="last").set_index("norm")["role"])
        if mlb_b.exists() and mlb_t.exists():
            # Resumed games can appear twice in older files; duplicates would multiply in the merges below.
            bat = pd.read_parquet(mlb_b).drop_duplicates(["game_pk", "player_id"], keep="last")
            teams = pd.read_parquet(mlb_t).drop_duplicates(["game_pk", "team"], keep="last")
            lines_path = data_dir / "mlb_lines.parquet"
            if lines_path.exists():
                lines = pd.read_parquet(lines_path).drop_duplicates(["game_pk", "team"])
                teams = teams.merge(lines[["game_pk", "team", "exp_margin", "implied_runs"]],
                                    on=["game_pk", "team"], how="left")
                teams["runs_resid"] = teams["runs"] - teams["implied_runs"]
                teams["margin_resid"] = teams["margin"] - teams["exp_margin"]
            else:
                teams["runs_resid"] = teams["margin_resid"] = np.nan
                self.team_cols["MLB"] = MLB_RAW_TEAM_COL
            teams = add_mlb_game_columns(teams)
            bat["norm"] = bat["name"].fillna("").map(norm_name)
            bat["bucket"] = bat["slot"].map(mlb_slot_bucket)
            self.frames["MLB_team"] = teams
            if "home" in teams:
                self.frames["MLB_game"] = teams[teams["home"] == 1]  # one row per game
            own = self._outcomes("MLB", teams)
            self.frames["MLB_player"] = bat.merge(own, on=["game_pk", "team"])
            pit_path = data_dir / "mlb_pitchers.parquet"
            if pit_path.exists():
                opp = teams[["game_pk", "team", "opp"]].merge(
                    teams[["game_pk", "team", "runs", "runs_resid"]].rename(
                        columns={"team": "opp", "runs": "opp_runs", "runs_resid": "opp_runs_resid"}),
                    on=["game_pk", "opp"])
                pitchers = pd.read_parquet(pit_path).drop_duplicates(["game_pk", "team"], keep="last")
                self.frames["MLB_pitcher"] = (pitchers.merge(own, on=["game_pk", "team"])
                                              .merge(opp[["game_pk", "team", "opp_runs", "opp_runs_resid"]],
                                                     on=["game_pk", "team"]))
            latest = bat[bat["season"] == bat["season"].max()]
            self.frames["MLB_slots"] = latest.groupby("norm")["slot"].agg(lambda s: int(s.mode().iloc[0]))
        self.ladder = Ladder.from_frames(self.frames)

    @staticmethod
    def _outcomes(sport: str, teams: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in OUTCOME_COLS[sport] if c in teams]
        # Batter/pitcher rows have their own "runs" column, so the team's is "team_runs".
        return teams[[GAME_ID[sport], "team", *cols]].rename(columns={"runs": "team_runs"})

    def has(self, sport: str) -> bool:
        return f"{sport}_team" in self.frames

    # ---- roles -------------------------------------------------------------------------------

    def role(self, leg: Leg) -> str:
        if leg.kind not in PLAYER_KINDS:
            return "TEAM"
        if leg.kind in PITCHER_KINDS:
            return "SP"
        name = norm_name(leg.player or "")
        if leg.sport == "NFL":
            if leg.kind in QB_KINDS:
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
            if role == "SP":
                pools.append(("SP", self.frames.get("MLB_pitcher", pd.DataFrame())))
            elif sport == "MLB":
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

    def _pair_pool(self, sport: str, ra: str, pool_a: pd.DataFrame, col_a: str,
                   rb: str, pool_b: pd.DataFrame, col_b: str) -> pd.DataFrame:
        """Same-game, same-team rows for two different players: columns xa, xb (memoized)."""
        key = ("pair", sport, ra, col_a, rb, col_b)
        if key not in self._pools:
            if col_a not in pool_a or col_b not in pool_b:
                self._pools[key] = pd.DataFrame(columns=["xa", "xb"])
            else:
                on = [GAME_ID[sport], "team"]
                left = pool_a[[*on, "player_id", col_a]].rename(columns={"player_id": "pid_a", col_a: "xa"})
                right = pool_b[[*on, "player_id", col_b]].rename(columns={"player_id": "pid_b", col_b: "xb"})
                m = left.merge(right, on=on)
                self._pools[key] = m[m["pid_a"] != m["pid_b"]]
        return self._pools[key]

    def _complete_rows(self, frame: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
        """Rows where both columns exist (e.g. MLB residuals only exist for games with lines); memoized."""
        key = ("complete", id(frame), col_a, col_b)
        if key not in self._pools:
            self._pools[key] = frame[frame[col_a].notna() & frame[col_b].notna()][[col_a, col_b]]
        return self._pools[key]

    # ---- binarizing a leg on history -----------------------------------------------------------

    @staticmethod
    def _leg_key(leg: Leg, p: float) -> tuple:
        """How to rebuild the leg on history. NO-side legs: rebuild the YES market, then invert."""
        if leg.side == "no":
            return (*Model._leg_key(dataclasses.replace(leg, side="yes"), 1 - p), "no")
        if leg.kind in BINARY_KINDS:
            return ("win",)
        if leg.kind in YARD_KINDS or leg.kind in QUANTILE_TEAM_KINDS:
            return ("q", float(f"{p:.2g}"))  # 2 significant figures keeps tail thresholds distinct
        return ("k", leg.count_k)

    def _binarize(self, frame: pd.DataFrame, col: str, key: tuple) -> np.ndarray:
        if key[-1] == "no":
            return ~self._binarize(frame, col, key[:-1])
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

    # ---- which history describes a pair ----------------------------------------------------------

    def _team_col(self, sport: str, kind: str, player_frame: bool) -> str | None:
        col = self.team_cols[sport].get(kind)
        return "team_runs" if (player_frame and col == "runs") else col

    def _candidates(self, sport: str, a: Leg, b: Leg, role_a: str, role_b: str) -> list[tuple] | None:
        """(role_a, frame, col_a, role_b, col_b) from most to least specific; None if unsupported.

        `a` is the player leg whenever there is one (estimate() orders them).
        """
        a_player, b_player = a.kind in PLAYER_KINDS, b.kind in PLAYER_KINDS
        if a_player and b_player:
            if a.team != b.team:
                return None  # opposing players aren't modeled
            if a.player == b.player:  # same player, two stats: both columns on his own rows
                return [(ra, pool, a.kind, ra, b.kind) for ra, pool in self._player_pool(sport, role_a)]
            return [(ra, self._pair_pool(sport, ra, pool_a, a.kind, rb, pool_b, b.kind), "xa", rb, "xb")
                    for ra, pool_a in self._player_pool(sport, role_a)
                    for rb, pool_b in self._player_pool(sport, role_b)
                    if not (ra == rb and ra in ("SKILL", "ANY"))]
        if a_player:
            col_b = self._team_col(sport, b.kind, player_frame=True)
            if b.team and a.team and b.team != a.team and b.kind not in GAME_KINDS:
                col_b = OPP_COL.get(col_b)  # pitcher Ks + OPPONENT team total
            if col_b is None:
                return None
            return [(ra, pool, a.kind, "TEAM", col_b) for ra, pool in self._player_pool(sport, role_a)]
        a_game, b_game = a.kind in GAME_KINDS, b.kind in GAME_KINDS
        if not a_game and not b_game and a.team != b.team:
            return None  # two team legs must be the same team
        frame = self.frames.get(f"{sport}_game") if (a_game and b_game) else self.frames[f"{sport}_team"]
        col_a, col_b = self._team_col(sport, a.kind, False), self._team_col(sport, b.kind, False)
        if frame is None or col_a is None or col_b is None:
            return None
        return [("TEAM", frame, col_a, "TEAM", col_b)]

    @staticmethod
    def _ordered(a: Leg, b: Leg) -> tuple[Leg, Leg]:
        """Player leg first, then team, then game leg."""
        rank = lambda leg: 0 if leg.kind in PLAYER_KINDS else 2 if leg.kind in GAME_KINDS else 1  # noqa: E731
        return (a, b) if rank(a) <= rank(b) else (b, a)

    def atlas_rho(self, a: Leg, b: Leg) -> float | None:
        """Cheap screen: rank correlation of the two underlying stats on the most specific pool,
        signed for NO-side legs. Cached per (kinds, roles, columns, sides)."""
        if not self.has(a.sport):
            return None
        a, b = self._ordered(a, b)
        role_a, role_b = self.role(a), self.role(b)
        cands = self._candidates(a.sport, a, b, role_a, role_b)
        if not cands:
            return None
        ra, frame, col_a, rb, col_b = cands[0]
        key = (a.sport, a.kind, ra, b.kind, rb, col_a, col_b, a.player == b.player, a.side, b.side)
        if key not in self._atlas:
            rho = None
            if col_a in frame and col_b in frame:
                rows = self._complete_rows(frame, col_a, col_b)
                if len(rows) >= MIN_N and rows[col_a].nunique() > 1 and rows[col_b].nunique() > 1:
                    rho = float(spearmanr(rows[col_a], rows[col_b]).statistic)
                    if (a.side == "no") != (b.side == "no"):
                        rho = -rho
            self._atlas[key] = rho
        return self._atlas[key]

    # ---- main entry ---------------------------------------------------------------------------

    def estimate(self, a: Leg, b: Leg, haircut: float = MODEL_LEG_HAIRCUT) -> Estimate | None:
        """a/b must be same game; returns None when the pair has no historical analogue."""
        sport = a.sport
        if not self.has(sport):
            return None
        a, b = self._ordered(a, b)
        pa, pb = a.p, b.p

        if {a.kind, b.kind} == {"team_win", "team_spread"} and a.team == b.team:
            # "wins by over X" implies "wins": the combo is just the spread leg.
            spread = a if a.kind == "team_spread" else b
            q = min(pa, pb)
            return Estimate(q, spread.p_lo(haircut), spread.p_hi(haircut), None, None, None, 0, 0, "TEAM", "TEAM",
                            "subset")

        role_a, role_b = self.role(a), self.role(b)
        stats = self._fit(sport, a, b, role_a, role_b, pa, pb)
        if stats is None:
            return None
        q = joint_prob(pa, pb, stats["rho"])
        # Pessimistic/optimistic: uncertainty in both the correlation and each leg's price.
        q_lo = joint_prob(max(a.p_lo(haircut), 1e-6), max(b.p_lo(haircut), 1e-6), stats["rho_lo"])
        q_hi = joint_prob(a.p_hi(haircut), b.p_hi(haircut), stats["rho_hi"])
        return Estimate(q, q_lo, q_hi, stats["rho"], stats["rho_lo"], stats["rho_hi"], stats["n"], stats["n11"],
                        stats["role_a"], stats["role_b"], "copula")

    def estimate_n(self, legs: list[Leg], haircut: float = MODEL_LEG_HAIRCUT) -> Estimate | None:
        """Joint probability of 2+ same-game legs: pairwise ρ fits combined in one Gaussian copula.

        None when any pair has no historical analogue or one leg implies another (subset pairs).
        """
        if len(legs) == 2:
            return self.estimate(legs[0], legs[1], haircut)
        k = len(legs)
        r, r_lo, r_hi = np.eye(k), np.eye(k), np.eye(k)
        n, n11, roles = [], [], {}
        for i in range(k):
            for j in range(i + 1, k):
                est = self.estimate(legs[i], legs[j], haircut)
                if est is None or est.rho is None:
                    return None
                r[i, j] = r[j, i] = est.rho
                r_lo[i, j] = r_lo[j, i] = est.rho_lo
                r_hi[i, j] = r_hi[j, i] = est.rho_hi
                n.append(est.n)
                n11.append(est.n11)
        for leg in legs:
            roles[leg.ticker] = self.role(leg)
        q = joint_prob_n([leg.p for leg in legs], r)
        q_lo = joint_prob_n([max(leg.p_lo(haircut), 1e-6) for leg in legs], r_lo)
        q_hi = joint_prob_n([leg.p_hi(haircut) for leg in legs], r_hi)
        rho = float(np.mean(r[np.triu_indices(k, 1)]))
        return Estimate(q, min(q_lo, q), max(q_hi, q), rho, float(np.min(r_lo[np.triu_indices(k, 1)])),
                        float(np.max(r_hi[np.triu_indices(k, 1)])), min(n), min(n11),
                        "/".join(roles[leg.ticker] for leg in legs), "", "copula")

    def _fit(self, sport, a: Leg, b: Leg, role_a, role_b, pa, pb) -> dict | None:
        """Tetrachoric fit on the most specific historical pool with enough joint hits."""
        candidates = self._candidates(sport, a, b, role_a, role_b)
        if not candidates:
            return None
        key_a, key_b = self._leg_key(a, pa), self._leg_key(b, pb)
        same_player = a.player is not None and a.player == b.player
        for i, (ra, frame, col_a, rb, col_b) in enumerate(candidates):
            cache_key = (sport, a.kind, ra, key_a, b.kind, rb, key_b, col_b, same_player)
            if cache_key not in self._cache:
                result = None
                if col_a in frame and col_b in frame:
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
