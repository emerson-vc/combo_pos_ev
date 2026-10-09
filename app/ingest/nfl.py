"""Download NFL player-game and team-game history from nflverse into data/*.parquet.

Team rows carry the closing spread/total so team outcomes can be expressed relative to
pre-game expectation (residuals), which is what a same-game combo is really betting on.
"""

import nflreadpy as nfl
import numpy as np
import pandas as pd

from app import config

PLAYER_COLS = {
    "player_id": "player_id", "player_display_name": "name", "position": "position", "team": "team",
    "season": "season", "week": "week", "game_id": "game_id", "targets": "targets",
    "receptions": "rec", "receiving_yards": "rec_yds", "receiving_tds": "rec_td",
    "carries": "carries", "rushing_yards": "rush_yds", "rushing_tds": "rush_td",
    "attempts": "pass_att", "completions": "pass_cmp", "passing_interceptions": "pass_int",
    "passing_yards": "pass_yds", "passing_tds": "pass_td",
}
PBP_COLS = ["game_id", "play_id", "qtr", "game_seconds_remaining", "home_team", "away_team",
            "total_home_score", "total_away_score", "touchdown", "td_player_id", "td_team"]


def build_teams(sched: pd.DataFrame) -> pd.DataFrame:
    s = sched[sched["home_score"].notna() & sched["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])]
    rows = []
    for side, opp, sign in (("home", "away", 1), ("away", "home", -1)):
        df = pd.DataFrame({
            "game_id": s["game_id"], "season": s["season"], "week": s["week"],
            "team": s[f"{side}_team"], "opp": s[f"{opp}_team"],
            "pts": s[f"{side}_score"], "opp_pts": s[f"{opp}_score"],
            "spread_line": s["spread_line"], "total_line": s["total_line"],
        })
        # nflverse spread_line = expected home margin.
        df["exp_margin"] = sign * df["spread_line"]
        rows.append(df)
    t = pd.concat(rows, ignore_index=True)
    t["margin"] = t["pts"] - t["opp_pts"]
    t = t[t["margin"] != 0]  # drop ties so team_win is binary
    t["won"] = (t["margin"] > 0).astype(int)
    t["implied_pts"] = (t["total_line"] + t["exp_margin"]) / 2
    t["pts_resid"] = t["pts"] - t["implied_pts"]
    t["margin_resid"] = t["margin"] - t["exp_margin"]
    return t.dropna(subset=["implied_pts"]).reset_index(drop=True)


def game_splits(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """From play-by-play: per team-game 1st-quarter / 1st-half points and first-TD flags, and per
    player-game first-TD flags (game's first TD; his team's first TD)."""
    pbp = pbp.sort_values(["game_id", "game_seconds_remaining", "play_id"], ascending=[True, False, True])
    rows = []
    for (game, home, away), g in pbp.groupby(["game_id", "home_team", "away_team"], sort=False):
        q1, h1 = g[g["qtr"] <= 1], g[g["qtr"] <= 2]
        tds = g[(g["touchdown"] == 1) & g["td_team"].notna()]
        first_team = tds["td_team"].iloc[0] if len(tds) else None
        for team, side, opp_side in ((home, "home", "away"), (away, "away", "home")):
            rows.append({"game_id": game, "team": team,
                         "q1_pts": q1[f"total_{side}_score"].max(), "q1_opp_pts": q1[f"total_{opp_side}_score"].max(),
                         "h1_pts": h1[f"total_{side}_score"].max(), "h1_opp_pts": h1[f"total_{opp_side}_score"].max(),
                         "first_td_team": int(first_team == team)})
    teams = pd.DataFrame(rows).fillna({"q1_pts": 0, "q1_opp_pts": 0, "h1_pts": 0, "h1_opp_pts": 0})
    tds = pbp[(pbp["touchdown"] == 1) & pbp["td_player_id"].notna()]
    first = tds.groupby("game_id").head(1)[["game_id", "td_player_id"]].assign(first_td=1)
    team_first = tds.groupby(["game_id", "td_team"]).head(1)[["game_id", "td_player_id"]].assign(team_first_td=1)
    players = first.merge(team_first, on=["game_id", "td_player_id"], how="outer").rename(
        columns={"td_player_id": "player_id"})
    return teams, players.fillna(0)


def assign_roles(p: pd.DataFrame) -> pd.DataFrame:
    """Season-level depth role per team: WR1/WR2/WR3, TE1/TE2, RB1/RB2, QB1/QB2."""
    usage = {"WR": "targets", "TE": "targets", "RB": "carries", "QB": "pass_att"}
    caps = {"WR": 3, "TE": 2, "RB": 2, "QB": 2}
    tot = p.groupby(["season", "team", "player_id", "position"], as_index=False)[
        ["targets", "carries", "pass_att"]].sum()
    tot = tot[tot["position"].isin(usage)]
    tot["usage"] = [getattr(r, usage[r.position]) for r in tot.itertuples()]
    tot["rank"] = tot.groupby(["season", "team", "position"])["usage"].rank(ascending=False, method="first")
    tot["role"] = [f"{r.position}{min(int(r.rank), caps[r.position])}" for r in tot.itertuples()]
    out = p.merge(tot[["season", "team", "player_id", "role"]], on=["season", "team", "player_id"], how="left")
    out["role"] = out["role"].fillna("OTH")
    return out


def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    seasons = config.NFL_SEASONS
    print(f"Loading nflverse schedules + player stats for {seasons} ...")
    sched = nfl.load_schedules(seasons).to_pandas()
    teams = build_teams(sched)
    print("Loading play-by-play for quarter/half scores and first TDs ...")
    pbp = nfl.load_pbp(seasons).to_pandas()[PBP_COLS]
    split_teams, split_players = game_splits(pbp)
    teams = teams.merge(split_teams, on=["game_id", "team"], how="left")

    ps = nfl.load_player_stats(seasons, summary_level="week").to_pandas()
    ps = ps[ps["season_type"].isin(["REG", "POST"])][list(PLAYER_COLS)].rename(columns=PLAYER_COLS)
    num = ["targets", "rec", "rec_yds", "rec_td", "carries", "rush_yds", "rush_td", "pass_att", "pass_cmp",
           "pass_int", "pass_yds", "pass_td"]
    ps[num] = ps[num].fillna(0)
    ps["td"] = ps["rec_td"] + ps["rush_td"]
    ps["rush_att"] = ps["carries"]
    ps["rr_yds"] = ps["rush_yds"] + ps["rec_yds"]
    ps = ps.merge(split_players, on=["game_id", "player_id"], how="left")
    ps[["first_td", "team_first_td"]] = ps[["first_td", "team_first_td"]].fillna(0)
    ps = ps[ps["game_id"].isin(teams["game_id"])]
    ps = assign_roles(ps)

    teams.to_parquet(config.DATA_DIR / "nfl_teams.parquet", index=False)
    ps.to_parquet(config.DATA_DIR / "nfl_players.parquet", index=False)
    print(f"nfl_teams: {len(teams):,} team-games   nfl_players: {len(ps):,} player-games")
    print(ps["role"].value_counts().to_string())
    print("team pts_resid sd:", np.round(teams["pts_resid"].std(), 2))


if __name__ == "__main__":
    main()
