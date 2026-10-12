import sys, numpy as np, pandas as pd
d = sys.argv[1]
ps = []
for y in range(2021, 2026):
    p = pd.read_parquet(f"{d}/ps_{y}.parquet")
    if "recent_team" in p: p = p.rename(columns={"recent_team": "team"})
    p = p[p.season_type == "REG"]
    ps.append(p[["player_id","player_display_name","position","team","season","week","attempts","completions","passing_yards","passing_tds",
                 "carries","rushing_yards","rushing_tds","receptions","targets","receiving_yards","receiving_tds"]])
ps = pd.concat(ps)
ps["tds"] = ps.rushing_tds + ps.receiving_tds
g = pd.read_csv(f"{d}/games.csv")
g = g[(g.season.between(2021, 2025)) & (g.game_type == "REG") & g.home_score.notna() & g.total_line.notna()]
def side(home):
    t = pd.DataFrame({"season": g.season, "week": g.week, "game_id": g.game_id,
        "team": g.home_team if home else g.away_team,
        "pts": g.home_score if home else g.away_score, "opp_pts": g.away_score if home else g.home_score,
        "total_line": g.total_line, "spread": g.spread_line if home else -g.spread_line})
    return t
t = pd.concat([side(True), side(False)])
t["implied"] = (t.total_line + t.spread) / 2
t["won"] = t.pts > t.opp_pts
t["over_tt"] = t.pts > t.implied
t["over_tt7"] = t.pts > t.implied + 7
t["cover"] = (t.pts - t.opp_pts) > t.spread
t["game_over"] = (t.pts + t.opp_pts) > t.total_line
t["game_under"] = ~t.game_over
# strata: pregame expectation buckets (so lift isn't just "good teams do everything")
t["stratum"] = pd.cut(t.spread, [-99, -7, -3, 0, 3, 7, 99]).astype(str) + "|" + pd.cut(t.total_line, [0, 41, 45, 49, 99]).astype(str)
ps["team"] = ps.team.replace({"LA": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA"})
# roles: rank on the team by season volume (QB1 by attempts, RB1 by carries, WR1/TE1 by targets)
tot = ps.groupby(["season","team","player_id","position"]).agg(att=("attempts","sum"), car=("carries","sum"), tgt=("targets","sum")).reset_index()
role = {}
for (s, tm), grp in tot.groupby(["season","team"]):
    for pos, col, lab in [("QB","att","QB1"),("RB","car","RB1"),("WR","tgt","WR1"),("WR","tgt","WR2"),("TE","tgt","TE1")]:
        c = grp[grp.position == pos].sort_values(col, ascending=False)
        idx = 1 if lab == "WR2" else 0
        if len(c) > idx: role[(s, tm, c.iloc[idx].player_id)] = lab
ps["role"] = [role.get(k) for k in zip(ps.season, ps.team, ps.player_id)]
ps = ps[ps.role.notna()].merge(t, on=["season","week","team"])
# only games where the player actually played meaningfully
ps = ps[(ps.role != "QB1") | (ps.attempts >= 15)]
legs = {
 "QB1": {"2+ pass TD": lambda f: f.passing_tds >= 2, "3+ pass TD": lambda f: f.passing_tds >= 3, "4+ pass TD": lambda f: f.passing_tds >= 4,
         "275+ pass yds": lambda f: f.passing_yards >= 275, "325+ pass yds": lambda f: f.passing_yards >= 325},
 "RB1": {"1+ TD": lambda f: f.tds >= 1, "2+ TD": lambda f: f.tds >= 2, "100+ rush yds": lambda f: f.rushing_yards >= 100},
 "WR1": {"1+ TD": lambda f: f.tds >= 1, "2+ TD": lambda f: f.tds >= 2, "100+ rec yds": lambda f: f.receiving_yards >= 100, "8+ rec": lambda f: f.receptions >= 8},
 "TE1": {"1+ TD": lambda f: f.tds >= 1, "75+ rec yds": lambda f: f.receiving_yards >= 75},
}
team_legs = {"team wins": "won", "team over its total": "over_tt", "team over total +7": "over_tt7", "covers spread": "cover",
             "game over": "game_over", "game under": "game_under"}
def mh_lift(a, b, strata):
    df = pd.DataFrame({"a": a.values, "b": b.values, "s": strata.values})
    obs, exp = 0.0, 0.0
    for _, x in df.groupby("s"):
        n = len(x); obs += (x.a & x.b).sum(); exp += x.a.sum() * x.b.sum() / n
    return obs / exp if exp else np.nan, int(obs)
rows = []
for r, ls in legs.items():
    f = ps[ps.role == r]
    for ln, fn in ls.items():
        a = fn(f)
        for tn, col in team_legs.items():
            lift, n11 = mh_lift(a, f[col], f.stratum)
            rows.append((f"{r} {ln}", tn, a.mean(), f[col].mean(), lift, n11, len(f)))
# teammate stacks
qb = ps[ps.role == "QB1"][["season","week","team","passing_tds","passing_yards","stratum","won"]]
for r2, ls2 in [("WR1", {"1+ TD": lambda f: f.tds >= 1, "100+ rec yds": lambda f: f.receiving_yards >= 100}),
                ("TE1", {"1+ TD": lambda f: f.tds >= 1}), ("RB1", {"1+ TD": lambda f: f.tds >= 1, "100+ rush yds": lambda f: f.rushing_yards >= 100})]:
    m = ps[ps.role == r2].merge(qb, on=["season","week","team"], suffixes=("", "_qb"))
    for ln, fn in ls2.items():
        b = fn(m)
        for qn, qa in [("QB1 3+ pass TD", m.passing_tds_qb >= 3), ("QB1 300+ pass yds", m.passing_yards_qb >= 300)]:
            lift, n11 = mh_lift(qa, b, m.stratum)
            rows.append((qn, f"{r2} {ln}", qa.mean(), b.mean(), lift, n11, len(m)))
out = pd.DataFrame(rows, columns=["leg_a","leg_b","p_a","p_b","lift","n11","n"])
out = out[out.n11 >= 15].sort_values("lift", ascending=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(out.round(3).to_string(index=False))
out.to_csv(f"{d}/nfl_edge_map.csv", index=False)
