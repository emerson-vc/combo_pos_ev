import sys; sys.argv=["x", "."]
exec(open("nfl_edge_map.py").read().split("legs = {")[0])
import pandas as pd
wide = None
for r in ["QB1","RB1","WR1","TE1"]:
    f = ps[ps.role == r][["season","week","team","passing_tds","passing_yards","tds","rushing_yards","receiving_yards"]].add_suffix("_"+r)
    f = f.rename(columns={f"season_{r}":"season", f"week_{r}":"week", f"team_{r}":"team"}).drop_duplicates(["season","week","team"])
    wide = f if wide is None else wide.merge(f, on=["season","week","team"])
w = wide.merge(t, on=["season","week","team"])
def lift3(cs):
    obs = exp = 0.0
    for _, x in w.assign(**{f"c{i}": c for i, c in enumerate(cs)}).groupby("stratum"):
        n = len(x); a = [x[f"c{i}"] for i in range(len(cs))]
        allh = a[0].copy()
        for c in a[1:]: allh &= c
        obs += allh.sum(); e = n
        for c in a: e *= c.mean()
        exp += e
    return obs/exp, int(obs), len(w)
tests = {
 "team wins + QB 2+ TD + RB1 TD (Emerson's Bills 3-leg)": [w.won, w.passing_tds_QB1 >= 2, w.tds_RB1 >= 1],
 "QB 3+ TD + WR1 TD + team over its total": [w.passing_tds_QB1 >= 3, w.tds_WR1 >= 1, w.over_tt],
 "QB 2+ TD + WR1 TD + team over its total": [w.passing_tds_QB1 >= 2, w.tds_WR1 >= 1, w.over_tt],
 "QB 300+ yds + WR1 100+ yds + game over": [w.passing_yards_QB1 >= 300, w.receiving_yards_WR1 >= 100, w.game_over],
 "RB1 TD + team wins + team over its total": [w.tds_RB1 >= 1, w.won, w.over_tt],
 "QB 2+ TD + TE1 TD + team wins": [w.passing_tds_QB1 >= 2, w.tds_TE1 >= 1, w.won],
 "QB 3+ TD + WR1 TD + TE1 TD": [w.passing_tds_QB1 >= 3, w.tds_WR1 >= 1, w.tds_TE1 >= 1],
 "team wins + team over + game over": [w.won, w.over_tt, w.game_over],
}
for k, cs in tests.items():
    l, n11, n = lift3(cs); print(f"{l:5.2f}  hits={n11:4d}/{n}  {k}")
