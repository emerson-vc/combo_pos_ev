import numpy as np, pandas as pd
b = pd.read_parquet("mlb_bat.parquet"); p = pd.read_parquet("mlb_pit.parquet"); g = pd.read_parquet("mlb_games.parquet")
g = g[g.gametype == "regular"]
t = pd.concat([
    pd.DataFrame({"game": g.game, "season": g.season, "side": 0, "team": g.away, "opp": g.home, "runs": g.runs0, "opp_runs": g.runs1, "home": 0}),
    pd.DataFrame({"game": g.game, "season": g.season, "side": 1, "team": g.home, "opp": g.away, "runs": g.runs1, "opp_runs": g.runs0, "home": 1})])
off = t.groupby(["season", "team"]).runs.mean().rename("off"); dfn = t.groupby(["season", "team"]).opp_runs.mean().rename("dfn")
lg = t.groupby("season").runs.mean().rename("lg")
t = t.join(off, on=["season", "team"]).join(dfn, on=["season", "team"]).join(lg, on="season")
t = t.join(dfn.rename("opp_dfn"), on=["season", "opp"]).join(off.rename("opp_off"), on=["season", "opp"])
t["exp"] = t.off * t.opp_dfn / t.lg * np.where(t.home == 1, 1.02, 0.98)
t["opp_exp"] = t.opp_off * t.dfn / t.lg * np.where(t.home == 1, 0.98, 1.02)
# opposing starter quality proxy: his season Ks per start
p["k_avg"] = p.groupby(["season", "pid"]).k.transform("mean")
sp = p[["game", "side", "pid", "k", "k_avg", "name"]]
t = t.merge(sp.rename(columns={"side": "opp_side", "pid": "opp_sp", "k": "opp_sp_k", "k_avg": "opp_sp_kavg", "name": "opp_sp_name"}).assign(side=lambda x: 1 - x.opp_side).drop(columns="opp_side"), on=["game", "side"], how="left")
t = t.merge(sp.rename(columns={"pid": "sp", "k": "sp_k", "k_avg": "sp_kavg", "name": "sp_name"}), on=["game", "side"], how="left")
t["won"] = t.runs > t.opp_runs
t["line"] = np.floor(t.exp) + 0.5
t["over_tt"] = t.runs > t.line; t["over_tt2"] = t.runs > t.line + 2
t["gline"] = np.floor(t.exp + t.opp_exp) + 0.5
t["game_over"] = t.runs + t.opp_runs > t.gline; t["game_under"] = ~t.game_over
t["opp_under"] = t.opp_runs < np.floor(t.opp_exp) + 0.5
t["stratum"] = (pd.qcut(t.exp, 4, labels=False).astype(str) + "|" + pd.qcut(t.exp - t.opp_exp, 4, labels=False).astype(str)
                + "|" + pd.qcut(t.opp_sp_kavg.fillna(t.opp_sp_kavg.median()), 3, labels=False, duplicates="drop").astype(str))
b = b[b.pa >= 3].merge(t, on=["game", "season", "side"])
b["bucket"] = np.where(b.slot <= 2, "TOP", np.where(b.slot <= 5, "MID", "BOT"))
# batter power stratum (season HR per PA) so 'HR' compares like with like
hr_rate = b.groupby(["season", "pid"]).hr.transform("sum") / b.groupby(["season", "pid"]).pa.transform("sum")
b["pw"] = pd.cut(hr_rate, [-1, 0.02, 0.035, 0.05, 1], labels=False).astype(str)
def mh(a, c, s):
    df = pd.DataFrame({"a": np.asarray(a, bool), "c": np.asarray(c, bool), "s": np.asarray(s)})
    obs = exp = 0.0
    for _, x in df.groupby("s"):
        obs += (x.a & x.c).sum(); exp += x.a.sum() * x.c.sum() / len(x)
    return obs / exp, int(obs)
rows = []
bl = {"1+ HR": b.hr >= 1, "2+ HR": b.hr >= 2, "2+ hits": b.hit >= 2, "3+ hits": b.hit >= 3, "3+ TB": b.tb >= 3,
      "5+ TB": b.tb >= 5, "2+ RBI": b.rbi >= 2, "3+ RBI": b.rbi >= 3, "4+ RBI": b.rbi >= 4}
tl = {"team wins": "won", "team over its total": "over_tt", "team over total +2": "over_tt2", "game over": "game_over",
      "game under": "game_under", "opp UNDER its total": "opp_under"}
strat = b.stratum + "|" + b.pw
for bucket in ["TOP", "MID", "BOT", "ALL"]:
    m = (b.bucket == bucket) if bucket != "ALL" else pd.Series(True, index=b.index)
    for ln, a in bl.items():
        for tn, col in tl.items():
            lift, n11 = mh(a[m], b.loc[m, col], strat[m])
            rows.append(("batter " + bucket, ln, tn, a[m].mean(), b.loc[m, col].mean(), lift, n11, int(m.sum())))
# starting pitcher Ks
s = t[t.sp_k.notna()].copy()
s["kav"] = pd.cut(s.sp_kavg, [0, 4, 5.5, 7, 20], labels=False).astype(str)
for k in (6, 8, 9, 10):
    a = s.sp_k >= k
    for tn, col in [("team wins", "won"), ("opp UNDER its total", "opp_under"), ("game under", "game_under"), ("team over its total", "over_tt")]:
        lift, n11 = mh(a, s[col], s.stratum + "|" + s.kav)
        rows.append(("SP", f"{k}+ K", tn, a.mean(), s[col].mean(), lift, n11, len(s)))
# teammate stacks (same team, same game)
mid = b[b.bucket.isin(["TOP", "MID"])][["game", "side", "pid", "slot", "hr", "hit", "rbi", "tb", "stratum", "pw"]]
pr = mid.merge(mid, on=["game", "side"], suffixes=("_a", "_b")); pr = pr[pr.slot_a < pr.slot_b]
for n1, a, n2, c in [("1+ HR", pr.hr_a >= 1, "1+ HR", pr.hr_b >= 1), ("2+ hits", pr.hit_a >= 2, "2+ RBI", pr.rbi_b >= 2),
                     ("1+ HR", pr.hr_a >= 1, "2+ RBI", pr.rbi_b >= 2), ("3+ TB", pr.tb_a >= 3, "3+ TB", pr.tb_b >= 3)]:
    lift, n11 = mh(a, c, pr.stratum_a + pr.pw_a + pr.pw_b)
    rows.append(("teammates top/mid", n1, "teammate " + n2, a.mean(), c.mean(), lift, n11, len(pr)))
out = pd.DataFrame(rows, columns=["who", "leg_a", "leg_b", "p_a", "p_b", "lift", "n11", "n"])
out = out[out.n11 >= 20].sort_values("lift", ascending=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 300)
print(out.round(3).to_string(index=False))
out.to_csv("mlb_edge_map.csv", index=False)
