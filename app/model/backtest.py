"""Out-of-sample check of the copula model (`make backtest`).

For representative combo shapes: fit ρ on earlier seasons, then predict the held-out season's
joint hit rate from that season's own marginals. If correlation is real and stable, the model
column should track `actual` far better than the independence column.
"""

import math

import numpy as np
import pandas as pd

from app.model.copula import joint_prob, tetrachoric
from app.model.joint import Model

NFL_PT = [  # (label, role, A(frame), B(frame)) on player x team rows
    ("WR1 1+ TD & team beats implied total", "WR1", lambda f: f.td >= 1, lambda f: f.pts_resid > 0),
    ("WR1 2+ TD & team beats total by 7+", "WR1", lambda f: f.td >= 2, lambda f: f.pts_resid > 7),
    ("WR1 100+ yds & team beats total", "WR1", lambda f: f.rec_yds >= 100, lambda f: f.pts_resid > 0),
    ("RB1 1+ TD & team wins", "RB1", lambda f: f.td >= 1, lambda f: f.won == 1),
    ("RB1 2+ TD & team beats total by 7+", "RB1", lambda f: f.td >= 2, lambda f: f.pts_resid > 7),
    ("TE1 1+ TD & team beats total", "TE1", lambda f: f.td >= 1, lambda f: f.pts_resid > 0),
    ("QB1 3+ pass TD & covers by 7+", "QB1", lambda f: f.pass_td >= 3, lambda f: f.margin_resid > 7),
]
MLB_PT = [
    ("Top-order 1+ HR & team wins", "TOP", lambda f: f.hr >= 1, lambda f: f.won == 1),
    ("Mid-order 1+ HR & team wins", "MID", lambda f: f.hr >= 1, lambda f: f.won == 1),
    ("Mid-order 2+ HR & team wins", "MID", lambda f: f.hr >= 2, lambda f: f.won == 1),
    ("Mid-order 2+ hits & team 6+ runs", "MID", lambda f: f.hit >= 2, lambda f: f.team_runs >= 6),
    ("Mid-order 2+ RBI & team 6+ runs", "MID", lambda f: f.rbi >= 2, lambda f: f.team_runs >= 6),
    ("Bottom-order 4+ TB & team wins", "BOT", lambda f: f.tb >= 4, lambda f: f.won == 1),
]


MLB_RESID = [  # team outcomes vs Kalshi pre-game expectation (data/mlb_lines.parquet)
    ("Mid-order 1+ HR & team beats implied runs", "MID", lambda f: f.hr >= 1, lambda f: f.runs_resid > 0),
    ("Mid-order 2+ hits & beats implied runs by 2+", "MID", lambda f: f.hit >= 2, lambda f: f.runs_resid > 2),
    ("Top-order 2+ RBI & beats expected margin by 2+", "TOP", lambda f: f.rbi >= 2, lambda f: f.margin_resid > 2),
]
MLB_RESID_SPLIT = "2026-07-15"
# Pair families from the all-pairs search: (label, role prefix, A, B) on NFL player rows.
NFL_WIDE = [
    ("RB1 1+ TD & 70+ rush yds (same player)", "RB1", lambda f: f.td >= 1, lambda f: f.rush_yds >= 70),
    ("WR1 6+ rec & 80+ rec yds (same player)", "WR1", lambda f: f.rec >= 6, lambda f: f.rec_yds >= 80),
    ("RB1 first TD & team wins", "RB1", lambda f: f.first_td == 1, lambda f: f.won == 1),
    ("QB1 40+ pass att & team loses", "QB1", lambda f: f.pass_att >= 40, lambda f: f.won == 0),
]
# (label, frame, A, B, residual column or None): new MLB combo types. Rows that use a residual
# vs Kalshi's pre-game line only exist for 2026, so those use the in-season date split.
MLB_NEW = [
    ("SP 7+ K & team wins", "MLB_pitcher", lambda f: f.ks >= 7, lambda f: f.won == 1, None),
    ("SP 8+ K & opponent under expectation", "MLB_pitcher", lambda f: f.ks >= 8,
     lambda f: f.opp_runs_resid < 0, "opp_runs_resid"),
    ("Leads after 5 & wins", "MLB_team", lambda f: f.f5_won == 1, lambda f: f.won == 1, None),
    ("Run in 1st & game beats implied total", "MLB_game", lambda f: f.first_inning_runs >= 1,
     lambda f: f.total_resid > 0, "total_resid"),
    ("F5 total 6+ & game beats implied total by 3+", "MLB_game", lambda f: f.f5_total >= 6,
     lambda f: f.total_resid > 3, "total_resid"),
]


def evaluate(label, frame, season_col, test_season, fa, fb, split=None) -> dict:
    if split is None:
        train, test = frame[frame[season_col] < test_season], frame[frame[season_col] == test_season]
    else:
        train, test = frame[~split], frame[split]
    fit = tetrachoric(fa(train).to_numpy(), fb(train).to_numpy())
    a, b = fa(test).to_numpy(), fb(test).to_numpy()
    pa, pb, actual = a.mean(), b.mean(), (a & b).mean()
    model = joint_prob(pa, pb, fit["rho"])
    se = math.sqrt(max(model * (1 - model), 1e-12) / len(test))
    return {"combo": label, "n_test": len(test), "P(A)": pa, "P(B)": pb, "rho_train": fit["rho"],
            "actual": actual, "indep": pa * pb, "model": model, "z_model": (actual - model) / se,
            "z_indep": (actual - pa * pb) / math.sqrt(max(pa * pb * (1 - pa * pb), 1e-12) / len(test))}


def main():
    m = Model()
    rows = []
    if m.has("NFL"):
        f = m.frames["NFL_player"]
        weeks = f.groupby("season")["week"].max()
        test = int(weeks[weeks >= 18].index.max())  # latest complete season
        for label, role, fa, fb in NFL_PT:
            rows.append({"sport": f"NFL {test}", **evaluate(label, f[f.role == role], "season", test, fa, fb)})
        t = m.frames["NFL_team"]
        rows.append({"sport": f"NFL {test}", **evaluate("Covers by 7+ & beats total by 7+", t, "season", test,
                                                         lambda x: x.margin_resid > 7, lambda x: x.pts_resid > 7)})
    if m.has("MLB"):
        f = m.frames["MLB_player"]
        test = int(f["season"].max())
        for label, bucket, fa, fb in MLB_PT:
            rows.append({"sport": f"MLB {test}", **evaluate(label, f[f.bucket == bucket], "season", test, fa, fb)})
        r = f[f["runs_resid"].notna()]
        if len(r) > 5000:
            for label, bucket, fa, fb in MLB_RESID:
                sub = r[r.bucket == bucket]
                rows.append({"sport": "MLB resid", **evaluate(label, sub, "season", None, fa, fb,
                                                               split=sub["date"] >= MLB_RESID_SPLIT)})
    if m.has("NFL"):
        f = m.frames["NFL_player"]
        weeks = f.groupby("season")["week"].max()
        test = int(weeks[weeks >= 18].index.max())
        for label, role, fa, fb in NFL_WIDE:
            rows.append({"sport": f"NFL wide", **evaluate(label, f[f.role.str.startswith(role)], "season", test,
                                                         fa, fb)})
        g = m.frames["NFL_game"]
        rows.append({"sport": "NFL wide", **evaluate("1H total 24+ & game beats total", g, "season", test,
                                                     lambda x: x.h1_total >= 24, lambda x: x.total_resid > 0)})
    if m.has("MLB"):
        b = m.frames["MLB_player"]
        mid = b[b.bucket == "MID"][["game_pk", "team", "player_id", "season", "hr"]]
        pair = mid.merge(mid, on=["game_pk", "team", "season"], suffixes=("_a", "_b"))
        pair = pair[pair.player_id_a < pair.player_id_b]
        rows.append({"sport": "MLB wide", **evaluate("Mid-order teammates both 1+ HR", pair, "season",
                                                     int(pair.season.max()), lambda x: x.hr_a >= 1,
                                                     lambda x: x.hr_b >= 1)})
    if "MLB_pitcher" in m.frames and "MLB_game" in m.frames:
        for label, name, fa, fb, resid_col in MLB_NEW:
            f = m.frames[name]
            if resid_col:
                f = f[f[resid_col].notna()]
                row = evaluate(label, f, "season", None, fa, fb, split=f["date"] >= MLB_RESID_SPLIT)
            else:
                row = evaluate(label, f, "season", int(f["season"].max()), fa, fb)
            rows.append({"sport": "MLB new", **row})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    fmt = {c: "{:.4f}".format for c in ["P(A)", "P(B)", "actual", "indep", "model"]}
    fmt.update({"rho_train": "{:.2f}".format, "z_model": "{:+.1f}".format, "z_indep": "{:+.1f}".format})
    print(df.to_string(index=False, formatters=fmt))
    print(f"\nmean |z| model: {np.abs(df.z_model).mean():.2f}   mean |z| independence: {np.abs(df.z_indep).mean():.2f}")
    print("(|z| < 2 means the held-out joint rate is within sampling noise of the prediction)")


if __name__ == "__main__":
    main()
