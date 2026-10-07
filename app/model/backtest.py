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
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    fmt = {c: "{:.4f}".format for c in ["P(A)", "P(B)", "actual", "indep", "model"]}
    fmt.update({"rho_train": "{:.2f}".format, "z_model": "{:+.1f}".format, "z_indep": "{:+.1f}".format})
    print(df.to_string(index=False, formatters=fmt))
    print(f"\nmean |z| model: {np.abs(df.z_model).mean():.2f}   mean |z| independence: {np.abs(df.z_indep).mean():.2f}")
    print("(|z| < 2 means the held-out joint rate is within sampling noise of the prediction)")


if __name__ == "__main__":
    main()
