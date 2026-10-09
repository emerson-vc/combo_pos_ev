import asyncio

import numpy as np
import pandas as pd
import pytest

from app import config, db
from app.config import DEFAULT_SETTINGS
from app.kalshi import combos
from app.legs import Leg, parse_market
from app.model.joint import Model
from app.scan import pick_diverse

TEAMS = ["MIL", "SD"]


def market(ticker, title, strike=None, bid="0.40", ask="0.42"):
    event = "-".join(ticker.split("-")[:2])  # SERIES-GAMEKEY
    return {"ticker": ticker, "event_ticker": event, "title": title, "yes_sub_title": title,
            "floor_strike": strike, "yes_bid_dollars": bid, "yes_ask_dollars": ask, "status": "active"}


def test_parse_new_mlb_markets():
    ks = parse_market(market("KXMLBKS-26OCT072200MILSD-SDWBUEHLER10-7", "Walker Buehler: 7+ strikeouts?", 6.5), TEAMS)
    assert (ks.kind, ks.team, ks.player, ks.count_k) == ("ks", "SD", "Walker Buehler", 7)
    tie = parse_market(market("KXMLBF5-26OCT072200MILSD-TIE", "first 5 innings tie"), TEAMS)
    assert tie.kind == "f5_win" and tie.team is None
    f5 = parse_market(market("KXMLBF5-26OCT072200MILSD-SD", "San Diego first 5 innings winner"), TEAMS)
    assert (f5.kind, f5.team) == ("f5_win", "SD")
    rfi = parse_market(market("KXMLBRFI-26OCT072200MILSD", "1st inning: Over 0.5 runs", 1), TEAMS)
    assert (rfi.kind, rfi.team, rfi.count_k) == ("rfi", None, 1)


def leg(kind, team, p, player=None, threshold=None, ticker=None):
    return Leg("MLB", "G", "S", f"E-{kind}", ticker or f"T-{kind}-{team}-{player}-{threshold}", kind, team,
               player, threshold, p - 0.01, p + 0.01, f"{player or team} {kind} {threshold}")


def test_search_builds_pitcher_opponent_under():
    from app.pairs import candidate_pairs

    class Stub:
        def atlas_rho(self, a, b):
            return 0.4

    legs = [leg("ks", "SD", 0.2, "Ace", 6.5), leg("team_total", "MIL", 0.6, threshold=3.5)]
    pairs = candidate_pairs(legs, TEAMS, Stub(), DEFAULT_SETTINGS)
    unders = [(a, b) for t, a, b in pairs if t == "mlb_sp_opp_under"]
    assert len(unders) == 1
    assert unders[0][1].side == "no" and unders[0][1].p == pytest.approx(0.4)


def test_no_side_combo_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    db.init()
    sent = []

    class Client:
        async def post(self, path, json):
            sent.append(json)
            return {"market_ticker": "C", "event_ticker": "CE"}

    a, b = leg("ks", "SD", 0.2, "Ace", 6.5), leg("team_total", "MIL", 0.6, threshold=3.5).negate()
    asyncio.run(combos.combo_market(Client(), "COL", a, b))
    sides = {m["market_ticker"]: m["side"] for m in sent[0]["selected_markets"]}
    assert sides == {a.ticker: "yes", b.ticker: "no"}
    assert combos.legs_key(a, b) != combos.legs_key(a, leg("team_total", "MIL", 0.6, threshold=3.5))


@pytest.fixture
def pitcher_model(tmp_path):
    """Synthetic MLB history: strikeout-heavy starts suppress the opponent's runs."""
    rng = np.random.default_rng(1)
    n = 3000
    ks = rng.poisson(5, n)
    opp_runs = rng.poisson(np.clip(5.5 - 0.35 * ks, 0.3, None))
    rows = []
    for i in range(n):
        rows.append({"game_pk": i, "team": "SD", "opp": "MIL", "runs": 4, "opp_runs": opp_runs[i],
                     "margin": 4 - opp_runs[i], "won": int(opp_runs[i] < 4), "date": "2026-05-01", "season": 2026})
        rows.append({"game_pk": i, "team": "MIL", "opp": "SD", "runs": opp_runs[i], "opp_runs": 4,
                     "margin": opp_runs[i] - 4, "won": int(opp_runs[i] > 4), "date": "2026-05-01", "season": 2026})
    pd.DataFrame(rows).to_parquet(tmp_path / "mlb_teams.parquet")
    pd.DataFrame({"game_pk": [0], "team": ["SD"], "player_id": [1], "name": ["X"], "slot": [1], "season": [2026],
                  "hit": [1], "hr": [0], "rbi": [0], "runs": [0], "tb": [1], "hrr": [1], "sb": [0]}
                 ).to_parquet(tmp_path / "mlb_batters.parquet")
    pd.DataFrame({"game_pk": range(n), "team": "SD", "player_id": 7, "name": "Ace", "season": 2026,
                  "ks": ks, "outs": 18}).to_parquet(tmp_path / "mlb_pitchers.parquet")
    return Model(tmp_path)


def test_pitcher_ks_with_opponent_under_is_positively_correlated(pitcher_model):
    ks = leg("ks", "SD", 0.25, "Ace", 6.5)
    under = leg("team_total", "MIL", 0.55, threshold=2.5).negate()  # MIL under 2.5 runs
    est = pitcher_model.estimate(ks, under)
    assert est is not None and est.role_a == "SP"
    assert est.rho > 0.2 and est.q > ks.p * under.p


def test_longshot_slots_reserved():
    def row(i, p_long, lift):
        return {"template": "t", "game": "G", "p_a": p_long, "p_b": 0.5, "lift": lift, "q": 0.01,
                "leg_a": {"ticker": f"a{i}", "p": p_long}, "leg_b": {"ticker": f"b{i}", "p": 0.5}}
    rows = [row(i, 0.3, 3.0) for i in range(20)] + [row(100 + i, 0.02, 1.2) for i in range(10)]
    picked = pick_diverse(rows, 10, None, longshot_share=0.4)
    assert sum(r["p_a"] <= 0.05 for r in picked) >= 4


def test_hr_ml_combos_always_quoted(monkeypatch):
    from app import scan
    from app.legs import Leg
    from app.model.joint import Estimate

    class FakeModel:
        ladder = type("L", (), {"apply": staticmethod(lambda legs: legs)})()

        def has(self, sport):
            return True

        def atlas_rho(self, a, b):
            return 0.5

        def estimate(self, a, b, haircut=0.3):
            lift = 1.5 if a.kind == "hr" else 3.0  # other pairs outrank 2+ HR + ML on lift
            q = lift * a.p * b.p
            return Estimate(q, q * 0.9, q * 1.1, 0.3, 0.2, 0.4, 1000, 50, "MID", "TEAM", "copula")

    def L(kind, p, player=None, k=None, i=0):
        return Leg("MLB", "G", "S", f"E-{kind}", f"T-{kind}-{player}-{k}-{i}", kind, "SD", player,
                   None if k is None else k - 0.5, p - 0.005, p + 0.005, f"{player} {kind} {k}")

    legs = [L("team_win", 0.5)] + [L("hr", 0.03, f"Slugger{i}", 2, i) for i in range(6)]
    legs += [L("hit", 0.03, f"Hitter{i}", 4, i) for i in range(30)]
    game = type("G", (), {"sport": "MLB", "key": "G", "label": "MIL @ SD", "teams": ["MIL", "SD"], "legs": legs,
                          "start": __import__("datetime").datetime(2026, 10, 8)})()
    settings = {**DEFAULT_SETTINGS, "mlb_quotes_per_game": 5}
    rows, _ = scan.build_candidates([game], FakeModel(), settings, None)
    assert sum(r["template"] == "mlb_hr_win" for r in rows) == 6  # all pinned, on top of the 5 slots
    assert len(rows) == 11
    rows, _ = scan.build_candidates([game], FakeModel(), {**settings, "always_quote_hr_ml": False}, None)
    assert len(rows) == 5  # setting off: they compete for the 5 slots like anything else
