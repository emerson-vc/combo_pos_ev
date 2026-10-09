import asyncio
import math

import numpy as np
import pandas as pd
import pytest

from app import config, db
from app.config import DEFAULT_SETTINGS
from app.kalshi import combos
from app.kalshi.client import KalshiError
from app.legs import Leg, parse_market
from app.model.joint import Model
from app.model.ladder import curve_fit
from app.templates import usable

EV = "KXNFLTEAMTOTAL-26OCT11CINMIA"


def tt(x, bid, ask):
    return Leg("NFL", "26OCT11CINMIA", "KXNFLTEAMTOTAL", EV, f"{EV}-MIA{int(x + .5)}", "team_total", "MIA", None,
               x, bid, ask, f"MIA over {x}")


def test_curve_prices_thin_tail_from_liquid_rungs():
    # Liquid rungs follow logit P = 2 - 0.12x; the 42.5 rung is quoted 0c/6c.
    rungs = [tt(x, round(1 / (1 + math.exp(-(2 - .12 * x))) - .01, 2), round(1 / (1 + math.exp(-(2 - .12 * x))) + .01, 2))
             for x in (14.5, 17.5, 20.5, 24.5, 28.5)] + [tt(42.5, 0.0, 0.06)]
    out = {leg.threshold: leg for leg in curve_fit(rungs)}
    assert out[42.5].p_source == "curve"
    assert out[42.5].p == pytest.approx(1 / (1 + math.exp(-(2 - .12 * 42.5))), abs=0.003)  # ~4%, not the 3c mid
    assert out[24.5].p_source == "market"


def test_unbid_market_leg_unusable_and_p_lo_uses_bid():
    s = DEFAULT_SETTINGS
    assert not usable(tt(35.5, 0.0, 0.06), s)
    leg = tt(24.5, 0.21, 0.23)
    assert usable(leg, s) and leg.p_lo() == 0.21 and leg.p == pytest.approx(0.22)


def test_parse_new_nfl_markets():
    m = {"ticker": "KXNFLTEAMFIRSTTD-26OCT11PHIJAC-JAC-TLAWRENCE16", "event_ticker": "KXNFLTEAMFIRSTTD-26OCT11PHIJAC-JAC",
         "title": "Trevor Lawrence: 1st Jacksonville Touchdown", "yes_sub_title": "Trevor Lawrence",
         "yes_bid_dollars": "0.05", "yes_ask_dollars": "0.07", "status": "active"}
    leg = parse_market(m, ["PHI", "JAC"])
    assert (leg.kind, leg.team, leg.player) == ("team_first_td", "JAC", "Trevor Lawrence")
    h1 = parse_market({"ticker": "KXNFL1HTOTAL-26OCT12BUFLAR-43", "event_ticker": "KXNFL1HTOTAL-26OCT12BUFLAR",
                       "title": "Will there be over 42.5 1H points scored?", "floor_strike": 42.5,
                       "yes_bid_dollars": "0.05", "yes_ask_dollars": "0.07", "status": "active"}, ["BUF", "LAR"])
    assert (h1.kind, h1.team, h1.threshold) == ("h1_total", None, 42.5)


@pytest.fixture
def same_player_model(tmp_path):
    """RB history where TDs come with rushing yards."""
    rng = np.random.default_rng(3)
    n = 3000
    yds = rng.gamma(4, 15, n)
    td = rng.poisson(np.exp(-1.6 + 0.012 * yds))
    teams = pd.DataFrame({"game_id": np.arange(n), "team": "MIA", "won": rng.integers(0, 2, n),
                          "margin_resid": rng.normal(0, 10, n), "pts_resid": rng.normal(0, 9, n)})
    players = pd.DataFrame({"game_id": np.arange(n), "team": "MIA", "player_id": "rb", "name": "Runner",
                            "role": "RB1", "season": 2025, "week": 1, "td": td, "rush_yds": yds, "rec_yds": 0,
                            "rec": 0, "pass_yds": 0, "pass_td": 0})
    teams.to_parquet(tmp_path / "nfl_teams.parquet")
    players.to_parquet(tmp_path / "nfl_players.parquet")
    return Model(tmp_path)


def test_same_player_pair_is_modeled_and_screened(same_player_model):
    def L(kind, ev, k, p):
        return Leg("NFL", "G", "S", ev, f"{ev}-MIARUNNER-{k}", kind, "MIA", "Runner", k - 0.5, p - .01, p + .01, kind)
    td, yds = L("td", "E-TD", 2, 0.15), L("rush_yds", "E-YDS", 100, 0.2)
    assert same_player_model.atlas_rho(td, yds) > 0.15
    est = same_player_model.estimate(td, yds)
    assert est is not None and est.rho > 0.3 and est.q > td.p * yds.p
    assert est.q_lo < est.q  # bid prices + rho CI


def test_duplicated_legs_rejection_is_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    db.init()
    calls = []

    class Client:
        async def post(self, path, json):
            calls.append(path)
            raise KalshiError(400, '{"error":{"code":"duplicated_legs"}}', path)

    a, b = tt(24.5, .2, .22), tt(28.5, .1, .12)
    for _ in range(2):
        with pytest.raises(combos.ComboRejected):
            asyncio.run(combos.combo_market(Client(), "COL", a, b))
    assert len(calls) == 1  # second attempt answered from the cache
    assert combos.legs_key(a, b) in db.rejected_combo_keys() and db.creations_this_week() == 0
