import asyncio
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app import config, db, miner
from app.kalshi import combos
from app.kalshi.client import KalshiError
from app.kalshi.discovery import ComboRules
from app.legs import Leg
from app.model.copula import joint_prob, joint_prob_n
from app.model.joint import Estimate


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    db.init()


def L(kind, p, team="DAL", player=None, i=0, spread=0.02):
    t = f"KX{kind.upper()}-G-{team}{player or ''}{i}"
    return Leg("NFL", "G", "S", f"EV{kind}{player or ''}{team}", t, kind, team, player, 0.5,
               round(p - spread / 2, 4), round(p + spread / 2, 4), f"{player or team} {kind} {i}")


# ---- Emerson's hand-found combos (research/DESIGN.md section 4) -------------------------------------

@pytest.mark.parametrize("legs_x,kalshi_x,lift,edge", [
    ((1.73, 46.73), 106.38, 1.48, True),  # Freeman 2+ HR + Dodgers ML
    ((5.56, 1.79), 10.00, 1.9, True),  # Stafford 4+ pass TD + Rams over 28.5
    ((2.14, 8.55), 17.04, 1.21, True),  # Henderson 9+ K + Brewers ML
    ((2.46, 7.25), 17.67, 1.04, False),  # Allen 325+ pass yds + Bills ML: the trap
])
def test_hand_found_combos(legs_x, kalshi_x, lift, edge):
    q = lift / float(np.prod(legs_x))  # leg asks carry vig, so this is conservative
    ev = miner.ev_at(q, kalshi_x)
    assert (ev > 0.02) == edge


# ---- joint model ---------------------------------------------------------------------------------

def test_joint_prob_n_matches_two_leg_and_independence():
    assert joint_prob_n([0.1, 0.3], np.array([[1, .4], [.4, 1]])) == pytest.approx(joint_prob(0.1, 0.3, 0.4))
    assert joint_prob_n([0.2, 0.3, 0.5], np.eye(3)) == pytest.approx(0.03, rel=1e-4)
    r = np.array([[1, .4, .3], [.4, 1, .5], [.3, .5, 1]])
    assert joint_prob_n([0.1, 0.3, 0.5], r) > 0.015 * 2  # positive correlation lifts the 3-way hit rate


def test_inconsistent_pairwise_rhos_are_repaired():
    r = np.array([[1, .9, -.9], [.9, 1, .9], [-.9, .9, 1]])  # impossible together
    q = joint_prob_n([0.2, 0.3, 0.5], r)
    assert 0 <= q <= 0.2


# ---- Kalshi rules + N-leg combos -----------------------------------------------------------------

def test_combo_rules_respect_yes_only_and_leg_limits():
    a, b, c = L("td", 0.3), L("team_total", 0.5), L("game_total", 0.5)
    rules = ComboRules("COL", {a.event_ticker, b.event_ticker, c.event_ticker}, yes_only={b.event_ticker}, size_max=2)
    assert rules.allows([a, b])
    assert not rules.allows([a, b.negate()])
    assert not rules.allows([a, b, c])


class ComboClient:
    def __init__(self, fail=None):
        self.posts, self.fail = [], fail

    async def post(self, path, json):
        self.posts.append(json)
        if self.fail:
            raise KalshiError(400, self.fail, path)
        return {"market_ticker": "C", "event_ticker": "CE"}


def test_three_leg_combo_and_rejections_are_cached():
    a, b, c = L("td", 0.3), L("team_total", 0.5), L("pass_td", 0.4)
    client = ComboClient()
    asyncio.run(combos.combo_market(client, "COL", a, b, c))
    assert len(client.posts[0]["selected_markets"]) == 3
    bad = ComboClient(fail='{"code":"invalid_combination"}')
    with pytest.raises(combos.ComboRejected):
        asyncio.run(combos.combo_market(bad, "COL", a, c))
    assert combos.legs_key(a, c) in db.rejected_combo_keys()


# ---- maker model + mining loop -------------------------------------------------------------------

def test_maker_model_learns_and_shrinks():
    legs = (L("td", 0.3), L("team_total", 0.5))
    rows = [{"d": 0.7, "levels": miner.type_levels(legs)} for _ in range(30)]
    mean, sd, n = miner.MakerModel(rows).posterior(miner.type_levels(legs))
    assert n == 30 and math.exp(mean) == pytest.approx(0.7, abs=0.05) and sd < 0.1
    other = (L("rec_yds", 0.3), L("team_win", 0.5))
    mean2, sd2, n2 = miner.MakerModel(rows).posterior(miner.type_levels(other))
    assert n2 == 0 and mean2 > mean and sd2 > sd  # unseen type: pulled toward the sport/global level only


class FakeModel:
    ladder = type("Lad", (), {"apply": staticmethod(lambda legs: legs)})()
    lineup_slots = {}

    def has(self, sport):
        return True

    def atlas_rho(self, a, b):
        return 0.3

    def role(self, leg):
        return "X"

    def estimate(self, a, b, haircut=0.3):
        q = 1.6 * a.p * b.p
        return Estimate(q, q * 0.85, q * 1.15, 0.3, 0.2, 0.4, 1000, 50, "X", "X", "copula")

    def estimate_n(self, legs, haircut=0.3):
        q = 2.0 * float(np.prod([leg.p for leg in legs]))
        return Estimate(q, q * 0.8, q * 1.2, 0.3, 0.2, 0.4, 1000, 30, "X", "", "copula")


class MarketClient:
    """Fake Kalshi: makers pay legs_x / d, with d = 0.75 for combos holding a TD leg and 2.2 otherwise."""

    def __init__(self, legs):
        self.legs = {leg.ticker: leg for leg in legs}
        self.combos, self.rfqs, self.n = {}, {}, 0

    async def post(self, path, json):
        if path.startswith("/multivariate_event_collections/"):
            self.n += 1
            ticker = f"COMBO{self.n}"
            self.combos[ticker] = [m["market_ticker"] + ":" + m["side"] for m in json["selected_markets"]]
            return {"market_ticker": ticker, "event_ticker": "CE", "market": {}}
        if path == "/communications/rfqs":
            rid = f"r{len(self.rfqs)}"
            self.rfqs[rid] = json["market_ticker"]
            return {"id": rid}
        raise AssertionError(path)

    async def get(self, path, params=None, auth=False):
        if path == "/communications/quotes":
            sel = self.combos[self.rfqs[params["rfq_id"]]]
            legs = []
            for s in sel:
                t, side = s.rsplit(":", 1)
                legs.append(self.legs[t] if side == "yes" else self.legs[t].negate())
            legs_x = float(np.prod([1 / miner.leg_price(leg) for leg in legs]))
            d = 0.75 if any(leg.kind == "td" for leg in legs) else 2.2
            return {"quotes": [{"no_bid_dollars": f"{1 - d / legs_x:.4f}"}]}
        if path.startswith("/markets/"):
            return {"market": {"result": "yes"}}
        raise AssertionError(path)

    async def delete(self, path):
        return {}


def test_mine_finds_the_soft_combo_type_and_logs_everything():
    legs = [L("td", 0.30, player="WR1"), L("td", 0.25, player="RB1", team="DAL", i=1),
            L("rec_yds", 0.35, player="WR2"), L("team_total", 0.5), L("game_total", 0.5),
            L("pass_yds", 0.4, player="QB1")]
    game = type("G", (), {"sport": "NFL", "key": "G", "label": "NYG @ DAL", "teams": ["NYG", "DAL"], "legs": legs,
                          "start": datetime.now(timezone.utc) + timedelta(hours=5)})()
    rules = ComboRules("COL", {leg.event_ticker for leg in legs})
    client = MarketClient(legs)
    cands, rows = asyncio.run(miner.mine(client, [game], FakeModel(), dict(config.DEFAULT_SETTINGS), rules,
                                         n_quotes=60, batch=10, wait=0.6, requote=3, seed=1, log=lambda *_: None))
    assert any(len(c.legs) == 3 for c in cands)
    ledger = db.ledger_rows()
    assert len(ledger) >= len(rows) > 10
    td = [r["d"] for r in ledger if any(leg["kind"] == "td" for leg in r["legs"])]
    assert td and np.median(td) == pytest.approx(0.75, abs=0.02)
    top = miner.top_bets(cands)
    assert top and all("td" in r["type"] for r in top)  # only the underpriced type is +EV
    assert asyncio.run(miner.grade(client)) > 0
    board = miner.type_board(db.ledger_rows())
    assert board[0]["graded"] > 0 and board[0]["mean_ev"] > 0
