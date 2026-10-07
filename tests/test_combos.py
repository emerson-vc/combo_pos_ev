import asyncio

import pytest

from app import config, db
from app.kalshi import combos
from app.legs import Leg


def leg(t):
    return Leg("NFL", "G", "S", f"EV-{t}", t, "td", "DAL", "X", 0.5, 0.1, 0.12, t)


class FakeClient:
    def __init__(self, quotes):
        self.quotes = quotes
        self.calls = []

    async def post(self, path, json):
        self.calls.append(("POST", path, json))
        if path.startswith("/multivariate_event_collections/"):
            return {"market_ticker": "COMBO-1", "event_ticker": "COMBOEV", "market": {"yes_ask_dollars": "0.0000"}}
        if path == "/communications/rfqs":
            return {"id": "rfq-1"}
        raise AssertionError(path)

    async def get(self, path, params=None, auth=False):
        self.calls.append(("GET", path, params))
        if path == "/communications/quotes":
            assert params["rfq_user_filter"] == "self"  # Kalshi 403s without a user filter
            return {"quotes": self.quotes}
        return {"market": {}}

    async def delete(self, path):
        self.calls.append(("DELETE", path, None))
        return {}


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    db.init()


def test_quote_takes_best_yes_price_and_deletes_rfq():
    client = FakeClient([{"yes_bid_dollars": "0.0100", "no_bid_dollars": "0.9750"},
                         {"yes_bid_dollars": "0.0120", "no_bid_dollars": "0.9790"}])
    q = asyncio.run(combos.quote_combo(client, "COL", leg("A"), leg("B"), 5.0, 1.0))
    assert q.yes_price == pytest.approx(0.021)  # 1 - best no_bid
    assert q.source == "rfq" and q.n_quotes == 2
    assert ("DELETE", "/communications/rfqs/rfq-1", None) in client.calls
    assert not any("accept" in c[1] for c in client.calls)  # never trades


def test_combo_ticker_cached_across_calls():
    client = FakeClient([])
    asyncio.run(combos.quote_combo(client, "COL", leg("A"), leg("B"), 5.0, 0.6))
    asyncio.run(combos.quote_combo(client, "COL", leg("B"), leg("A"), 5.0, 0.6))
    creates = [c for c in client.calls if c[1].startswith("/multivariate_event_collections/")]
    assert len(creates) == 1
    assert db.creations_this_week() == 1
