import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app import autobet, config, db
from app.config import DEFAULT_SETTINGS
from app.kalshi import trading
from app.legs import Leg, market_key

FUTURE = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()


def leg(ticker, kind="team_win", player=None, sport="MLB"):
    return Leg(sport, "G1", "S", f"E-{ticker}", ticker, kind, "SD", player, 1.5, 0.1, 0.12, ticker)


def row(i, a, b, ev_lo, game="G1"):
    return {"game": game, "start": FUTURE, "combo_ticker": f"C{i}", "ev_lo": ev_lo, "q": 0.03, "q_lo": 0.025,
            "kelly_full": 0.02, "_legs": (a, b),
            "leg_a": {"ticker": a.ticker, "event_ticker": a.event_ticker},
            "leg_b": {"ticker": b.ticker, "event_ticker": b.event_ticker}}


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    db.init()


def test_select_caps_per_game_counting_held_and_never_shares_legs():
    win = leg("SDwin")
    rows = [row(1, leg("hr1", "hr", "A"), win, 0.30), row(2, leg("hr2", "hr", "B"), win, 0.25),
            row(3, leg("hr3", "hr", "C"), leg("SDtt"), 0.20), row(4, leg("hr4", "hr", "D"), leg("SDtt5"), 0.15)]
    picks = autobet.select(rows, {}, 2)
    assert [r["combo_ticker"] for r in picks] == ["C1", "C3"]  # C2 shares the SD-win leg with C1
    held = {"G1": [{"heldX", "heldY"}]}
    assert [r["combo_ticker"] for r in autobet.select(rows, held, 2)] == ["C1"]  # 1 already held
    held_overlap = {"G1": [{market_key("E-hr1", "hr1"), "other"}]}  # held combo uses the hr1 market
    assert "C1" not in [r["combo_ticker"] for r in autobet.select(rows, held_overlap, 2)]


def test_eligible_rules():
    s = DEFAULT_SETTINGS
    now = datetime.now(timezone.utc)
    hr = leg("hr1", "hr", "Manny Machado")
    r = row(1, hr, leg("SDwin"), 0.15)
    assert not autobet.eligible(r, {}, s, set(), now)  # not in posted lineup
    assert autobet.eligible(r, {"manny machado": 3}, s, set(), now)
    assert not autobet.eligible({**r, "ev_lo": 0.09}, {"manny machado": 3}, s, set(), now)  # needs > 10%
    assert not autobet.eligible(r, {"manny machado": 3}, s, {"C1"}, now)  # already bet/held
    nfl = row(2, leg("td", "td", "CeeDee Lamb", "NFL"), leg("DALtt", sport="NFL"), 0.2)
    assert autobet.eligible(nfl, {}, s, set(), now)  # lineup rule is MLB-only


class FakeClient:
    def __init__(self, no_bid, balance="100.00"):
        self.no_bid, self.balance_ = no_bid, balance
        self.calls = []

    async def get(self, path, params=None, auth=False):
        self.calls.append(("GET", path))
        if path == "/portfolio/balance":
            return {"balance_dollars": self.balance_}
        if path == "/communications/quotes":
            return {"quotes": [{"id": "q1", "yes_bid_dollars": "0", "no_bid_dollars": self.no_bid}]}
        if path == "/communications/quotes/q1":
            return {"quote": {"status": "executed"}}
        if path == "/portfolio/fills":
            return {"fills": [{"count_fp": "50", "yes_price_dollars": "0.0200", "fee_cost": "0.07", "side": "yes"}]}
        return {}

    async def paginate(self, path, key, params=None):
        return []

    async def post(self, path, json):
        self.calls.append(("POST", path))
        if path == "/communications/rfqs":
            return {"id": "rfq1"}
        return {"market_ticker": "C1", "event_ticker": "CE"}

    async def put(self, path, json):
        self.calls.append(("PUT", path, json))

    async def delete(self, path):
        self.calls.append(("DELETE", path))


def place(client, dry, q_lo=0.03):
    return asyncio.run(trading.place_combo(client, "COL", leg("a"), leg("b"), 1.0, 0.04, q_lo, 0.10, dry, "no", 0.6))


def test_dry_run_never_accepts_and_deletes_rfq():
    c = FakeClient("0.9800")  # YES at 2c; q_lo 3% -> pessimistic EV ~ +45%
    res = place(c, dry=True)
    assert res.status == "paper"
    assert not any(call[0] == "PUT" for call in c.calls)
    assert ("DELETE", "/communications/rfqs/rfq1") in c.calls


def test_live_rechecks_fresh_quote_before_accepting():
    worse = FakeClient("0.9700")  # fresh price 3c: q_lo 3% -> pessimistic EV < 0
    res = place(worse, dry=False)
    assert res.status == "skipped" and "pessimistic EV" in res.reason
    assert not any(call[0] == "PUT" for call in worse.calls)

    good = FakeClient("0.9800")
    res = place(good, dry=False)
    assert res.status == "placed"
    assert ("PUT", "/communications/rfqs/rfq1/quotes/q1/accept", {"accepted_side": "no"}) in good.calls
    assert res.contracts == pytest.approx(50) and res.spent == pytest.approx(50 * 0.02 + 0.07)


def test_run_respects_daily_cap_and_logs_paper_bets():
    settings = {**DEFAULT_SETTINGS, "auto_bet_mode": "dry_run", "daily_max_spend_pct": 0.02, "quote_wait_seconds": 0.6}
    rows = [row(1, leg("x1"), leg("y1"), 0.5, game="G1"), row(2, leg("x2"), leg("y2"), 0.4, game="G2")]
    for r in rows:
        r["kelly_full"] = 0.5  # quarter Kelly -> capped at 2% of $100 = $2 each
    attempts = asyncio.run(autobet.run(rows, FakeClient("0.9800"), "COL", {}, settings))
    statuses = [r["auto_bet"]["status"] for r in rows]
    assert statuses == ["paper", "skipped"]  # $2 daily cap used by the first bet
    assert db.list_bets()[0]["source"] == "paper"
    assert len(attempts) == 1


def test_small_kelly_stake_rounds_up_to_minimum():
    settings = {**DEFAULT_SETTINGS, "auto_bet_mode": "dry_run", "quote_wait_seconds": 0.6}
    r = row(1, leg("x1"), leg("y1"), 0.5)
    r["kelly_full"] = 0.001  # quarter Kelly on $100 -> $0.03
    asyncio.run(autobet.run([r], FakeClient("0.9800"), "COL", {}, settings))
    assert r["auto_bet"]["status"] == "paper" and r["stake"] == 1.0
    tiny = FakeClient("0.9800", balance="20.00")  # 2% of $20 = $0.40 < $1 minimum
    r2 = row(2, leg("x2"), leg("y2"), 0.5, game="G2")
    asyncio.run(autobet.run([r2], tiny, "COL", {}, settings))
    assert r2["auto_bet"]["status"] == "skipped" and "per-bet cap" in r2["auto_bet"]["reason"]


def test_same_market_different_line_counts_as_shared():
    def L(ticker, event):
        return Leg("NFL", "G1", "S", event, ticker, "team_spread", "MIA", None, 9.5, 0.1, 0.12, ticker)
    spread, total = "KXNFLSPREAD-26OCT11CINMIA", "KXNFLTEAMTOTAL-26OCT11CINMIA"
    a = row(1, L(f"{spread}-MIA10", spread), L(f"{total}-MIA43", total), 0.9)
    b = row(2, L(f"{spread}-MIA14", spread), L(f"{total}-MIA39", total), 0.8)  # same two markets, other lines
    c = row(3, L("KXNFLTD-26OCT11CINMIA-MIATHILL10-2", "KXNFLTD-26OCT11CINMIA"), L(f"{total}-MIA36", total), 0.7)
    assert [r["combo_ticker"] for r in autobet.select([a, b, c], {}, 2)] == ["C1"]  # b and c reuse MIA's total


def test_settings_reject_null_and_nan_modes(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        assert client.put("/api/settings", json={"auto_bet_mode": None}).status_code == 400
        assert client.put("/api/settings", json={"auto_bet_mode": "yolo"}).status_code == 400
        assert client.put("/api/settings", json={"bankroll": "abc"}).status_code == 400
        assert client.put("/api/settings", json={"auto_bet_mode": "dry_run"}).json()["auto_bet_mode"] == "dry_run"
    with db.connect() as c:  # a bad value already in the DB reads back as the safe default
        c.execute("INSERT OR REPLACE INTO settings VALUES ('auto_bet_mode', '\"None\"')")
    assert db.get_settings()["auto_bet_mode"] == "off"


def test_page_uses_versioned_assets_matching_status_build():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        page = client.get("/")
        build = client.get("/api/status").json()["build"]
    assert page.headers["cache-control"] == "no-store"
    assert f'/static/app.js?v={build}' in page.text and f'content="{build}"' in page.text
