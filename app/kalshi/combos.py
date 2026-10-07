"""Price a 2-leg combo on Kalshi: create (or reuse) the combo market, then RFQ for a quote.

Read-only by design: RFQs are deleted after reading quotes; nothing is ever accepted.
"""

import asyncio
import time
from dataclasses import dataclass

from app import config, db
from app.kalshi.client import KalshiClient, KalshiError
from app.legs import Leg


@dataclass
class ComboQuote:
    market_ticker: str
    yes_price: float | None  # what we'd pay per $1 YES contract, before fees
    source: str  # "rfq" | "book" | "none"
    n_quotes: int
    raw: list[dict]  # raw maker quotes, kept to verify quote semantics against the app
    quoted_at: float


def legs_key(a: Leg, b: Leg) -> str:
    return "|".join(sorted([a.ticker, b.ticker]))


async def combo_market(client: KalshiClient, collection: str, a: Leg, b: Leg) -> tuple[str, dict | None]:
    key = legs_key(a, b)
    cached = db.get_combo_ticker(key)
    if cached:
        return cached["market_ticker"], None
    if db.creations_this_week() >= config.WEEKLY_COMBO_CREATION_LIMIT:
        raise RuntimeError("Weekly Kalshi combo-creation budget (5,000) used up")
    resp = await client.post(f"/multivariate_event_collections/{collection}", {
        "selected_markets": [
            {"market_ticker": leg.ticker, "event_ticker": leg.event_ticker, "side": "yes"} for leg in (a, b)
        ],
        "with_market_payload": True,
    })
    db.put_combo_ticker(key, collection, resp["market_ticker"], resp["event_ticker"])
    return resp["market_ticker"], resp.get("market")


def _yes_cost_from_quote(q: dict) -> float | None:
    """A maker bidding `no_bid` for NO lets us buy YES at 1 - no_bid."""
    no_bid = float(q.get("no_bid_dollars") or 0)
    return round(1 - no_bid, 4) if no_bid > 0 else None


async def quote_combo(client: KalshiClient, collection: str, a: Leg, b: Leg, size_dollars: float,
                      wait_seconds: float) -> ComboQuote:
    ticker, market = await combo_market(client, collection, a, b)
    if market is None:
        market = (await client.get(f"/markets/{ticker}")).get("market", {})
    book_ask = float(market.get("yes_ask_dollars") or 0)
    best = book_ask if 0 < book_ask < 1 else None
    source = "book" if best else "none"

    raw: list[dict] = []
    rfq_id = None
    try:
        rfq = await client.post("/communications/rfqs", {
            "market_ticker": ticker,
            "rest_remainder": False,
            "target_cost_dollars": f"{size_dollars:.2f}",
        })
        rfq_id = rfq["id"]
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            raw = (await client.get("/communications/quotes",
                                    {"rfq_id": rfq_id, "rfq_user_filter": "self"})).get("quotes", [])
            if raw and time.monotonic() > deadline - wait_seconds / 2:
                break
    finally:
        if rfq_id:
            try:
                await client.delete(f"/communications/rfqs/{rfq_id}")
            except KalshiError:
                pass

    for q in raw:
        cost = _yes_cost_from_quote(q)
        if cost is not None and (best is None or cost < best):
            best, source = cost, "rfq"
    return ComboQuote(ticker, best, source, len(raw), raw, time.time())
