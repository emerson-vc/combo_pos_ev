"""Price a 2-leg combo on Kalshi: create (or reuse) the combo market, then RFQ for a quote.

Quoting never trades: RFQs are deleted after reading quotes. Accepting a quote (placing a bet)
lives in app/kalshi/trading.py and only runs under the auto-bet rules.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from app import config, db
from app.kalshi.client import KalshiClient, KalshiError
from app.legs import Leg


REJECTED = "REJECTED"  # combo_tickers marker: Kalshi refused this leg combination


class ComboRejected(Exception):
    """Kalshi won't build this combo (e.g. one leg implies the other)."""


@dataclass
class ComboQuote:
    market_ticker: str
    yes_price: float | None  # what we'd pay per $1 YES contract, before fees
    source: str  # "rfq" | "book" | "none"
    n_quotes: int
    raw: list[dict]  # raw maker quotes, kept to verify quote semantics against the app
    quoted_at: float


def legs_key(a: Leg, b: Leg) -> str:
    # YES legs keep the bare ticker so combos cached before NO legs existed still match.
    return "|".join(sorted(leg.ticker if leg.side == "yes" else f"{leg.ticker}:no" for leg in (a, b)))


async def combo_market(client: KalshiClient, collection: str, a: Leg, b: Leg) -> tuple[str, dict | None]:
    key = legs_key(a, b)
    cached = db.get_combo_ticker(key)
    if cached:
        if cached["market_ticker"] == REJECTED:
            raise ComboRejected(f"Kalshi rejects this combination ({cached['event_ticker']})")
        return cached["market_ticker"], None
    if db.creations_this_week() >= config.WEEKLY_COMBO_CREATION_LIMIT:
        raise RuntimeError("Weekly Kalshi combo-creation budget (5,000) used up")
    try:
        resp = await client.post(f"/multivariate_event_collections/{collection}", {
            "selected_markets": [
                {"market_ticker": leg.ticker, "event_ticker": leg.event_ticker, "side": leg.side} for leg in (a, b)
            ],
            "with_market_payload": True,
        })
    except KalshiError as e:
        if e.status == 400 and "duplicated_legs" in str(e):
            db.put_combo_ticker(key, collection, REJECTED, "duplicated_legs")  # never ask again
            raise ComboRejected("Kalshi rejects this combination (one leg implies the other)") from e
        raise
    db.put_combo_ticker(key, collection, resp["market_ticker"], resp["event_ticker"])
    return resp["market_ticker"], resp.get("market")


def _yes_cost_from_quote(q: dict) -> float | None:
    """A maker bidding `no_bid` for NO lets us buy YES at 1 - no_bid."""
    no_bid = float(q.get("no_bid_dollars") or 0)
    return round(1 - no_bid, 4) if no_bid > 0 else None


def best_quote(quotes: list[dict]) -> tuple[dict | None, float | None]:
    """The maker quote that lets us buy YES cheapest, and that YES price."""
    best, best_cost = None, None
    for q in quotes:
        cost = _yes_cost_from_quote(q)
        if cost is not None and (best_cost is None or cost < best_cost):
            best, best_cost = q, cost
    return best, best_cost


@asynccontextmanager
async def open_rfq(client: KalshiClient, ticker: str, size_dollars: float):
    """Request quotes for `size_dollars` (fees included); the RFQ is always deleted on exit."""
    rfq = await client.post("/communications/rfqs", {
        "market_ticker": ticker,
        "rest_remainder": False,
        "target_cost_dollars": f"{size_dollars:.2f}",
    })
    try:
        yield rfq["id"]
    finally:
        try:
            await client.delete(f"/communications/rfqs/{rfq['id']}")
        except KalshiError:
            pass


async def collect_quotes(client: KalshiClient, rfq_id: str, wait_seconds: float) -> list[dict]:
    """Poll maker quotes for up to `wait_seconds` (stops early once quotes have settled in)."""
    raw: list[dict] = []
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        raw = (await client.get("/communications/quotes",
                                {"rfq_id": rfq_id, "rfq_user_filter": "self"})).get("quotes", [])
        if raw and time.monotonic() > deadline - wait_seconds / 2:
            break
    return raw


async def quote_combo(client: KalshiClient, collection: str, a: Leg, b: Leg, size_dollars: float,
                      wait_seconds: float) -> ComboQuote:
    ticker, market = await combo_market(client, collection, a, b)
    if market is None:
        market = (await client.get(f"/markets/{ticker}")).get("market", {})
    book_ask = float(market.get("yes_ask_dollars") or 0)
    best = book_ask if 0 < book_ask < 1 else None
    source = "book" if best else "none"

    async with open_rfq(client, ticker, size_dollars) as rfq_id:
        raw = await collect_quotes(client, rfq_id, wait_seconds)

    _, cost = best_quote(raw)
    if cost is not None and (best is None or cost < best):
        best, source = cost, "rfq"
    return ComboQuote(ticker, best, source, len(raw), raw, time.time())
