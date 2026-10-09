"""Placing combo bets on Kalshi by accepting an RFQ quote.

Only app/autobet.py calls this, and only under the user's auto-bet rules. Every attempt re-quotes
at the real stake and re-checks the pessimistic-EV rule at that price before accepting.
"""

import asyncio
import time
from dataclasses import dataclass

from app import db
from app.kalshi.client import KalshiClient
from app.kalshi.combos import best_quote, collect_quotes, combo_market, open_rfq
from app.legs import Leg, game_key, market_key
from app.pricing import all_in_cost

EXECUTION_TIMEOUT = 15.0  # seconds to wait for the maker to confirm and the trade to execute


@dataclass
class PlaceResult:
    status: str  # "placed" | "paper" | "skipped" | "failed"
    reason: str
    combo_ticker: str
    yes_price: float | None = None
    cost: float | None = None  # all-in per $1 contract
    ev: float | None = None
    ev_lo: float | None = None
    contracts: float | None = None
    spent: float | None = None  # dollars actually paid, fees included


async def balance(client: KalshiClient) -> float:
    return float((await client.get("/portfolio/balance"))["balance_dollars"])


async def held_combos(client: KalshiClient) -> dict[str, dict]:
    """Open combo positions: combo ticker -> {"game": game key, "legs": {leg market keys}}."""
    positions = await client.paginate("/portfolio/positions", "market_positions", {"limit": 200})
    out = {}
    for p in positions:
        ticker = p["ticker"]
        if not ticker.startswith("KXMVE") or float(p.get("position_fp") or 0) == 0:
            continue
        legs = db.get_combo_legs(ticker)
        if legs is None:
            market = (await client.get(f"/markets/{ticker}")).get("market", {})
            legs = market.get("mve_selected_legs") or []
            db.put_combo_legs(ticker, legs)
        if legs:
            out[ticker] = {"game": game_key(legs[0]["event_ticker"]),
                           "legs": {market_key(leg["event_ticker"], leg["market_ticker"]) for leg in legs}}
    return out


async def _wait_executed(client: KalshiClient, quote_id: str) -> str:
    deadline = time.monotonic() + EXECUTION_TIMEOUT
    status = "accepted"
    while time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        status = (await client.get(f"/communications/quotes/{quote_id}")).get("quote", {}).get("status", status)
        if status in ("executed", "cancelled"):
            break
    return status


async def _fills(client: KalshiClient, ticker: str, since: float) -> tuple[float, float, str | None]:
    """(contracts, dollars spent incl. fees, side) for our fills on `ticker` since `since`."""
    fills = (await client.get("/portfolio/fills", {"ticker": ticker, "min_ts": int(since) - 5})).get("fills", [])
    contracts = spent = 0.0
    side = None
    for f in fills:
        n = float(f.get("count_fp") or 0)
        yes_price = float(f.get("yes_price_dollars") or (1 - float(f.get("no_price_dollars") or 1)))
        price = yes_price if f.get("side") == "yes" else 1 - yes_price
        contracts += n
        spent += n * price + float(f.get("fee_cost") or 0)
        side = f.get("side")
    return contracts, spent, side


async def place_combo(client: KalshiClient, collection: str, a: Leg, b: Leg, stake: float, q: float, q_lo: float,
                      min_ev_lo: float, dry_run: bool, accept_side: str | None, wait_seconds: float) -> PlaceResult:
    ticker, _ = await combo_market(client, collection, a, b)
    async with open_rfq(client, ticker, stake) as rfq_id:
        quote, price = best_quote(await collect_quotes(client, rfq_id, wait_seconds))
        if quote is None:
            return PlaceResult("skipped", "no maker quotes at the real stake", ticker)
        cost = all_in_cost(price)
        ev, ev_lo = q / cost - 1, q_lo / cost - 1
        result = PlaceResult("skipped", "", ticker, price, cost, ev, ev_lo, stake / cost, stake)
        if ev_lo <= min_ev_lo:
            result.reason = f"pessimistic EV {ev_lo:+.1%} at real size (needs > {min_ev_lo:.0%})"
            return result
        if dry_run:
            result.status, result.reason = "paper", "dry run: would have placed"
            return result
        if accept_side not in ("yes", "no"):
            result.status, result.reason = "failed", "KALSHI_ACCEPT_SIDE not set (run make side-test)"
            return result
        # Stay inside the RFQ context until execution: deleting the RFQ early could cancel the trade.
        accepted_at = time.time()
        await client.put(f"/communications/rfqs/{rfq_id}/quotes/{quote['id']}/accept",
                         {"accepted_side": accept_side})
        status = await _wait_executed(client, quote["id"])
        if status != "executed":
            result.status, result.reason = "failed", f"quote {status} (maker didn't confirm)"
            return result
    contracts, spent, side = await _fills(client, ticker, accepted_at)
    result.status, result.reason = "placed", f"filled {contracts:.2f} contracts ({side})"
    result.contracts, result.spent = contracts or result.contracts, spent or stake
    if side and side != "yes":
        result.reason += " -- WARNING: not a YES position, check KALSHI_ACCEPT_SIDE"
    return result
