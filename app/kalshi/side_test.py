"""One real $1 combo bet to learn which `accepted_side` buys YES on an RFQ quote.

Kalshi's docs don't say whether accepting side "yes" or "no" of a maker's quote leaves you holding
YES. Makers on combos only bid NO, so the hypothesis is accepted_side="no" (we hit their NO bid and
take YES). This places ONE bet of $1 on the best-EV combo from the latest live scan, then reads the
fill and position to confirm, and prints the KALSHI_ACCEPT_SIDE value for .env.

    make side-test           # explains, does nothing
    make side-test CONFIRM=1 # places the $1 bet
"""

import asyncio
import sys
import time
from datetime import datetime, timezone

from app import db
from app.kalshi.client import KalshiClient
from app.kalshi.combos import best_quote, collect_quotes, open_rfq
from app.kalshi.trading import _fills, _wait_executed

STAKE = 1.00
HYPOTHESIS = "no"


def pick_combo() -> dict | None:
    scan_id = db.latest_scan_id()
    scan = db.get_scan(scan_id) if scan_id else None
    now = datetime.now(timezone.utc)
    rows = [r for r in (scan or {}).get("results", [])
            if r.get("combo_ticker") and r.get("ev_lo") is not None and r["kalshi_x"] <= 60
            and datetime.fromisoformat(r["start"]) > now]
    return max(rows, key=lambda r: r["ev_lo"], default=None)  # a sensible bet, not a lottery ticket


async def main(confirm: bool):
    row = pick_combo()
    if row is None:
        print("No quoted combo for a game that hasn't started. Run a live scan first.")
        return
    print(f"Combo: {row['leg_a']['label']} + {row['leg_b']['label']} ({row['game_label']})")
    print(f"  ticker {row['combo_ticker']}  scan quote {row['kalshi_x']:.1f}x  EV {row['ev']:+.1%}")
    if not confirm:
        print(f"\nDry: would place a ${STAKE:.2f} bet with accepted_side={HYPOTHESIS!r}. "
              "Re-run with CONFIRM=1 to place it.")
        return
    client = KalshiClient()
    try:
        async with open_rfq(client, row["combo_ticker"], STAKE) as rfq_id:
            quote, price = best_quote(await collect_quotes(client, rfq_id, 4.0))
            if quote is None:
                print("No maker quotes right now; try again or after another scan.")
                return
            print(f"Best quote: YES at ${price:.4f} (maker no_bid {quote['no_bid_dollars']}). "
                  f"Accepting side {HYPOTHESIS!r}...")
            accepted_at = time.time()
            await client.put(f"/communications/rfqs/{rfq_id}/quotes/{quote['id']}/accept",
                             {"accepted_side": HYPOTHESIS})
            status = await _wait_executed(client, quote["id"])
        print(f"Quote status: {status}")
        if status != "executed":
            print("Not executed (maker didn't confirm). Nothing was bought; try again.")
            return
        contracts, spent, side = await _fills(client, row["combo_ticker"], accepted_at)
        positions = await client.paginate("/portfolio/positions", "market_positions", {"limit": 200})
        pos = next((float(p["position_fp"]) for p in positions if p["ticker"] == row["combo_ticker"]), 0.0)
        print(f"Filled {contracts:.2f} contracts, ${spent:.2f} incl. fees, fill side={side!r}, position={pos:+.2f}")
        if side == "yes" and pos > 0:
            print(f"\nConfirmed: accepted_side={HYPOTHESIS!r} buys YES.\nAdd to .env:  KALSHI_ACCEPT_SIDE={HYPOTHESIS}")
        else:
            other = "yes" if HYPOTHESIS == "no" else "no"
            print(f"\nThat bought NO, so YES needs accepted_side={other!r}.\nAdd to .env:  KALSHI_ACCEPT_SIDE={other}\n"
                  "(This $1 NO position can be left to settle or sold in the Kalshi app.)")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main("--confirm" in sys.argv))
