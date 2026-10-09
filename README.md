# Combo +EV

Local dashboard that finds Kalshi same-game 2-leg combos (MLB + NFL) where correlated legs are
priced as if they were close to independent, and sizes bets with fractional Kelly. It only reads
prices and requests quotes. It never accepts quotes or places orders.

## Setup

```bash
make setup        # venv + dependencies
make ingest       # NFL (nflverse) + MLB (StatsAPI) history + MLB pre-game lines from Kalshi -> data/
                  # (first run ~30 min for the Kalshi archive backfill; reruns are incremental)
make backtest     # optional: out-of-sample check of the correlation model
make run          # http://localhost:8000
```

**Kalshi API key (for live combo quotes):** Kalshi → Settings → API → create a key, save the
private key as `kalshi.pem` in this folder, then `cp .env.example .env` and set `KALSHI_KEY_ID`.
Without a key the app runs in **model-only** mode. You type the payout the Kalshi app shows
($1 pays $x) into a row's Kalshi column and it prices the row.

## How a scan works

1. **Discover**: every open MLB/NFL game on Kalshi in the next ~2 days (MLB) / ~8 days (NFL), and
   all single-leg markets for it (team win/spread/total, player HR/hits/RBI/TB/H+R+RBI, TDs, yards...).
2. **Pair**: every pair of legs in a game is considered (`app/pairs.py`): same player two stats,
   teammates, player + own team, player + game, team + team, game + team, game + game (plus pitcher
   Ks + opponent UNDER). Pairs survive if their legs are in different Kalshi events, they pay ≥ 3x
   if independent, and a correlation atlas (rank correlation of the underlying stats on history)
   is ≥ 0.15. Combo types are generated (e.g. "Same player: TDs + rush yds"). Combos Kalshi
   rejects as duplicated legs are remembered and skipped.
3. **Model**: each leg's probability = Kalshi bid/ask midpoint, except thin rungs: count props
   (2+ HR at 0¢/1¢, 3+ TD...) are priced from the player's liquid lower rung with a
   dispersion-corrected Poisson ladder, and line ladders (team totals, spreads, half/quarter lines,
   yardage) from a logit curve through the rungs that trade (both shown as `≈`). Legs with no bid
   and no curve/ladder price are dropped. **Pessimistic EV** prices each leg conservatively (its
   bid, or the curve/ladder value minus 30%) on top of the correlation's 95% range. The joint probability uses a
   Gaussian copula whose correlation ρ is fit on the last 3+ seasons of history: the same
   role (WR1/RB1/TE1/QB1, lineup slot), the same count threshold (3+ TDs), or a quantile-matched
   threshold for yardage and team scoring. Team outcomes are measured vs pre-game expectation:
   NFL closing lines (nflverse), MLB Kalshi's own pre-game moneyline/total prices.
4. **Filter**: keep pairs whose model probability beats independence by `min_lift`. Pick up to
   `max_quotes_per_game` per game. Once 30+ live quotes exist for a combo type, slots go to the
   highest *predicted* EV (learned from how makers priced that type over the last 21 days), with
   20% kept for exploring less-quoted types. History tab → "Maker pricing by combo type".
5. **Quote** (live mode only): create the combo market (cached; 5,000 creations/week limit),
   send an RFQ for `quote_size` dollars, take the best maker quote, delete the RFQ.
6. **Price**: EV = q / (price + fee) − 1, with fee = 0.07·P·(1−P). Stake = Kelly fraction × full
   Kelly × bankroll, capped per bet and per game.

## Auto-betting (off by default)

Settings bar → **Auto-bet**: Off / Dry run (paper bets) / Live. Only runs during scans you start:
pessimistic EV > 10% (rechecked on a fresh quote at the real stake), at most 2 combos per game
counting combos already held on Kalshi, no shared legs, MLB hitters must be in the posted lineup,
quarter Kelly from the real Kalshi balance (2%/bet, 5%/game, $1 minimum), daily cap 10% of balance.
Bets land in Bet history tagged Auto/Paper and settle automatically. Live requires
`KALSHI_ACCEPT_SIDE` in `.env` (confirmed `no` by `make side-test CONFIRM=1`, a real $1 bet).

## Caveats

- **Quote semantics:** buying YES costs `1 − maker no_bid`. Confirmed on live quotes: makers
  answer combo RFQs with `yes_bid = 0` and only bid NO. Expand a row to see the raw maker quotes.
- The 95% range on "Fair" comes from the historical sample size. Extreme tails (3+ TDs, 2+ HRs)
  rest on a few dozen joint hits, so the expanded row also shows a pessimistic EV.
- Leg midpoints are only as good as the market. Rows flagged "wide leg spread" have a leg
  with a bid/ask gap above 6¢.
