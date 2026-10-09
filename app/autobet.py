"""Auto-place qualifying combos during a user-started scan.

Rules (set by the user, all in settings):
- pessimistic EV > `min_pess_ev` on the scan quote AND again on a fresh quote at the real stake;
- at most `max_bets_per_game` combos per game, counting combos already held on Kalshi (dry run: open
  paper bets); highest pessimistic EV first; no two bets on a game share a market at any line
  (incl. held combos);
- MLB player legs only once the player is in the posted starting lineup;
- quarter Kelly with the usual caps, sized from the real Kalshi balance (stakes under the $1 minimum
  round up to it while the per-bet cap allows); daily spend cap.
"""

from collections import defaultdict
from datetime import datetime, timezone

from app import config, db
from app.kalshi import trading
from app.kalshi.client import KalshiClient
from app.legs import PLAYER_KINDS, Leg, market_key
from app.model.joint import norm_name
from app.pricing import size_stakes


def row_markets(r: dict) -> set[str]:
    return {market_key(leg["event_ticker"], leg["ticker"]) for leg in (r["leg_a"], r["leg_b"])}


def lineup_ok(leg: Leg, lineup_slots: dict[str, int]) -> bool:
    if leg.sport != "MLB" or leg.kind not in PLAYER_KINDS or leg.kind in ("ks", "outs"):
        return True  # starting pitchers are announced with the Kalshi market itself
    return norm_name(leg.player or "") in lineup_slots


def eligible(r: dict, lineup_slots: dict[str, int], settings: dict, already: set[str], now: datetime) -> bool:
    a, b = r["_legs"]
    return (r.get("ev_lo") is not None and r["ev_lo"] > settings["min_pess_ev"]
            and r.get("combo_ticker") not in already
            and datetime.fromisoformat(r["start"]) > now
            and lineup_ok(a, lineup_slots) and lineup_ok(b, lineup_slots))


def select(rows: list[dict], held_by_game: dict[str, list[set[str]]], max_per_game: int) -> list[dict]:
    """Per game: fill the remaining slots by pessimistic EV, never sharing a leg."""
    by_game = defaultdict(list)
    for r in rows:
        by_game[r["game"]].append(r)
    picks = []
    for game, cands in by_game.items():
        held = held_by_game.get(game, [])
        used = set().union(*held) if held else set()
        slots = max_per_game - len(held)
        for r in sorted(cands, key=lambda r: -r["ev_lo"]):
            if slots <= 0:
                break
            legs = row_markets(r)  # same market at any line counts as shared
            if legs & used:
                continue
            picks.append(r)
            used |= legs
            slots -= 1
    return picks


async def run(rows: list[dict], client: KalshiClient, collection: str, lineup_slots: dict[str, int],
              settings: dict, progress=lambda done, total: None) -> list[dict]:
    mode = settings["auto_bet_mode"]
    if mode not in ("dry_run", "live"):
        return []
    dry = mode == "dry_run"
    source = "paper" if dry else "auto"

    bankroll = await trading.balance(client)
    held = await trading.held_combos(client)
    held_by_game: dict[str, list[set[str]]] = defaultdict(list)
    for h in held.values():
        held_by_game[h["game"]].append(h["legs"])
    if dry:  # open paper bets count like held positions so dry runs follow the same per-game limit
        for game, bets in db.bets_by_game("paper").items():
            for bet in bets:
                held_by_game[game].append(row_markets(bet["data"]))
    already = set(held) | db.bet_tickers(source)

    now = datetime.now(timezone.utc)
    picks = select([r for r in rows if eligible(r, lineup_slots, settings, already, now)],
                   held_by_game, settings["max_bets_per_game"])
    size_stakes(picks, {**settings, "bankroll": bankroll})
    budget = settings["daily_max_spend_pct"] * bankroll - db.today_auto_spend(source)

    attempts = []
    for i, r in enumerate(sorted(picks, key=lambda r: -r["ev_lo"])):
        progress(i, len(picks))
        stake = r["stake"]
        if stake < settings["min_auto_stake"]:
            # Qualifying bet but Kelly says < $1 (small bankroll): round up if the per-bet cap allows.
            if settings["min_auto_stake"] > settings["max_bet_pct"] * bankroll:
                r["auto_bet"] = {"status": "skipped", "reason": f"${settings['min_auto_stake']:.2f} minimum "
                                 f"exceeds the {settings['max_bet_pct']:.0%} per-bet cap"}
                continue
            stake = r["stake"] = settings["min_auto_stake"]
        if stake > budget:
            r["auto_bet"] = {"status": "skipped", "reason": "daily auto-bet spend limit reached"}
            continue
        a, b = r["_legs"]
        res = await trading.place_combo(client, collection, a, b, stake, r["q"], r["q_lo"],
                                        settings["min_pess_ev"], dry, config.KALSHI_ACCEPT_SIDE,
                                        settings["quote_wait_seconds"])
        r["auto_bet"] = {"status": res.status, "reason": res.reason, "ev_lo": res.ev_lo, "spent": res.spent}
        if res.status in ("placed", "paper"):
            data = {k: v for k, v in r.items() if k not in ("_legs", "raw_quotes")}
            db.add_bet(r.get("result_id"), data, round(res.spent, 2), 1 / res.cost, source=source,
                       combo_ticker=res.combo_ticker, game=r["game"], contracts=res.contracts)
            budget -= res.spent
        attempts.append(r)
    return attempts
