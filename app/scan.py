"""One manual scan: discover games -> generate correlated pairs -> model -> quote -> EV/Kelly."""

import asyncio
import traceback
from collections import Counter

import httpx

from app import autobet, config, db
from app.kalshi.client import KalshiClient
from app.kalshi.combos import legs_key, quote_combo
from app.kalshi.discovery import discover, find_combo_collection
from app.legs import ET, Leg
from app.model.joint import Model, norm_name
from app.pricing import all_in_cost, ev_pct, kelly_fraction, size_stakes
from app.quote_policy import MIN_FIT, QuotePolicy
from app.pairs import candidate_pairs, template_name
from app.templates import PINNED_TEMPLATE

PROGRESS: dict[int, dict] = {}
# Per game: max candidates sharing one long-shot leg, so quotes aren't spent on near-duplicates
# (e.g. Lamb 3+ TD against six different team totals).
PER_LEG_CAP = 3
EXPLORE_SHARE = 0.2  # of each game's quote slots, reserved for learning about less-quoted combo types
MLB_LONGSHOT_SHARE = 0.4  # of each MLB game's quote slots, reserved for combos with a long-shot leg
LONGSHOT_P = 0.05
QUOTE_CONCURRENCY = 5


def _progress(scan_id: int, phase: str, done: int = 0, total: int = 0, **extra):
    PROGRESS[scan_id] = {"phase": phase, "done": done, "total": total, **extra}


async def fetch_mlb_lineups(dates: set[str]) -> dict[str, int]:
    """Today's posted MLB lineups: normalized player name -> batting slot (1-9)."""
    slots: dict[str, int] = {}
    async with httpx.AsyncClient(timeout=20) as http:
        for d in dates:
            try:
                r = await http.get("https://statsapi.mlb.com/api/v1/schedule",
                                   params={"sportId": 1, "date": d, "hydrate": "lineups"})
                r.raise_for_status()
            except httpx.HTTPError:
                continue
            for day in r.json().get("dates", []):
                for g in day["games"]:
                    for side in ("homePlayers", "awayPlayers"):
                        for i, p in enumerate(g.get("lineups", {}).get(side, [])):
                            slots[norm_name(p["fullName"])] = i + 1
    return slots


def _leg_view(leg: Leg, role: str) -> dict:
    return {**leg.to_dict(), "role": role}


def _long_leg(r: dict) -> str:
    return min(r["leg_a"], r["leg_b"], key=lambda leg: leg["p"])["ticker"]


def _round_robin(game_rows: list[dict], limit: int, per_leg: Counter) -> list[dict]:
    """Highest-lift rows, taken round-robin across templates, skipping repeats of a long-shot leg."""
    by_template: dict[str, list[dict]] = {}
    for r in sorted(game_rows, key=lambda r: r["lift"], reverse=True):
        by_template.setdefault(r["template"], []).append(r)
    kept = []
    queues = list(by_template.values())
    while queues and len(kept) < limit:
        for q in list(queues):
            while q:
                r = q.pop(0)
                if per_leg[_long_leg(r)] < PER_LEG_CAP:
                    per_leg[_long_leg(r)] += 1
                    kept.append(r)
                    break
            if not q:
                queues.remove(q)
            if len(kept) >= limit:
                break
    return kept


def _take(rows: list[dict], n: int, per_leg: Counter) -> list[dict]:
    kept = []
    for r in rows:
        if len(kept) >= n:
            break
        if per_leg[_long_leg(r)] < PER_LEG_CAP:
            per_leg[_long_leg(r)] += 1
            kept.append(r)
    return kept


def pick_diverse(game_rows: list[dict], limit: int, policy: QuotePolicy | None = None,
                 longshot_share: float = 0.0) -> list[dict]:
    """Choose which candidates get a Kalshi quote.

    1. `longshot_share` of slots go to combos with a long-shot leg (p <= LONGSHOT_P), the shape of
       the edges found by hand (2+ HR + team win).
    2. With a fitted policy, most remaining slots go to the highest predicted EV (from how makers
       priced each combo type before); EXPLORE_SHARE stays round-robin so rare types keep getting data.
    """
    per_leg = Counter()
    ready = policy is not None and policy.ready
    if ready:
        for r in game_rows:
            r["pred_ev"] = policy.predicted_ev(r)
    kept = []
    if longshot_share > 0:
        longshots = [r for r in game_rows if min(r["p_a"], r["p_b"]) <= LONGSHOT_P]
        longshots.sort(key=lambda r: (r.get("pred_ev") is None, -(r.get("pred_ev") or 0), -r["lift"]))
        kept += _take(longshots, round(limit * longshot_share), per_leg)
    taken = {id(r) for r in kept}
    rest = [r for r in game_rows if id(r) not in taken]
    if not ready:
        return kept + _round_robin(rest, limit - len(kept), per_leg)
    n_exploit = limit - len(kept) - max(1, round(limit * EXPLORE_SHARE))
    kept += _take(sorted((r for r in rest if r["pred_ev"] is not None), key=lambda r: -r["pred_ev"]),
                  max(n_exploit, 0), per_leg)
    # Explore: combo types without enough logged quotes to fit first, then anything left.
    for pool in ([r for r in game_rows if policy.count[r["template"]] < MIN_FIT], game_rows):
        taken = {id(r) for r in kept}
        kept += _round_robin([r for r in pool if id(r) not in taken], limit - len(kept), per_leg)
    return kept


def build_candidates(games, model: Model, settings: dict, combinable: set[str] | None,
                      policy: QuotePolicy | None = None) -> tuple[list[dict], int]:
    rows, n_pairs = [], 0
    rejected = db.rejected_combo_keys()
    for g in games:
        if not model.has(g.sport):
            continue
        pairs = candidate_pairs(model.ladder.apply(g.legs), g.teams, model, settings)
        pairs = [(t, a, b) for t, a, b in pairs if legs_key(a, b) not in rejected]
        n_pairs += len(pairs)
        game_rows = []
        for template, a, b in pairs:
            if combinable is not None and not (a.event_ticker in combinable and b.event_ticker in combinable):
                continue
            est = model.estimate(a, b, settings["model_leg_haircut"])
            if est is None:
                continue
            indep = a.p * b.p
            lift = est.q / indep
            if template == PINNED_TEMPLATE:
                max_fair_x = settings["pinned_max_fair_x"]
            else:
                max_fair_x = settings["mlb_max_fair_x"] if g.sport == "MLB" else settings["max_fair_x"]
            if lift < settings["min_lift"] or est.q < 1 / max_fair_x:
                continue
            game_rows.append({
                "sport": g.sport, "game": g.key, "game_label": g.label, "start": g.start.isoformat(),
                "template": template, "template_name": template_name(template),
                "leg_a": _leg_view(a, est.role_a), "leg_b": _leg_view(b, est.role_b),
                "p_a": a.p, "p_b": b.p, "indep_prob": indep, "indep_x": 1 / indep,
                "q": est.q, "q_lo": est.q_lo, "q_hi": est.q_hi, "fair_x": 1 / est.q,
                "rho": est.rho, "rho_lo": est.rho_lo, "rho_hi": est.rho_hi,
                "n_hist": est.n, "n11": est.n11, "method": est.method, "lift": lift,
                "wide_leg": max(a.spread, b.spread) > 0.06,
                "_legs": (a, b),  # the exact legs to quote (a NO-side leg shares its YES ticker)
            })
        if settings["always_quote_hr_ml"]:
            # Pinned rows are quoted every scan and don't use up the game's quote slots.
            rows += [r for r in game_rows if r["template"] == PINNED_TEMPLATE]
            game_rows = [r for r in game_rows if r["template"] != PINNED_TEMPLATE]
        if g.sport == "MLB":
            rows += pick_diverse(game_rows, settings["mlb_quotes_per_game"], policy, MLB_LONGSHOT_SHARE)
        else:
            rows += pick_diverse(game_rows, settings["max_quotes_per_game"], policy)
    return rows, n_pairs


def price_row(r: dict, yes_price: float | None):
    """Fill EV/Kelly fields from a (pre-fee) YES price; None leaves the row model-only."""
    r["kalshi_price"] = yes_price
    if yes_price is None or not (0 < yes_price < 1):
        r.update(kalshi_cost=None, kalshi_x=None, ev=None, ev_lo=None, kelly_full=None)
        return
    cost = all_in_cost(yes_price)
    r["kalshi_cost"] = cost
    r["kalshi_x"] = 1 / yes_price
    r["kalshi_x_after_fees"] = 1 / cost
    r["ev"] = ev_pct(r["q"], cost)
    r["ev_lo"] = ev_pct(r["q_lo"], cost)
    r["kelly_full"] = kelly_fraction(r["q"], cost)


async def run_scan(scan_id: int, model: Model):
    settings = db.get_settings()
    client = KalshiClient()
    live = client.has_auth
    try:
        _progress(scan_id, "Discovering games")
        games = await discover(client)
        combinable = None
        collection = None
        if live:
            col = await find_combo_collection(client)
            if col:
                collection, combinable = col

        mlb_dates = {g.start.astimezone(ET).date().isoformat() for g in games if g.sport == "MLB"}
        model.lineup_slots = await fetch_mlb_lineups(mlb_dates)

        _progress(scan_id, "Modeling pairs")
        policy = QuotePolicy.from_db() if live else None
        rows, n_pairs = await asyncio.to_thread(build_candidates, games, model, settings, combinable, policy)

        for r in rows:
            price_row(r, None)
        n_quoted = 0
        if live and collection:
            budget_left = config.WEEKLY_COMBO_CREATION_LIMIT - db.creations_this_week()
            sem = asyncio.Semaphore(QUOTE_CONCURRENCY)
            done = 0
            _progress(scan_id, "Requesting Kalshi quotes", 0, len(rows))
            async def quote(r):
                nonlocal done, n_quoted
                async with sem:
                    a, b = r["_legs"]
                    try:
                        cq = await quote_combo(client, collection, a, b, settings["quote_size_dollars"],
                                               settings["quote_wait_seconds"])
                        r.update(combo_ticker=cq.market_ticker, quote_source=cq.source, n_quotes=cq.n_quotes,
                                 quoted_at=cq.quoted_at, raw_quotes=cq.raw)
                        price_row(r, cq.yes_price)
                        n_quoted += cq.yes_price is not None
                    except Exception as e:  # one bad combo shouldn't kill the scan
                        r["quote_error"] = str(e)[:300]
                    done += 1
                    _progress(scan_id, "Requesting Kalshi quotes", done, len(rows))

            await asyncio.gather(*[quote(r) for r in rows[:max(budget_left, 0)]])

        priced = [r for r in rows if r.get("ev") is not None and r["ev"] > 0]
        size_stakes(priced, settings)
        for r in rows:
            r.setdefault("stake", 0.0)

        n_auto, auto_error = 0, None
        if live and collection and settings["auto_bet_mode"] in ("dry_run", "live"):
            _progress(scan_id, "Placing bets")
            try:
                attempts = await autobet.run(
                    rows, client, collection, model.lineup_slots, settings,
                    progress=lambda d, t: _progress(scan_id, "Placing bets", d, t))
                n_auto = sum(r["auto_bet"]["status"] in ("placed", "paper") for r in attempts)
            except Exception as e:  # never lose the scan's results over a betting error
                traceback.print_exc()
                auto_error = f"auto-bet: {type(e).__name__}: {e}"[:300]

        for r in rows:
            r.pop("_legs", None)
        db.save_results(scan_id, rows)
        db.finish_scan(scan_id, "done", error=auto_error, n_games=len(games), n_pairs=n_pairs,
                       n_candidates=len(rows), n_quoted=n_quoted)
        _progress(scan_id, "Done", len(rows), len(rows), auto_bets=n_auto)
    except Exception as e:
        traceback.print_exc()
        db.finish_scan(scan_id, "error", error=f"{type(e).__name__}: {e}")
        _progress(scan_id, "Error", error=str(e))
    finally:
        await client.close()

