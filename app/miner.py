"""Edge miner: quote combos in bulk, learn how makers price each combo type, rank the best bets.

The scanner quotes ~20 hand-ranked pairs per game. The miner instead treats finding edges as a search:

1. Generate every positively correlated 2-leg combo in each game (all rungs, YES and NO sides), plus 3-leg
   combos that extend the best pairs with a leg correlated to both.
2. Price each with the joint model (q, and pessimistic q_lo).
3. Predict what makers will pay for it from the quote ledger: makers quote roughly
   (product of leg payouts) / d, with a discount d that depends on the combo type. d is learned per type with
   shrinkage (type -> kinds -> sport and leg count -> everything), starting from d = 1.
4. Quote in rounds. Each round samples d from its uncertainty (Thompson sampling), so unexplored combo types
   get tried and soft ones get quoted more. Every quote goes to the ledger and the model refits.
5. Re-quote the best few once more, then print the top bets and a combo-type leaderboard.

It never accepts a quote. Run `make mine`, `make mine-report`, `make grade`.
"""

import argparse
import asyncio
import csv
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations

import numpy as np

from app import config, db
from app.kalshi.client import KalshiClient
from app.kalshi.combos import ComboRejected, legs_key, quote_legs
from app.kalshi.discovery import ComboRules, combo_rules, discover
from app.legs import GAME_KINDS, PLAYER_KINDS, Leg, market_key
from app.model.joint import Model
from app.pricing import all_in_cost
from app.templates import usable

PRIOR_WEIGHT = 5.0  # pseudo-quotes pulling a type's mean log d toward its parent level
DEFAULT_SIGMA = 0.20  # spread of log d within a type before the ledger says otherwise
NEGATABLE = {"team_total", "game_total", "f5_total", "h1_total", "h1_team_total", "q1_total"}  # unders
MIN_LIFT = 1.05  # the edge needs positive correlation: skip pairs the model sees as ~independent
MIN_ATLAS = 0.05
PAIRS_TO_EXTEND = 40  # per game: best 2-leg combos tried with a third leg
MAX_CANDIDATES_PER_GAME = 1500
LEG_CAP = 4  # per run: quotes sharing one long-shot leg
TYPE_CAP_PER_ROUND = 4
RESERVE_CREATIONS = 500  # of the weekly 5,000, left for scans and auto-bets
QUOTE_CONCURRENCY = 5


def leg_price(leg: Leg) -> float:
    """What the leg costs alone: its ask, or the model price for thin rungs."""
    return leg.yes_ask if 0 < leg.yes_ask < 1 else leg.p


def _tier(p: float) -> str:
    return "L" if p < 0.12 else "M" if p < 0.4 else "H"


def _token(leg: Leg, tier: bool) -> str:
    t = leg.kind + ("~no" if leg.side == "no" else "")
    return f"{t}:{_tier(leg.p)}" if tier else t


def _cross_team(legs) -> bool:
    teams = {leg.team for leg in legs if leg.team and leg.kind not in GAME_KINDS}
    return len(teams) > 1


def combo_type(legs, tier: bool = True) -> str:
    """Stable, readable combo type: sport, sorted leg kinds (NO side marked, probability tier), and
    whether the legs span both teams."""
    tokens = "+".join(sorted(_token(leg, tier) for leg in legs))
    return f"{legs[0].sport}:{tokens}{'|x' if _cross_team(legs) else ''}"


def type_levels(legs) -> list[str]:
    """Most to least specific grouping used to share what the ledger knows."""
    return [combo_type(legs), combo_type(legs, tier=False), f"{legs[0].sport}:{len(legs)}", "all"]


# ---- what makers pay ------------------------------------------------------------------------------

class MakerModel:
    """log d per combo type, shrunk toward coarser groupings (empirical Bayes)."""

    def __init__(self, rows: list[dict]):
        self.stats: dict[str, list[float]] = defaultdict(list)
        resid = []
        for r in rows:
            if not r.get("d") or r["d"] <= 0:
                continue
            for lvl in r["levels"]:
                self.stats[lvl].append(math.log(r["d"]))
        for vals in self.stats.values():
            if len(vals) >= 3:
                m = float(np.mean(vals))
                resid += [v - m for v in vals]
        self.sigma = float(np.std(resid)) if len(resid) >= 20 else DEFAULT_SIGMA
        self.sigma = max(self.sigma, 0.03)

    def posterior(self, levels: list[str]) -> tuple[float, float, int]:
        """(mean log d, sd of that mean, quotes at the most specific level)."""
        mean, weight = 0.0, PRIOR_WEIGHT  # prior: makers pay the legs multiplied (d = 1)
        for lvl in reversed(levels):
            vals = self.stats.get(lvl, [])
            mean = (sum(vals) + PRIOR_WEIGHT * mean) / (len(vals) + PRIOR_WEIGHT)
            weight = len(vals) + PRIOR_WEIGHT
        return mean, self.sigma / math.sqrt(weight), len(self.stats.get(levels[0], []))


def ev_at(q: float, payout_x: float) -> float:
    """EV per $1 at a pre-fee payout, after Kalshi's fee."""
    price = 1 / payout_x
    if not 0 < price < 1:
        return -1.0
    return q / all_in_cost(price) - 1


# ---- candidates -----------------------------------------------------------------------------------

@dataclass
class Candidate:
    sport: str
    game: str
    game_label: str
    start: str
    legs: tuple[Leg, ...]
    q: float
    q_lo: float
    lift: float
    key: str = ""
    levels: list[str] = field(default_factory=list)
    legs_x: float = 0.0
    quotes: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.key = legs_key(*self.legs)
        self.levels = type_levels(self.legs)
        self.legs_x = float(np.prod([1 / leg_price(leg) for leg in self.legs]))

    @property
    def type(self) -> str:
        return self.levels[0]

    def long_leg(self) -> str:
        return min(self.legs, key=lambda leg: leg.p).ticker


def game_legs(game, model: Model, rules: ComboRules | None, settings: dict) -> list[Leg]:
    legs = [leg for leg in model.ladder.apply(game.legs) if usable(leg, settings)]
    legs += [u for u in (leg.negate() for leg in legs if leg.kind in NEGATABLE and leg.side == "yes")
             if usable(u, settings)]
    if rules is not None:
        legs = [leg for leg in legs if rules.allows([leg])]
    return legs


def _compatible(legs, rejected: set[str], rules: ComboRules | None) -> bool:
    per_event = Counter(leg.event_ticker for leg in legs)
    for ev, n in per_event.items():
        cap = rules.event_size_max.get(ev, 1) if rules is not None else 1
        if n > cap:
            return False  # one line per event per combo, unless the collection says more are allowed
    if len({market_key(leg.event_ticker, leg.ticker) for leg in legs}) < len(legs):
        return False
    if any(legs_key(*pair) in rejected for pair in combinations(legs, 2)):
        return False  # Kalshi refused a pair inside it
    return rules is None or rules.allows(legs)


def build_candidates(games, model: Model, settings: dict, rules: ComboRules | None, rejected: set[str],
                     max_legs: int = 3, min_x: float = 2.0, max_x: float = 300.0) -> list[Candidate]:
    out = []
    haircut = settings["model_leg_haircut"]
    for g in games:
        if not model.has(g.sport):
            continue
        legs = game_legs(g, model, rules, settings)
        start = g.start.isoformat() if g.start else ""
        pairs: list[Candidate] = []
        for a, b in combinations(legs, 2):
            if not _compatible((a, b), rejected, rules) or legs_key(a, b) in rejected:
                continue
            rho = model.atlas_rho(a, b)
            if rho is None or rho < MIN_ATLAS:
                continue
            est = model.estimate(a, b, haircut)
            if est is None or est.q <= 0:
                continue
            lift = est.q / (a.p * b.p)
            if lift < MIN_LIFT or not (min_x <= 1 / est.q <= max_x):
                continue
            pairs.append(Candidate(g.sport, g.key, g.label, start, (a, b), est.q, est.q_lo, lift))
        game_cands = list(pairs)
        if max_legs >= 3 and pairs:
            seen = {c.key for c in pairs}
            for base in sorted(pairs, key=lambda c: -c.lift * c.q)[:PAIRS_TO_EXTEND]:
                for c3 in legs:
                    trio = (*base.legs, c3)
                    if c3 in base.legs or not _compatible(trio, rejected, rules):
                        continue
                    if any((model.atlas_rho(x, c3) or -1) < MIN_ATLAS for x in base.legs):
                        continue
                    key = legs_key(*trio)
                    if key in seen or key in rejected:
                        continue
                    seen.add(key)
                    est = model.estimate_n(list(trio), haircut)
                    if est is None or est.q <= 0:
                        continue
                    lift = est.q / float(np.prod([leg.p for leg in trio]))
                    if lift < MIN_LIFT or not (min_x <= 1 / est.q <= max_x):
                        continue
                    game_cands.append(Candidate(g.sport, g.key, g.label, start, trio, est.q, est.q_lo, lift))
        game_cands.sort(key=lambda c: -c.lift)
        out += game_cands[:MAX_CANDIDATES_PER_GAME]
    return out


# ---- choosing what to quote -----------------------------------------------------------------------

def predicted(c: Candidate, maker: MakerModel, rng: random.Random | None = None) -> float:
    """EV at the payout makers are predicted to quote; with rng, d is sampled (Thompson sampling)."""
    mean, sd, _ = maker.posterior(c.levels)
    log_d = rng.gauss(mean, sd) if rng else mean
    return ev_at(c.q, c.legs_x / math.exp(log_d))


def pick_batch(cands: list[Candidate], maker: MakerModel, n: int, quoted: set[str], leg_use: Counter,
               rng: random.Random, min_score: float = -0.10) -> list[Candidate]:
    scored = sorted(((predicted(c, maker, rng), c) for c in cands if c.key not in quoted), key=lambda t: -t[0])
    batch, per_type = [], Counter()
    for score, c in scored:
        if len(batch) >= n or score < min_score:
            break
        if per_type[c.type] >= TYPE_CAP_PER_ROUND or leg_use[c.long_leg()] >= LEG_CAP:
            continue
        batch.append(c)
        per_type[c.type] += 1
        leg_use[c.long_leg()] += 1
    return batch


# ---- quoting + ledger -----------------------------------------------------------------------------

def _leg_row(leg: Leg) -> dict:
    return {"ticker": leg.ticker, "event_ticker": leg.event_ticker, "side": leg.side, "kind": leg.kind,
            "team": leg.team, "player": leg.player, "threshold": leg.threshold, "label": leg.label,
            "p": leg.p, "bid": leg.yes_bid, "ask": leg.yes_ask, "price": leg_price(leg)}


def ledger_row(c: Candidate, run_id: str, ticker: str | None, yes_price: float | None, n_quotes: int) -> dict:
    combo_x = 1 / yes_price if yes_price and 0 < yes_price < 1 else None
    return {
        "quoted_at": db.now_iso(), "run_id": run_id, "sport": c.sport, "game": c.game, "start": c.start,
        "type": c.type, "levels": c.levels, "n_legs": len(c.legs), "legs": [_leg_row(leg) for leg in c.legs],
        "combo_ticker": ticker, "legs_x": c.legs_x, "combo_x": combo_x,
        "d": c.legs_x / combo_x if combo_x else None, "q": c.q, "q_lo": c.q_lo,
        "ev": ev_at(c.q, combo_x) if combo_x else None, "ev_lo": ev_at(c.q_lo, combo_x) if combo_x else None,
        "n_quotes": n_quotes,
    }


def ledger_history(days: int = 21) -> list[dict]:
    rows = db.ledger_rows(days)
    for r in rows:
        legs = [Leg(r["sport"], r["game"], "", leg["event_ticker"], leg["ticker"], leg["kind"], leg["team"],
                    leg["player"], leg["threshold"], leg["bid"], leg["ask"], leg["label"],
                    p_model=leg["p"], side=leg["side"]) for leg in r["legs"]]
        r["levels"] = type_levels(legs)
    return rows


async def quote_one(client, collection: str, c: Candidate, run_id: str, settings: dict, wait: float) -> dict:
    try:
        cq = await quote_legs(client, collection, c.legs, settings["quote_size_dollars"], wait)
        row = ledger_row(c, run_id, cq.market_ticker, cq.yes_price, cq.n_quotes)
    except ComboRejected:
        row = ledger_row(c, run_id, None, None, 0)
        row["rejected"] = True
    except Exception as e:  # one bad combo shouldn't end the run
        row = ledger_row(c, run_id, None, None, 0)
        row["error"] = str(e)[:200]
    if not row.get("rejected") and not row.get("error"):
        db.add_ledger_row(row)
    c.quotes.append(row)
    return row


async def mine(client, games, model: Model, settings: dict, rules: ComboRules, n_quotes: int = 300,
               batch: int = 40, max_legs: int = 3, wait: float = 2.0, requote: int = 10, seed: int | None = None,
               log=print) -> tuple[list[Candidate], list[dict]]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rejected = db.rejected_combo_keys()
    cands = build_candidates(games, model, settings, rules, rejected, max_legs=max_legs)
    log(f"{len(cands)} candidate combos across {len(games)} games "
        f"({sum(len(c.legs) == 3 for c in cands)} with 3 legs)")
    history = ledger_history()
    budget = min(n_quotes, config.WEEKLY_COMBO_CREATION_LIMIT - RESERVE_CREATIONS - db.creations_this_week())
    rng = random.Random(seed)
    quoted, leg_use, rows = set(), Counter(), []
    sem = asyncio.Semaphore(QUOTE_CONCURRENCY)

    async def run(c):
        async with sem:
            return await quote_one(client, rules.collection, c, run_id, settings, wait)

    while len(quoted) < budget:
        maker = MakerModel(history + rows)
        picks = pick_batch(cands, maker, min(batch, budget - len(quoted)), quoted, leg_use, rng)
        if not picks:
            break
        quoted |= {c.key for c in picks}
        new = await asyncio.gather(*[run(c) for c in picks])
        rows += [r for r in new if r.get("d")]
        best = max((r["ev"] for r in new if r.get("ev") is not None), default=None)
        log(f"quoted {len(quoted)}/{budget}: {sum(1 for r in new if r.get('d'))} priced, "
            f"{sum(1 for r in new if r.get('rejected'))} rejected"
            + (f", best EV {best:+.0%}" if best is not None else ""))
    # Quotes drift for the same combo: re-quote the best few once (no new creations needed).
    top = sorted((c for c in cands if c.quotes and c.quotes[-1].get("ev_lo") is not None),
                 key=lambda c: -c.quotes[-1]["ev_lo"])[:requote]
    if top:
        await asyncio.gather(*[run(c) for c in top])
        log(f"re-quoted the top {len(top)}")
    return cands, rows


# ---- reports --------------------------------------------------------------------------------------

def top_bets(cands: list[Candidate], n: int = 25) -> list[dict]:
    """Best combos by pessimistic EV on their latest quote (worst of the last two when re-quoted)."""
    out = []
    for c in cands:
        priced = [r for r in c.quotes if r.get("ev_lo") is not None]
        if not priced:
            continue
        last = min(priced[-2:], key=lambda r: r["ev_lo"])
        if last["ev"] <= 0:
            continue
        out.append({"game": c.game_label, "start": c.start, "type": c.type,
                    "legs": " + ".join(leg.label for leg in c.legs), "kalshi_x": last["combo_x"],
                    "legs_x": c.legs_x, "fair_x": 1 / c.q, "d": last["d"], "lift": c.lift, "ev": last["ev"],
                    "ev_lo": last["ev_lo"], "combo_ticker": last["combo_ticker"],
                    "drift": (priced[-1]["combo_x"] / priced[-2]["combo_x"] - 1) if len(priced) > 1 else None})
    return sorted(out, key=lambda r: -r["ev_lo"])[:n]


def type_board(rows: list[dict], min_n: int = 3) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    out = []
    for t, rs in by.items():
        if len(rs) < min_n:
            continue
        graded = [r for r in rs if r.get("result") in ("yes", "no")]
        out.append({"type": t, "quotes": len(rs), "median_d": float(np.median([r["d"] for r in rs])),
                    "mean_ev": float(np.mean([r["ev"] for r in rs])),
                    "share_pos_ev_lo": float(np.mean([r["ev_lo"] > 0 for r in rs])),
                    "graded": len(graded),
                    "hit_rate": float(np.mean([r["result"] == "yes" for r in graded])) if graded else None,
                    "pred_rate": float(np.mean([r["q"] for r in graded])) if graded else None,
                    "roi": float(np.mean([(r["result"] == "yes") / all_in_cost(1 / r["combo_x"]) - 1
                                          for r in graded])) if graded else None})
    return sorted(out, key=lambda r: -r["mean_ev"])


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def print_reports(top: list[dict], board: list[dict], log=print):
    log("\nTOP BETS (pessimistic EV on the latest quote)")
    for r in top:
        drift = f"  drift {r['drift']:+.0%}" if r["drift"] is not None else ""
        log(f"  {r['ev_lo']:+6.0%} (EV {r['ev']:+.0%})  Kalshi {r['kalshi_x']:.1f}x  fair {r['fair_x']:.1f}x  "
            f"d {r['d']:.2f}  {r['game']}: {r['legs']}{drift}")
    log("\nCOMBO TYPES (ledger, last 21 days)")
    for r in board[:30]:
        graded = f"  hit {r['hit_rate']:.0%} vs model {r['pred_rate']:.0%}, ROI {r['roi']:+.0%} ({r['graded']})" \
            if r["graded"] else ""
        log(f"  EV {r['mean_ev']:+5.0%}  d {r['median_d']:.2f}  n={r['quotes']:3d}  {r['type']}{graded}")


async def grade(client) -> int:
    """Grade every settled combo in the ledger from its Kalshi market result."""
    n = 0
    for ticker in db.ungraded_ledger_tickers():
        try:
            m = (await client.get(f"/markets/{ticker}")).get("market", {})
        except Exception:
            continue
        if m.get("result") in ("yes", "no"):
            db.grade_ledger(ticker, m["result"])
            n += 1
    return n


async def _main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["mine", "report", "grade"])
    ap.add_argument("--quotes", type=int, default=300, help="max combos to quote this run")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--legs", type=int, default=3, help="max legs per combo (2 or 3)")
    ap.add_argument("--wait", type=float, default=2.0, help="seconds to wait for maker quotes")
    ap.add_argument("--sports", default="MLB,NFL")
    args = ap.parse_args()
    db.init()
    client = KalshiClient()
    try:
        if args.cmd == "grade":
            print(f"graded {await grade(client)} combos")
            return
        if args.cmd == "report":
            print_reports([], type_board(db.ledger_rows(21)))
            return
        if not client.has_auth:
            raise SystemExit("make mine needs a Kalshi API key (see .env.example)")
        print("Loading model...")
        model = Model()
        settings = db.get_settings()
        rules = await combo_rules(client)
        if rules is None:
            raise SystemExit("No open combo collection found")
        print(f"Combo collection {rules.collection}: {len(rules.events)} events, max legs {rules.size_max}")
        t0 = time.monotonic()
        games = await discover(client, tuple(args.sports.split(",")))
        from app.scan import fetch_mlb_lineups
        from app.legs import ET
        model.lineup_slots = await fetch_mlb_lineups(
            {g.start.astimezone(ET).date().isoformat() for g in games if g.sport == "MLB" and g.start})
        cands, _ = await mine(client, games, model, settings, rules, args.quotes, args.batch, args.legs, args.wait)
        top, board = top_bets(cands), type_board(db.ledger_rows(21))
        print_reports(top, board)
        _write_csv(config.DATA_DIR / "miner_top.csv", top)
        _write_csv(config.DATA_DIR / "miner_types.csv", board)
        print(f"\nSaved data/miner_top.csv and data/miner_types.csv ({time.monotonic() - t0:.0f}s)")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(_main())
