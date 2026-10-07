"""Find upcoming MLB/NFL games on Kalshi and every single-leg market we can combine.

Run `make discover` to print what is live right now.
"""

import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from app import config
from app.kalshi.client import KalshiClient
from app.legs import SERIES, Leg, game_key, game_start, game_teams_blob, parse_market

GAME_SERIES = {"MLB": "KXMLBGAME", "NFL": "KXNFLGAME"}
HORIZON = {"MLB": timedelta(days=2), "NFL": timedelta(days=8)}


@dataclass
class Game:
    sport: str
    key: str
    teams: list[str]
    start: datetime | None
    legs: list[Leg] = field(default_factory=list)

    @property
    def label(self) -> str:
        return " @ ".join(self.teams) if len(self.teams) == 2 else game_teams_blob(self.key)


def _ordered_teams(key: str, teams: set[str]) -> list[str]:
    """Kalshi game keys are away+home, e.g. TBDAL -> [TB, DAL]."""
    blob = game_teams_blob(key)
    return sorted(teams, key=lambda t: blob.find(t) if blob.find(t) >= 0 else 99)


async def find_games(client: KalshiClient, sport: str, now: datetime) -> dict[str, Game]:
    markets = await client.paginate(
        "/markets", "markets", {"series_ticker": GAME_SERIES[sport], "status": "open", "limit": 1000}
    )
    by_game: dict[str, list[dict]] = defaultdict(list)
    for m in markets:
        by_game[game_key(m["event_ticker"])].append(m)
    games = {}
    for key, ms in by_game.items():
        start = game_start(key, ms[0].get("occurrence_datetime"))
        if start is None or not (now < start < now + HORIZON[sport]):
            continue
        teams = {m["ticker"][len(m["event_ticker"]) + 1:] for m in ms}
        games[key] = Game(sport, key, _ordered_teams(key, teams), start)
    return games


async def discover(client: KalshiClient, sports=("MLB", "NFL"), now: datetime | None = None) -> list[Game]:
    now = now or datetime.now(timezone.utc)
    games: dict[str, Game] = {}
    for sport in sports:
        games.update(await find_games(client, sport, now))
    series_list = [s for s, (sp, _) in SERIES.items() if sp in sports]
    results = await asyncio.gather(*[
        client.paginate("/markets", "markets", {"series_ticker": s, "status": "open", "limit": 1000})
        for s in series_list
    ])
    for markets in results:
        for m in markets:
            g = games.get(game_key(m["event_ticker"]))
            if g is None:
                continue
            leg = parse_market(m, g.teams)
            if leg is not None:
                g.legs.append(leg)
    return sorted(games.values(), key=lambda g: g.start)


async def find_combo_collection(client: KalshiClient) -> tuple[str, set[str]] | None:
    """Return (collection_ticker, combinable event tickers) for the sports combo collection."""
    cols = await client.paginate(
        "/multivariate_event_collections", "multivariate_contracts",
        {"status": "open", "series_ticker": config.COMBO_COLLECTION_SERIES, "limit": 200},
    )
    if not cols:
        return None
    col = cols[0]
    events = set(col.get("associated_event_tickers") or [e["ticker"] for e in col.get("associated_events", [])])
    return col["collection_ticker"], events


async def _main():
    client = KalshiClient()
    try:
        games = await discover(client)
        col = await find_combo_collection(client)
        print(f"Combo collection: {col[0] if col else 'NOT FOUND'} ({len(col[1]) if col else 0} events)\n")
        for g in games:
            counts = Counter(leg.kind for leg in g.legs)
            combinable = sum(1 for leg in g.legs if col and leg.event_ticker in col[1])
            print(f"{g.sport} {g.label:12} {g.start:%a %m/%d %H:%MZ}  legs={len(g.legs)} combinable={combinable}  "
                  + " ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(_main())
