"""Pre-game MLB expectations from Kalshi's own settled markets -> data/mlb_lines.parquet.

For every settled game: the last moneyline / total / team-total prices before first pitch,
turned into each team's implied runs and expected margin. The model then measures team
outcomes as residuals vs those expectations (like the NFL model does with closing lines),
so "over 5.5 runs" means "beat expectation" rather than mixing Coors games with duels.

Kalshi keeps ~2 months of settled markets in the live API (batch candlesticks, cheap); older
markets are archived under /historical (one request per market), so for those we only fetch
what the model needs: one moneyline side plus the 3.5/4.5/5.5 team-total rungs. Prices are
cached per ticker in data/mlb_pregame_prices.json, so reruns are incremental and resumable.

Run `make ingest-mlb-lines` after `make ingest-mlb` (it joins onto data/mlb_teams.parquet).
"""

import asyncio
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import logit, ndtri

from app import config
from app.kalshi.client import KalshiClient, KalshiError
from app.legs import ET, game_key, game_start, team_of

SERIES = ["KXMLBGAME", "KXMLBTOTAL", "KXMLBTEAMTOTAL"]
ARCHIVE_TT_RUNGS = {3.5, 4.5, 5.5}  # enough to locate each team's median runs
# Kalshi team codes that differ from MLB StatsAPI abbreviations.
TEAM_ALIAS = {"ARI": "AZ", "OAK": "ATH", "CHW": "CWS", "WAS": "WSH", "WSN": "WSH", "KCR": "KC", "SDP": "SD",
              "SFG": "SF", "TBR": "TB"}
CACHE = config.DATA_DIR / "mlb_pregame_prices.json"
NO_TRADES = -1.0  # cached marker: archived market had no candles in the window; don't refetch
# The archive only returns candles for hours with trades (no "latest before start" option), so
# look back far enough to find the last pre-game trade.
ARCHIVE_LOOKBACK = 48 * 3600


def _close(c: dict, side: str) -> float:
    d = c.get(side) or {}
    return float(d.get("close_dollars") or d.get("close") or 0)  # archive candles use "close"


def _candle_mid(c: dict) -> float | None:
    bid, ask = _close(c, "yes_bid"), _close(c, "yes_ask")
    if 0 < bid < ask < 1:
        return (bid + ask) / 2
    last = _close(c, "price")
    return last if last > 0 else None


def _last_before(candles: list[dict], start: int) -> float | None:
    before = [c for c in candles if c["end_period_ts"] <= start]
    return _candle_mid(before[-1]) if before else None


def median_line(points: list[tuple[float, float]]) -> float | None:
    """Threshold X where P(over X) = 0.5, from a logit-linear fit of the ladder."""
    pts = [(x, p) for x, p in points if 0.03 < p < 0.97]
    if len(pts) < 2:
        return None
    x, p = np.array(pts).T
    slope, icpt = np.polyfit(x, logit(p), 1)
    return float(-icpt / slope) if slope < 0 else None


async def fetch_markets(client: KalshiClient) -> list[dict]:
    out = []
    for s in SERIES:
        live = await client.paginate("/markets", "markets", {"series_ticker": s, "status": "settled", "limit": 1000})
        hist = await client.paginate("/historical/markets", "markets", {"series_ticker": s, "limit": 1000})
        for m in live:
            m["_archived"] = False
        for m in hist:
            m["_archived"] = True
        out += live + hist
        print(f"{s}: {len(live):,} live + {len(hist):,} archived settled markets")
    return out


def _candle_chunks(markets: list[dict], starts: dict[str, int], max_candles: int = 9000):
    """Batches of <=100 markets whose shared hourly window stays under Kalshi's candle cap."""
    chunk, lo = [], None
    for m in markets:
        st = starts[m["event_ticker"]]
        lo = st if lo is None else lo
        hours = (st - lo) // 3600 + 4
        if chunk and (len(chunk) >= 100 or (len(chunk) + 1) * hours > max_candles):
            yield chunk
            chunk, lo = [], st
        chunk.append(m)
    if chunk:
        yield chunk


def wanted_archived(m: dict, picked_ml: set[str]) -> bool:
    series = m["event_ticker"].split("-")[0]
    if series == "KXMLBGAME":
        return m["event_ticker"] not in picked_ml  # one side is enough: p_opp = 1 - p
    if series == "KXMLBTEAMTOTAL":
        return m.get("floor_strike") in ARCHIVE_TT_RUNGS
    return False


async def fetch_prices(client: KalshiClient, markets: list[dict], starts: dict[str, int], cache: dict) -> None:
    todo_live = sorted((m for m in markets if not m["_archived"] and m["ticker"] not in cache),
                       key=lambda m: starts[m["event_ticker"]])
    for chunk in _candle_chunks(todo_live, starts):
        ts = [starts[m["event_ticker"]] for m in chunk]
        data = await client.get("/markets/candlesticks", {
            "market_tickers": ",".join(m["ticker"] for m in chunk), "start_ts": min(ts) - 3 * 3600,
            "end_ts": max(ts), "period_interval": 60, "include_latest_before_start": "true"})
        got = {e.get("market_ticker") or e.get("ticker"): e.get("candlesticks", []) for e in data.get("markets", [])}
        for m in chunk:
            cache[m["ticker"]] = _last_before(got.get(m["ticker"], []), starts[m["event_ticker"]])
    print(f"live prices: {len(todo_live):,} fetched")

    picked_ml: set[str] = set()
    todo_hist = []
    for m in sorted((m for m in markets if m["_archived"]), key=lambda m: m["ticker"]):
        if wanted_archived(m, picked_ml):
            if m["event_ticker"].startswith("KXMLBGAME"):
                picked_ml.add(m["event_ticker"])
            if cache.get(m["ticker"]) is None:  # missing, or blank from an older narrower window
                todo_hist.append(m)
    print(f"archived prices to fetch: {len(todo_hist):,} (one request each; resumable)")
    for i, m in enumerate(todo_hist):
        st = starts[m["event_ticker"]]
        try:
            data = await client.get(f"/historical/markets/{m['ticker']}/candlesticks",
                                    {"start_ts": st - ARCHIVE_LOOKBACK, "end_ts": st, "period_interval": 60})
            p = _last_before(data.get("candlesticks", []), st)
            cache[m["ticker"]] = NO_TRADES if p is None else p
        except KalshiError:
            cache[m["ticker"]] = None
        if i % 500 == 499:
            CACHE.write_text(json.dumps(cache))
            print(f"  archived {i + 1:,}/{len(todo_hist):,}")
    CACHE.write_text(json.dumps(cache))


def build_lines(markets: list[dict], starts: dict[str, int], teams_by_game: dict, cache: dict) -> pd.DataFrame:
    games = defaultdict(lambda: {"win": {}, "total": [], "tt": defaultdict(list)})
    for m in markets:
        p = cache.get(m["ticker"])
        if p is None or p < 0:
            continue
        series, key = m["event_ticker"].split("-")[0], game_key(m["event_ticker"])
        suffix = m["ticker"][len(m["event_ticker"]) + 1:]
        g = games[key]
        if series == "KXMLBGAME":
            g["win"][suffix] = p
        elif m.get("floor_strike") is not None:
            if series == "KXMLBTOTAL":
                g["total"].append((float(m["floor_strike"]), p))
            else:
                team = team_of(suffix, sorted(teams_by_game[key]))
                if team:
                    g["tt"][team].append((float(m["floor_strike"]), p))

    rows = []
    for key, g in games.items():
        teams = sorted(teams_by_game[key])
        if len(teams) != 2 or not g["win"]:
            continue
        a, b = teams
        if a in g["win"] and b in g["win"]:
            pa = g["win"][a] / (g["win"][a] + g["win"][b])
        else:
            pa = g["win"][a] if a in g["win"] else 1 - g["win"][b]
        date = pd.Timestamp(starts[f"KXMLBGAME-{key}"], unit="s", tz="UTC").tz_convert(ET).date().isoformat()
        total_med = median_line(g["total"])
        for team, opp, pw in ((a, b, pa), (b, a, 1 - pa)):
            rows.append({"date": date, "team": TEAM_ALIAS.get(team, team), "opp": TEAM_ALIAS.get(opp, opp),
                         "p_win": pw, "team_med": median_line(g["tt"].get(team, [])), "total_med": total_med})
    return pd.DataFrame(rows)


async def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    client = KalshiClient(key_id=None)  # public market data only
    try:
        markets = await fetch_markets(client)
        starts, teams_by_game = {}, defaultdict(set)
        for m in markets:
            if m["event_ticker"].startswith("KXMLBGAME-"):
                key = game_key(m["event_ticker"])
                start = game_start(key, m.get("occurrence_datetime") or m.get("expected_expiration_time"))
                if start is None:
                    continue
                st = int(start.timestamp())
                teams_by_game[key].add(m["ticker"][len(m["event_ticker"]) + 1:])
                for s in SERIES:
                    starts[f"{s}-{key}"] = st
        markets = [m for m in markets if m["event_ticker"] in starts]
        await fetch_prices(client, markets, starts, cache)
    finally:
        CACHE.write_text(json.dumps(cache))
        await client.close()

    lines = build_lines(markets, starts, teams_by_game, cache)
    teams = pd.read_parquet(config.DATA_DIR / "mlb_teams.parquet")
    # Doubleheaders can't be matched by date+team alone; drop them rather than guess.
    lines = lines[~lines.duplicated(["date", "team"], keep=False)]
    single = teams[~teams.duplicated(["date", "team"], keep=False)]
    j = single.merge(lines, on=["date", "team", "opp"], how="inner")

    z = ndtri(j["p_win"].clip(0.02, 0.98))
    s = float((z * j["margin"]).sum() / (z * z).sum())  # OLS through origin: margin ≈ s·Φ⁻¹(p_win)
    j["exp_margin"] = s * z
    j["implied_runs"] = j["team_med"].fillna(j["total_med"] / 2 + j["exp_margin"] / 2)
    out = j[["game_pk", "team", "p_win", "exp_margin", "implied_runs"]].drop_duplicates(["game_pk", "team"])
    out.to_parquet(config.DATA_DIR / "mlb_lines.parquet", index=False)

    print(f"\nmatched {len(out):,} team-games ({out['implied_runs'].notna().sum():,} with implied runs); "
          f"margin scale s = {s:.2f} runs per σ")
    print("by season:", j.groupby("season").size().to_dict())
    unmatched = sorted(set(lines["team"]) - set(teams["team"]))
    if unmatched:
        print("Kalshi team codes not in StatsAPI (add to TEAM_ALIAS):", unmatched)
    resid = (j["runs"] - j["implied_runs"]).dropna()
    print(f"runs residual mean {resid.mean():+.2f}  sd {resid.std():.2f}  (raw runs sd {j['runs'].std():.2f})")


if __name__ == "__main__":
    asyncio.run(main())
