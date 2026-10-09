"""Download MLB batter-, pitcher- and team-game history from the MLB StatsAPI into data/*.parquet.

Per game: the box score (batters, starting pitchers, team runs) and the line score (runs by
inning -> first-inning / first-3 / first-5 results). Incremental: games already stored are
skipped, so re-running only fetches new results. Bump SCHEMA_VERSION when the stored columns
change and everything is refetched once.
"""

import asyncio
import json

import httpx
import pandas as pd

from app import config

API = "https://statsapi.mlb.com/api/v1"
GAME_TYPES = "R,F,D,L,W"  # regular season + all postseason rounds
BOX_FIELDS = ("teams,home,away,team,abbreviation,teamStats,batting,runs,players,person,id,fullName,"
              "battingOrder,stats,hits,homeRuns,rbi,totalBases,stolenBases,pitchers,pitching,strikeOuts,outs")
LINE_FIELDS = "innings,num,home,away,runs"
SCHEMA_VERSION = 2
FILES = {"teams": "mlb_teams.parquet", "batters": "mlb_batters.parquet", "pitchers": "mlb_pitchers.parquet"}
VERSION_FILE = config.DATA_DIR / "mlb_schema.json"
DEDUP_KEYS = {"teams": ["game_pk", "team"], "batters": ["game_pk", "player_id"], "pitchers": ["game_pk", "team"]}


async def final_games(http: httpx.AsyncClient, season: int) -> list[tuple[int, str, str]]:
    r = await http.get(f"{API}/schedule", params={
        "sportId": 1, "season": season, "gameType": GAME_TYPES,
        "fields": "dates,date,games,gamePk,gameType,status,codedGameState",
    })
    r.raise_for_status()
    out = []
    for d in r.json().get("dates", []):
        for g in d["games"]:
            if g["status"]["codedGameState"] == "F":
                out.append((g["gamePk"], d["date"], g["gameType"]))
    return out


def _inning_runs(line: dict, side: str, upto: int) -> int:
    return sum((inn.get(side) or {}).get("runs", 0) or 0 for inn in line.get("innings", []) if inn["num"] <= upto)


def parse_game(pk: int, date: str, gtype: str, box: dict, line: dict) -> tuple[list, list, list]:
    batters, teams, pitchers = [], [], []
    sides = box["teams"]
    season = int(date[:4])
    first_inning = _inning_runs(line, "home", 1) + _inning_runs(line, "away", 1)
    for side, opp in (("home", "away"), ("away", "home")):
        t = sides[side]
        abbr = t["team"]["abbreviation"]
        runs = t["teamStats"]["batting"]["runs"]
        opp_runs = sides[opp]["teamStats"]["batting"]["runs"]
        f5, f5_opp = _inning_runs(line, side, 5), _inning_runs(line, opp, 5)
        f3, f3_opp = _inning_runs(line, side, 3), _inning_runs(line, opp, 3)
        teams.append({"game_pk": pk, "date": date, "season": season, "game_type": gtype, "team": abbr,
                      "opp": sides[opp]["team"]["abbreviation"], "home": int(side == "home"),
                      "runs": runs, "opp_runs": opp_runs, "margin": runs - opp_runs, "won": int(runs > opp_runs),
                      "f5_runs": f5, "f5_opp_runs": f5_opp, "f3_runs": f3, "f3_opp_runs": f3_opp,
                      "first_inning_runs": first_inning})
        for p in t["players"].values():
            bo = p.get("battingOrder")
            if not bo or not bo.endswith("00"):  # starters only; subs have partial games
                continue
            b = p["stats"]["batting"]
            h, hr, rbi, r, tb, sb = (b.get(k, 0) for k in ("hits", "homeRuns", "rbi", "runs", "totalBases",
                                                          "stolenBases"))
            batters.append({"game_pk": pk, "date": date, "season": season, "team": abbr,
                            "player_id": p["person"]["id"], "name": p["person"]["fullName"],
                            "slot": int(bo) // 100, "hit": h, "hr": hr, "rbi": rbi, "runs": r, "tb": tb,
                            "hrr": h + r + rbi, "sb": sb})
        if t.get("pitchers"):
            sp = t["players"].get(f"ID{t['pitchers'][0]}")  # first pitcher listed = starter
            if sp:
                pit = sp["stats"].get("pitching", {})
                pitchers.append({"game_pk": pk, "date": date, "season": season, "team": abbr,
                                 "player_id": sp["person"]["id"], "name": sp["person"]["fullName"],
                                 "ks": pit.get("strikeOuts", 0), "outs": pit.get("outs", 0)})
    return batters, teams, pitchers


async def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    paths = {k: config.DATA_DIR / v for k, v in FILES.items()}
    version = json.loads(VERSION_FILE.read_text()).get("version") if VERSION_FILE.exists() else 1
    fresh = version != SCHEMA_VERSION or not all(p.exists() for p in paths.values())
    old = {k: (pd.DataFrame() if fresh else pd.read_parquet(p)) for k, p in paths.items()}
    have = set() if fresh else set(old["teams"]["game_pk"])
    if fresh:
        print(f"schema v{version} -> v{SCHEMA_VERSION}: refetching all games")

    async with httpx.AsyncClient(timeout=30) as http:
        games = []
        for season in config.MLB_SEASONS:
            games += await final_games(http, season)
        # Suspended-then-resumed games are listed on both dates with the same gamePk; keep the last.
        games = list({g[0]: g for g in games}.values())
        todo = [g for g in games if g[0] not in have]
        print(f"{len(games):,} final games, {len(todo):,} to fetch")

        sem = asyncio.Semaphore(16)
        new = {"batters": [], "teams": [], "pitchers": []}
        failed = 0

        async def fetch(pk, date, gtype):
            nonlocal failed
            async with sem:
                for attempt in range(3):
                    try:
                        box = await http.get(f"{API}/game/{pk}/boxscore", params={"fields": BOX_FIELDS})
                        line = await http.get(f"{API}/game/{pk}/linescore", params={"fields": LINE_FIELDS})
                        box.raise_for_status()
                        line.raise_for_status()
                        b, t, p = parse_game(pk, date, gtype, box.json(), line.json())
                        new["batters"] += b
                        new["teams"] += t
                        new["pitchers"] += p
                        return
                    except (httpx.HTTPError, KeyError):
                        await asyncio.sleep(1 + attempt)
                failed += 1

        tasks = [fetch(*g) for g in todo]
        for i in range(0, len(tasks), 400):
            await asyncio.gather(*tasks[i:i + 400])
            print(f"  {min(i + 400, len(tasks)):,}/{len(tasks):,}", flush=True)

    for k, p in paths.items():
        df = pd.concat([old[k], pd.DataFrame(new[k])], ignore_index=True) if new[k] else old[k]
        df = df.drop_duplicates(DEDUP_KEYS[k], keep="last")
        df.to_parquet(p, index=False)
        print(f"{FILES[k]}: {len(df):,} rows")
    VERSION_FILE.write_text(json.dumps({"version": SCHEMA_VERSION}))
    print(f"failed: {failed}")


if __name__ == "__main__":
    asyncio.run(main())
