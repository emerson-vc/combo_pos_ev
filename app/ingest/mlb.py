"""Download MLB batter-game and team-game history from the MLB StatsAPI into data/*.parquet.

Incremental: games already in data/mlb_teams.parquet are skipped, so re-running only fetches
new results.
"""

import asyncio

import httpx
import pandas as pd

from app import config

API = "https://statsapi.mlb.com/api/v1"
GAME_TYPES = "R,F,D,L,W"  # regular season + all postseason rounds
BOX_FIELDS = ("teams,home,away,team,abbreviation,teamStats,batting,runs,players,person,id,fullName,"
              "battingOrder,stats,hits,homeRuns,rbi,totalBases")


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


def parse_box(pk: int, date: str, gtype: str, box: dict) -> tuple[list[dict], list[dict]]:
    batters, teams = [], []
    sides = box["teams"]
    for side, opp in (("home", "away"), ("away", "home")):
        t = sides[side]
        abbr = t["team"]["abbreviation"]
        runs = t["teamStats"]["batting"]["runs"]
        opp_runs = sides[opp]["teamStats"]["batting"]["runs"]
        teams.append({"game_pk": pk, "date": date, "season": int(date[:4]), "game_type": gtype, "team": abbr,
                      "opp": sides[opp]["team"]["abbreviation"], "runs": runs, "opp_runs": opp_runs,
                      "margin": runs - opp_runs, "won": int(runs > opp_runs)})
        for p in t["players"].values():
            bo = p.get("battingOrder")
            if not bo or not bo.endswith("00"):  # starters only; subs have partial games
                continue
            b = p["stats"]["batting"]
            h, hr, rbi, r, tb = (b.get(k, 0) for k in ("hits", "homeRuns", "rbi", "runs", "totalBases"))
            batters.append({"game_pk": pk, "date": date, "season": int(date[:4]), "team": abbr,
                            "player_id": p["person"]["id"], "name": p["person"]["fullName"],
                            "slot": int(bo) // 100, "hit": h, "hr": hr, "rbi": rbi, "runs": r, "tb": tb,
                            "hrr": h + r + rbi})
    return batters, teams


async def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    teams_path = config.DATA_DIR / "mlb_teams.parquet"
    bat_path = config.DATA_DIR / "mlb_batters.parquet"
    old_teams = pd.read_parquet(teams_path) if teams_path.exists() else pd.DataFrame(columns=["game_pk"])
    old_bat = pd.read_parquet(bat_path) if bat_path.exists() else pd.DataFrame()
    have = set(old_teams["game_pk"])

    async with httpx.AsyncClient(timeout=30) as http:
        games = []
        for season in config.MLB_SEASONS:
            games += await final_games(http, season)
        todo = [g for g in games if g[0] not in have]
        print(f"{len(games):,} final games, {len(todo):,} to fetch")

        sem = asyncio.Semaphore(16)
        batters, teams, failed = [], [], 0

        async def fetch(pk, date, gtype):
            nonlocal failed
            async with sem:
                for attempt in range(3):
                    try:
                        r = await http.get(f"{API}/game/{pk}/boxscore", params={"fields": BOX_FIELDS})
                        r.raise_for_status()
                        b, t = parse_box(pk, date, gtype, r.json())
                        batters.extend(b)
                        teams.extend(t)
                        return
                    except (httpx.HTTPError, KeyError):
                        await asyncio.sleep(1 + attempt)
                failed += 1

        tasks = [fetch(*g) for g in todo]
        for i in range(0, len(tasks), 400):
            await asyncio.gather(*tasks[i:i + 400])
            print(f"  {min(i + 400, len(tasks)):,}/{len(tasks):,}")

    new_teams = pd.concat([old_teams, pd.DataFrame(teams)], ignore_index=True) if teams else old_teams
    new_bat = pd.concat([old_bat, pd.DataFrame(batters)], ignore_index=True) if batters else old_bat
    new_teams.to_parquet(teams_path, index=False)
    new_bat.to_parquet(bat_path, index=False)
    print(f"mlb_teams: {len(new_teams):,} team-games   mlb_batters: {len(new_bat):,} batter-games   failed: {failed}")


if __name__ == "__main__":
    asyncio.run(main())
