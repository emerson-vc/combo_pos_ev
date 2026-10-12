import glob, re, sys
import pandas as pd
HIT = re.compile(r"^(?:(S|D|T)(?=\d|/|\.|$|\?)|DGR|(HR|H)(?=\d|/|\.|$))")
def hit_bases(ev):
    m = HIT.match(ev)
    if not m: return 0
    s = m.group(0)
    return {"S": 1, "D": 2, "DGR": 2, "T": 3, "HR": 4, "H": 4}[s]
bat_rows, pit_rows, games = [], [], []
for season in range(2021, 2026):
    for fn in sorted(glob.glob(f"rs/seasons/{season}/{season}*.EV[AN]")):
        gid = None
        def flush():
            if gid is None: return
            games.append({"game": gid, "season": season, "away": info["visteam"], "home": info["hometeam"], "runs0": runs[0], "runs1": runs[1], "gametype": info.get("gametype", "regular")})
            for pid, b in bats.items(): bat_rows.append({"game": gid, "season": season, "pid": pid, **b})
            for side in (0, 1):
                sp = starters[side]
                if sp: pit_rows.append({"game": gid, "season": season, "pid": sp, "side": side, "k": ks[side][sp], "name": names[sp]})
        for line in open(fn, encoding="latin-1"):
            parts = line.rstrip("\n").split(",")
            tag = parts[0]
            if tag == "id":
                flush()
                gid, info, runs, bats, names = parts[1], {}, [0, 0], {}, {}
                starters, cur_p, ks = [None, None], [None, None], [dict(), dict()]
            elif tag == "info":
                info[parts[1]] = parts[2]
            elif tag in ("start", "sub"):
                pid, name, side, slot, pos = parts[1], parts[2].strip('"'), int(parts[3]), int(parts[4]), int(parts[5])
                names[pid] = name
                if tag == "start" and slot > 0:
                    bats[pid] = {"side": side, "slot": slot, "name": name, "pa": 0, "hr": 0, "hit": 0, "tb": 0, "rbi": 0}
                if pos == 1:
                    cur_p[side] = pid
                    if tag == "start": starters[side] = pid
                    ks[side].setdefault(pid, 0)
            elif tag == "play":
                side, pid, ev = int(parts[2]), parts[3], parts[6]
                if ev == "NP": continue
                main, _, adv = ev.partition(".")
                head = main.split("/")[0]
                scored = [a for a in adv.split(";") if re.match(r"^[B123]-H", a)]
                hb = hit_bases(head)
                n_runs = len(scored) + (1 if hb == 4 and not any(a.startswith("B") for a in scored) else 0)
                runs[side] += n_runs
                if head.startswith("K") and cur_p[1 - side]:
                    ks[1 - side][cur_p[1 - side]] = ks[1 - side].get(cur_p[1 - side], 0) + 1
                b = bats.get(pid)
                if b is not None and b["side"] == side:
                    is_pa = not re.match(r"^(SB|CS|PO|POCS|WP|PB|BK|DI|OA|FLE|C/)", head)
                    if is_pa:
                        b["pa"] += 1
                        if hb:
                            b["hit"] += 1; b["tb"] += hb; b["hr"] += hb == 4
                        no_rbi = head.startswith("E") or "DP" in main
                        if not no_rbi:
                            b["rbi"] += sum(1 for a in scored if "NR" not in a and "NORBI" not in a) + (1 if hb == 4 and not any(a.startswith("B") for a in scored) else 0)
        flush()
pd.DataFrame(bat_rows).to_parquet("mlb_bat.parquet"); pd.DataFrame(pit_rows).to_parquet("mlb_pit.parquet"); pd.DataFrame(games).to_parquet("mlb_games.parquet")
print(len(games), len(bat_rows), len(pit_rows))
