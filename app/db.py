"""SQLite persistence: settings, scan history, results, combo-ticker cache, bet log."""

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, mode TEXT NOT NULL,
    n_games INTEGER DEFAULT 0, n_pairs INTEGER DEFAULT 0, n_candidates INTEGER DEFAULT 0,
    n_quoted INTEGER DEFAULT 0, error TEXT
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL REFERENCES scans(id),
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS results_scan ON results(scan_id);
CREATE TABLE IF NOT EXISTS combo_tickers (
    legs_key TEXT PRIMARY KEY, collection TEXT NOT NULL, market_ticker TEXT NOT NULL,
    event_ticker TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS combo_legs (combo_ticker TEXT PRIMARY KEY, legs TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    placed_at TEXT NOT NULL, result_id INTEGER, data TEXT NOT NULL,
    stake REAL NOT NULL, payout_x REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open', pnl REAL, settled_at TEXT
);
-- Every combo the edge miner quoted (app/miner.py), graded once it settles.
CREATE TABLE IF NOT EXISTS quote_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    quoted_at TEXT NOT NULL, run_id TEXT NOT NULL, sport TEXT NOT NULL, game TEXT NOT NULL,
    start TEXT, type TEXT NOT NULL, n_legs INTEGER NOT NULL, legs TEXT NOT NULL,
    combo_ticker TEXT, legs_x REAL NOT NULL, combo_x REAL, d REAL,
    q REAL NOT NULL, q_lo REAL NOT NULL, ev REAL, ev_lo REAL, n_quotes INTEGER,
    result TEXT, graded_at TEXT
);
CREATE INDEX IF NOT EXISTS quote_ledger_type ON quote_ledger(type);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    config.DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# Columns added after the first release; added in place on older databases.
ADDED_COLUMNS = {
    "bets": {"source": "TEXT NOT NULL DEFAULT 'manual'", "combo_ticker": "TEXT", "game": "TEXT",
             "contracts": "REAL"},
    "scans": {"auto_bet_mode": "TEXT"},
}


def init():
    with connect() as c:
        c.executescript(SCHEMA)
        for table, cols in ADDED_COLUMNS.items():
            have = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
            for col, decl in cols.items():
                if col not in have:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


# ---- settings ---------------------------------------------------------------------------------

AUTO_BET_MODES = ("off", "dry_run", "live")


def coerce_setting(key: str, value):
    """Validate a settings value against its default's type; raises ValueError on anything else."""
    default = config.DEFAULT_SETTINGS[key]
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false")
        return value
    if isinstance(default, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a number")
        return type(default)(value)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text")
    if key == "auto_bet_mode" and value not in AUTO_BET_MODES:
        raise ValueError("auto_bet_mode must be off, dry_run or live")
    return value


def get_settings() -> dict:
    with connect() as c:
        stored = {r["key"]: json.loads(r["value"]) for r in c.execute("SELECT key, value FROM settings")}
    out = dict(config.DEFAULT_SETTINGS)
    for k, v in stored.items():
        if k not in out:
            continue
        try:
            out[k] = coerce_setting(k, v)
        except ValueError:  # a bad stored value (e.g. an old "None" mode) falls back to the safe default
            pass
    return out


def update_settings(values: dict) -> dict:
    clean = {k: coerce_setting(k, v) for k, v in values.items() if k in config.DEFAULT_SETTINGS}
    with connect() as c:
        for k, v in clean.items():
            c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (k, json.dumps(v)))
    return get_settings()


# ---- scans + results --------------------------------------------------------------------------

def create_scan(mode: str, auto_bet_mode: str = "off") -> int:
    with connect() as c:
        return c.execute("INSERT INTO scans (started_at, status, mode, auto_bet_mode) VALUES (?, 'running', ?, ?)",
                         (now_iso(), mode, auto_bet_mode)).lastrowid


def finish_scan(scan_id: int, status: str, error: str | None = None, **counts):
    with connect() as c:
        sets = ", ".join(f"{k} = ?" for k in counts)
        c.execute(f"UPDATE scans SET finished_at = ?, status = ?, error = ?{', ' + sets if sets else ''} WHERE id = ?",
                  (now_iso(), status, error, *counts.values(), scan_id))


def save_results(scan_id: int, rows: list[dict]) -> list[int]:
    with connect() as c:
        ids = []
        for r in rows:
            ids.append(c.execute("INSERT INTO results (scan_id, data) VALUES (?, ?)",
                                 (scan_id, json.dumps(r, default=str))).lastrowid)
        return ids


def get_scan(scan_id: int) -> dict | None:
    with connect() as c:
        row = c.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
        if row is None:
            return None
        results = [{**json.loads(r["data"]), "result_id": r["id"]}
                   for r in c.execute("SELECT id, data FROM results WHERE scan_id = ?", (scan_id,))]
    return {**dict(row), "results": results}


def latest_scan_id() -> int | None:
    with connect() as c:
        row = c.execute("SELECT id FROM scans WHERE status = 'done' ORDER BY id DESC LIMIT 1").fetchone()
    return row["id"] if row else None


def list_scans(limit: int = 50) -> list[dict]:
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,))]


def recent_quoted_rows(days: int) -> list[dict]:
    """Results that got a Kalshi quote in live scans from the last `days` days."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    with connect() as c:
        rows = c.execute("""SELECT r.data FROM results r JOIN scans s ON s.id = r.scan_id
                            WHERE s.mode = 'live' AND s.started_at >= ?""", (since,)).fetchall()
    out = []
    for row in rows:
        d = json.loads(row["data"])
        if d.get("kalshi_x_after_fees"):
            d.pop("raw_quotes", None)
            out.append(d)
    return out


def get_result(result_id: int) -> dict | None:
    with connect() as c:
        row = c.execute("SELECT data FROM results WHERE id = ?", (result_id,)).fetchone()
    return json.loads(row["data"]) if row else None


# ---- combo ticker cache + weekly creation budget ----------------------------------------------

def get_combo_ticker(legs_key: str) -> dict | None:
    with connect() as c:
        row = c.execute("SELECT * FROM combo_tickers WHERE legs_key = ?", (legs_key,)).fetchone()
    return dict(row) if row else None


def put_combo_ticker(legs_key: str, collection: str, market_ticker: str, event_ticker: str):
    with connect() as c:
        c.execute("INSERT OR REPLACE INTO combo_tickers VALUES (?, ?, ?, ?, ?)",
                  (legs_key, collection, market_ticker, event_ticker, now_iso()))


def rejected_combo_keys() -> set[str]:
    with connect() as c:
        return {r[0] for r in c.execute("SELECT legs_key FROM combo_tickers WHERE market_ticker = 'REJECTED'")}


def creations_this_week() -> int:
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(timespec="seconds")
    with connect() as c:
        return c.execute("SELECT COUNT(*) FROM combo_tickers WHERE created_at >= ? AND market_ticker != 'REJECTED'",
                         (since,)).fetchone()[0]


# ---- bets -------------------------------------------------------------------------------------

def add_bet(result_id: int | None, data: dict, stake: float, payout_x: float, source: str = "manual",
            combo_ticker: str | None = None, game: str | None = None, contracts: float | None = None) -> int:
    with connect() as c:
        return c.execute(
            """INSERT INTO bets (placed_at, result_id, data, stake, payout_x, source, combo_ticker, game, contracts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now_iso(), result_id, json.dumps(data, default=str), stake, payout_x, source, combo_ticker, game,
             contracts)).lastrowid


def today_auto_spend(source: str) -> float:
    """Dollars staked today (UTC) by auto-betting of the given source ("auto" or "paper")."""
    today = datetime.now(timezone.utc).date().isoformat()
    with connect() as c:
        return c.execute("SELECT COALESCE(SUM(stake), 0) FROM bets WHERE source = ? AND placed_at >= ?",
                         (source, today)).fetchone()[0]


def bet_tickers(source: str) -> set[str]:
    with connect() as c:
        return {r[0] for r in c.execute("SELECT combo_ticker FROM bets WHERE source = ? AND combo_ticker IS NOT NULL",
                                        (source,))}


def bets_by_game(source: str) -> dict[str, list[dict]]:
    """Open bets of a source grouped by game key (paper bets stand in for held positions in dry runs)."""
    out: dict[str, list[dict]] = {}
    for b in list_bets():
        if b["source"] == source and b["status"] == "open" and b["game"]:
            out.setdefault(b["game"], []).append(b)
    return out


def list_bets() -> list[dict]:
    with connect() as c:
        return [{**dict(r), "data": json.loads(r["data"])} for r in c.execute("SELECT * FROM bets ORDER BY id DESC")]


def settle_bet(bet_id: int, status: str, pnl: float):
    with connect() as c:
        c.execute("UPDATE bets SET status = ?, pnl = ?, settled_at = ? WHERE id = ?", (status, pnl, now_iso(), bet_id))


def get_combo_legs(combo_ticker: str) -> list[dict] | None:
    with connect() as c:
        row = c.execute("SELECT legs FROM combo_legs WHERE combo_ticker = ?", (combo_ticker,)).fetchone()
    return json.loads(row["legs"]) if row else None


def put_combo_legs(combo_ticker: str, legs: list[dict]):
    with connect() as c:
        c.execute("INSERT OR REPLACE INTO combo_legs VALUES (?, ?)", (combo_ticker, json.dumps(legs)))


def delete_bet(bet_id: int):
    with connect() as c:
        c.execute("DELETE FROM bets WHERE id = ?", (bet_id,))


# ---- edge-miner quote ledger --------------------------------------------------------------------

LEDGER_COLS = ("quoted_at", "run_id", "sport", "game", "start", "type", "n_legs", "legs", "combo_ticker", "legs_x",
               "combo_x", "d", "q", "q_lo", "ev", "ev_lo", "n_quotes")


def add_ledger_row(row: dict) -> int:
    vals = [json.dumps(row[k]) if k == "legs" else row.get(k) for k in LEDGER_COLS]
    with connect() as c:
        return c.execute(f"INSERT INTO quote_ledger ({', '.join(LEDGER_COLS)}) VALUES "
                         f"({', '.join('?' * len(LEDGER_COLS))})", vals).lastrowid


def ledger_rows(days: int | None = None, priced_only: bool = True) -> list[dict]:
    sql, args = "SELECT * FROM quote_ledger WHERE 1=1", []
    if days is not None:
        sql += " AND quoted_at >= ?"
        args.append((datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds"))
    if priced_only:
        sql += " AND d IS NOT NULL"
    with connect() as c:
        return [{**dict(r), "legs": json.loads(r["legs"])} for r in c.execute(sql + " ORDER BY id", args)]


def ungraded_ledger_tickers() -> list[str]:
    with connect() as c:
        return [r[0] for r in c.execute("""SELECT DISTINCT combo_ticker FROM quote_ledger
                                           WHERE result IS NULL AND combo_ticker IS NOT NULL""")]


def grade_ledger(combo_ticker: str, result: str):
    with connect() as c:
        c.execute("UPDATE quote_ledger SET result = ?, graded_at = ? WHERE combo_ticker = ? AND result IS NULL",
                  (result, now_iso(), combo_ticker))
