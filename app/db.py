"""SQLite persistence: settings, scan history, results, combo-ticker cache, bet log."""

import json
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
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    placed_at TEXT NOT NULL, result_id INTEGER, data TEXT NOT NULL,
    stake REAL NOT NULL, payout_x REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open', pnl REAL, settled_at TEXT
);
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


def init():
    with connect() as c:
        c.executescript(SCHEMA)


# ---- settings ---------------------------------------------------------------------------------

def get_settings() -> dict:
    with connect() as c:
        stored = {r["key"]: json.loads(r["value"]) for r in c.execute("SELECT key, value FROM settings")}
    return {**config.DEFAULT_SETTINGS, **{k: v for k, v in stored.items() if k in config.DEFAULT_SETTINGS}}


def update_settings(values: dict) -> dict:
    with connect() as c:
        for k, v in values.items():
            if k in config.DEFAULT_SETTINGS:
                c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                          (k, json.dumps(type(config.DEFAULT_SETTINGS[k])(v))))
    return get_settings()


# ---- scans + results --------------------------------------------------------------------------

def create_scan(mode: str) -> int:
    with connect() as c:
        return c.execute("INSERT INTO scans (started_at, status, mode) VALUES (?, 'running', ?)",
                         (now_iso(), mode)).lastrowid


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


def creations_this_week() -> int:
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(timespec="seconds")
    with connect() as c:
        return c.execute("SELECT COUNT(*) FROM combo_tickers WHERE created_at >= ?", (since,)).fetchone()[0]


# ---- bets -------------------------------------------------------------------------------------

def add_bet(result_id: int | None, data: dict, stake: float, payout_x: float) -> int:
    with connect() as c:
        return c.execute("INSERT INTO bets (placed_at, result_id, data, stake, payout_x) VALUES (?, ?, ?, ?, ?)",
                         (now_iso(), result_id, json.dumps(data, default=str), stake, payout_x)).lastrowid


def list_bets() -> list[dict]:
    with connect() as c:
        return [{**dict(r), "data": json.loads(r["data"])} for r in c.execute("SELECT * FROM bets ORDER BY id DESC")]


def settle_bet(bet_id: int, status: str, pnl: float):
    with connect() as c:
        c.execute("UPDATE bets SET status = ?, pnl = ?, settled_at = ? WHERE id = ?", (status, pnl, now_iso(), bet_id))


def delete_bet(bet_id: int):
    with connect() as c:
        c.execute("DELETE FROM bets WHERE id = ?", (bet_id,))
