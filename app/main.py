"""Local dashboard server: `make run`, then open http://localhost:8000."""

import asyncio
import hashlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config, db
from app.kalshi import trading
from app.kalshi.client import KalshiClient
from app.model.joint import Model
from app.pricing import size_stakes
from app.quote_policy import QuotePolicy
from app.scan import PROGRESS, run_scan

STATIC = config.ROOT / "app" / "static"
state: dict = {"running": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    state["model"] = Model()
    yield


app = FastAPI(title="Combo +EV", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def no_stale_dashboard(request, call_next):
    """Make the browser revalidate the page and its JS/CSS (ETags keep this cheap), so a cached old
    app.js can't run against new settings."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def build_hash() -> str:
    """Short hash of the dashboard's script + styles; changes whenever either file changes."""
    h = hashlib.sha256()
    for name in ("app.js", "style.css"):
        h.update((STATIC / name).read_bytes())
    return h.hexdigest()[:10]


@app.get("/")
def index():
    """The page, with versioned asset URLs so the browser can never run a stale cached script."""
    build = build_hash()
    html = (STATIC / "index.html").read_text()
    html = html.replace("/static/app.js", f"/static/app.js?v={build}").replace(
        "/static/style.css", f"/static/style.css?v={build}")
    html = html.replace("<head>", f'<head>\n  <meta name="build" content="{build}">', 1)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/api/status")
async def status():
    model: Model = state["model"]
    balance = None
    if config.has_kalshi_auth():
        client = KalshiClient()
        try:
            balance = await trading.balance(client)
        except Exception:  # status must load even if Kalshi is unreachable
            pass
        finally:
            await client.close()
    return {
        "auth": config.has_kalshi_auth(),
        "balance": balance,
        "auto_bet_mode": db.get_settings()["auto_bet_mode"],
        "accept_side_set": config.KALSHI_ACCEPT_SIDE in ("yes", "no"),
        "env": config.KALSHI_ENV,
        "fee_rate": config.FEE_RATE,
        "history": {"NFL": model.has("NFL"), "MLB": model.has("MLB")},
        "creations_this_week": db.creations_this_week(),
        "creation_limit": config.WEEKLY_COMBO_CREATION_LIMIT,
        "running_scan": state["running"],
        "latest_scan": db.latest_scan_id(),
        "build": build_hash(),
    }


@app.post("/api/scan")
async def start_scan():
    if state["running"] is not None:
        return {"scan_id": state["running"]}
    scan_id = db.create_scan("live" if config.has_kalshi_auth() else "model", db.get_settings()["auto_bet_mode"])
    state["running"] = scan_id

    async def runner():
        try:
            await run_scan(scan_id, state["model"])
        finally:
            state["running"] = None

    asyncio.create_task(runner())
    return {"scan_id": scan_id}


@app.get("/api/scan/{scan_id}")
def get_scan(scan_id: int):
    scan = db.get_scan(scan_id)
    if scan is None:
        raise HTTPException(404)
    # Re-size stakes with current settings so bankroll/Kelly edits apply without rescanning.
    priced = [r for r in scan["results"] if r.get("ev") is not None and r["ev"] > 0]
    for r in scan["results"]:
        r["stake"] = 0.0
    size_stakes(priced, db.get_settings())
    scan["progress"] = PROGRESS.get(scan_id)
    return scan


@app.get("/api/scans")
def scans():
    return db.list_scans()


@app.get("/api/quote-stats")
def quote_stats():
    return QuotePolicy.from_db().stats()


@app.get("/api/settings")
def get_settings():
    return db.get_settings()


@app.put("/api/settings")
def put_settings(values: dict):
    if values.get("auto_bet_mode") == "live" and config.KALSHI_ACCEPT_SIDE not in ("yes", "no"):
        raise HTTPException(400, "Run `make side-test` and set KALSHI_ACCEPT_SIDE in .env before going live")
    try:
        return db.update_settings(values)
    except ValueError as e:  # e.g. a null/NaN from a stale dashboard script
        raise HTTPException(400, str(e))


class BetIn(BaseModel):
    result_id: int | None = None
    stake: float
    payout_x: float  # all-in payout per $1 staked, as the Kalshi app shows it


@app.post("/api/bets")
def add_bet(bet: BetIn):
    data = db.get_result(bet.result_id) if bet.result_id else None
    if data is None:
        raise HTTPException(400, "unknown result_id")
    data.pop("raw_quotes", None)
    return {"id": db.add_bet(bet.result_id, data, bet.stake, bet.payout_x)}


@app.get("/api/bets")
def bets():
    return db.list_bets()


@app.delete("/api/bets/{bet_id}")
def delete_bet(bet_id: int):
    db.delete_bet(bet_id)
    return {"ok": True}


@app.post("/api/bets/settle")
async def settle_bets():
    """Settle open bets from the two legs' Kalshi results (public market data)."""
    client = KalshiClient(key_id=None)
    settled = 0
    try:
        for bet in db.list_bets():
            if bet["status"] != "open":
                continue
            outcomes = []  # per leg: True won / False lost / None unresolved
            for leg in (bet["data"]["leg_a"], bet["data"]["leg_b"]):
                m = (await client.get(f"/markets/{leg['ticker']}")).get("market", {})
                result = m.get("result") or ""
                outcomes.append(None if result not in ("yes", "no") else result == leg.get("side", "yes"))
            if False in outcomes:
                db.settle_bet(bet["id"], "lost", -bet["stake"])
                settled += 1
            elif outcomes == [True, True]:
                db.settle_bet(bet["id"], "won", round(bet["stake"] * (bet["payout_x"] - 1), 2))
                settled += 1
    finally:
        await client.close()
    return {"settled": settled}
