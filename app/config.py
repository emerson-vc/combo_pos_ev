import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "scanner.db"

load_dotenv(ROOT / ".env")

KALSHI_URLS = {
    "prod": "https://external-api.kalshi.com/trade-api/v2",
    "demo": "https://external-api.demo.kalshi.co/trade-api/v2",
}

KALSHI_ENV = os.getenv("KALSHI_ENV", "prod")
KALSHI_BASE_URL = KALSHI_URLS[KALSHI_ENV]
KALSHI_KEY_ID = os.getenv("KALSHI_KEY_ID") or None
_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH") or ""
KALSHI_PRIVATE_KEY_PATH = (ROOT / _key_path).resolve() if _key_path else None

FEE_RATE = float(os.getenv("FEE_RATE", "0.07"))
# Which `accepted_side` buys YES on a combo RFQ quote. Set from `make side-test`; live auto-betting
# refuses to run until it is "yes" or "no".
KALSHI_ACCEPT_SIDE = os.getenv("KALSHI_ACCEPT_SIDE") or None

# Collection used to build combos; it holds every MLB/NFL game event.
COMBO_COLLECTION_SERIES = "KXMVESPORTSMULTIGAMEEXTENDED"
WEEKLY_COMBO_CREATION_LIMIT = 5000

# Historical seasons used by the correlation model ("last 3 seasons" + current season to date).
MLB_SEASONS = [2024, 2025, 2026]
NFL_SEASONS = [2023, 2024, 2025, 2026]

DEFAULT_SETTINGS = {
    "bankroll": 1000.0,
    "kelly_fraction": 0.25,
    "max_bet_pct": 0.02,
    "max_game_pct": 0.05,
    # Only request Kalshi quotes for pairs whose model joint prob beats independence by this ratio.
    "min_lift": 1.10,
    "max_quotes_per_game": 15,
    # All-pairs search: keep pairs whose underlying stats rank-correlate at least this much...
    "min_atlas_rho": 0.15,
    # ...and that pay at least this much if the legs were independent (short x short can't win).
    "min_indep_x": 3.0,
    # Skip lottery tickets: combos whose fair payout exceeds this multiple.
    "max_fair_x": 150.0,
    # MLB: the edges found by hand were long shots (2+ HR + team win ~100x+), so allow more room/quotes.
    "mlb_max_fair_x": 200.0,
    "mlb_quotes_per_game": 20,
    # Always quote every 2+ HR + own team ML combo, outside the per-game quote slots.
    "always_quote_hr_ml": True,
    "pinned_max_fair_x": 250.0,
    # Auto-betting (only during scans the user starts): "off" | "dry_run" (paper bets) | "live".
    "auto_bet_mode": "off",
    "min_pess_ev": 0.10,  # pessimistic EV must beat this on a fresh quote at the real stake
    "max_bets_per_game": 2,  # counts combos already held on Kalshi for that game
    "daily_max_spend_pct": 0.10,  # of the Kalshi balance
    "min_auto_stake": 1.0,
    "quote_size_dollars": 5.0,
    "quote_wait_seconds": 4.0,
    "min_leg_prob": 0.02,
    # Thin long-shot rungs priced from the player's liquid lower rung may go this low.
    "min_ladder_prob": 0.005,
    # Pessimistic EV prices market legs at their bid and curve/ladder legs this much lower.
    "model_leg_haircut": 0.3,
    "max_leg_prob": 0.97,
    "max_leg_spread": 0.10,
}


def has_kalshi_auth() -> bool:
    return bool(KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH and KALSHI_PRIVATE_KEY_PATH.exists())
