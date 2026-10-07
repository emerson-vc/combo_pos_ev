"""Normalize Kalshi single-leg markets into typed Legs."""

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# series ticker -> (sport, leg kind)
SERIES = {
    "KXMLBGAME": ("MLB", "team_win"),
    "KXMLBSPREAD": ("MLB", "team_spread"),
    "KXMLBTEAMTOTAL": ("MLB", "team_total"),
    "KXMLBTOTAL": ("MLB", "game_total"),
    "KXMLBHR": ("MLB", "hr"),
    "KXMLBHIT": ("MLB", "hit"),
    "KXMLBRBI": ("MLB", "rbi"),
    "KXMLBTB": ("MLB", "tb"),
    "KXMLBHRR": ("MLB", "hrr"),
    "KXNFLGAME": ("NFL", "team_win"),
    "KXNFLSPREAD": ("NFL", "team_spread"),
    "KXNFLTEAMTOTAL": ("NFL", "team_total"),
    "KXNFLTOTAL": ("NFL", "game_total"),
    "KXNFLTD": ("NFL", "td"),
    "KXNFLPASSYDS": ("NFL", "pass_yds"),
    "KXNFLPASSTDS": ("NFL", "pass_td"),
    "KXNFLRECYDS": ("NFL", "rec_yds"),
    "KXNFLRSHYDS": ("NFL", "rush_yds"),
    "KXNFLREC": ("NFL", "rec"),
}

TEAM_KINDS = {"team_win", "team_spread", "team_total"}
PLAYER_KINDS = {"hr", "hit", "rbi", "tb", "hrr", "td", "pass_yds", "pass_td", "rec_yds", "rush_yds", "rec"}

ET = ZoneInfo("America/New_York")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
_GAME_KEY = re.compile(r"^(\d{2})([A-Z]{3})(\d{2})(\d{4})?([A-Z]+)$")


@dataclass(frozen=True)
class Leg:
    sport: str
    game: str  # game key, e.g. "26OCT08TBDAL"
    series: str
    event_ticker: str
    ticker: str
    kind: str
    team: str | None
    player: str | None
    threshold: float | None  # Kalshi floor_strike: YES if stat > threshold
    yes_bid: float
    yes_ask: float
    label: str
    p_model: float | None = None  # set when the market is too thin to trust (see model/ladder.py)
    p_source: str = "market"  # "market" | "ladder"

    @property
    def mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2

    @property
    def p(self) -> float:
        """Probability used by the model: ladder-implied for thin long shots, else the midpoint."""
        return self.p_model if self.p_model is not None else self.mid

    @property
    def spread(self) -> float:
        return self.yes_ask - self.yes_bid

    @property
    def count_k(self) -> int | None:
        """For 'N+' markets: the N (threshold 2.5 -> 3)."""
        return None if self.threshold is None else int(self.threshold + 0.5)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mid"] = self.mid
        d["p"] = self.p
        d["spread"] = self.spread
        return d


def game_key(event_ticker: str) -> str:
    return event_ticker.split("-")[1]


def game_teams_blob(key: str) -> str:
    m = _GAME_KEY.match(key)
    return m.group(5) if m else key


def game_start(key: str, occurrence: str | None) -> datetime | None:
    """MLB keys carry the ET first-pitch time; otherwise Kalshi's occurrence time is ~start + 3h."""
    m = _GAME_KEY.match(key)
    if m and m.group(4):
        yy, mon, dd, hhmm = m.group(1), m.group(2), m.group(3), m.group(4)
        local = datetime(2000 + int(yy), _MONTHS[mon], int(dd), int(hhmm[:2]), int(hhmm[2:]), tzinfo=ET)
        return local.astimezone(timezone.utc)
    if occurrence:
        return datetime.fromisoformat(occurrence.replace("Z", "+00:00")) - timedelta(hours=3)
    return None


def _price(market: dict, field: str) -> float:
    v = market.get(f"{field}_dollars")
    return float(v) if v not in (None, "") else 0.0


def team_of(suffix: str, teams: list[str]) -> str | None:
    for t in sorted(teams, key=len, reverse=True):
        if suffix.startswith(t):
            return t
    return None


def parse_market(market: dict, teams: list[str]) -> Leg | None:
    event_ticker = market["event_ticker"]
    series = event_ticker.split("-")[0]
    if series not in SERIES or market.get("status") not in ("active", "open"):
        return None
    sport, kind = SERIES[series]
    suffix = market["ticker"][len(event_ticker) + 1:]
    team = team_of(suffix, teams) if kind != "game_total" else None
    player = None
    if kind in PLAYER_KINDS:
        player = (market.get("yes_sub_title") or market.get("title") or "").split(":")[0].strip() or None
    threshold = market.get("floor_strike")
    return Leg(
        sport=sport,
        game=game_key(event_ticker),
        series=series,
        event_ticker=event_ticker,
        ticker=market["ticker"],
        kind=kind,
        team=team,
        player=player,
        threshold=float(threshold) if threshold is not None else None,
        yes_bid=_price(market, "yes_bid"),
        yes_ask=_price(market, "yes_ask"),
        label=(market.get("title") or market.get("yes_sub_title") or market["ticker"]).rstrip("?"),
    )
