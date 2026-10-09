"""Normalize Kalshi single-leg markets into typed Legs."""

import dataclasses
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
    "KXMLBSB": ("MLB", "sb"),
    "KXMLBKS": ("MLB", "ks"),
    "KXMLBOUTS": ("MLB", "outs"),
    "KXMLBF5": ("MLB", "f5_win"),
    "KXMLBF3": ("MLB", "f3_win"),
    "KXMLBF5SPREAD": ("MLB", "f5_spread"),
    "KXMLBF5TOTAL": ("MLB", "f5_total"),
    "KXMLBRFI": ("MLB", "rfi"),
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
    "KXNFLPASSATT": ("NFL", "pass_att"),
    "KXNFLPASSCOMP": ("NFL", "pass_cmp"),
    "KXNFLPASSINT": ("NFL", "pass_int"),
    "KXNFLRSHATT": ("NFL", "rush_att"),
    "KXNFLRRYDS": ("NFL", "rr_yds"),
    "KXNFLFIRSTTD": ("NFL", "first_td"),
    "KXNFLTEAMFIRSTTD": ("NFL", "team_first_td"),
    "KXNFLFIRSTTDTEAM": ("NFL", "first_td_team"),
    "KXNFL1H": ("NFL", "h1_win"),
    "KXNFL1HSPREAD": ("NFL", "h1_spread"),
    "KXNFL1HTEAMTOTAL": ("NFL", "h1_team_total"),
    "KXNFL1HTOTAL": ("NFL", "h1_total"),
    "KXNFL1Q": ("NFL", "q1_win"),
    "KXNFL1QSPREAD": ("NFL", "q1_spread"),
    "KXNFL1QTOTAL": ("NFL", "q1_total"),
}

TEAM_KINDS = {"team_win", "team_spread", "team_total", "f5_win", "f3_win", "f5_spread", "first_td_team",
              "h1_win", "h1_spread", "h1_team_total", "q1_win", "q1_spread"}
GAME_KINDS = {"game_total", "f5_total", "rfi", "h1_total", "q1_total"}  # whole-game outcomes, no team
PITCHER_KINDS = {"ks", "outs"}
PLAYER_KINDS = {"hr", "hit", "rbi", "tb", "hrr", "sb", "td", "pass_yds", "pass_td", "rec_yds", "rush_yds", "rec",
                "pass_att", "pass_cmp", "pass_int", "rush_att", "rr_yds", "first_td", "team_first_td",
                *PITCHER_KINDS}

ET = ZoneInfo("America/New_York")
# Uncertainty on curve/ladder-priced legs, used for the pessimistic EV (settings: model_leg_haircut).
MODEL_LEG_HAIRCUT = 0.3
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
    side: str = "yes"  # "no" = betting the market doesn't happen (e.g. team total UNDER); prices are for this side

    @property
    def mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2

    @property
    def p(self) -> float:
        """Probability used by the model: ladder-implied for thin long shots, else the midpoint."""
        return self.p_model if self.p_model is not None else self.mid

    def p_lo(self, haircut: float = MODEL_LEG_HAIRCUT) -> float:
        """Conservative probability: the bid for market-priced legs; a haircut on curve/ladder prices."""
        if self.p_source == "market":
            return self.yes_bid
        return self.p * (1 - haircut)

    def p_hi(self, haircut: float = MODEL_LEG_HAIRCUT) -> float:
        if self.p_source == "market":
            return self.yes_ask if self.yes_ask > 0 else self.p
        return min(self.p * (1 + haircut), 0.999)

    @property
    def spread(self) -> float:
        return self.yes_ask - self.yes_bid

    @property
    def count_k(self) -> int | None:
        """For 'N+' markets: the N (threshold 2.5 -> 3)."""
        return None if self.threshold is None else int(self.threshold + 0.5)

    def negate(self) -> "Leg":
        """The NO side of this (YES) market, priced and labeled as its own leg."""
        assert self.side == "yes", "only YES legs are negated"
        if self.kind == "team_total":
            team = self.label.split(" over ")[0].removeprefix("Will ").removesuffix(" score")
            label = f"{team} under {self.threshold:g} runs"
        else:
            label = f"NOT {self.label}"
        return dataclasses.replace(
            self, side="no", yes_bid=round(1 - self.yes_ask, 4),
            yes_ask=round(1 - self.yes_bid, 4), label=label,
            p_model=None if self.p_model is None else 1 - self.p_model)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mid"] = self.mid
        d["p"] = self.p
        d["p_lo"] = self.p_lo()
        d["spread"] = self.spread
        return d


def game_key(event_ticker: str) -> str:
    return event_ticker.split("-")[1]


def market_key(event_ticker: str, market_ticker: str) -> str:
    """A leg's market, ignoring its line: 'MIA by 9.5+' and 'MIA by 13.5+' are the same market.

    KXNFLSPREAD-26OCT11CINMIA-MIA10 -> KXNFLSPREAD-26OCT11CINMIA|MIA   (team line: trailing number)
    KXNFLTD-26OCT08TBDAL-DALCLAMB88-3 -> KXNFLTD-26OCT08TBDAL|DALCLAMB88 (player rung: "-3")
    """
    suffix = market_ticker[len(event_ticker) + 1:] if market_ticker.startswith(event_ticker) else market_ticker
    if re.search(r"-\d+$", suffix):
        suffix = re.sub(r"-\d+$", "", suffix)
    else:
        suffix = re.sub(r"\d+$", "", suffix)
    return f"{event_ticker}|{suffix}"


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
    team = team_of(suffix, teams) if kind not in GAME_KINDS else None  # F5 "TIE" also -> None
    if series == "KXNFLTEAMFIRSTTD":  # team lives in the event ticker: ...-26OCT11PHIJAC-JAC
        team = event_ticker.split("-")[-1]
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
