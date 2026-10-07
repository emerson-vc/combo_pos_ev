from datetime import datetime, timezone

from app.legs import game_start, parse_market

LAMB = {
    "ticker": "KXNFLTD-26OCT08TBDAL-DALCLAMB88-3", "event_ticker": "KXNFLTD-26OCT08TBDAL",
    "title": "CeeDee Lamb: 3+ touchdowns", "yes_sub_title": "CeeDee Lamb: 3+", "floor_strike": 2.5,
    "yes_bid_dollars": "0.0300", "yes_ask_dollars": "0.0400", "status": "active",
}
TEAM_TOTAL = {
    "ticker": "KXNFLTEAMTOTAL-26OCT08TBDAL-DAL29", "event_ticker": "KXNFLTEAMTOTAL-26OCT08TBDAL",
    "title": "DAL Cowboys over 28.5 points scored", "yes_sub_title": "DAL Cowboys over 28.5 points",
    "floor_strike": 28.5, "yes_bid_dollars": "0.5000", "yes_ask_dollars": "0.5100", "status": "active",
}
WIN = {
    "ticker": "KXMLBGAME-26OCT071800LADATL-LAD", "event_ticker": "KXMLBGAME-26OCT071800LADATL",
    "title": "Los Angeles D wins", "yes_sub_title": "Los Angeles D", "yes_bid_dollars": "0.5600",
    "yes_ask_dollars": "0.5800", "status": "active",
}


def test_parse_player_leg():
    leg = parse_market(LAMB, ["TB", "DAL"])
    assert (leg.sport, leg.kind, leg.team, leg.player, leg.count_k) == ("NFL", "td", "DAL", "CeeDee Lamb", 3)
    assert leg.mid == 0.035


def test_parse_team_legs():
    tt = parse_market(TEAM_TOTAL, ["TB", "DAL"])
    assert (tt.kind, tt.team, tt.threshold, tt.player) == ("team_total", "DAL", 28.5, None)
    win = parse_market(WIN, ["LAD", "ATL"])
    assert (win.sport, win.kind, win.team, win.threshold) == ("MLB", "team_win", "LAD", None)


def test_team_prefix_prefers_longest():
    leg = parse_market({**TEAM_TOTAL, "ticker": "KXNFLTEAMTOTAL-26OCT08LALAC-LAC22",
                        "event_ticker": "KXNFLTEAMTOTAL-26OCT08LALAC"}, ["LA", "LAC"])
    assert leg.team == "LAC"


def test_game_start():
    # MLB keys carry ET first pitch; NFL uses occurrence time minus 3h.
    assert game_start("26OCT071800LADATL", None) == datetime(2026, 10, 7, 22, 0, tzinfo=timezone.utc)
    assert game_start("26OCT08TBDAL", "2026-10-09T03:15:00Z") == datetime(2026, 10, 9, 0, 15, tzinfo=timezone.utc)
