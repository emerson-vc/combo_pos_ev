"""Every correlated pair of legs in a game (replaces the hand-written templates).

Relationships searched (the shapes that move together and that Kalshi lets you combine):
  same_player   two different stats of one player       RB 1+ TD + RB 70+ rush yds
  teammates     two players on the same team            QB 300+ yds + WR 100+ yds
  player_team   a player + his own team's outcome       2+ HR + team ML
  player_game   a player + a whole-game outcome          QB 3+ TD + game over 49.5
  team_team     two outcomes of the same team           wins + team total over
  game_team     a game outcome + one team's outcome     game over + team wins 1st half
  game_game     two whole-game outcomes                 1st half over + game over
Opposing-team pairs are skipped, except pitcher Ks/outs + opponent team total UNDER (NO side).

A pair survives only if (1) its legs are in different Kalshi events (one line per event per
combo), (2) it pays at least `min_indep_x` if the legs were independent (short × short pairs
can't beat the makers' margin), and (3) the correlation atlas says the two underlying stats
move together (rank correlation ≥ `min_atlas_rho`).
"""

from itertools import combinations

from app.legs import GAME_KINDS, PLAYER_KINDS, Leg
from app.model.joint import Model
from app.templates import PINNED_TEMPLATE, usable

REL_NAMES = {
    "same_player": "Same player", "teammates": "Teammates", "player_team": "Player + team",
    "player_game": "Player + game", "team_team": "Team", "game_team": "Game + team", "game_game": "Game",
}
KIND_NAMES = {
    "hr": "HR", "hit": "hits", "rbi": "RBI", "tb": "total bases", "hrr": "H+R+RBI", "sb": "steals",
    "ks": "strikeouts", "outs": "outs", "td": "TDs", "pass_yds": "pass yds", "pass_td": "pass TDs",
    "pass_att": "pass att", "pass_cmp": "completions", "pass_int": "INTs", "rec_yds": "rec yds",
    "rush_yds": "rush yds", "rr_yds": "rush+rec yds", "rec": "receptions", "rush_att": "rush att",
    "first_td": "first TD", "team_first_td": "team's first TD", "team_win": "wins", "team_spread": "spread",
    "team_total": "team total", "game_total": "game total", "f5_win": "F5 win", "f3_win": "F3 win",
    "f5_spread": "F5 spread", "f5_total": "F5 total", "rfi": "1st-inning run", "first_td_team": "scores 1st TD",
    "h1_win": "1H win", "h1_spread": "1H spread", "h1_team_total": "1H team total", "h1_total": "1H total",
    "q1_win": "1Q win", "q1_spread": "1Q spread", "q1_total": "1Q total",
}
SPECIAL_NAMES = {PINNED_TEMPLATE: "2+ HR + team ML", "mlb_sp_opp_under": "Pitcher Ks/outs + opponent UNDER"}


def _scope(leg: Leg) -> int:
    return 0 if leg.kind in PLAYER_KINDS else 2 if leg.kind in GAME_KINDS else 1


def relationship(a: Leg, b: Leg) -> str | None:
    """`a`, `b` ordered player → team → game. None for pairs we don't search (opposing sides)."""
    sa, sb = _scope(a), _scope(b)
    if sa == 0 and sb == 0:
        if a.team != b.team:
            return None
        return "same_player" if a.player == b.player else "teammates"
    if sa == 0:
        return "player_game" if sb == 2 else ("player_team" if a.team == b.team else None)
    if sa == 1:
        return "game_team" if sb == 2 else ("team_team" if a.team == b.team else None)
    return "game_game"


def template_id(a: Leg, b: Leg, rel: str) -> str:
    if rel == "player_team" and a.kind == "hr" and (a.count_k or 0) >= 2 and b.kind == "team_win":
        return PINNED_TEMPLATE
    return f"{a.sport.lower()}:{rel}:{a.kind}+{b.kind}"


def template_name(tid: str) -> str:
    if tid in SPECIAL_NAMES:
        return SPECIAL_NAMES[tid]
    try:
        _, rel, kinds = tid.split(":")
        ka, kb = kinds.split("+")
    except ValueError:
        return tid
    return f"{REL_NAMES.get(rel, rel)}: {KIND_NAMES.get(ka, ka)} + {KIND_NAMES.get(kb, kb)}"


def candidate_pairs(legs: list[Leg], teams: list[str], model: Model, settings: dict) -> list[tuple[str, Leg, Leg]]:
    """Correlated pairs worth modeling, each as (template id, leg a, leg b), player leg first."""
    legs = [leg for leg in legs if usable(leg, settings)]
    if not legs:
        return []
    legs.sort(key=_scope)  # combinations() then yields pairs already ordered player → team → game
    max_indep_p = 1 / settings["min_indep_x"]
    out = []
    for a, b in combinations(legs, 2):
        if a.event_ticker == b.event_ticker or a.p * b.p > max_indep_p:
            continue
        rel = relationship(a, b)
        if rel is None:
            continue
        rho = model.atlas_rho(a, b)
        if rho is None or rho < settings["min_atlas_rho"]:
            continue
        out.append((template_id(a, b, rel), a, b))
    if legs[0].sport == "MLB":
        out += _pitcher_opponent_unders(legs, teams, model, settings, max_indep_p)
    return out


def _pitcher_opponent_unders(legs, teams, model, settings, max_indep_p) -> list[tuple[str, Leg, Leg]]:
    """The one opposing-team shape: a strikeout-heavy start + the other team UNDER its total."""
    out = []
    for p in (leg for leg in legs if leg.kind in ("ks", "outs")):
        for t in (leg for leg in legs if leg.kind == "team_total" and leg.team and leg.team != p.team):
            under = t.negate()
            if not usable(under, settings) or p.p * under.p > max_indep_p:
                continue
            rho = model.atlas_rho(p, under)
            if rho is not None and rho >= settings["min_atlas_rho"]:
                out.append(("mlb_sp_opp_under", p, under))
    return out
