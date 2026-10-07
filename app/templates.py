"""Generate same-team 2-leg combos worth pricing, per the user's chosen templates."""

from itertools import product

from app.legs import Leg

MLB_BATTER = {"hr", "hit", "rbi", "tb", "hrr"}
NFL_SKILL = {"td", "rec_yds", "rush_yds", "rec"}
NFL_QB = {"pass_yds", "pass_td"}

TEMPLATE_NAMES = {
    "mlb_bat_win": "Batter prop + team wins",
    "mlb_bat_tt": "Batter prop + team total over",
    "mlb_win_tt": "Team wins + team total over",
    "nfl_skill_tt": "Skill player + team total over",
    "nfl_spread_tt": "Spread + team total over",
    "nfl_spread_player": "Spread + player prop",
    "nfl_qb_rec": "QB + receiver stack",
}


def usable(leg: Leg, settings: dict) -> bool:
    if leg.p_source == "ladder":  # thin long shot priced off a liquid lower rung
        return settings["min_ladder_prob"] <= leg.p <= settings["max_leg_prob"]
    return (leg.yes_ask > 0
            and settings["min_leg_prob"] <= leg.p <= settings["max_leg_prob"]
            and leg.spread <= settings["max_leg_spread"])


def pairs_for_game(legs: list[Leg], teams: list[str], settings: dict) -> list[tuple[str, Leg, Leg]]:
    """Same-team pairs; the player leg always comes first (the model expects that order)."""
    legs = [leg for leg in legs if usable(leg, settings)]
    out = []
    for team in teams:
        mine = [leg for leg in legs if leg.team == team]

        def kind(*kinds):
            return [leg for leg in mine if leg.kind in kinds]

        wins, spreads, totals = kind("team_win"), kind("team_spread"), kind("team_total")
        if legs and legs[0].sport == "MLB":
            bats = kind(*MLB_BATTER)
            out += [("mlb_bat_win", a, b) for a, b in product(bats, wins)]
            out += [("mlb_bat_tt", a, b) for a, b in product(bats, totals)]
            # (team wins + run line is rejected by Kalshi: the spread already implies the win)
            out += [("mlb_win_tt", a, b) for a, b in product(wins, totals)]
        else:
            skill, qbs = kind(*NFL_SKILL), kind(*NFL_QB)
            out += [("nfl_skill_tt", a, b) for a, b in product(skill, totals)]
            out += [("nfl_spread_tt", a, b) for a, b in product(spreads, totals)]
            out += [("nfl_spread_player", a, b) for a, b in product(skill + qbs, spreads)]
            out += [("nfl_qb_rec", a, b) for a, b in product(qbs, skill) if a.player != b.player]
    return out
