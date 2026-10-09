"""Leg usability rules and the pinned combo type (pair generation lives in app/pairs.py)."""

from app.legs import Leg

# The user's hand-found edge (e.g. Machado 2+ HR + SD ML paid 133x): always quoted, see scan.py.
PINNED_TEMPLATE = "mlb_hr_win"


def usable(leg: Leg, settings: dict) -> bool:
    if leg.p_source in ("ladder", "curve"):  # thin rung priced off its market's liquid rungs
        return settings["min_ladder_prob"] <= leg.p <= settings["max_leg_prob"]
    # A market-priced leg nobody is bidding on has no trustworthy probability.
    return (leg.yes_bid > 0 and leg.yes_ask > 0
            and settings["min_leg_prob"] <= leg.p <= settings["max_leg_prob"]
            and leg.spread <= settings["max_leg_spread"])
