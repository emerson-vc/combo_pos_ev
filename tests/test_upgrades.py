import math

import numpy as np
import pytest

from app.legs import Leg
from app.model.ladder import Ladder, is_thin, solve_lambda, tail
from app.quote_policy import QuotePolicy
from app.scan import pick_diverse


def rung(k, bid, ask, player="Slugger", kind="hr"):
    return Leg("MLB", "G", "KXMLBHR", "E", f"T-{player}-{kind}-{k}", kind, "LAD", player, k - 0.5, bid, ask,
               f"{player}: {k}+")


def test_poisson_tail_and_inverse():
    lam = solve_lambda(0.2, 1)
    assert lam == pytest.approx(-math.log(0.8))
    assert tail(lam, 2) == pytest.approx(1 - math.exp(-lam) * (1 + lam))


def test_thin_detection():
    assert is_thin(rung(2, 0.00, 0.01))
    assert is_thin(rung(2, 0.01, 0.04))  # 3c gap on a 2.5c midpoint
    assert not is_thin(rung(1, 0.20, 0.22))


def test_ladder_reprices_thin_rung_from_liquid_base():
    ladder = Ladder({("hr", 1, 2): 0.8})
    legs = ladder.apply([rung(1, 0.20, 0.22), rung(2, 0.00, 0.02)])
    base, thin = legs
    assert base.p_source == "market" and base.p == pytest.approx(0.21)
    lam = solve_lambda(0.21, 1)
    assert thin.p_source == "ladder" and thin.p == pytest.approx(0.8 * tail(lam, 2))


def test_ladder_leaves_rung_without_liquid_base():
    legs = Ladder().apply([rung(1, 0.00, 0.02), rung(2, 0.00, 0.01)])
    assert all(leg.p_source == "market" for leg in legs)


def row(template, sport, indep_x, fair_x, kalshi_x, q=None, lift=1.5, ticker="X"):
    q = q or 1 / fair_x
    return {"template": template, "sport": sport, "indep_x": indep_x, "fair_x": fair_x, "q": q, "lift": lift,
            "kalshi_x_after_fees": kalshi_x, "ev": q * kalshi_x - 1, "game": "G",
            "leg_a": {"ticker": ticker + "a", "p": 0.1}, "leg_b": {"ticker": ticker + "b", "p": 0.5}}


def test_quote_policy_learns_maker_pricing():
    rng = np.random.default_rng(0)
    quoted = []
    for i in range(60):  # makers pay 1.2x fair on this template
        fair = float(rng.uniform(10, 80))
        quoted.append(row("soft", "NFL", fair * 2, fair, 1.2 * fair))
    policy = QuotePolicy(quoted)
    assert policy.ready
    assert policy.predicted_ev(row("soft", "NFL", 60, 30, 0)) == pytest.approx(0.2, abs=1e-6)
    assert policy.predicted_ev(row("other", "MLB", 60, 30, 0)) is None


def test_pick_diverse_reserves_exploration_slots():
    quoted = [row("soft", "NFL", 40, 20, 24) for _ in range(40)]
    policy = QuotePolicy(quoted)
    cands = [row("soft", "NFL", 40, 20, 0, ticker=f"s{i}") for i in range(20)]
    cands += [row("new", "NFL", 40, 20, 0, ticker=f"n{i}", lift=3.0) for i in range(5)]
    picked = pick_diverse(cands, 10, policy)
    assert len(picked) == 10
    assert sum(r["template"] == "new" for r in picked) >= 2  # 20% exploration


def test_candle_mid_reads_live_and_archive_formats():
    from app.ingest.mlb_lines import _candle_mid, median_line
    live = {"yes_bid": {"close_dollars": "0.2100"}, "yes_ask": {"close_dollars": "0.2300"}}
    archive = {"yes_bid": {"close": "0.2300"}, "yes_ask": {"close": "0.2500"}, "price": {"close": "0.25"}}
    assert _candle_mid(live) == pytest.approx(0.22)
    assert _candle_mid(archive) == pytest.approx(0.24)
    assert _candle_mid({"price": {"close": "0.3"}}) == pytest.approx(0.3)
    # Team-total ladder generated from a logistic with median 4.5 runs.
    ladder = [(x, 1 / (1 + math.exp(0.8 * (x - 4.5)))) for x in (2.5, 3.5, 4.5, 5.5, 6.5)]
    assert median_line(ladder) == pytest.approx(4.5, abs=1e-6)
