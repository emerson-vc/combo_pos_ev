import pytest

from app.pricing import all_in_cost, ev_pct, fee_per_contract, kelly_fraction, size_stakes

SETTINGS = {"bankroll": 1000.0, "kelly_fraction": 0.25, "max_bet_pct": 0.02, "max_game_pct": 0.05}


def test_fee_matches_kalshi_formula():
    assert fee_per_contract(0.5, 0.07) == pytest.approx(0.0175)
    assert all_in_cost(0.02, 0.07) == pytest.approx(0.02 + 0.07 * 0.02 * 0.98)


def test_ev_screenshot_combo():
    # Lamb 3+ TD + DAL o28.5 paid 49.02x; model fair ~34x -> q = 1/34.
    cost = 1 / 49.02
    assert ev_pct(1 / 34, cost) == pytest.approx(49.02 / 34 - 1)


def test_kelly_zero_when_no_edge():
    assert kelly_fraction(0.02, 0.025) == 0.0
    assert kelly_fraction(0.5, 0.4) == pytest.approx((0.5 - 0.4) / 0.6)


def test_stakes_capped_per_bet_and_per_game():
    rows = [{"game": "G1", "kelly_full": 0.5} for _ in range(4)] + [{"game": "G2", "kelly_full": 0.01}]
    size_stakes(rows, SETTINGS)
    g1 = [r["stake"] for r in rows if r["game"] == "G1"]
    assert all(s == pytest.approx(12.5) for s in g1)  # 4 x $20 cap scaled to the $50 game cap
    assert rows[-1]["stake"] == pytest.approx(2.5)  # 0.25 * 0.01 * 1000
