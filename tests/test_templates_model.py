import numpy as np
import pandas as pd
import pytest

from app.config import DEFAULT_SETTINGS
from app.legs import Leg
from app.model.joint import Model
from app.templates import pairs_for_game


def leg(kind, team, mid, player=None, threshold=None, sport="NFL"):
    return Leg(sport, "26OCT08TBDAL", "S", "E-" + kind, f"T-{kind}-{team}-{player}-{threshold}", kind, team,
               player, threshold, mid - 0.005, mid + 0.005, f"{player or team} {kind} {threshold}")


def test_templates_same_team_only():
    legs = [leg("td", "DAL", 0.3, "A", 0.5), leg("team_total", "DAL", 0.5, threshold=28.5),
            leg("team_total", "TB", 0.5, threshold=20.5), leg("pass_yds", "DAL", 0.4, "Q", 274.5),
            leg("team_spread", "DAL", 0.4, threshold=6.5)]
    pairs = pairs_for_game(legs, ["TB", "DAL"], DEFAULT_SETTINGS)
    assert all(a.team == b.team for _, a, b in pairs)
    templates = {t for t, *_ in pairs}
    assert templates == {"nfl_skill_tt", "nfl_spread_tt", "nfl_spread_player", "nfl_qb_rec"}
    assert all(a.kind not in ("team_win", "team_spread", "team_total") or b.kind in ("team_win", "team_spread", "team_total")
               for _, a, b in pairs)  # player leg always first


@pytest.fixture
def synthetic_model(tmp_path):
    """Correlated synthetic NFL history: WR1 TDs rise with team points."""
    rng = np.random.default_rng(0)
    n = 4000
    pts_resid = rng.normal(0, 9, n)
    td = rng.poisson(np.exp(-0.9 + 0.06 * pts_resid))
    teams = pd.DataFrame({"game_id": np.arange(n), "team": "DAL", "won": (pts_resid > 0).astype(int),
                          "margin_resid": pts_resid + rng.normal(0, 5, n), "pts_resid": pts_resid})
    players = pd.DataFrame({"game_id": np.arange(n), "team": "DAL", "player_id": "p1", "name": "Wide Out",
                            "role": "WR1", "season": 2025, "week": 1, "td": td, "rec_yds": 0, "rush_yds": 0,
                            "rec": 0, "pass_yds": 0, "pass_td": 0})
    teams.to_parquet(tmp_path / "nfl_teams.parquet")
    players.to_parquet(tmp_path / "nfl_players.parquet")
    return Model(tmp_path)


def test_model_finds_positive_correlation(synthetic_model):
    a = leg("td", "DAL", 0.10, "Wide Out", 1.5)
    b = leg("team_total", "DAL", 0.50, threshold=28.5)
    est = synthetic_model.estimate(a, b)
    assert est.role_a == "WR1"
    assert est.rho > 0.2
    assert est.q > a.mid * b.mid
    assert est.q_lo <= est.q <= est.q_hi


def test_win_and_spread_is_subset(synthetic_model):
    win, spread = leg("team_win", "DAL", 0.7), leg("team_spread", "DAL", 0.4, threshold=6.5)
    est = synthetic_model.estimate(win, spread)
    assert est.method == "subset" and est.q == pytest.approx(0.4)
