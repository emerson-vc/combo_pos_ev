import dataclasses

import numpy as np
import pandas as pd
import pytest

from app.config import DEFAULT_SETTINGS
from app.legs import Leg
from app.model.joint import Model


def leg(kind, team, mid, player=None, threshold=None, sport="NFL"):
    return Leg(sport, "26OCT08TBDAL", "S", "E-" + kind, f"T-{kind}-{team}-{player}-{threshold}", kind, team,
               player, threshold, mid - 0.005, mid + 0.005, f"{player or team} {kind} {threshold}")


class StubModel:
    """atlas_rho stand-in: everything correlates except pairs involving a 'junk' player."""
    def atlas_rho(self, a, b):
        return 0.0 if "junk" in (a.player or "", b.player or "") else 0.5


def test_search_relationships_and_screens():
    from app.pairs import candidate_pairs, relationship
    td = leg("td", "DAL", 0.3, "Runner", 0.5)
    yds = dataclasses.replace(leg("rush_yds", "DAL", 0.3, "Runner", 69.5), event_ticker="E-yds")
    team_tt = leg("team_total", "DAL", 0.3, threshold=28.5)
    opp_tt = leg("team_total", "TB", 0.3, threshold=20.5)
    game = leg("game_total", None, 0.3, threshold=48.5)
    mate = dataclasses.replace(leg("rec_yds", "DAL", 0.3, "Catcher", 79.5), event_ticker="E-rec")
    junk = dataclasses.replace(leg("rec", "DAL", 0.3, "junk", 4.5), event_ticker="E-junk")
    fav = dataclasses.replace(leg("team_win", "DAL", 0.8), event_ticker="E-win")
    assert relationship(td, yds) == "same_player"
    assert relationship(td, mate) == "teammates"
    assert relationship(td, team_tt) == "player_team"
    assert relationship(td, opp_tt) is None  # opposing side not searched
    assert relationship(team_tt, game) == "game_team"
    pairs = candidate_pairs([td, yds, team_tt, opp_tt, game, mate, junk, fav], ["TB", "DAL"], StubModel(),
                            DEFAULT_SETTINGS)
    ids = {t for t, *_ in pairs}
    assert "nfl:same_player:td+rush_yds" in ids and "nfl:player_team:td+team_total" in ids
    assert not any("junk" in (a.player or "") + (b.player or "") for _, a, b in pairs)  # atlas screen
    assert not any(b is opp_tt or a is opp_tt for _, a, b in pairs if a.kind in ("td", "rush_yds"))
    assert all(_scope_ok(a, b) for _, a, b in pairs)


def _scope_ok(a, b):
    from app.pairs import _scope
    return _scope(a) <= _scope(b)  # player leg first, game leg last


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
