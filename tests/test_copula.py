import numpy as np
import pytest
from scipy.stats import multivariate_normal
from scipy.special import ndtri

from app.model.copula import bvn_cdf, joint_prob, rho_for_joint, tetrachoric


@pytest.mark.parametrize("h,k,r", [(-2.0, 0.3, 0.4), (1.1, -0.7, -0.5), (-3.0, -2.5, 0.8), (0.0, 0.5, 0.2)])
def test_bvn_matches_scipy(h, k, r):
    ref = multivariate_normal([0, 0], [[1, r], [r, 1]]).cdf([h, k])
    assert bvn_cdf(h, k, r) == pytest.approx(ref, rel=1e-6, abs=1e-9)


def test_independence_and_comonotone_limits():
    assert joint_prob(0.035, 0.5, 0.0) == pytest.approx(0.035 * 0.5)
    assert joint_prob(0.035, 0.5, 0.999) == pytest.approx(0.035, rel=1e-3)


def test_rho_round_trip():
    assert rho_for_joint(0.1, 0.3, joint_prob(0.1, 0.3, 0.37)) == pytest.approx(0.37, abs=1e-4)


def test_tetrachoric_recovers_latent_rho():
    rng = np.random.default_rng(1)
    z = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=200_000)
    a = z[:, 0] < ndtri(0.1)
    b = z[:, 1] < ndtri(0.4)
    fit = tetrachoric(a, b)
    assert fit["rho"] == pytest.approx(0.5, abs=0.03)
    assert fit["rho_lo"] < fit["rho"] < fit["rho_hi"]
