"""Gaussian copula for two binary legs.

Each leg is YES when a latent standard normal falls below Φ⁻¹(p). The latent correlation ρ
(the tetrachoric correlation) is estimated from historical 2x2 tables, then reapplied to
today's Kalshi marginals: P(A∩B) = Φ2(Φ⁻¹(pA), Φ⁻¹(pB); ρ).
"""

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri, owens_t

RHO_MAX = 0.999


def bvn_cdf(h: float, k: float, rho: float) -> float:
    """P(X <= h, Y <= k) for standard bivariate normal with correlation rho (Owen's T form)."""
    if rho >= RHO_MAX:
        return float(ndtr(min(h, k)))
    if rho <= -RHO_MAX:
        return float(max(0.0, ndtr(h) + ndtr(k) - 1))
    # Owen's formula is singular at h=0 or k=0; nudging is far below any precision we use.
    h = h if abs(h) > 1e-10 else 1e-10
    k = k if abs(k) > 1e-10 else 1e-10
    s = math.sqrt(1 - rho * rho)
    a_h = (k - rho * h) / (h * s)
    a_k = (h - rho * k) / (k * s)
    beta = 0.0 if h * k > 0 else 0.5
    val = 0.5 * ndtr(h) + 0.5 * ndtr(k) - owens_t(h, a_h) - owens_t(k, a_k) - beta
    return float(min(max(val, 0.0), min(ndtr(h), ndtr(k))))


def joint_prob(p_a: float, p_b: float, rho: float) -> float:
    return bvn_cdf(float(ndtri(p_a)), float(ndtri(p_b)), rho)


def rho_for_joint(p_a: float, p_b: float, p_ab: float) -> float:
    """Solve for the copula ρ that reproduces joint probability p_ab given marginals."""
    lo_bound = max(0.0, p_a + p_b - 1)
    hi_bound = min(p_a, p_b)
    eps = 1e-9
    if p_ab <= joint_prob(p_a, p_b, -RHO_MAX) + eps or p_ab <= lo_bound + eps:
        return -RHO_MAX
    if p_ab >= joint_prob(p_a, p_b, RHO_MAX) - eps or p_ab >= hi_bound - eps:
        return RHO_MAX
    return brentq(lambda r: joint_prob(p_a, p_b, r) - p_ab, -RHO_MAX, RHO_MAX, xtol=1e-6)


def tetrachoric(a: np.ndarray, b: np.ndarray) -> dict:
    """Estimate ρ (with ~95% CI) from paired boolean outcomes.

    Adds a 0.5 continuity correction to every cell when any cell is empty, which keeps rare
    tail events (3+ TDs, 2+ HRs) from producing ρ = ±1 off a handful of games.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n11 = float(np.sum(a & b))
    n10 = float(np.sum(a & ~b))
    n01 = float(np.sum(~a & b))
    n00 = float(np.sum(~a & ~b))
    if min(n11, n10, n01, n00) == 0:
        n11, n10, n01, n00 = n11 + 0.5, n10 + 0.5, n01 + 0.5, n00 + 0.5
    n = n11 + n10 + n01 + n00
    pa, pb, pab = (n11 + n10) / n, (n11 + n01) / n, n11 / n
    rho = rho_for_joint(pa, pb, pab)
    se = math.sqrt(pab * (1 - pab) / n)
    rho_lo = rho_for_joint(pa, pb, max(pab - 1.96 * se, 1e-9))
    rho_hi = rho_for_joint(pa, pb, min(pab + 1.96 * se, min(pa, pb)))
    return {"rho": rho, "rho_lo": rho_lo, "rho_hi": rho_hi, "n": int(len(a)), "n11": int(np.sum(a & b)),
            "pa_hist": pa, "pb_hist": pb, "pab_hist": pab}
