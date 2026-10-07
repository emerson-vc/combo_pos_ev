"""Fees, EV and Kelly sizing for a $1-payout YES contract."""

from collections import defaultdict

from app import config


def fee_per_contract(price: float, rate: float = config.FEE_RATE) -> float:
    """Kalshi taker fee is ceil(rate * C * P * (1-P)) cents-rounded per order; per contract ≈ rate*P*(1-P)."""
    return rate * price * (1 - price)


def all_in_cost(price: float, rate: float = config.FEE_RATE) -> float:
    return price + fee_per_contract(price, rate)


def ev_pct(q: float, cost: float) -> float:
    """Expected return per $1 staked: q * (1/cost) - 1."""
    return q / cost - 1


def kelly_fraction(q: float, cost: float) -> float:
    """Full-Kelly bankroll fraction for paying `cost` to win $1 with probability q."""
    if cost <= 0 or cost >= 1:
        return 0.0
    return max(0.0, (q - cost) / (1 - cost))


def size_stakes(rows: list[dict], settings: dict) -> None:
    """Fill row['stake'] in place: fractional Kelly, per-bet cap, then per-game cap (pro-rata)."""
    bankroll = settings["bankroll"]
    per_bet_cap = settings["max_bet_pct"] * bankroll
    per_game_cap = settings["max_game_pct"] * bankroll
    by_game = defaultdict(list)
    for r in rows:
        k = r.get("kelly_full") or 0.0
        r["stake_uncapped"] = settings["kelly_fraction"] * k * bankroll
        r["stake"] = min(r["stake_uncapped"], per_bet_cap)
        if r["stake"] > 0:
            by_game[r["game"]].append(r)
    for game_rows in by_game.values():
        total = sum(r["stake"] for r in game_rows)
        if total > per_game_cap:
            scale = per_game_cap / total
            for r in game_rows:
                r["stake"] *= scale
    for r in rows:
        r["stake"] = round(r["stake"], 2)
