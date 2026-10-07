"""Learn how Kalshi market makers price each combo type, so quote slots go where they're soft.

Fit on our own logged quotes: log(kalshi_x) = a + b·log(indep_x) + c·log(fair_x), per template
(falling back to a per-sport fit). A candidate's predicted EV is then q · predicted_kalshi_x − 1.
"""

from collections import defaultdict

import numpy as np

from app import db

LOOKBACK_DAYS = 21
MIN_FIT = 30  # quotes needed before trusting a fit


def _design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[1.0, np.log(r["indep_x"]), np.log(r["fair_x"])] for r in rows])
    y = np.array([np.log(r["kalshi_x_after_fees"]) for r in rows])
    return x, y


class QuotePolicy:
    def __init__(self, quoted: list[dict]):
        self.rows = quoted
        self.count = defaultdict(int)
        by_template, by_sport = defaultdict(list), defaultdict(list)
        for r in quoted:
            self.count[r["template"]] += 1
            by_template[r["template"]].append(r)
            by_sport[r["sport"]].append(r)
        self.fits = {}
        for key, rows in [*by_template.items(), *by_sport.items()]:
            if len(rows) >= MIN_FIT:
                x, y = _design(rows)
                self.fits[key] = np.linalg.lstsq(x, y, rcond=None)[0]

    @classmethod
    def from_db(cls) -> "QuotePolicy":
        return cls(db.recent_quoted_rows(LOOKBACK_DAYS))

    @property
    def ready(self) -> bool:
        return bool(self.fits)

    def predicted_x(self, row: dict) -> float | None:
        coef = self.fits.get(row["template"])
        if coef is None:
            coef = self.fits.get(row["sport"])
        if coef is None:
            return None
        return float(np.exp(coef @ [1.0, np.log(row["indep_x"]), np.log(row["fair_x"])]))

    def predicted_ev(self, row: dict) -> float | None:
        x = self.predicted_x(row)
        return None if x is None else row["q"] * x - 1

    def stats(self) -> list[dict]:
        """Per combo type: how makers have priced it vs our fair value."""
        by_template = defaultdict(list)
        for r in self.rows:
            by_template[r["template"]].append(r)
        out = []
        for t, rows in by_template.items():
            ratio = [r["kalshi_x_after_fees"] / r["fair_x"] for r in rows]
            out.append({"template": t, "template_name": rows[0].get("template_name", t), "sport": rows[0]["sport"],
                        "quotes": len(rows), "median_kalshi_over_fair": float(np.median(ratio)),
                        "pct_pos_ev": float(np.mean([r["ev"] > 0 for r in rows])),
                        "fitted": t in self.fits})
        return sorted(out, key=lambda s: -s["median_kalshi_over_fair"])
