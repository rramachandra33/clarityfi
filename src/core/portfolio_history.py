"""'What if you had held today's mix?': a portfolio value history built from each holding's price history.

Pure functions, no network. This is a HYPOTHETICAL replay of the CURRENT holdings over a past window: it ignores when
positions were actually bought or sold, taxes and fees. Every sentence produced here is built from computed numbers.
"""
from __future__ import annotations

from src.core.market_calc import relative_performance
from src.core.models import Holding, PricePoint, TrendSummary

MIN_COMMON_DAYS = 10


def value_series(holdings: list[Holding], histories: dict[str, list[PricePoint]], cash: float = 0.0
                 ) -> tuple[list[PricePoint], list[str], float]:
    """Daily value of today's holdings (+ constant cash) on the dates every included holding has a price.

    Returns (series, excluded_tickers, coverage_pct). Holdings with no usable history are excluded; `coverage_pct`
    is the share of the latest invested value that the included holdings represent (100 when nothing is excluded).
    """
    usable = [h for h in holdings if h.shares > 0 and len(histories.get(h.ticker, [])) >= 2]
    excluded = sorted({h.ticker for h in holdings if h.shares > 0} - {h.ticker for h in usable})
    if not usable:
        return [], excluded, 0.0
    by_day = {h.ticker: {p.day: p.close for p in histories[h.ticker]} for h in usable}
    days = sorted(set.intersection(*(set(d) for d in by_day.values())))
    if len(days) < MIN_COMMON_DAYS:
        return [], sorted(excluded + [h.ticker for h in usable]), 0.0
    series = [PricePoint(day=d, close=round(sum(h.shares * by_day[h.ticker][d] for h in usable) + cash, 4)) for d in days]
    latest = {h.ticker: h.shares * histories[h.ticker][-1].close for h in usable}
    unknown = sum(h.cost_basis or 0.0 for h in holdings if h.ticker in excluded)
    total = sum(latest.values()) + unknown
    coverage = round(sum(latest.values()) / total * 100, 1) if total > 0 else 100.0
    return series, excluded, coverage


def _worse_drop(a: float, b: float) -> str:
    return "smaller" if a > b else "larger"


def interpret(port: TrendSummary, bench: TrendSummary | None, bench_name: str | None, *, coverage_pct: float,
              excluded: list[str]) -> list[str]:
    """Plain-language reading of the comparison: descriptive, never a recommendation."""
    out: list[str] = []
    if port.return_3m_pct is not None:
        line = f"Had you held today's mix for the last 3 months, it would have returned {port.return_3m_pct:+.1f}%"
        if bench is not None and bench.return_3m_pct is not None and bench_name:
            gap = port.return_3m_pct - bench.return_3m_pct
            line += (f", compared with {bench.return_3m_pct:+.1f}% for {bench_name}: {abs(gap):.1f} percentage points "
                     f"{'ahead' if gap > 0 else 'behind'}.")
        else:
            line += "."
        out.append(line)
    elif port.return_1m_pct is not None:
        out.append(f"Had you held today's mix for the last month, it would have returned {port.return_1m_pct:+.1f}%.")
    if bench is not None and bench_name:
        drop_p, drop_b = port.max_drawdown_pct, bench.max_drawdown_pct
        if drop_p is not None and drop_b is not None and (drop_p < -1 or drop_b < -1):
            out.append(f"Its largest peak-to-trough fall in this window was {abs(drop_p):.1f}%, a {_worse_drop(drop_p, drop_b)} drop "
                       f"than {bench_name}'s {abs(drop_b):.1f}%.")
        rel = relative_performance(port, bench)["volatility_diff_pts"]
        if rel is not None and abs(rel) >= 2:
            out.append(f"Its day-to-day swings were {'larger' if rel > 0 else 'smaller'} than {bench_name}'s "
                       f"(about {port.volatility_annual_pct:.0f}% vs {bench.volatility_annual_pct:.0f}% a year), "
                       f"{'so results were bumpier' if rel > 0 else 'so results were steadier'}.")
    if excluded:
        out.append(f"{', '.join(excluded)} could not be included (no usable price history), so this covers about {coverage_pct:.0f}% of your invested value.")
    out.append("This replays your current holdings over past prices. It ignores when you actually bought, taxes and fees, and it does "
               "not predict future results or recommend any change.")
    return out
