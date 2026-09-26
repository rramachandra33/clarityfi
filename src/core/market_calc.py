"""Pure trend/volatility math over a price history (no network, no LLM).

Everything here describes the PAST. Labels are educational descriptors, not predictions.
"""
from __future__ import annotations

import math
import statistics

from src.core.models import PricePoint, TrendSummary

TRADING_DAYS_1M = 21
TRADING_DAYS_3M = 63
TRADING_DAYS_YEAR = 252


def sma(values: list[float], window: int) -> float | None:
    return round(sum(values[-window:]) / window, 4) if len(values) >= window else None


def pct_change(new: float, old: float) -> float | None:
    return round((new / old - 1) * 100, 2) if old else None


def max_drawdown_pct(values: list[float]) -> float:
    """Largest peak-to-trough decline as a negative percent (0.0 if it never fell)."""
    peak, worst = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = min(worst, v / peak - 1)
    return round(worst * 100, 2)


def annualized_volatility_pct(values: list[float]) -> float | None:
    """Std-dev of daily log returns x sqrt(252), in percent. Needs >= 10 prices."""
    if len(values) < 10:
        return None
    returns = [math.log(b / a) for a, b in zip(values, values[1:]) if a > 0 and b > 0]
    if len(returns) < 9:
        return None
    return round(statistics.stdev(returns) * math.sqrt(TRADING_DAYS_YEAR) * 100, 2)


def trend_label(last: float, sma20: float | None, sma50: float | None) -> str:
    if sma20 is None:
        return "Insufficient history"
    if sma50 is None:
        return "Above short-term average" if last > sma20 else "Below short-term average"
    if last > sma20 > sma50:
        return "Uptrend"
    if last < sma20 < sma50:
        return "Downtrend"
    return "Sideways / mixed"


def compute_trend(symbol: str, history: list[PricePoint]) -> TrendSummary:
    """Summarize a price history. Raises ValueError if fewer than 2 points."""
    points = sorted(history, key=lambda p: p.day)
    if len(points) < 2:
        raise ValueError("At least 2 price points are required for a trend summary")
    closes = [p.close for p in points]
    last = closes[-1]

    s20, s50 = sma(closes, 20), sma(closes, 50)
    r1m = pct_change(last, closes[-TRADING_DAYS_1M - 1]) if len(closes) > TRADING_DAYS_1M else None
    r3m = pct_change(last, closes[-TRADING_DAYS_3M - 1]) if len(closes) > TRADING_DAYS_3M else None
    vol = annualized_volatility_pct(closes)
    dd = max_drawdown_pct(closes)
    label = trend_label(last, s20, s50)

    notes = ["Based on past prices only; trends can reverse and do not predict future returns."]
    if vol is not None:
        if vol >= 40:
            notes.append(f"Prices have been very volatile (about {vol:.0f}% annualized).")
        elif vol >= 25:
            notes.append(f"Prices have been fairly volatile (about {vol:.0f}% annualized).")
        elif vol < 15:
            notes.append(f"Prices have been relatively steady (about {vol:.0f}% annualized).")
    if dd <= -20:
        notes.append(f"It fell as much as {abs(dd):.0f}% from a peak during this period, a reminder that drawdowns happen.")
    if len(closes) < 50:
        notes.append("Limited history available, so longer-term averages are not shown.")

    return TrendSummary(
        symbol=symbol,
        points=len(closes),
        last_close=round(last, 4),
        sma_20=s20,
        sma_50=s50,
        return_1m_pct=r1m,
        return_3m_pct=r3m,
        volatility_annual_pct=vol,
        max_drawdown_pct=dd,
        period_high=round(max(closes), 4),
        period_low=round(min(closes), 4),
        trend_label=label,
        notes=notes,
    )


# ------------------------------------------------------------------------------ market context
# Broad market segments shown in the overview. ETFs are used as proxies because indexes themselves are not priced.
SEGMENTS: list[tuple[str, str, str]] = [
    ("SPY", "US large companies", "S&P 500 proxy"),
    ("QQQ", "US tech & growth", "Nasdaq-100 proxy"),
    ("IWM", "US small companies", "Russell 2000 proxy"),
    ("VXUS", "International stocks", "developed + emerging markets ex-US"),
    ("BND", "US bonds", "broad investment-grade bond market"),
]
DEFAULT_BENCHMARK = "SPY"


def rebased(series: dict[str, list[PricePoint]]) -> dict[str, list[dict]]:
    """Put several histories on a common footing: only dates every series has, each starting at 100.

    Lets a chart show 'what 100 would have become' for a symbol and its benchmark on the same axis.
    """
    if not series:
        return {}
    common = set.intersection(*({p.day for p in pts} for pts in series.values()))
    out: dict[str, list[dict]] = {}
    for name, pts in series.items():
        kept = sorted((p for p in pts if p.day in common), key=lambda p: p.day)
        if len(kept) < 2 or kept[0].close <= 0:
            return {}
        base = kept[0].close
        out[name] = [{"day": p.day.isoformat(), "value": round(p.close / base * 100, 2)} for p in kept]
    return out


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else round(a - b, 2)


def relative_performance(trend: TrendSummary, bench: TrendSummary) -> dict[str, float | None]:
    """Percentage-point difference between a symbol and its benchmark over 1 and 3 months."""
    return {"return_1m_diff_pts": _diff(trend.return_1m_pct, bench.return_1m_pct),
            "return_3m_diff_pts": _diff(trend.return_3m_pct, bench.return_3m_pct),
            "volatility_diff_pts": _diff(trend.volatility_annual_pct, bench.volatility_annual_pct)}


def _vol_word(vol: float) -> str:
    return "low" if vol < 12 else "moderate" if vol < 25 else "high" if vol < 40 else "very high"


def interpret(trend: TrendSummary, bench: TrendSummary | None = None, bench_name: str | None = None) -> list[str]:
    """Plain-language reading of the numbers. Descriptive and cautious: no forecasts, no buy/sell language.

    Every sentence is built from computed values, so nothing here can state a figure the data does not support.
    """
    out: list[str] = []
    last, s20, s50 = trend.last_close, trend.sma_20, trend.sma_50
    if s20 is not None and s50 is not None:
        if last > s20 > s50:
            out.append("The price is above its 20-day and 50-day averages, and the shorter average is above the longer one. "
                       "That pattern is what people usually call an uptrend: recent prices have been higher than older ones.")
        elif last < s20 < s50:
            out.append("The price is below both averages, and the shorter one is below the longer one, the pattern usually called "
                       "a downtrend: recent prices have been lower than older ones.")
        else:
            out.append("The price and its 20-day and 50-day averages point in different directions, so recent movement is mixed rather "
                       "than a clear trend.")
    if trend.period_high > 0:
        gap = (last / trend.period_high - 1) * 100
        if gap > -1:
            out.append("It is at or very near its highest close of this period.")
        else:
            out.append(f"It is {abs(gap):.1f}% below its highest close of this period.")
    if trend.volatility_annual_pct is not None:
        vol = trend.volatility_annual_pct
        out.append(f"Day-to-day swings have been {_vol_word(vol)} (about {vol:.0f}% a year). Higher volatility means bigger ups and downs, "
                   "in both directions.")
    if bench is not None and bench_name:
        rel = relative_performance(trend, bench)
        d3, d1 = rel["return_3m_diff_pts"], rel["return_1m_diff_pts"]
        if d3 is not None:
            verb = "ahead of" if d3 > 0 else "behind"
            out.append(f"Over 3 months it is {abs(d3):.1f} percentage points {verb} {bench_name}"
                       + (f" ({trend.return_3m_pct:+.1f}% vs {bench.return_3m_pct:+.1f}%)." if trend.return_3m_pct is not None and bench.return_3m_pct is not None else "."))
        elif d1 is not None:
            verb = "ahead of" if d1 > 0 else "behind"
            out.append(f"Over 1 month it is {abs(d1):.1f} percentage points {verb} {bench_name}.")
        vd = rel["volatility_diff_pts"]
        if vd is not None and abs(vd) >= 3:
            out.append(f"It has been {'more' if vd > 0 else 'less'} volatile than {bench_name} ({abs(vd):.0f} points {'higher' if vd > 0 else 'lower'}).")
    out.append("This describes the past only. Past movement does not predict future results, and it is not a recommendation.")
    return out
