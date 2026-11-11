"""The two rule sets. Nothing here knows about brokers or files: prices in, decisions out.

Pullback sleeve (holds 2-10 trading days)
    enter : close > SMA(trend_days)  and  RSI(rsi_days) < rsi_entry
    exit  : close > SMA(exit_days)   or   held >= max_hold_days
    Decisions use the completed daily bar; orders fill at the next open.

Trend sleeve (holds weeks to months)
    an asset is ON when its weekly close is above its SMA over sma_weeks weeks.
    ON assets get an equal slice of the sleeve; OFF slices sit in cash.
    Checked once a week on the last trading day of the week.
"""
from dataclasses import dataclass
import pandas as pd
from .indicators import sma, rsi_wilder, weekly_closes


@dataclass(frozen=True)
class PullbackParams:
    trend_days: int = 200
    rsi_days: int = 2
    rsi_entry: float = 10.0
    exit_days: int = 5
    max_hold_days: int = 10


@dataclass(frozen=True)
class TrendParams:
    sma_weeks: int = 40
    drift_band: float = 0.25     # resize a slice when it drifts 25% away from its target weight


def pullback_frame(close: pd.Series, p: PullbackParams) -> pd.DataFrame:
    """Per-day entry/exit conditions for one instrument (no position state)."""
    f = pd.DataFrame({"close": close})
    f["trend_ok"] = close > sma(close, p.trend_days)
    f["rsi"] = rsi_wilder(close, p.rsi_days)
    f["entry"] = f["trend_ok"] & (f["rsi"] < p.rsi_entry)
    f["exit"] = close > sma(close, p.exit_days)
    return f


def pullback_decision(close: pd.Series, in_position: bool, days_held: int, p: PullbackParams) -> str:
    """Decision for the latest completed bar: 'enter', 'exit' or 'hold'."""
    f = pullback_frame(close, p)
    last = f.iloc[-1]
    if pd.isna(last["rsi"]) or pd.isna(sma(close, p.trend_days).iloc[-1]):
        return "hold"
    if in_position:
        return "exit" if (bool(last["exit"]) or days_held >= p.max_hold_days) else "hold"
    return "enter" if bool(last["entry"]) else "hold"


def trend_on(close: pd.Series, p: TrendParams) -> bool:
    """Is this asset above its weekly moving average as of the last completed week?"""
    w = weekly_closes(close)
    m = sma(w, p.sma_weeks)
    if pd.isna(m.iloc[-1]):
        return False
    return bool(w.iloc[-1] > m.iloc[-1])


def trend_targets(closes: dict, p: TrendParams, sleeve_weight: float) -> dict:
    """Target portfolio weight per trend asset (0 when OFF)."""
    n = len(closes)
    return {sym: (sleeve_weight / n if trend_on(c, p) else 0.0) for sym, c in closes.items()}


def needs_resize(current_w: float, target_w: float, band: float) -> bool:
    """Trade a trend slice when it flips on/off, or when it drifts outside the band."""
    if target_w == 0.0:
        return current_w > 0.0
    if current_w == 0.0:
        return True
    return abs(current_w - target_w) > band * target_w


# ----------------------------------------------------------------------------- momentum sleeve
import datetime as _dt

MOMENTUM_WEEKS = (13, 26, 52)
RERANK_EPOCH = _dt.date(2026, 1, 2)      # a Friday; re-rank weeks are counted from here


def week_friday(d: _dt.date) -> _dt.date:
    return d + _dt.timedelta(days=4 - d.weekday())


def rerank_due(d: _dt.date, every_weeks: int = 4) -> bool:
    """True when the week containing d is a re-rank week. Depends only on the calendar."""
    return ((week_friday(d) - RERANK_EPOCH).days // 7) % every_weeks == 0


def last_rerank_friday(d: _dt.date, every_weeks: int = 4) -> _dt.date:
    f = week_friday(d)
    while not rerank_due(f, every_weeks):
        f -= _dt.timedelta(days=7)
    return f


def momentum_picks_weekly(w: pd.DataFrame, n: int = 3, sma_weeks: int = 40) -> list:
    """w: weekly closes, one column per fund, last row = the re-rank week.
    Score = average of the 13-, 26- and 52-week returns. Keep the top n, and of those
    only the ones above their own weekly average. Returns the symbols to hold."""
    if len(w) < max(MOMENTUM_WEEKS) + 1:
        return []
    score = sum(w.iloc[-1] / w.iloc[-1 - k] - 1 for k in MOMENTUM_WEEKS) / len(MOMENTUM_WEEKS)
    above = w.iloc[-1] > w.iloc[-sma_weeks:].mean()
    top = score.sort_values(ascending=False, kind="mergesort").index[:n]
    return [s for s in top if bool(above[s])]


def momentum_targets(closes: dict, today: _dt.date, sleeve_weight: float, n: int = 3,
                     every_weeks: int = 4, sma_weeks: int = 40) -> dict:
    """Target weight per fund from the momentum sleeve, as decided at the most recent re-rank week.
    Recomputed from prices each time, so no stored state is needed."""
    cutoff = pd.Timestamp(last_rerank_friday(today, every_weeks))
    w = pd.DataFrame({s: weekly_closes(c[c.index <= cutoff]) for s, c in closes.items()}).dropna()
    picks = momentum_picks_weekly(w, n, sma_weeks)
    return {s: (sleeve_weight / n if s in picks else 0.0) for s in closes}
