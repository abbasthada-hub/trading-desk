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
