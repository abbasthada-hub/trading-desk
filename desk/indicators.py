"""Indicators. Pure functions on pandas Series; the backtest and the live runner both import these."""
import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi_wilder(close: pd.Series, n: int = 2) -> pd.Series:
    """Wilder's RSI. Values run 0-100; low means the last n days were mostly down."""
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(ad != 0.0, 100.0)          # no down moves at all -> 100
    out[au.isna() | ad.isna()] = np.nan
    return out


def weekly_closes(close: pd.Series) -> pd.Series:
    """Last close of each calendar week (weeks end Friday). Index = last trading day of that week."""
    idx = pd.DatetimeIndex(close.index)
    wk = idx.to_period("W-FRI")
    last = pd.Series(idx, index=idx).groupby(wk).max()
    return close.loc[last.values]
