"""Research backtests for the desk. Usage: python -m backtest.backtest <data_dir> [out_json]

Data files expected in <data_dir>:
    SPY_daily.csv, IDX_IXIC_daily.csv, IDX_RUT_daily.csv, IDX_DJI_daily.csv   (date,open,high,low,close,volume)
    <SYM>_weekly.csv for EFA EEM VNQ IEF TLT GLD DBC SHY QQQ                  (date,...,adjusted close,...)

Conventions
    * Signals use completed bars only. Base case fills at the NEXT bar's open (daily) or
      one bar later (weekly), so nothing is traded on information that was not yet known.
    * Costs are charged per side on traded notional.
    * Idle cash earns 0%, which is what a paper account pays.
"""
import json
import sys
import numpy as np
import pandas as pd

from desk.indicators import sma, rsi_wilder
from desk.strategy import PullbackParams, pullback_frame, needs_resize

COST = 0.0005          # 5 basis points per side
TRADING_DAYS = 252


# ----------------------------------------------------------------------------- helpers
def load_daily(path):
    d = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return d[["open", "high", "low", "close"]].astype(float)


def load_weekly(path):
    d = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return d["adjusted close"].astype(float)


def perf(r: pd.Series, periods: int, rf: pd.Series | None = None) -> dict:
    """Stats from a return series. Sharpe uses returns over cash when rf is given."""
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / periods
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(periods)
    ex = r - rf.reindex(r.index).fillna(0) if rf is not None else r
    sharpe = ex.mean() / r.std() * np.sqrt(periods) if r.std() > 0 else np.nan
    dd = eq / eq.cummax() - 1
    return {"years": round(yrs, 1), "cagr": cagr, "vol": vol, "sharpe": sharpe,
            "max_dd": dd.min(), "total": eq.iloc[-1] - 1}


def trade_stats(tr: pd.DataFrame) -> dict:
    if len(tr) == 0:
        return {"trades": 0}
    r = tr["ret"]
    wins, losses = r[r > 0], r[r <= 0]
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 and r.std() > 0 else np.nan
    return {"trades": int(len(r)), "win_rate": float((r > 0).mean()), "avg": float(r.mean()),
            "median": float(r.median()), "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "worst": float(r.min()), "best": float(r.max()),
            "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.inf,
            "t_stat": float(t), "avg_hold": float(tr["hold"].mean())}


def bootstrap_mean_ci(x, n=5000, seed=7):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ----------------------------------------------------------------------------- pullback sleeve
def run_pullback(px: pd.DataFrame, p: PullbackParams, fill: str = "next_open",
                 cost: float = COST, use_trend: bool = True):
    """One instrument, all-in/all-out. fill: next_open | next_close | same_close.
    Returns (daily return series of the sleeve when 100% allocated, trades frame)."""
    f = pullback_frame(px["close"], p)
    entry = f["entry"] if use_trend else (f["rsi"] < p.rsi_entry)
    entry = entry.fillna(False).values
    exit_ = f["exit"].fillna(False).values
    o, c = px["open"].values, px["close"].values
    n = len(px)
    ret = np.zeros(n)
    trades = []
    pos = False
    pend_in = pend_out = False
    ep = np.nan
    ei = -1
    for i in range(n):
        day = 0.0
        # execute what was decided on the previous bar
        if pend_in:
            if fill == "next_open":
                ep = o[i]; day = c[i] / o[i] - 1 - cost
            else:                                   # next_close: in from tonight's close
                ep = c[i]; day = -cost
            pos, pend_in, ei = True, False, i
        elif pend_out:
            if fill == "next_open":
                xp = o[i]; day = o[i] / c[i - 1] - 1 - cost
            else:
                xp = c[i]; day = c[i] / c[i - 1] - 1 - cost
            trades.append((px.index[ei], px.index[i], xp / ep - 1 - 2 * cost, i - ei))
            pos, pend_out = False, False
        elif pos:
            day = c[i] / c[i - 1] - 1
        ret[i] = day
        # decide on this bar's close
        if fill == "same_close":
            if not pos and entry[i]:
                pos, ep, ei = True, c[i], i
                ret[i] -= cost
            elif pos and i > ei and (exit_[i] or i - ei >= p.max_hold_days):
                trades.append((px.index[ei], px.index[i], c[i] / ep - 1 - 2 * cost, i - ei))
                ret[i] -= cost
                pos = False
        else:
            if not pos and not pend_in and entry[i]:
                pend_in = True
            elif pos and not pend_out and (exit_[i] or i - ei >= p.max_hold_days):
                pend_out = True
    tr = pd.DataFrame(trades, columns=["entry", "exit", "ret", "hold"])
    return pd.Series(ret, index=px.index), tr


def pullback_report(px, label, p=PullbackParams(), fill="next_open", start=None, end=None, **kw):
    r, tr = run_pullback(px, p, fill, **kw)
    if start is not None:
        r, tr = r[r.index >= start], tr[tr["entry"] >= start]
    if end is not None:
        r, tr = r[r.index < end], tr[tr["entry"] < end]
    out = {"label": label, **trade_stats(tr), **perf(r, TRADING_DAYS)}
    out["exposure"] = float((r != 0).mean())
    if len(tr) > 5:
        lo, hi = bootstrap_mean_ci(tr["ret"].values)
        out["avg_ci95"] = [lo, hi]
    return out, r, tr


# ----------------------------------------------------------------------------- trend sleeve
def run_trend(weekly: pd.DataFrame, sma_weeks: int = 40, lag: int = 1, band: float = 0.25,
              cost: float = COST, rule: str = "sma"):
    """Equal slices; a slice is invested when its asset is ON, else cash at 0%.
    lag=1: signal at week t is traded at the close of week t+1 (conservative).
    lag=0: traded at the close of week t (optimistic; live trading sits between the two).
    Slices are resized only on a flip or when drift exceeds the band."""
    px = weekly.dropna()
    n_assets = px.shape[1]
    tgt_w = 1.0 / n_assets
    if rule == "sma":
        on = px > px.rolling(sma_weeks).mean()
    else:                                           # 'mom': positive total return over sma_weeks
        on = px / px.shift(sma_weeks) - 1 > 0
    on = on.shift(lag).fillna(False)
    rets = px.pct_change().fillna(0.0)
    start = sma_weeks + lag
    w = np.zeros(n_assets)
    out, turn = [], []
    onv, rv = on.values, rets.values
    for i in range(start, len(px)):
        # weights were set at the close of week i-1; earn week i
        port = float((w * rv[i]).sum())
        gross = w * (1 + rv[i])
        w = gross / (1 + port)                      # drifted weights (cash share drifts too)
        # reset slices as the rules say, at this week's close
        new_w = w.copy()
        for j in range(n_assets):
            target = tgt_w if onv[i, j] else 0.0
            if needs_resize(w[j], target, band):
                new_w[j] = target
        traded = float(np.abs(new_w - w).sum())
        if new_w.sum() > 1.0:                       # never borrow
            new_w = new_w / new_w.sum()
        w = new_w
        out.append(port - traded * cost)
        turn.append(traded)
    idx = px.index[start:]
    return pd.Series(out, index=idx), pd.Series(turn, index=idx), on.iloc[start:]


def buy_hold_equal(weekly: pd.DataFrame, start):
    px = weekly.dropna()
    r = px.pct_change().mean(axis=1)                # weekly rebalanced equal weight
    return r[r.index >= start]


# ----------------------------------------------------------------------------- main
def main(data_dir: str, out_json: str | None = None):
    R = {}
    spy = load_daily(f"{data_dir}/SPY_daily.csv")
    idx = {k: load_daily(f"{data_dir}/IDX_{k}_daily.csv") for k in ["IXIC", "RUT", "DJI"]}

    # 0. data check: are index opens real?
    R["open_equals_prev_close"] = {
        k: float((v["open"].round(2) == v["close"].shift().round(2)).mean()) for k, v in {**idx, "SPY": spy}.items()}

    # 1. where SPY's return comes from: overnight vs intraday
    on = spy["open"] / spy["close"].shift() - 1
    intr = spy["close"] / spy["open"] - 1
    yrs = len(spy) / TRADING_DAYS
    R["overnight_intraday"] = {
        "period": [str(spy.index[0].date()), str(spy.index[-1].date())],
        "buy_hold_total": float(spy["close"].iloc[-1] / spy["close"].iloc[0] - 1),
        "overnight_total": float((1 + on.dropna()).prod() - 1),
        "intraday_total": float((1 + intr).prod() - 1),
        "intraday_total_after_1bp_per_side": float((1 + intr - 0.0002).prod() - 1),
        "intraday_cagr": float((1 + intr).prod() ** (1 / yrs) - 1),
        "overnight_cagr": float((1 + on.dropna()).prod() ** (1 / yrs) - 1),
    }

    # 2. pullback sleeve on SPY: full period, before and after publication (2008 book)
    pub = pd.Timestamp("2009-01-01")
    P = PullbackParams()
    R["pullback_spy"] = {}
    for name, kw in {
        "full_next_open": dict(),
        "pre2009_next_open": dict(end=pub),
        "post2009_next_open": dict(start=pub),
        "post2009_next_close": dict(start=pub, fill="next_close"),
        "post2009_same_close": dict(start=pub, fill="same_close"),
        "post2009_no_trend_filter": dict(start=pub, use_trend=False),
        "post2009_cost_10bps": dict(start=pub, cost=0.0010),
        "last10y_next_open": dict(start=pd.Timestamp("2016-11-10")),
        "last5y_next_open": dict(start=pd.Timestamp("2021-11-10")),
    }.items():
        R["pullback_spy"][name], _, _ = pullback_report(spy, name, P, **kw)

    # buy and hold SPY for the same windows
    sr = spy["close"].pct_change().fillna(0)
    R["spy_buy_hold"] = {"full": perf(sr, TRADING_DAYS), "post2009": perf(sr[sr.index >= pub], TRADING_DAYS)}

    # 3. parameter grid on SPY, post-2009: is the default sitting on a lucky peak?
    grid = []
    for rsi_entry in (5, 10, 15, 20, 25):
        for exit_days in (3, 5, 7, 10):
            for trend_days in (100, 200):
                p = PullbackParams(trend_days=trend_days, rsi_entry=rsi_entry, exit_days=exit_days)
                s, _, _ = pullback_report(spy, "g", p, start=pub)
                grid.append({"rsi_entry": rsi_entry, "exit_days": exit_days, "trend_days": trend_days,
                             "trades": s["trades"], "avg": s["avg"], "win_rate": s["win_rate"],
                             "cagr": s["cagr"], "max_dd": s["max_dd"], "t_stat": s["t_stat"]})
    g = pd.DataFrame(grid)
    R["pullback_grid_post2009"] = {
        "cells": len(g), "cells_positive_avg": int((g["avg"] > 0).sum()),
        "avg_min": float(g["avg"].min()), "avg_median": float(g["avg"].median()), "avg_max": float(g["avg"].max()),
        "cagr_min": float(g["cagr"].min()), "cagr_median": float(g["cagr"].median()), "cagr_max": float(g["cagr"].max()),
        "default_rank_by_avg": int((g["avg"] > g[(g.rsi_entry == 10) & (g.exit_days == 5) & (g.trend_days == 200)]["avg"].iloc[0]).sum() + 1),
        "table": grid}

    # 4. other indexes (price indexes, no dividends; opens unreliable -> close fills only)
    R["pullback_indexes"] = {}
    for k, v in idx.items():
        for name, kw in {"post2009_next_close": dict(start=pub, fill="next_close"),
                         "pre2009_next_close": dict(end=pub, fill="next_close"),
                         "post2009_same_close": dict(start=pub, fill="same_close")}.items():
            R["pullback_indexes"][f"{k}_{name}"], _, _ = pullback_report(v, f"{k}_{name}", P, **kw)

    # 5. trend sleeve on ETFs, weekly
    syms = ["SPY", "EFA", "EEM", "VNQ", "IEF", "TLT", "GLD", "DBC"]
    wk = {s: load_weekly(f"{data_dir}/{s}_weekly.csv") for s in syms if s != "SPY"}
    shy = load_weekly(f"{data_dir}/SHY_weekly.csv")
    ref = wk["EFA"].index
    spy_w = spy["close"].reindex(spy.index.union(ref)).ffill().reindex(ref)   # SPY close on the same week-end dates
    wk["SPY"] = spy_w
    W = pd.DataFrame(wk)[syms].dropna()
    W = W.iloc[:-1] if W.index[-1].dayofweek != 4 else W          # drop a partial final week
    rf = shy.pct_change().reindex(W.index).fillna(0)
    R["trend"] = {"universe": syms, "first_week": str(W.index[0].date()), "last_week": str(W.index[-1].date())}
    base_r, base_turn, base_on = run_trend(W, 40, lag=1)
    start = base_r.index[0]
    variants = {"sma40_lag1 (base)": dict(sma_weeks=40, lag=1), "sma40_lag0": dict(sma_weeks=40, lag=0),
                "sma30_lag1": dict(sma_weeks=30, lag=1), "sma50_lag1": dict(sma_weeks=50, lag=1),
                "mom52_lag1": dict(sma_weeks=52, lag=1, rule="mom"), "mom26_lag1": dict(sma_weeks=26, lag=1, rule="mom"),
                "sma40_lag1_cost10": dict(sma_weeks=40, lag=1, cost=0.0010)}
    R["trend"]["variants"] = {}
    for name, kw in variants.items():
        r, t, _ = run_trend(W, **kw)
        r = r[r.index >= start]
        s = perf(r, 52, rf)
        s["turnover_per_year"] = float(t[t.index >= start].sum() / (len(r) / 52))
        R["trend"]["variants"][name] = s
    R["trend"]["avg_share_invested"] = float(base_on.mean(axis=1).mean())
    bh = buy_hold_equal(W, start)
    spy_wr = W["SPY"].pct_change()[lambda x: x.index >= start]
    b6040 = (0.6 * W["SPY"].pct_change() + 0.4 * W["IEF"].pct_change())[lambda x: x.index >= start]
    R["trend"]["benchmarks"] = {"equal_weight_buy_hold": perf(bh, 52, rf), "spy": perf(spy_wr, 52, rf),
                                "60_40": perf(b6040, 52, rf)}
    # calendar years
    def by_year(r):
        return {int(y): float((1 + x).prod() - 1) for y, x in r.groupby(r.index.year)}
    R["trend"]["by_year"] = {"trend": by_year(base_r), "spy": by_year(spy_wr), "equal_weight": by_year(bh)}

    # 6. the combined desk, weekly: 60% trend sleeve + 40% pullback sleeve.
    #    Pullback sleeve = 4 equal slots: SPY (ETF, next-open fills) and three index proxies (next-close fills).
    slots = {"SPY": run_pullback(spy, P, "next_open")[0]}
    for k, v in idx.items():
        slots[k] = run_pullback(v, P, "next_close")[0]
    pb_daily = pd.DataFrame(slots).mean(axis=1)                    # each slot is a quarter of the sleeve
    wk_id = pd.Series(np.searchsorted(W.index.values, pb_daily.index.values, side="left"), index=pb_daily.index)
    pb_weekly = (1 + pb_daily).groupby(wk_id).prod() - 1
    pb_weekly = pb_weekly[pb_weekly.index < len(W)]
    pb_weekly.index = W.index[pb_weekly.index]
    pb_weekly = pb_weekly.reindex(base_r.index).fillna(0)
    combo = 0.6 * base_r + 0.4 * pb_weekly
    R["combined"] = {"weights": {"trend": 0.6, "pullback": 0.4},
                     "period": [str(combo.index[0].date()), str(combo.index[-1].date())],
                     "desk": perf(combo, 52, rf), "trend_only": perf(base_r, 52, rf),
                     "pullback_only": perf(pb_weekly, 52, rf),
                     "spy": perf(spy_wr.reindex(combo.index), 52, rf),
                     "60_40": perf(b6040.reindex(combo.index), 52, rf),
                     "corr_trend_pullback": float(base_r.corr(pb_weekly)),
                     "by_year": {"desk": by_year(combo), "spy": by_year(spy_wr.reindex(combo.index)),
                                 "60_40": by_year(b6040.reindex(combo.index))}}
    # what a short demo window looks like: rolling 13-week and 26-week returns
    for wks in (13, 26, 52):
        roll = (1 + combo).rolling(wks).apply(np.prod, raw=True).dropna() - 1
        sroll = (1 + spy_wr.reindex(combo.index)).rolling(wks).apply(np.prod, raw=True).dropna() - 1
        R["combined"][f"rolling_{wks}w"] = {
            "p05": float(roll.quantile(0.05)), "p25": float(roll.quantile(0.25)), "median": float(roll.median()),
            "p75": float(roll.quantile(0.75)), "p95": float(roll.quantile(0.95)), "worst": float(roll.min()),
            "share_negative": float((roll < 0).mean()), "share_behind_spy": float((roll < sroll).mean())}
    # pullback trade frequency per year across the four slots, post-2009
    n_tr = 0
    for k, px_ in {"SPY": spy, **idx}.items():
        _, tr = run_pullback(px_, P, "next_open" if k == "SPY" else "next_close")
        n_tr += int((tr["entry"] >= pub).sum())
    yrs_post = (spy.index[-1] - pub).days / 365.25
    R["combined"]["pullback_trades_per_year_4_slots"] = n_tr / yrs_post
    R["combined"]["trend_trades_per_year"] = float((base_turn > 0).sum() / (len(base_r) / 52))

    if out_json:
        with open(out_json, "w") as fh:
            json.dump(R, fh, indent=1, default=float)
        combo.to_frame("desk").assign(trend=base_r, pullback=pb_weekly, spy=spy_wr.reindex(combo.index),
                                      b6040=b6040.reindex(combo.index)).to_csv(out_json.replace(".json", "_weekly_returns.csv"))
    return R


if __name__ == "__main__":
    res = main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    res.get("pullback_grid_post2009", {}).pop("table", None)
    print(json.dumps(res, indent=1, default=float))
