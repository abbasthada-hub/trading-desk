"""Backtest of the version-2 split. Usage: python -m backtest.v2 backtest/data
Weights come from config.yaml. Writes backtest/RESULTS_V2.md and backtest/model.json (used by the report).

Leveraged piece: simulated as 3x the plain fund's move on each day held, less 2 basis points a day
carrying cost and 5 basis points per side. It does not use the real 3x funds' prices.
Day-trading piece: counted as cash (0%). It could not be backtested here.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import yaml

from backtest.backtest import load_daily, load_weekly, perf, run_trend, run_pullback, COST
from desk.strategy import PullbackParams, momentum_picks_weekly

HERE = os.path.dirname(os.path.abspath(__file__))


def lev3(px, fill, p):
    r, tr = run_pullback(px, p, fill, cost=0.0)
    held = r != 0
    enter = held & ~held.shift(1, fill_value=False)
    leave = ~held & held.shift(1, fill_value=False)
    out = 3 * r - 0.0002 * held - 0.0005 * enter
    out[leave[leave].index] -= 0.0005
    return out, tr


def rotation(W, n=3, hold=4, lag=1, sma_weeks=40, cost=COST):
    rets = W.pct_change().fillna(0).values
    start = 52 + lag
    w = np.zeros(W.shape[1])
    out = []
    cols = list(W.columns)
    for i in range(start, len(W)):
        port = float((w * rets[i]).sum())
        w = w * (1 + rets[i]) / (1 + port)
        new = w
        if (i - start) % hold == 0:
            picks = momentum_picks_weekly(W.iloc[: i - lag + 1], n, sma_weeks)
            new = np.array([1.0 / n if c in picks else 0.0 for c in cols])
        out.append(port - float(np.abs(new - w).sum()) * cost)
        w = new
    return pd.Series(out, index=W.index[start:])


def main(data_dir):
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(HERE), "config.yaml")))
    wts = {"trend": cfg["trend_sleeve"]["weight"], "momentum": cfg["momentum_sleeve"]["weight"],
           "pullback": cfg["pullback_sleeve"]["weight"], "leveraged": cfg["leveraged_sleeve"]["weight"]}
    P = PullbackParams()
    spy = load_daily(f"{data_dir}/SPY_daily.csv")
    idx = {k: load_daily(f"{data_dir}/IDX_{k}_daily.csv") for k in ["IXIC", "RUT", "DJI"]}
    syms = cfg["trend_sleeve"]["symbols"]
    wk = {s: load_weekly(f"{data_dir}/{s}_weekly.csv") for s in syms if s != "SPY"}
    ref = wk["EFA"].index
    wk["SPY"] = spy["close"].reindex(spy.index.union(ref)).ffill().reindex(ref)
    W = pd.DataFrame(wk)[syms].dropna().iloc[:-1]
    rf = load_weekly(f"{data_dir}/SHY_weekly.csv").pct_change().reindex(W.index).fillna(0)

    def to_weekly(daily):
        wid = np.searchsorted(W.index.values, daily.index.values, side="left")
        s = (1 + daily).groupby(wid).prod() - 1
        s = s[s.index < len(W)]
        s.index = W.index[s.index]
        return s.reindex(W.index).fillna(0)

    trend, _, _ = run_trend(W, cfg["trend_sleeve"]["sma_weeks"], lag=1)
    slots = {"SPY": run_pullback(spy, P, "next_open")[0], **{k: run_pullback(v, P, "next_close")[0] for k, v in idx.items()}}
    pull = to_weekly(pd.DataFrame(slots).mean(axis=1))
    lu, tr_u = lev3(spy, "next_open", P)
    lt, tr_t = lev3(idx["IXIC"], "next_close", P)
    lev = to_weekly((lu + lt) / 2)
    mom = rotation(W, cfg["momentum_sleeve"]["top_n"], cfg["momentum_sleeve"]["rerank_every_weeks"])
    start = max(trend.index[0], mom.index[0])
    X = pd.DataFrame({"trend": trend, "momentum": mom, "pullback": pull, "leveraged": lev,
                      "spy": W["SPY"].pct_change(), "b6040": 0.6 * W["SPY"].pct_change() + 0.4 * W["IEF"].pct_change()}).loc[start:].fillna(0)
    desk = sum(X[k] * v for k, v in wts.items())

    def stats(r):
        a = perf(r, 52, rf)
        m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
        y = (1 + r).groupby(r.index.year).prod() - 1
        return {"cagr": a["cagr"], "vol": a["vol"], "sharpe": a["sharpe"], "max_dd": a["max_dd"],
                "months_up": float((m > 0).mean()), "avg_month": float(m.mean()), "m_p10": float(m.quantile(.1)),
                "m_p90": float(m.quantile(.9)), "worst_month": float(m.min()), "best_month": float(m.max()),
                "neg_years": int((y < 0).sum()), "years": int(len(y)), "worst_year": float(y.min()),
                "by_year": {int(k): float(v) for k, v in y.items()}}

    R = {"weights": wts, "period": [str(X.index[0].date()), str(X.index[-1].date())],
         "desk": stats(desk), **{k: stats(X[k]) for k in X}}
    roll = (1 + desk).rolling(13).apply(np.prod, raw=True).dropna() - 1
    model = {"version": 2, "p05": float(roll.quantile(.05)), "median": float(roll.median()), "p95": float(roll.quantile(.95)),
             "worst_13w": float(roll.min()), "share_13w_negative": float((roll < 0).mean()), "max_dd": R["desk"]["max_dd"]}
    R["model_13w"] = model
    R["leveraged_trades_per_year"] = float(((tr_u.entry >= "2009") .sum() + (tr_t.entry >= "2009").sum()) / ((spy.index[-1] - pd.Timestamp("2009-01-01")).days / 365.25))
    R["leveraged_worst_trade"] = float(3 * min(tr_u.ret.min(), tr_t.ret.min()))
    json.dump(model, open(os.path.join(HERE, "model.json"), "w"), indent=1)

    pc = lambda x: f"{x * 100:+.1f}%"
    L = ["# Backtest results, version 2\n",
         f"Generated by `python -m backtest.v2 backtest/data`. Weekly returns, {R['period'][0]} to {R['period'][1]}, after costs. "
         "The day-trading piece (5%) is counted as cash.\n",
         "| | Share | Return per year | Volatility | Sharpe | Worst drop | Months up | Typical month (8 in 10) | Worst month | Losing years |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    names = [("desk", "The desk, version 2", "95% + 5% day trade"), ("trend", "Trend", f"{wts['trend']:.0%}"),
             ("momentum", "Momentum (top 3)", f"{wts['momentum']:.0%}"), ("pullback", "Dip-buying", f"{wts['pullback']:.0%}"),
             ("leveraged", "Leveraged dip-buying (3x)", f"{wts['leveraged']:.0%}"), ("b6040", "60% SPY / 40% IEF, held", ""), ("spy", "SPY, held", "")]
    for k, n, sh in names:
        s = R[k]
        L.append(f"| {n} | {sh} | {pc(s['cagr'])} | {s['vol'] * 100:.1f}% | {s['sharpe']:.2f} | {s['max_dd'] * 100:.1f}% | {s['months_up'] * 100:.0f}% | "
                 f"{pc(s['m_p10'])} to {pc(s['m_p90'])} | {pc(s['worst_month'])} | {s['neg_years']} of {s['years']} |")
    L += ["", f"Rolling 13 weeks for the desk: 9 in 10 windows between {pc(model['p05'])} and {pc(model['p95'])}, median {pc(model['median'])}, "
              f"worst {pc(model['worst_13w'])}; {model['share_13w_negative'] * 100:.0f}% of windows lost money.", "",
          f"Leveraged piece: about {R['leveraged_trades_per_year']:.0f} trades a year across its two slots; worst single trade about {pc(R['leveraged_worst_trade'])} of the slot.", "",
          "| Year | Desk v2 | 60/40 | SPY |", "| --- | --- | --- | --- |"]
    L += [f"| {y} | {pc(v)} | {pc(R['b6040']['by_year'][y])} | {pc(R['spy']['by_year'][y])} |" for y, v in R["desk"]["by_year"].items()]
    L += ["", "## Limits", "",
          "- The leveraged piece is simulated from the plain index's daily moves, not from UPRO and TQQQ prices. The Nasdaq slot uses the Nasdaq Composite.",
          "- The day-trading rule could not be tested: no long intraday history was available. Its published result is 6.3% a year before costs on 1999 to 2012 data.",
          "- The momentum piece was first tested as an alternative to the trend piece and set aside for its 29% drop. It is used here on purpose, as the higher-risk part.",
          "- Forex: nine swing rules on seven major pairs, 2008 to 2026, in `backtest/fx_test.py`. None made money reliably after costs, so no forex piece.",
          "- Everything in `RESULTS.md` under limits applies here too."]
    open(os.path.join(HERE, "RESULTS_V2.md"), "w").write("\n".join(L) + "\n")
    return R


if __name__ == "__main__":
    out = main(sys.argv[1])
    print(open(os.path.join(HERE, "RESULTS_V2.md")).read())
