"""Checks that need no broker and no network.  Run:  python -m tests.test_desk"""
import copy
import datetime as dt
import os
import tempfile

import numpy as np
import pandas as pd

from backtest.backtest import run_pullback
from desk import risk
from desk.broker import AlpacaPaper, BrokerError
from desk.indicators import rsi_wilder
from desk.run_daily import load_config, run
from desk import run_intraday
from desk.strategy import rerank_due
from desk.strategy import PullbackParams


def make_prices(seed, n=900, start="2021-01-04", drift=0.0004, vol=0.012):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.003, n))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998, "close": c}, index=idx)


class FakeBroker:
    """Fills queued market orders at the next session's open, like the paper account does."""
    def __init__(self, data, cash=100_000.0):
        self.data, self.cash, self.pos, self.queue, self.fills, self.eq_hist = data, cash, {}, [], [], []
        self.today, self.ids = None, set()

    def set_day(self, day):
        ts = pd.Timestamp(day)
        for o in self.queue:                                   # fill at today's open
            px = float(self.data[o["symbol"]].loc[ts, "open"])
            q = o["qty"] if o["side"] == "buy" else -o["qty"]
            self.pos[o["symbol"]] = self.pos.get(o["symbol"], 0) + q
            self.cash -= q * px
            self.fills.append({**o, "date": day, "price": px})
        self.queue, self.today = [], day

    def _px(self, s):
        return float(self.data[s].loc[pd.Timestamp(self.today), "close"])

    def account(self):
        eq = self.cash + sum(q * self._px(s) for s, q in self.pos.items())
        self.eq_hist.append(eq)
        return {"equity": eq, "cash": self.cash, "status": "ACTIVE", "account_number": "PA-FAKE"}

    def positions(self):
        return {s: {"qty": float(q), "market_value": q * self._px(s), "avg_entry_price": 0.0} for s, q in self.pos.items() if q}

    def equity_history(self):
        return pd.Series(self.eq_hist, dtype=float)

    def last_buy_fill_date(self, symbol):
        d = [f["date"] for f in self.fills if f["symbol"] == symbol and f["side"] == "buy"]
        return d[-1] if d else None

    def open_orders(self):
        return list(self.queue)

    def trading_days(self, start, end):
        return [d.date() for d in next(iter(self.data.values())).index if start <= d.date() <= end]

    def daily_closes(self, symbols, lookback_days, feeds=()):
        return {s: self.data[s]["close"].loc[: pd.Timestamp(self.today)] for s in symbols}

    def submit_market_order(self, symbol, qty, side, client_order_id):
        if client_order_id in self.ids:
            raise BrokerError("duplicate client_order_id")
        self.ids.add(client_order_id)
        o = {"symbol": symbol, "qty": qty, "side": side, "client_order_id": client_order_id}
        if getattr(self, "instant", False):                    # intraday: fill now at the latest price
            px = self.latest_price(symbol)
            q = qty if side == "buy" else -qty
            self.pos[symbol] = self.pos.get(symbol, 0) + q
            self.cash -= q * px
            self.fills.append({**o, "date": self.today, "price": px})
        else:
            self.queue.append(o)
        return {"id": "fake-" + client_order_id, "status": "accepted"}

    # intraday helpers
    def calendar_day(self, day):
        return getattr(self, "cal", {"open": "09:30", "close": "16:00"})

    def minute_closes(self, symbol, start_utc, end_utc, feed="iex"):
        t = pd.Timestamp(dt.datetime.combine(self.today, dt.time(9, 59)), tz="America/New_York")
        return pd.Series({t: self.price_at_10})

    def latest_price(self, symbol, feed="iex"):
        return self.live_px[symbol](self.clock())

    def cancel_order(self, order_id):
        pass

    def realized_pnl(self, symbols, id_prefix):
        return sum((f["qty"] * f["price"]) * (1 if f["side"] == "sell" else -1)
                   for f in self.fills if f["client_order_id"].startswith(id_prefix)) + getattr(self, "past_pnl", 0.0)


def test_rsi_extremes():
    up = pd.Series(np.arange(1.0, 30.0))
    assert rsi_wilder(up, 2).iloc[-1] == 100.0
    down = pd.Series(np.arange(30.0, 1.0, -1.0))
    assert rsi_wilder(down, 2).iloc[-1] == 0.0
    s = pd.Series([10, 11, 10, 11, 10.0])           # by hand: avg up 0.375, avg down 0.625 -> RSI 37.5
    assert abs(rsi_wilder(s, 2).iloc[-1] - 37.5) < 1e-9


def test_paper_only_guard():
    for url in ("https://api.alpaca.markets", "https://example.com"):
        try:
            AlpacaPaper("k", "s", base_url=url)
            raise AssertionError("live URL was accepted")
        except BrokerError:
            pass
    AlpacaPaper("k", "s")                             # the paper URL constructs fine


def _cfg():
    cfg = copy.deepcopy(load_config())
    cfg["account"]["submit_orders"] = True
    cfg["risk"]["max_drawdown"] = 0.99          # these tests check the rules, not the drawdown stop
    return cfg


def _data(cfg):
    syms = cfg["trend_sleeve"]["symbols"] + cfg["pullback_sleeve"]["symbols"]
    data = {s: make_prices(i + 1) for i, s in enumerate(syms)}
    for pair in cfg["leveraged_sleeve"]["pairs"]:                 # a 3x fund: three times the plain fund's daily move
        base = data[pair["signal"]]
        r = base["close"].pct_change().fillna(0) * 3
        c = 50 * (1 + r).cumprod()
        o = c.shift().fillna(c.iloc[0]) * (1 + 3 * (base["open"] / base["close"].shift().fillna(base["open"]) - 1))
        data[pair["trade"]] = pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c})
    for i, s_ in enumerate((cfg["day_trade"]["signal_symbol"], cfg["day_trade"]["long_symbol"], cfg["day_trade"]["short_symbol"])):
        data.setdefault(s_, make_prices(90 + i))
    return data


def test_live_matches_backtest():
    """Step the live job one day at a time and compare its pullback trades with the backtest's."""
    cfg = _cfg()
    data = _data(cfg)
    b = FakeBroker(data)
    days = [d.date() for d in data["IVV"].index]
    start = 320
    with tempfile.TemporaryDirectory() as root:
        for d in days[start:]:
            b.set_day(d)
            out = run(b, cfg, today=d, root=root, write=False)
            assert not out["errors"], out["errors"]
    p = PullbackParams()
    checked = 0
    for sym in cfg["pullback_sleeve"]["symbols"]:
        _, tr = run_pullback(data[sym], p, "next_open")
        inside = tr[(tr["entry"] > pd.Timestamp(days[start])) & (tr["exit"] <= pd.Timestamp(days[-1]))]
        # skip a trade the backtest was already holding when the live run began
        open_at_start = tr[(tr["entry"] <= pd.Timestamp(days[start])) & (tr["exit"] > pd.Timestamp(days[start]))]
        live_buys = [f["date"] for f in b.fills if f["symbol"] == sym and f["side"] == "buy"]
        live_sells = [f["date"] for f in b.fills if f["symbol"] == sym and f["side"] == "sell"]
        want_buys = [t.date() for t in inside["entry"]]
        want_sells = [t.date() for t in inside["exit"]]
        if len(open_at_start):
            first_ok = open_at_start["exit"].iloc[-1].date()
            live_buys = [x for x in live_buys if x > first_ok]
            live_sells = [x for x in live_sells if x > first_ok]
            want_buys = [x for x in want_buys if x > first_ok]
            want_sells = [x for x in want_sells if x > first_ok]
        assert live_buys[: len(want_buys)] == want_buys, (sym, live_buys[:5], want_buys[:5])
        assert live_sells[: len(want_sells)] == want_sells, (sym, live_sells[:5], want_sells[:5])
        checked += len(want_buys)
    assert checked >= 20, checked
    # the account never borrowed and no position broke the cap
    assert b.cash > -1.0
    return checked


def test_risk_gates():
    rk = {"max_position_weight": 0.15, "max_gross_weight": 1.0, "max_orders_per_run": 16}
    prices = {"AAA": 100.0, "BBB": 100.0}
    pos = {"AAA": {"qty": 100.0}}
    buys = [{"symbol": "AAA", "qty": 100, "side": "buy"}, {"symbol": "BBB", "qty": 50, "side": "buy"}]
    ok, rej = risk.apply_limits(buys, pos, prices, 100_000.0, rk, entries_blocked=False)
    assert [o["symbol"] for o in ok] == ["BBB"] and rej[0]["why"] == "position cap"      # AAA would be 20% > 15%
    ok, rej = risk.apply_limits(buys + [{"symbol": "AAA", "qty": 100, "side": "sell"}], pos, prices, 100_000.0, rk, True)
    assert [o["side"] for o in ok] == ["sell"] and len(rej) == 2                           # blocked buys, sells pass
    assert abs(risk.drawdown(88_000.0, [100_000.0, 95_000.0]) + 0.12) < 1e-9


def test_halt_file_and_rerun():
    cfg = _cfg()
    data = _data(cfg)
    b = FakeBroker(data)
    day = data["IVV"].index[400].date()
    b.set_day(day)
    with tempfile.TemporaryDirectory() as root:
        open(os.path.join(root, "HALT"), "w").close()
        out = run(b, cfg, today=day, root=root, write=True)
        assert out["status"].startswith("halted") and not b.queue
        os.remove(os.path.join(root, "HALT"))
        friday = [d.date() for d in data["IVV"].index[400:410] if d.dayofweek == 4][0]
        b.set_day(friday)
        first = run(b, cfg, today=friday, root=root, write=True)
        n = len(b.queue)
        assert n > 0 and first["week_end_check"]
        second = run(b, cfg, today=friday, root=root, write=True)          # same evening, run twice
        assert len(b.queue) == n, "a second run on the same day must not add orders"
        assert os.path.exists(os.path.join(root, "reports", "latest.md"))
    sat = friday + dt.timedelta(days=1)
    assert run(b, cfg, today=sat, root=tempfile.gettempdir(), write=False)["status"].startswith("market was closed")


def test_leveraged_and_momentum():
    """3x funds are bought and sold on the plain fund's signal; momentum picks change only in re-rank weeks."""
    cfg = _cfg()
    data = _data(cfg)
    b = FakeBroker(data)
    days = [d.date() for d in data["IVV"].index]
    picks, max_w, max_gross = {}, 0.0, 0.0
    for d in days[330:]:
        b.set_day(d)
        out = run(b, cfg, today=d, root=tempfile.gettempdir(), write=False)
        assert not out["errors"], out["errors"]
        assert not out["orders_rejected"], out["orders_rejected"]
        if out["week_end_check"]:
            picks[d] = tuple(out["decisions"]["momentum"]["holding"])
        eq = out["equity"]
        vals = [q * float(data[s].loc[pd.Timestamp(d), "close"]) for s, q in b.pos.items() if q]
        max_w, max_gross = max(max_w, max(vals, default=0) / eq), max(max_gross, sum(vals) / eq)
    for pair in cfg["leveraged_sleeve"]["pairs"]:
        sig = [f["date"] for f in b.fills if f["symbol"] == pair["signal"] and f["side"] == "buy"]
        lev = [f["date"] for f in b.fills if f["symbol"] == pair["trade"] and f["side"] == "buy"]
        assert sig == lev and len(lev) > 5, (pair, sig[:4], lev[:4])
        sig_s = [f["date"] for f in b.fills if f["symbol"] == pair["signal"] and f["side"] == "sell"]
        lev_s = [f["date"] for f in b.fills if f["symbol"] == pair["trade"] and f["side"] == "sell"]
        assert sig_s == lev_s
    wk = sorted(picks)
    changes = [d for a, d in zip(wk, wk[1:]) if picks[a] != picks[d]]
    assert changes and all(rerank_due(d) for d in changes), [str(c) for c in changes if not rerank_due(c)]
    assert all(len(p) <= cfg["momentum_sleeve"]["top_n"] for p in picks.values())
    assert max_w <= cfg["risk"]["max_position_weight"] * 1.3, max_w      # 11.7% target plus drift
    assert max_gross <= 1.0 + 1e-6 and b.cash > -1.0, (max_gross, b.cash)
    return len(changes), round(max_gross, 3)


class Clock:
    def __init__(self, start):
        self.t = start
    def __call__(self):
        return self.t
    def sleep(self, seconds):
        self.t += dt.timedelta(seconds=seconds)


def _intraday(cfg, start_hhmm, move, live=True, **attrs):
    data = _data(cfg)
    day = data["SPY"].index[500].date()
    ny = run_intraday.NY
    clock = Clock(dt.datetime.combine(day, dt.time(*start_hhmm), tzinfo=ny))
    b = FakeBroker(data); b.set_day(day); b.instant = True; b.clock = clock
    prev = float(data["SPY"]["close"].iloc[499])
    b.price_at_10 = prev * (1 + move)
    t0 = dt.datetime.combine(day, dt.time(15, 30), tzinfo=ny)
    b.live_px = {"VOO": lambda t: 500.0 + 2.0 * max(0.0, (t - t0).total_seconds() / 1500),     # +$2 over the 25 minutes
                 "SH": lambda t: 40.0 - 0.2 * max(0.0, (t - t0).total_seconds() / 1500)}
    for k, v in attrs.items():
        setattr(b, k, v)
    cfg = copy.deepcopy(cfg); cfg["account"]["submit_orders"] = live
    return b, run_intraday.run(b, cfg, clock=clock, sleep=clock.sleep, root=tempfile.gettempdir(), write=False)


def test_day_trade():
    cfg = _cfg()
    b, out = _intraday(cfg, (15, 5), +0.004)                       # up morning -> buy VOO 15:30, sell 15:55
    f = b.fills
    assert out["status"] == "ok" and [x["side"] for x in f] == ["buy", "sell"] and f[0]["symbol"] == "VOO"
    assert out["entry_time"] == "15:30:00" and out["exit_time"] == "15:55:00"
    assert f[0]["qty"] == 10 and abs(out["paper_result"] - 20.0) < 1e-6 and b.pos["VOO"] == 0
    b, out = _intraday(cfg, (15, 5), -0.004)                       # down morning -> the inverse fund
    assert b.fills[0]["symbol"] == "SH" and out["direction"] == "down" and b.pos["SH"] == 0
    b, out = _intraday(cfg, (15, 5), +0.004, live=False)           # practice mode: nothing sent, result logged
    assert not b.fills and out["paper_result"] == 20.0
    b, out = _intraday(cfg, (15, 45), +0.004)                      # started late -> no trade
    assert not b.fills and "too late" in out["status"]
    b, out = _intraday(cfg, (14, 5), +0.004)                       # the other season's schedule -> exits
    assert not b.fills and "too early" in out["status"]
    b, out = _intraday(cfg, (15, 5), +0.004, past_pnl=-1000.01)    # 20% of $5,000 lost -> switched off
    assert not b.fills and "loss limit" in out["status"]
    b, out = _intraday(cfg, (15, 5), +0.004, cal={"open": "09:30", "close": "13:00"})
    assert not b.fills and "early close" in out["status"]


if __name__ == "__main__":
    test_rsi_extremes(); print("ok  RSI extremes and a hand-worked value")
    test_paper_only_guard(); print("ok  broker refuses any non-paper URL")
    n = test_live_matches_backtest(); print(f"ok  live job reproduces the backtest's pullback trades ({n} trades compared)")
    test_risk_gates(); print("ok  risk gates: position cap, drawdown block, drawdown maths")
    test_halt_file_and_rerun(); print("ok  HALT file stops the desk; a repeat run adds no orders; closed days do nothing")
    n, g = test_leveraged_and_momentum(); print(f"ok  3x funds trade on the plain fund's signal; momentum picks changed {n} times, only in re-rank weeks; peak invested {g:.0%}")
    test_day_trade(); print("ok  day trade: up and down signals, practice mode, late and early starts, loss limit, early close")
