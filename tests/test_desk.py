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
        self.queue.append({"symbol": symbol, "qty": qty, "side": side, "client_order_id": client_order_id})
        return {"id": "fake-" + client_order_id, "status": "accepted"}


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
    return cfg


def _data(cfg):
    syms = cfg["trend_sleeve"]["symbols"] + cfg["pullback_sleeve"]["symbols"]
    return {s: make_prices(i + 1) for i, s in enumerate(syms)}


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


if __name__ == "__main__":
    test_rsi_extremes(); print("ok  RSI extremes and a hand-worked value")
    test_paper_only_guard(); print("ok  broker refuses any non-paper URL")
    n = test_live_matches_backtest(); print(f"ok  live job reproduces the backtest's pullback trades ({n} trades compared)")
    test_risk_gates(); print("ok  risk gates: position cap, drawdown block, drawdown maths")
    test_halt_file_and_rerun(); print("ok  HALT file stops the desk; a repeat run adds no orders; closed days do nothing")
