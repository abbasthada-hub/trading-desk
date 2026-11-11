"""The day-trading piece. Run on trading afternoons:  python -m desk.run_intraday   (add --selftest to only check data)

Rule (Gao, Han and Zhou, "Market intraday momentum"): the S&P 500's move from yesterday's close to 10:00
tends to carry into the last half-hour. So, once a day at most:
    up by 10:00   -> at 15:30 buy the long fund,  sell at 15:55
    down by 10:00 -> at 15:30 buy the inverse fund, sell at 15:55
The piece is switched off once it has lost stop_loss_fraction of its allocation.
"""
import datetime as dt
import json
import math
import os
import sys
import time
from zoneinfo import ZoneInfo

from . import risk
from .run_daily import ROOT, load_config

NY = ZoneInfo("America/New_York")


def _at(day: dt.date, hhmm: str) -> dt.datetime:
    h, m = map(int, hhmm.split(":"))
    return dt.datetime(day.year, day.month, day.day, h, m, tzinfo=NY)


def first_half_hour_signal(broker, symbol: str, day: dt.date):
    """Return (move from the previous close to 10:00, previous close, price at 10:00)."""
    daily = broker.daily_closes([symbol], 12)[symbol]
    prev = daily[daily.index.date < day]
    if len(prev) == 0:
        return None, None, None
    start = _at(day, "09:30").astimezone(dt.timezone.utc)
    end = _at(day, "10:00").astimezone(dt.timezone.utc)
    bars = broker.minute_closes(symbol, start, end)
    bars = bars[bars.index < _at(day, "10:00")]
    if len(bars) == 0:
        return None, float(prev.iloc[-1]), None
    return float(bars.iloc[-1] / prev.iloc[-1] - 1), float(prev.iloc[-1]), float(bars.iloc[-1])


def run(broker, cfg=None, clock=None, sleep=time.sleep, selftest=False, root=ROOT, write=True) -> dict:
    cfg = cfg or load_config()
    clock = clock or (lambda: dt.datetime.now(NY))
    now = clock()
    today = now.date()
    dc = cfg["day_trade"]
    live = bool(cfg["account"]["submit_orders"])
    e = {"date": str(today), "piece": "day trade", "mode": "live-paper" if live else "dry-run", "errors": []}

    def done(status):
        e["status"] = status
        return _finish(e, root, write and not selftest)

    reason = risk.halt_reason(root, cfg)
    if reason:
        return done(f"halted: {reason}")
    if not dc.get("enabled", True):
        return done("day trading is switched off in config.yaml")

    if selftest:                                    # data check only: uses the most recent session, sends nothing
        days = broker.trading_days(today - dt.timedelta(days=10), today)
        day = [d for d in days if _at(d, "10:00") < now][-1]
        r1, prev, p10 = first_half_hour_signal(broker, dc["signal_symbol"], day)
        e.update({"selftest_session": str(day), "signal_move": r1, "prev_close": prev, "price_at_10": p10,
                  "latest": {s: broker.latest_price(s) for s in (dc["long_symbol"], dc["short_symbol"])},
                  "calendar": broker.calendar_day(day), "result_so_far": broker.realized_pnl([dc["long_symbol"], dc["short_symbol"]], "dtx-")})
        return done("selftest ok")

    cal = broker.calendar_day(today)
    if cal is None:
        return done("market is closed today")
    if cal["close"] != "16:00":
        return done("early close today: no day trade")
    t_in, t_out = _at(today, dc["entry_time"]), _at(today, dc["exit_time"])
    if now < t_in - dt.timedelta(minutes=45):
        return done("started too early for today's window (the other seasonal schedule handles it)")
    if now > t_in + dt.timedelta(minutes=10):
        return done("started too late for today's window: no trade")

    acct = broker.account()
    equity = acct["equity"]
    budget = dc["weight"] * equity
    syms = [dc["long_symbol"], dc["short_symbol"]]
    result = broker.realized_pnl(syms, "dtx-")
    e.update({"equity": equity, "budget": round(budget, 2), "result_so_far": round(result, 2)})
    if result <= -dc["stop_loss_fraction"] * budget:
        return done("loss limit reached: the day-trading piece is switched off")
    if risk.drawdown(equity, list(broker.equity_history().values)) <= -cfg["risk"]["max_drawdown"]:
        return done("account drawdown limit reached: no new buying")
    if any(broker.positions().get(s, {}).get("qty", 0) > 0 for s in syms):
        return done("a day-trade position is already open: no new trade")

    r1, prev, p10 = first_half_hour_signal(broker, dc["signal_symbol"], today)
    e.update({"signal_move": r1, "prev_close": prev, "price_at_10": p10})
    if r1 is None or r1 == 0:
        return done("no signal today")
    sym = dc["long_symbol"] if r1 > 0 else dc["short_symbol"]
    e["direction"] = "up" if r1 > 0 else "down"

    while clock() < t_in:
        sleep(min(20, max(1, (t_in - clock()).total_seconds())))
    px_in = broker.latest_price(sym)
    qty = math.floor(budget / px_in)
    e.update({"symbol": sym, "qty": qty, "entry_ref_price": px_in, "entry_time": str(clock().time())[:8]})
    if qty <= 0:
        return done("budget too small for one share")
    order_id = None
    if live:
        try:
            order_id = broker.submit_market_order(sym, qty, "buy", f"dtx-{sym}-{today:%Y%m%d}-buy").get("id")
        except Exception as ex:
            e["errors"].append(f"entry order: {ex}")
            return done("entry order failed")

    while clock() < t_out:
        sleep(min(20, max(1, (t_out - clock()).total_seconds())))
    px_out = broker.latest_price(sym)
    e.update({"exit_ref_price": px_out, "exit_time": str(clock().time())[:8],
              "paper_result": round(qty * (px_out - px_in), 2)})
    if live:
        try:
            held = int(broker.positions().get(sym, {}).get("qty", 0))
            if held > 0:
                broker.submit_market_order(sym, held, "sell", f"dtx-{sym}-{today:%Y%m%d}-sell")
            elif order_id:
                broker.cancel_order(order_id)
                e["errors"].append("entry order had not filled by the exit time and was cancelled")
        except Exception as ex:
            e["errors"].append(f"exit order: {ex}. The nightly job will sell any leftover position at the next open.")
    return done("ok" if not e["errors"] else "completed with errors")


def _finish(e: dict, root: str, write: bool) -> dict:
    if write:
        d = os.path.join(root, "journal", "intraday")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{e['date']}.json"), "w") as fh:
            json.dump(e, fh, indent=1, default=str)
        if "paper_result" in e:
            path = os.path.join(root, "journal", "daytrade.csv")
            new = not os.path.exists(path)
            with open(path, "a") as fh:
                if new:
                    fh.write("date,direction,symbol,qty,entry_ref,exit_ref,paper_result,mode\n")
                fh.write(f"{e['date']},{e['direction']},{e['symbol']},{e['qty']},{e['entry_ref_price']},"
                         f"{e['exit_ref_price']},{e['paper_result']},{e['mode']}\n")
    return e


if __name__ == "__main__":
    from .broker import AlpacaPaper
    out = run(AlpacaPaper(), selftest="--selftest" in sys.argv)
    print(json.dumps(out, indent=1, default=str))
    sys.exit(0 if not out["errors"] else 1)
