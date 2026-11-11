"""The daily job. Run once each evening after the US close:  python -m desk.run_daily

What it does, in order:
  1. stop if the desk is switched off (HALT file or DESK_HALT=1)
  2. read the paper account, positions and today's completed daily bars
  3. pullback sleeve: decide enter / exit / hold for each symbol
  4. trend sleeve: on the last trading day of the week, decide which slices are on and resize
  5. pass every order through the risk gates
  6. send whole-share market orders that fill at the next open (or only log them in dry-run mode)
  7. write the journal entry and refresh the report
"""
import datetime as dt
import json
import math
import os
import sys
from zoneinfo import ZoneInfo

import yaml

from . import risk
from .strategy import PullbackParams, TrendParams, pullback_decision, trend_targets, needs_resize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None) -> dict:
    with open(path or os.path.join(ROOT, "config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    both = set(cfg["trend_sleeve"]["symbols"]) & set(cfg["pullback_sleeve"]["symbols"])
    if both:
        raise ValueError(f"A symbol cannot be in both sleeves: {sorted(both)}")
    if cfg["trend_sleeve"]["weight"] + cfg["pullback_sleeve"]["weight"] > 1.0 + 1e-9:
        raise ValueError("Sleeve weights add up to more than 100% of the account.")
    return cfg


def is_last_trading_day_of_week(today: dt.date, trading_days: list) -> bool:
    later = [d for d in trading_days if d > today]
    return (not later) or later[0].isocalendar()[:2] != today.isocalendar()[:2]


def decide(cfg, equity, positions, closes, today, week_end, days_held_fn):
    """Pure decision step. Returns (orders, notes). An order is a dict: symbol, qty, side, sleeve, reason."""
    orders, notes = [], {"pullback": {}, "trend": {}, "skipped": []}
    rk = cfg["risk"]
    prices = {s: float(c.iloc[-1]) for s, c in closes.items()}

    # ---- pullback sleeve
    pc = cfg["pullback_sleeve"]
    pp = PullbackParams(pc["trend_days"], pc["rsi_days"], pc["rsi_entry"], pc["exit_days"], pc["max_hold_days"])
    slot_value = pc["weight"] / len(pc["symbols"]) * equity
    for sym in pc["symbols"]:
        c = closes[sym]
        if risk.bad_data(c, rk["max_daily_move"]):
            notes["skipped"].append(f"{sym}: one-day move above {rk['max_daily_move']:.0%}, treated as bad data")
            continue
        held = positions.get(sym, {}).get("qty", 0.0)
        in_pos = held > 0
        days = days_held_fn(sym, c) if in_pos else 0
        d = pullback_decision(c, in_pos, days, pp)
        notes["pullback"][sym] = {"decision": d, "in_position": in_pos, "days_held": days, "close": prices[sym]}
        if d == "enter":
            qty = math.floor(slot_value / prices[sym])
            if qty > 0:
                orders.append({"symbol": sym, "qty": qty, "side": "buy", "sleeve": "pb", "reason": "pullback entry"})
        elif d == "exit":
            orders.append({"symbol": sym, "qty": int(held), "side": "sell", "sleeve": "pb",
                           "reason": "time stop" if days >= pp.max_hold_days else "closed above short average"})

    # ---- trend sleeve (weekly)
    tc = cfg["trend_sleeve"]
    tp = TrendParams(tc["sma_weeks"], tc["drift_band"])
    notes["trend"]["checked"] = bool(week_end)
    if week_end:
        tcloses = {s: closes[s] for s in tc["symbols"] if not risk.bad_data(closes[s], rk["max_daily_move"])}
        for s in set(tc["symbols"]) - set(tcloses):
            notes["skipped"].append(f"{s}: one-day move above {rk['max_daily_move']:.0%}, treated as bad data")
        n_all = len(tc["symbols"])
        targets = {s: (tc["weight"] / n_all if w > 0 else 0.0)
                   for s, w in trend_targets(tcloses, tp, 1.0).items()}
        for sym, tw in targets.items():
            held = positions.get(sym, {}).get("qty", 0.0)
            cw = held * prices[sym] / equity
            notes["trend"][sym] = {"on": tw > 0, "current_weight": round(cw, 4), "target_weight": round(tw, 4)}
            if needs_resize(cw, tw, tp.drift_band):
                diff = math.floor(tw * equity / prices[sym]) - int(held)
                if diff > 0:
                    orders.append({"symbol": sym, "qty": diff, "side": "buy", "sleeve": "tr",
                                   "reason": "trend on" if held == 0 else "resize up"})
                elif diff < 0:
                    orders.append({"symbol": sym, "qty": -diff, "side": "sell", "sleeve": "tr",
                                   "reason": "trend off" if tw == 0 else "resize down"})
    return orders, notes, prices


def run(broker, cfg=None, today=None, root=ROOT, write=True) -> dict:
    cfg = cfg or load_config()
    # The run belongs to the last session that has closed. Shifting the clock back 9 hours means a run
    # that starts late (after midnight in New York) still works on the right day.
    today = today or (dt.datetime.now(ZoneInfo("America/New_York")) - dt.timedelta(hours=9)).date()
    entry = {"date": str(today), "mode": "live-paper" if cfg["account"]["submit_orders"] else "dry-run",
             "orders_sent": [], "orders_rejected": [], "errors": []}

    reason = risk.halt_reason(root, cfg)
    if reason:
        entry["status"] = f"halted: {reason}"
        return _finish(entry, root, write)

    days = broker.trading_days(today - dt.timedelta(days=10), today + dt.timedelta(days=10))
    if today not in days:
        entry["status"] = "market was closed today: nothing to do"
        return _finish(entry, root, write)

    acct = broker.account()
    positions = broker.positions()
    equity = acct["equity"]
    symbols = sorted(set(cfg["trend_sleeve"]["symbols"]) | set(cfg["pullback_sleeve"]["symbols"]))
    closes = broker.daily_closes(symbols, cfg["data"]["lookback_days"], tuple(cfg["data"]["feeds"]))

    stale = [s for s, c in closes.items() if c.index[-1].date() != today]
    if stale:
        entry["status"] = "stopped: today's bar is missing for " + ", ".join(stale)
        entry["errors"].append(entry["status"])
        return _finish(entry, root, write)

    unknown = sorted(set(positions) - set(symbols))
    if unknown:
        entry["errors"].append("positions the desk does not manage: " + ", ".join(unknown))

    def days_held(sym, c):
        d0 = broker.last_buy_fill_date(sym)
        return int((c.index.date > d0).sum()) if d0 else cfg["pullback_sleeve"]["max_hold_days"]

    week_end = is_last_trading_day_of_week(today, days)
    orders, notes, prices = decide(cfg, equity, positions, closes, today, week_end, days_held)
    waiting = {o["symbol"] for o in broker.open_orders()}       # an earlier order has not filled yet
    if waiting:
        notes["skipped"] += [f"{o['symbol']}: an order is already waiting to fill" for o in orders if o["symbol"] in waiting]
        orders = [o for o in orders if o["symbol"] not in waiting]

    dd = risk.drawdown(equity, list(broker.equity_history().values))
    blocked = dd <= -cfg["risk"]["max_drawdown"]
    ok, rejected = risk.apply_limits(orders, positions, prices, equity, cfg["risk"], blocked)

    entry.update({"equity": equity, "cash": acct["cash"], "drawdown_from_peak": round(dd, 4),
                  "new_buying_blocked": blocked, "week_end_check": week_end, "decisions": notes,
                  "positions": {s: p["qty"] for s, p in positions.items()}, "orders_rejected": rejected})
    for o in ok:
        cid = f"{o['sleeve']}-{o['symbol']}-{today:%Y%m%d}-{o['side']}"
        rec = {**o, "client_order_id": cid, "ref_price": prices[o["symbol"]]}
        if cfg["account"]["submit_orders"]:
            try:
                resp = broker.submit_market_order(o["symbol"], o["qty"], o["side"], cid)
                rec["broker_order_id"] = resp.get("id")
                rec["broker_status"] = resp.get("status")
            except Exception as e:                      # one failed order must not stop the others
                rec["error"] = str(e)
                entry["errors"].append(f"{cid}: {e}")
        entry["orders_sent"].append(rec)
    entry["status"] = "ok" if not entry["errors"] else "completed with errors"
    return _finish(entry, root, write)


def _finish(entry: dict, root: str, write: bool) -> dict:
    if write:
        jd = os.path.join(root, "journal")
        os.makedirs(jd, exist_ok=True)
        with open(os.path.join(jd, f"{entry['date']}.json"), "w") as fh:
            json.dump(entry, fh, indent=1, default=str)
        if "equity" in entry:
            path = os.path.join(jd, "equity.csv")
            new = not os.path.exists(path)
            with open(path, "a") as fh:
                if new:
                    fh.write("date,equity,cash,drawdown_from_peak,orders,mode\n")
                fh.write(f"{entry['date']},{entry['equity']:.2f},{entry['cash']:.2f},"
                         f"{entry['drawdown_from_peak']},{len(entry['orders_sent'])},{entry['mode']}\n")
        try:
            from .report import write_report
            write_report(root)
        except Exception as e:
            entry["errors"].append(f"report: {e}")
    return entry


if __name__ == "__main__":
    from .broker import AlpacaPaper
    result = run(AlpacaPaper())
    print(json.dumps(result, indent=1, default=str))
    sys.exit(0 if not result["errors"] else 1)
