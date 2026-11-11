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
from .strategy import (PullbackParams, TrendParams, pullback_decision, trend_targets, needs_resize,
                       momentum_targets)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None) -> dict:
    with open(path or os.path.join(ROOT, "config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    groups = [cfg["trend_sleeve"]["symbols"], cfg["pullback_sleeve"]["symbols"],
              [p["trade"] for p in cfg.get("leveraged_sleeve", {}).get("pairs", [])],
              [cfg["day_trade"]["long_symbol"], cfg["day_trade"]["short_symbol"]] if "day_trade" in cfg else []]
    flat = [s for g in groups for s in g]
    dup = sorted({s for s in flat if flat.count(s) > 1})
    if dup:
        raise ValueError(f"Each traded symbol must belong to one piece of the desk only: {dup}")
    total = sum(cfg.get(k, {}).get("weight", 0.0) for k in
                ("trend_sleeve", "momentum_sleeve", "pullback_sleeve", "leveraged_sleeve", "day_trade"))
    if total > 1.0 + 1e-9:
        raise ValueError(f"The pieces add up to {total:.0%} of the account, which is more than 100%.")
    return cfg


def is_last_trading_day_of_week(today: dt.date, trading_days: list) -> bool:
    later = [d for d in trading_days if d > today]
    return (not later) or later[0].isocalendar()[:2] != today.isocalendar()[:2]


def weekly_check_due(root: str, days: list, today, mode: str, use_journal: bool = True):
    """(due, why). Due on the last trading day of the week. Also due when the most recent week-end
    check is missing from the journal, failed, or was made in a different mode - so a skipped Friday run
    is made up the next day, and the first live run sets up the weekly positions without waiting."""
    if is_last_trading_day_of_week(today, days):
        return True, "week end"
    if not use_journal:
        return False, ""
    past = [d for d in days if d < today]
    ends = [d for d in past if is_last_trading_day_of_week(d, days)]
    if not ends:
        return False, ""
    for d in [x for x in past if x >= ends[-1]]:
        try:
            with open(os.path.join(root, "journal", f"{d}.json")) as fh:
                j = json.load(fh)
        except (OSError, ValueError):
            continue
        if j.get("week_end_check") and j.get("mode") == mode and str(j.get("status", "")).startswith(("ok", "completed")):
            return False, ""
    return True, f"catch-up: no completed weekly check on record since {ends[-1]} in {mode} mode"


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

    # ---- leveraged sleeve: the dip rule read on the plain fund, traded in its 3x fund
    lc = cfg.get("leveraged_sleeve", {})
    notes["leveraged"] = {}
    for pair in lc.get("pairs", []):
        sig, trd = pair["signal"], pair["trade"]
        c = closes[sig]
        if risk.bad_data(c, rk["max_daily_move"]):
            continue
        held = positions.get(trd, {}).get("qty", 0.0)
        in_pos = held > 0
        days = days_held_fn(trd, c) if in_pos else 0
        d = pullback_decision(c, in_pos, days, pp)
        notes["leveraged"][trd] = {"decision": d, "signal_from": sig, "in_position": in_pos, "days_held": days}
        if d == "enter":
            qty = math.floor(lc["weight"] / len(lc["pairs"]) * equity / prices[trd])
            if qty > 0:
                orders.append({"symbol": trd, "qty": qty, "side": "buy", "sleeve": "lv", "reason": "leveraged pullback entry"})
        elif d == "exit":
            orders.append({"symbol": trd, "qty": int(held), "side": "sell", "sleeve": "lv",
                           "reason": "time stop" if days >= pp.max_hold_days else "closed above short average"})

    # ---- anything left over from the day-trading piece is sold at the next open
    dt_cfg = cfg.get("day_trade", {})
    for sym in (dt_cfg.get("long_symbol"), dt_cfg.get("short_symbol")):
        held = positions.get(sym, {}).get("qty", 0.0) if sym else 0.0
        if held > 0:
            orders.append({"symbol": sym, "qty": int(held), "side": "sell", "sleeve": "dt", "reason": "left over from day trade"})
            notes["skipped"].append(f"{sym}: a day-trade position was still open at the close")

    # ---- trend and momentum sleeves (weekly). Both hold the same 8 funds, so their targets are added up.
    tc = cfg["trend_sleeve"]
    mc = cfg.get("momentum_sleeve", {"weight": 0.0, "top_n": 3, "rerank_every_weeks": 4})
    tp = TrendParams(tc["sma_weeks"], tc["drift_band"])
    notes["trend"]["checked"] = bool(week_end)
    if week_end:
        tcloses = {s: closes[s] for s in tc["symbols"] if not risk.bad_data(closes[s], rk["max_daily_move"])}
        for s in set(tc["symbols"]) - set(tcloses):
            notes["skipped"].append(f"{s}: one-day move above {rk['max_daily_move']:.0%}, treated as bad data")
        n_all = len(tc["symbols"])
        t_tgt = {s: (tc["weight"] / n_all if w > 0 else 0.0) for s, w in trend_targets(tcloses, tp, 1.0).items()}
        m_tgt = momentum_targets(tcloses, today, mc["weight"], mc["top_n"], mc["rerank_every_weeks"], tc["sma_weeks"]) \
            if mc["weight"] > 0 else {}
        notes["momentum"] = {"holding": sorted(s for s, w in m_tgt.items() if w > 0)}
        for sym in tcloses:
            tw = t_tgt.get(sym, 0.0) + m_tgt.get(sym, 0.0)
            held = positions.get(sym, {}).get("qty", 0.0)
            cw = held * prices[sym] / equity
            notes["trend"][sym] = {"trend_on": t_tgt.get(sym, 0.0) > 0, "momentum_pick": m_tgt.get(sym, 0.0) > 0,
                                   "current_weight": round(cw, 4), "target_weight": round(tw, 4)}
            if needs_resize(cw, tw, tp.drift_band):
                diff = math.floor(tw * equity / prices[sym]) - int(held)
                if diff > 0:
                    orders.append({"symbol": sym, "qty": diff, "side": "buy", "sleeve": "tr",
                                   "reason": "new holding" if held == 0 else "resize up"})
                elif diff < 0:
                    orders.append({"symbol": sym, "qty": -diff, "side": "sell", "sleeve": "tr",
                                   "reason": "sold out" if tw == 0 else "resize down"})
    return orders, notes, prices


def run(broker, cfg=None, today=None, root=ROOT, write=True) -> dict:
    cfg = cfg or load_config()
    # The run belongs to the last session that has closed. Shifting the clock back 9 hours means a run
    # that starts late (after midnight in New York) still works on the right day.
    today = today or (dt.datetime.now(ZoneInfo("America/New_York")) - dt.timedelta(hours=9)).date()
    entry = {"date": str(today), "mode": "live-paper" if cfg["account"]["submit_orders"] else "dry-run",
             "orders_sent": [], "orders_rejected": [], "errors": []}
    try:
        return _run(broker, cfg, today, root, write, entry)
    except Exception as e:                          # a failed run still leaves a journal entry
        entry["errors"].append(f"{type(e).__name__}: {e}")
        entry["status"] = "failed: the run stopped early"
        return _finish(entry, root, write)


def _run(broker, cfg, today, root, write, entry) -> dict:

    reason = risk.halt_reason(root, cfg)
    if reason:
        entry["status"] = f"halted: {reason}"
        return _finish(entry, root, write)

    if broker.is_market_open():                     # e.g. started by hand at midday: today's bar is unfinished
        entry["status"] = "the market is open: the nightly job only runs after the close"
        return _finish(entry, root, write)

    days = broker.trading_days(today - dt.timedelta(days=10), today + dt.timedelta(days=10))
    if today not in days:
        entry["status"] = "market was closed today: nothing to do"
        return _finish(entry, root, write)

    acct = broker.account()
    positions = broker.positions()
    equity = acct["equity"]
    lev_syms = {x for p in cfg.get("leveraged_sleeve", {}).get("pairs", []) for x in (p["signal"], p["trade"])}
    dt_syms = {cfg["day_trade"]["long_symbol"], cfg["day_trade"]["short_symbol"]} if "day_trade" in cfg else set()
    # day-trade funds are priced too, so a position left open can be sold and passes the risk gates
    symbols = sorted(set(cfg["trend_sleeve"]["symbols"]) | set(cfg["pullback_sleeve"]["symbols"]) | lev_syms | dt_syms)
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

    week_end, why = weekly_check_due(root, days, today, entry["mode"], use_journal=write)
    orders, notes, prices = decide(cfg, equity, positions, closes, today, week_end, days_held)
    waiting = {o["symbol"] for o in broker.open_orders()}       # an earlier order has not filled yet
    if waiting:
        notes["skipped"] += [f"{o['symbol']}: an order is already waiting to fill" for o in orders if o["symbol"] in waiting]
        orders = [o for o in orders if o["symbol"] not in waiting]

    dd = risk.drawdown(equity, list(broker.equity_history().values))
    blocked = dd <= -cfg["risk"]["max_drawdown"]
    ok, rejected = risk.apply_limits(orders, positions, prices, equity, cfg["risk"], blocked)

    entry.update({"weekly_check_reason": why, "data_feed": getattr(broker, "last_feed", None)})
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
