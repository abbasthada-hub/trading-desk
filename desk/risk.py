"""Risk gates. Every order list passes through here before anything is sent."""
import os


def halt_reason(repo_root: str, cfg: dict):
    """A reason string if the desk is switched off by hand, else None."""
    if os.path.exists(os.path.join(repo_root, "HALT")):
        return "HALT file present"
    if os.environ.get("DESK_HALT", "").strip() in ("1", "true", "TRUE"):
        return "DESK_HALT is set"
    return None


def drawdown(equity_now: float, equity_history) -> float:
    peak = max([equity_now] + [float(x) for x in equity_history]) if len(equity_history) else equity_now
    return equity_now / peak - 1.0 if peak > 0 else 0.0


def bad_data(close, max_daily_move: float) -> bool:
    last = close.iloc[-1] / close.iloc[-2] - 1.0
    return (close.iloc[-1] <= 0) or abs(last) > max_daily_move


def apply_limits(orders: list, positions: dict, prices: dict, equity: float, rk: dict, entries_blocked: bool):
    """Return (orders_to_send, rejected). Sells always pass. Buys are checked against:
    drawdown block, per-position cap, gross cap and the order-count cap."""
    ok, rejected = [], []
    value = {s: p["qty"] * prices.get(s, 0.0) for s, p in positions.items()}
    for o in [o for o in orders if o["side"] == "sell"]:
        ok.append(o)
        value[o["symbol"]] = value.get(o["symbol"], 0.0) - o["qty"] * prices[o["symbol"]]
    gross = sum(v for v in value.values() if v > 0)
    for o in [o for o in orders if o["side"] == "buy"]:
        add = o["qty"] * prices[o["symbol"]]
        if entries_blocked:
            rejected.append({**o, "why": "drawdown limit reached: no new buying"}); continue
        if value.get(o["symbol"], 0.0) + add > rk["max_position_weight"] * equity * 1.001:
            rejected.append({**o, "why": "position cap"}); continue
        if gross + add > rk["max_gross_weight"] * equity * 1.001:
            rejected.append({**o, "why": "gross exposure cap"}); continue
        ok.append(o)
        value[o["symbol"]] = value.get(o["symbol"], 0.0) + add
        gross += add
    if len(ok) > rk["max_orders_per_run"]:
        sells = [o for o in ok if o["side"] == "sell"]
        buys = [o for o in ok if o["side"] == "buy"]
        keep = (sells + buys)[: rk["max_orders_per_run"]]
        rejected += [{**o, "why": "order-count cap"} for o in (sells + buys)[rk["max_orders_per_run"]:]]
        ok = keep
    return ok, rejected
