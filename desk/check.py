"""Connection check. Sends no orders.  python -m desk.check"""
import datetime as dt

from .broker import AlpacaPaper
from .run_daily import load_config


def main():
    cfg = load_config()
    b = AlpacaPaper()
    a = b.account()
    print(f"Paper account reached. Status {a['status']}, equity ${a['equity']:,.2f}, cash ${a['cash']:,.2f}")
    print("Positions:", {s: p["qty"] for s, p in b.positions().items()} or "none")
    syms = sorted(set(cfg["trend_sleeve"]["symbols"]) | set(cfg["pullback_sleeve"]["symbols"]))
    closes = b.daily_closes(syms, cfg["data"]["lookback_days"], tuple(cfg["data"]["feeds"]))
    for s, c in closes.items():
        flag = "" if len(c) >= 205 else "   <-- fewer than 205 bars: the 200-day average cannot be computed"
        print(f"  {s}: {len(c)} daily bars, last {c.index[-1].date()} close {c.iloc[-1]:.2f}{flag}")
    today = dt.date.today()
    print("Next trading days:", [str(d) for d in b.trading_days(today, today + dt.timedelta(days=7))][:3])
    print("submit_orders in config.yaml is", cfg["account"]["submit_orders"])


if __name__ == "__main__":
    main()
