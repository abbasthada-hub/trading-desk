"""Thin Alpaca REST adapter. Paper endpoint only: the class will not talk to the live trading URL."""
import os
import datetime as dt
import pandas as pd
import requests

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


class BrokerError(RuntimeError):
    pass


class AlpacaPaper:
    def __init__(self, key_id=None, secret=None, base_url=PAPER_URL, timeout=30):
        if base_url.rstrip("/") != PAPER_URL:
            raise BrokerError("This desk only trades the paper account. Refusing base URL: " + base_url)
        self.base = PAPER_URL
        key_id = key_id or os.environ.get("ALPACA_API_KEY_ID")
        secret = secret or os.environ.get("ALPACA_API_SECRET_KEY")
        if not key_id or not secret:
            raise BrokerError("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in the environment.")
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret})
        self.timeout = timeout

    def _req(self, method, url, **kw):
        r = self.s.request(method, url, timeout=self.timeout, **kw)
        if r.status_code >= 400:
            raise BrokerError(f"{method} {url.split('?')[0]} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.text else {}

    # --- account ---------------------------------------------------------------------------
    def account(self) -> dict:
        a = self._req("GET", f"{self.base}/v2/account")
        return {"equity": float(a["equity"]), "cash": float(a["cash"]), "status": a.get("status"),
                "account_number": a.get("account_number", "")}

    def positions(self) -> dict:
        return {p["symbol"]: {"qty": float(p["qty"]), "market_value": float(p["market_value"]),
                              "avg_entry_price": float(p["avg_entry_price"])}
                for p in self._req("GET", f"{self.base}/v2/positions")}

    def equity_history(self) -> pd.Series:
        h = self._req("GET", f"{self.base}/v2/account/portfolio/history", params={"period": "1A", "timeframe": "1D"})
        s = pd.Series(h.get("equity", []), index=pd.to_datetime(h.get("timestamp", []), unit="s"), dtype=float)
        return s[s > 0]

    def last_buy_fill_date(self, symbol: str):
        """Date of the most recent filled buy order for a symbol, or None."""
        orders = self._req("GET", f"{self.base}/v2/orders",
                           params={"status": "closed", "symbols": symbol, "limit": 50, "direction": "desc"})
        for o in orders:
            if o.get("side") == "buy" and o.get("filled_at"):
                return pd.Timestamp(o["filled_at"]).tz_convert("America/New_York").date()
        return None

    def open_orders(self) -> list:
        return self._req("GET", f"{self.base}/v2/orders", params={"status": "open", "limit": 200})

    # --- calendar --------------------------------------------------------------------------
    def trading_days(self, start: dt.date, end: dt.date) -> list:
        cal = self._req("GET", f"{self.base}/v2/calendar", params={"start": str(start), "end": str(end)})
        return [dt.date.fromisoformat(c["date"]) for c in cal]

    # --- data ------------------------------------------------------------------------------
    def daily_closes(self, symbols: list, lookback_days: int, feeds=("sip", "iex")) -> dict:
        """Split- and dividend-adjusted daily closes per symbol. Tries each feed in order."""
        end = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20)
        start = end - dt.timedelta(days=lookback_days)
        last_err = None
        for feed in feeds:
            try:
                out, token = {s: [] for s in symbols}, None
                while True:
                    params = {"symbols": ",".join(symbols), "timeframe": "1Day", "adjustment": "all", "feed": feed,
                              "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                              "limit": 10000}
                    if token:
                        params["page_token"] = token
                    j = self._req("GET", f"{DATA_URL}/v2/stocks/bars", params=params)
                    for sym, bars in (j.get("bars") or {}).items():
                        out[sym] += [(pd.Timestamp(b["t"]).tz_convert("America/New_York").date(), float(b["c"])) for b in bars]
                    token = j.get("next_page_token")
                    if not token:
                        break
                res = {}
                for sym, rows in out.items():
                    if rows:
                        s = pd.Series(dict(rows)).sort_index()
                        s.index = pd.DatetimeIndex(s.index)
                        res[sym] = s
                if len(res) == len(symbols):
                    return res
                last_err = BrokerError(f"feed {feed}: no bars for {sorted(set(symbols) - set(res))}")
            except BrokerError as e:
                last_err = e
        raise last_err

    # --- orders ----------------------------------------------------------------------------
    def submit_market_order(self, symbol: str, qty: int, side: str, client_order_id: str) -> dict:
        """Whole-share market order, good for the day. Sent in the evening, it queues for the next open.
        The client_order_id carries the date, so a second run on the same day cannot double up."""
        body = {"symbol": symbol, "qty": str(int(qty)), "side": side, "type": "market",
                "time_in_force": "day", "client_order_id": client_order_id}
        return self._req("POST", f"{self.base}/v2/orders", json=body)
