"""Builds reports/latest.md from the journal. No broker access needed."""
import glob
import json
import os

import pandas as pd

# From the 20-year backtest of the same rules (backtest/RESULTS.md): rolling 13-week desk returns.
MODEL_13W = {"p05": -0.025, "median": 0.009, "p95": 0.050}
MODEL_MAX_DD = -0.087


def write_report(root: str) -> str:
    jd = os.path.join(root, "journal")
    eq_path = os.path.join(jd, "equity.csv")
    lines = ["# Desk report", ""]
    if not os.path.exists(eq_path):
        lines.append("No trading days recorded yet.")
    else:
        eq = pd.read_csv(eq_path, parse_dates=["date"]).drop_duplicates("date", keep="last").set_index("date")
        e = eq["equity"]
        first, last = e.iloc[0], e.iloc[-1]
        dd = (e / e.cummax() - 1).min()
        lines += [f"Last run: {e.index[-1].date()} ({eq['mode'].iloc[-1]})", "",
                  "| Measure | Value |", "| --- | --- |",
                  f"| Equity | ${last:,.2f} |",
                  f"| Return since first run ({e.index[0].date()}) | {last / first - 1:+.2%} |",
                  f"| Deepest drawdown so far | {dd:.2%} (model's 20-year worst: {MODEL_MAX_DD:.1%}) |",
                  f"| Trading days recorded | {len(e)} |",
                  f"| Orders sent | {int(eq['orders'].sum())} |", ""]
        if len(e) >= 63:
            r13 = e.iloc[-1] / e.iloc[-63] - 1
            inside = MODEL_13W["p05"] <= r13 <= MODEL_13W["p95"]
            lines += [f"Last 13 weeks: {r13:+.2%}. Model range for 13 weeks is {MODEL_13W['p05']:+.1%} to "
                      f"{MODEL_13W['p95']:+.1%} (9 windows in 10). {'Inside' if inside else 'OUTSIDE'} the range.", ""]
        else:
            lines += [f"Fewer than 63 trading days recorded, so there is no 13-week comparison yet "
                      f"({len(e)} of 63).", ""]
    files = sorted(glob.glob(os.path.join(jd, "20*.json")))[-15:]
    rows, problems = [], []
    for f in files:
        j = json.load(open(f))
        for o in j.get("orders_sent", []):
            rows.append(f"| {j['date']} | {o['sleeve']} | {o['side']} | {o['qty']} | {o['symbol']} | "
                        f"{o['reason']} | {o.get('broker_status', j.get('mode'))} |")
        problems += [f"- {j['date']}: {x}" for x in j.get("errors", [])]
        problems += [f"- {j['date']}: rejected {o['side']} {o['symbol']} ({o['why']})" for o in j.get("orders_rejected", [])]
        if j.get("status", "ok") not in ("ok",):
            problems.append(f"- {j['date']}: {j['status']}")
    lines += ["## Orders, last 15 runs", ""]
    lines += (["| Date | Sleeve | Side | Shares | Symbol | Why | Status |", "| --- | --- | --- | --- | --- | --- | --- |"]
              + rows) if rows else ["None."]
    lines += ["", "## Things to look at", ""] + (problems if problems else ["Nothing flagged."])
    out = os.path.join(root, "reports", "latest.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    print(write_report(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
