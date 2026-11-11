# Trading desk (paper account)

Two rule sets run side by side on an Alpaca paper account. A scheduled job runs each weekday evening,
decides, and queues whole-share market orders that fill at the next open. Nobody has to be at a screen.

| Sleeve | Share of account | Holds | Rule |
| --- | --- | --- | --- |
| Trend | 60% in 8 equal slices: SPY, EFA, EEM, VNQ, IEF, TLT, GLD, DBC | Weeks to months | A slice is invested while its ETF's weekly close is above its 40-week average, otherwise cash. Checked on the last trading day of each week. |
| Pullback | 40% in 4 slots: IVV, QQQ, IWM, DIA | 2 to 10 trading days | Buy when the close is above the 200-day average and the 2-day RSI is below 10. Sell when the close is above the 5-day average, or after 10 trading days. |

Backtest numbers and their limits are in `backtest/RESULTS.md`.

## Files

    config.yaml               every setting: symbols, weights, risk limits, dry-run switch
    desk/strategy.py          the rules (shared by the backtest and the live job)
    desk/run_daily.py         the daily job
    desk/risk.py              risk gates and the off switch
    desk/broker.py            Alpaca adapter; refuses anything except the paper URL
    desk/report.py            writes reports/latest.md
    desk/check.py             connection check, sends no orders
    backtest/                 research backtest, data and results
    tests/test_desk.py        offline checks, including "live job reproduces the backtest's trades"
    .github/workflows/        the evening schedule
    journal/, reports/        written by each run

## Setup

1. Open a paper-only account at alpaca.markets and generate paper API keys.
2. Create a private GitHub repository and upload this folder.
3. In the repository: Settings > Secrets and variables > Actions. Add `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY`.
4. Actions tab > "Desk daily run" > Run workflow. The first runs are dry runs: decisions are logged, nothing is sent.
5. After a week of clean dry runs, set `submit_orders: true` in `config.yaml`.

To run on your own computer instead: `pip install -r requirements.txt`, copy `.env.example` to `.env`,
fill it in, export the two variables, then `python -m desk.check` and `python -m desk.run_daily`.

## Switching it off

Add an empty file named `HALT` to the top folder of the repository. The next run does nothing and says so
in the journal. Delete the file to resume. Positions stay as they are while halted.

## What has and has not been tested

Tested offline: the indicators, the risk gates, the off switch, repeat runs, and that the live job stepped day
by day through simulated prices makes the same pullback trades as the backtest.

Not tested: the adapter against Alpaca's real servers. The build environment could not reach them.
`python -m desk.check` is the first thing to run, and the dry-run week exists to catch anything it misses.
