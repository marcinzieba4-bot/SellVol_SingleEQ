"""
simulate_recent_periods.py
---------------------------
Extends results/buy_call_trades_detail.csv with recent periods that have
no S3 options data yet (no AWS/S3 access in this environment), using a
VIX-based Black-Scholes approximation for the ATM call premium instead of
a real option-chain quote.

Premium model (per period):
    sigma   = VIX level at period_start / 100
    T       = 28 / 365                      (one ~4-week cycle)
    premium = 0.398 * stock_entry * sigma * sqrt(T)
            (standard at-the-money Black-Scholes approximation,
             C ≈ 0.4 * S * sigma * sqrt(T) at zero rates)

The same VIX-derived sigma is applied to every ticker in a given period
(a market-wide volatility proxy) since real per-ticker implied vols are
not available for this window — this should be treated as indicative,
not a precise historical fill.

Signal logic mirrors strategy_vol_premium.run_strategy(mode="buy_call"):
buy ATM call only when the stock's price rose over the prior ~4-week
period. Same 30-active-slot cap, prioritized by premium_pct, as the rest
of the pipeline.

Run manually whenever new anchor dates need to be appended; re-run
build_signal_portfolio.py afterwards to fold the new rows into the
period/monthly CSVs.
"""

import csv
import math
import datetime
import urllib.request
import json
import time

TICKERS = [
    "AAPL","ABBV","ABT","ACN","ADBE","AMZN","AVGO","BAC","BX","CMCSA",
    "COST","CRM","CVX","DIS","GOOG","GOOGL","HD","IBM","INTU","JNJ",
    "JPM","KO","LLY","MA","META","MRK","MSFT","NFLX","NKE","NOW",
    "NVDA","ORCL","PEP","PG","QCOM","TMO","TSLA","UNH","V","WFC","WMT","XOM",
]

MAX_SLOTS = 30
T_YEARS   = 28 / 365.0
BS_COEF   = 0.398
DETAIL_CSV = "results/buy_call_trades_detail.csv"

FIELDNAMES = [
    "period_start", "period_end", "ticker",
    "premium", "premium_pct",
    "underlying_ret_in_prev_period",
    "current_return_for_underlying",
    "pnl_dollar", "pnl_pct",
    "contribution_pct", "active_slot", "n_signals_period",
    "stock_entry", "stock_expiry", "strike", "payoff",
    "atm_correction", "carried", "split_factor",
]


def fetch_series(symbol, start, end):
    p1 = int(datetime.datetime.combine(start, datetime.time()).timestamp())
    p2 = int(datetime.datetime.combine(end, datetime.time()).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?period1={p1}&period2={p2}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        j = json.load(r)
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    return [(datetime.datetime.utcfromtimestamp(t).date(), c)
            for t, c in zip(ts, closes) if c is not None]


def nearest(series, target):
    return min(series, key=lambda kv: abs((kv[0] - target).days))


def simulate(anchors, range_start, range_end):
    """anchors: list of ISO date strings, oldest first, spaced ~28 days.
    Simulates one period per consecutive pair (anchors[i] -> anchors[i+1])."""
    vix_series = fetch_series("%5EVIX", range_start, range_end)
    vix_at = {a: nearest(vix_series, datetime.date.fromisoformat(a))[1] for a in anchors}

    price_data = {}
    for i, t in enumerate(TICKERS, 1):
        price_data[t] = fetch_series(t, range_start, range_end)
        time.sleep(0.2)

    all_rows = []
    for i in range(1, len(anchors) - 1):
        ps, pe = anchors[i], anchors[i + 1]
        prev = anchors[i - 1]
        sigma = vix_at[ps] / 100.0
        bs_factor = BS_COEF * sigma * math.sqrt(T_YEARS)

        period_rows = []
        for ticker in TICKERS:
            series = price_data[ticker]
            entry_px = nearest(series, datetime.date.fromisoformat(ps))[1]
            prev_px = nearest(series, datetime.date.fromisoformat(prev))[1]
            expiry_px = nearest(series, datetime.date.fromisoformat(pe))[1]

            if not (entry_px > prev_px):
                continue  # no signal

            strike = entry_px
            premium = bs_factor * entry_px
            payoff = max(0.0, expiry_px - strike)
            pnl_dollar = payoff - premium
            pnl_pct = pnl_dollar / entry_px * 100
            premium_pct = premium / entry_px * 100

            period_rows.append({
                "period_start": ps, "period_end": pe, "ticker": ticker,
                "premium": round(premium, 4), "premium_pct": round(premium_pct, 4),
                "underlying_ret_in_prev_period": round((entry_px / prev_px - 1) * 100, 2),
                "current_return_for_underlying": round((expiry_px / entry_px - 1) * 100, 2),
                "pnl_dollar": round(pnl_dollar, 4), "pnl_pct": round(pnl_pct, 4),
                "stock_entry": round(entry_px, 4), "stock_expiry": round(expiry_px, 4),
                "strike": round(strike, 4), "payoff": round(payoff, 4),
                "atm_correction": 0.0, "carried": False, "split_factor": 1,
            })

        n_signals = len(period_rows)
        period_rows.sort(key=lambda r: -r["premium_pct"])
        active_ids = {id(r) for r in period_rows[:MAX_SLOTS]}
        for r in period_rows:
            r["n_signals_period"] = n_signals
            r["active_slot"] = 1 if id(r) in active_ids else 0
            r["contribution_pct"] = (round(r["pnl_pct"] / MAX_SLOTS, 6)
                                      if r["active_slot"] else 0.0)
        all_rows.extend(period_rows)
        print(f"{ps}: VIX={vix_at[ps]:.2f}  n_signals={n_signals}  "
              f"active={len(active_ids)}")

    return all_rows


if __name__ == "__main__":
    # anchors[0] is only used as the "previous" price reference for the
    # first simulated period; the period actually simulated starts at
    # anchors[1].
    anchors = ["2026-02-13", "2026-03-13", "2026-04-10", "2026-05-08", "2026-06-05"]
    rows = simulate(anchors, datetime.date(2026, 1, 1), datetime.date(2026, 6, 22))

    with open(DETAIL_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writerows(rows)

    print(f"\nAppended {len(rows)} simulated rows to {DETAIL_CSV}")
