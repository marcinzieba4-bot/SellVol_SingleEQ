"""
export_buy_call_trades.py
─────────────────────────
Generates a per-trade CSV for the Buy Call strategy with:
  • period_start
  • ticker
  • premium          — ATM-adjusted dollar premium paid at entry
  • premium_pct      — premium as % of stock_entry (cost / notional)
  • underlying_ret_in_prev_period  — 4-week return that triggered the signal (%)
  • current_return_for_underlying  — underlying return during the option period (%)
  • pnl_dollar       — option P&L in dollars per share
  • pnl_pct          — P&L as % of stock_entry (net return; max loss = −premium_pct)
  • contribution_pct — pnl_pct × (1/30): weighted contribution to equal-30 portfolio
                       (0 if crowded out — see active_slot below)
  • active_slot      — 1 if this trade occupies one of the (max 30) book slots
                       that period, 0 if more than 30 tickers signaled that
                       period and this one was crowded out (lower premium_pct
                       = lower priority; capital stays in cash instead)
  • n_signals_period — how many tickers signaled BUY CALL that period (can
                       exceed 30 — the portfolio cannot hold more than 30)
  • trade            — "BUY CALL" or "NO TRADE" / "NO DATA"
  • signal           — True/False
  • stock_entry      — stock price at entry
  • stock_expiry     — stock price at expiry (settlement proxy, split-corrected)
  • strike           — option strike used
  • payoff           — intrinsic value at expiry (split-corrected)
  • split_factor     — detected split ratio applied to expiry price (1 = no split)

Only rows where trade == "BUY CALL" are written (active trades only).

Usage:
    python export_buy_call_trades.py
Output:
    results/buy_call_trades_detail.csv
"""

import os, json, csv, io, contextlib, re
from collections import defaultdict
import sys
sys.path.insert(0, os.path.dirname(__file__))

MAX_SLOTS = 30

from strategy_vol_premium import run_strategy

SPX50_JSON = "spx50_periods.json"
OUT_PATH   = "results/buy_call_trades_detail.csv"

S3_TICKERS = frozenset({
    "AAPL","ABBV","ABT","ACN","ADBE","AMZN","AVGO","BAC","BX","CMCSA",
    "COST","CRM","CVX","DIS","GOOG","GOOGL","HD","IBM","INTU","JNJ",
    "JPM","KO","LLY","MA","META","MRK","MSFT","NFLX","NKE","NOW",
    "NVDA","ORCL","PEP","PG","QCOM","TMO","TSLA","UNH","V","WFC","WMT","XOM",
})

_PCT_RE = re.compile(r"\(([+-]?\d+\.?\d*)%\)")

def parse_signal_pct(signal_detail: str) -> float | None:
    """Extract the percentage change from signal_detail string like '150.00→160.00 (+6.7%)'."""
    if not signal_detail or signal_detail.startswith("n/a"):
        return None
    m = _PCT_RE.search(signal_detail)
    return float(m.group(1)) if m else None


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading SPX50 period rankings...")
    with open(SPX50_JSON) as f:
        spx50 = json.load(f)

    needed_tickers = sorted(S3_TICKERS & {s["ticker"] for v in spx50.values() for s in v})
    print(f"  Tickers to process: {len(needed_tickers)}")

    rows = []
    failed = []

    for i, ticker in enumerate(needed_tickers, 1):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                results = run_strategy(ticker, mode="buy_call")
        except Exception as e:
            failed.append((ticker, str(e)))
            print(f"  [{i:>2}/{len(needed_tickers)}] {ticker:<8} FAILED: {e}")
            continue

        for r in results:
            if r["trade"] != "BUY CALL":
                continue  # skip no-trades; only active buy-call rows

            se  = r["stock_entry"]
            sx  = r["stock_expiry"]
            prem = r["premium"]

            premium_pct = (prem / se * 100) if (prem and se) else None
            curr_ret    = ((sx / se - 1) * 100) if (se and sx) else None
            prev_ret    = parse_signal_pct(r.get("signal_detail", ""))

            rows.append({
                "period_start":                  r["period_start"],
                "period_end":                    r["period_end"],
                "ticker":                        ticker,
                "premium":                       prem,
                "premium_pct":                   round(premium_pct, 4) if premium_pct is not None else None,
                "underlying_ret_in_prev_period": round(prev_ret, 2)    if prev_ret    is not None else None,
                "current_return_for_underlying": round(curr_ret, 2)    if curr_ret    is not None else None,
                "pnl_dollar":                    r["pnl_dollar"],
                "pnl_pct":                       r["pnl_pct"],
                "stock_entry":                   se,
                "stock_expiry":                  sx,
                "strike":                        r["strike"],
                "payoff":                        r["payoff"],
                "atm_correction":                r["atm_correction"],
                "carried":                       r["carried"],
                "split_factor":                  r.get("split_factor", 1),
            })

        print(f"  [{i:>2}/{len(needed_tickers)}] {ticker:<8} "
              f"{sum(1 for r in results if r['trade'] == 'BUY CALL'):>3} trades", flush=True)

    if failed:
        print(f"\nWARNING — {len(failed)} failures:")
        for t, e in failed:
            print(f"  {t}: {e}")

    # ── cap at 30 active slots per period, prioritized by premium_pct ───────
    # A real book can hold at most 30 positions. When more than 30 tickers
    # signal in a period, keep the 30 with the highest premium_pct (a proxy
    # for richer/more-volatile options) and leave the rest uninvested
    # (contribution_pct = 0) instead of silently over-allocating capital.
    by_period = defaultdict(list)
    for row in rows:
        by_period[row["period_start"]].append(row)

    for period, prows in by_period.items():
        n_signals = len(prows)
        prows.sort(key=lambda r: (r["premium_pct"] or 0.0), reverse=True)
        active_ids = {id(r) for r in prows[:MAX_SLOTS]}
        if n_signals > MAX_SLOTS:
            print(f"  CAP  {period}: {n_signals} signals → kept top {MAX_SLOTS} by premium_pct")
        for r in prows:
            r["n_signals_period"] = n_signals
            r["active_slot"] = 1 if id(r) in active_ids else 0
            r["contribution_pct"] = (round(r["pnl_pct"] / MAX_SLOTS, 6)
                                     if (r["pnl_pct"] is not None and r["active_slot"]) else 0.0)

    # Sort by period_start then ticker
    rows.sort(key=lambda r: (r["period_start"], r["ticker"]))

    fieldnames = [
        "period_start", "period_end", "ticker",
        "premium", "premium_pct",
        "underlying_ret_in_prev_period",
        "current_return_for_underlying",
        "pnl_dollar", "pnl_pct",
        "contribution_pct", "active_slot", "n_signals_period",
        "stock_entry", "stock_expiry", "strike", "payoff",
        "atm_correction", "carried", "split_factor",
    ]
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. {len(rows)} buy-call trades written to {OUT_PATH}")


if __name__ == "__main__":
    main()
