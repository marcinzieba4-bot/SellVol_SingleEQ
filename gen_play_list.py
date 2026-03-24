#!/usr/bin/env python3
"""
Generate current_play_list.json for the next buy-call period.

Signal logic (mirrors backtest, S3-native):
  For each ticker, load all real S3 option periods.
  Signal period = second-to-last S3 period  (N-1)
  Data period   = last S3 period             (N)

  If stock_price[N] > stock_price[N-1]  →  BUY CALL
  (= positive return during the most recently recorded period)

  Premium  = entry_premium from period N (real S3 data, not BS estimate)
  Strike   = stock_price from period N   (ATM reference at last observation)

  No minimum-return filter applied — any positive return qualifies.

Output: current_play_list.json
"""

import json
import os
import sys
from datetime import datetime

# ── shared S3 loading from strategy_vol_premium ───────────────────────────────
from strategy_vol_premium import load_s3_periods, fill_gaps, MIN_PREMIUM_FLOOR, MIN_PREMIUM_PCT

from backtest import TICKERS

# ── Next play window (last Friday of March → last Friday of April 2026) ───────
NEXT_ENTRY  = "2026-03-28"
NEXT_EXPIRY = "2026-04-24"
DTE         = 27

# ──────────────────────────────────────────────────────────────────────────────

def build_play_list() -> list[dict]:
    plays      = []
    no_signal  = []
    skipped    = []

    for ticker in TICKERS:
        # ── load S3 periods ───────────────────────────────────────────────────
        try:
            records = load_s3_periods(ticker)
        except Exception as e:
            print(f"  ERROR loading S3 for {ticker}: {e}")
            skipped.append(ticker)
            continue

        if not records:
            print(f"  WARNING: no S3 data for {ticker}")
            skipped.append(ticker)
            continue

        # fill gaps so period numbering is stable (but skip synthetics for signal)
        records = fill_gaps(records)
        real    = [r for r in records if not r.get("synthetic")]

        if len(real) < 2:
            print(f"  WARNING: fewer than 2 real periods for {ticker}, skipping")
            skipped.append(ticker)
            continue

        period_n1 = real[-2]   # signal: one before last
        period_n  = real[-1]   # data:   most recent

        px_n1 = period_n1.get("stock_price")
        px_n  = period_n.get("stock_price")

        if px_n1 is None or px_n is None or px_n1 <= 0:
            print(f"  WARNING: missing prices for {ticker}, skipping")
            skipped.append(ticker)
            continue

        prev_return = (px_n - px_n1) / px_n1 * 100.0

        if prev_return <= 0:
            no_signal.append({
                "ticker":      ticker,
                "prev_return": round(prev_return, 2),
                "period_n1":   period_n1["period_start"],
                "period_n":    period_n["period_start"],
            })
            continue

        # ── extract premium from S3 period N ─────────────────────────────────
        raw_prem = period_n.get("entry_premium")
        strike   = period_n.get("strike") or px_n  # fallback to stock price if no strike

        if raw_prem is None or raw_prem < MIN_PREMIUM_FLOOR:
            print(f"  WARNING: {ticker} premium {raw_prem} below floor, skipping")
            skipped.append(ticker)
            continue

        prem_pct = raw_prem / px_n * 100.0
        if prem_pct < MIN_PREMIUM_PCT * 100:
            print(f"  WARNING: {ticker} premium_pct {prem_pct:.2f}% too low (near-expiry data?), skipping")
            skipped.append(ticker)
            continue

        plays.append({
            "ticker":        ticker,
            "signal":        "buy_call",
            "prev_return":   round(prev_return, 2),
            "period_n1":     period_n1["period_start"],
            "period_n":      period_n["period_start"],
            "period_n_end":  period_n["period_end"],
            "stock_price_n": round(px_n,    2),
            "strike":        round(strike,  2),
            "premium":       round(raw_prem, 2),   # dollars per share
            "premium_pct":   round(prem_pct, 3),   # % of stock price at last obs
        })

    # ── sort by prev_return descending ───────────────────────────────────────
    plays.sort(key=lambda x: -x["prev_return"])

    # ── equal weight across qualifying plays ──────────────────────────────────
    weight = 1.0 / len(plays) if plays else 0.0
    for p in plays:
        p["weight"] = round(weight, 4)

    return plays, no_signal, skipped


# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═'*65}")
    print("  Generating buy-call play list from S3 options data ...")
    print(f"  Next play: {NEXT_ENTRY} → {NEXT_EXPIRY}  ({DTE} DTE)")
    print(f"{'═'*65}\n")

    plays, no_signal, skipped = build_play_list()

    if not plays:
        print("\nNo qualifying stocks found.")
        sys.exit(1)

    # ── save ──────────────────────────────────────────────────────────────────
    output = {
        "generated_at":    datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "next_entry":      NEXT_ENTRY,
        "next_expiry":     NEXT_EXPIRY,
        "dte":             DTE,
        "premium_source":  "s3_last_period",
        "n_plays":         len(plays),
        "n_no_signal":     len(no_signal),
        "n_skipped":       len(skipped),
        "equal_weight":    round(1.0 / len(plays), 4) if plays else 0,
        "plays":           plays,
        "no_signal":       no_signal,
    }

    out_path = os.path.join(os.path.dirname(__file__), "current_play_list.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── print table ───────────────────────────────────────────────────────────
    print(f"\n  {'TICKER':<7} {'PREV_RET%':>9} {'S3_PRICE':>9} {'STRIKE':>8} "
          f"{'PREM$':>7} {'PREM%':>7}  PERIOD_N")
    print(f"  {'─'*7} {'─'*9} {'─'*9} {'─'*8} {'─'*7} {'─'*7}  {'─'*10}")
    for p in plays:
        print(f"  {p['ticker']:<7} {p['prev_return']:>+9.2f} "
              f"{p['stock_price_n']:>9.2f} {p['strike']:>8.2f} "
              f"{p['premium']:>7.2f} {p['premium_pct']:>7.3f}%"
              f"  {p['period_n']}")

    avg_prem = sum(p["premium_pct"] for p in plays) / len(plays)
    print(f"\n  {len(plays)} stocks qualifying  |  avg premium {avg_prem:.3f}%  "
          f"|  weight {100/len(plays):.1f}% each")
    print(f"  {len(no_signal)} stocks: no signal (prev return <= 0%)")
    if skipped:
        print(f"  {len(skipped)} skipped (data issues): {', '.join(skipped)}")
    print(f"\n  Saved → {out_path}\n")


if __name__ == "__main__":
    main()
