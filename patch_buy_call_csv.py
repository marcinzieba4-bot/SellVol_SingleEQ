"""
patch_buy_call_csv.py
─────────────────────
Post-processes results/buy_call_trades_detail.csv to:

  1. Detect split-affected rows (stock_entry/stock_expiry ≈ known split ratio
     AND strike ≈ stock_entry) and correct stock_expiry, payoff, pnl_dollar,
     pnl_pct, current_return_for_underlying.

  2. Cap the book at 30 active slots per period (a real portfolio can only
     ever hold 30 positions). In periods where MORE than 30 tickers signal
     BUY CALL, the previous version divided every signal's pnl by a fixed
     1/30 with no cap — implicitly assuming up to 38/30 = 1.27x notional
     capital in those periods, an unacknowledged leverage bug (confirmed in
     10 of 74 periods, e.g. 2023-04-17 had 38 signals). This version caps
     the book at 30 per period, prioritizing the higher-premium_pct names
     (≈ richer/more volatile option, a proxy for realized volatility) when
     more than 30 signal that period. Crowded-out names get
     contribution_pct = 0 (their capital sits in cash instead of being
     phantom-deployed).

  3. Add new columns:
       contribution_pct  — pnl_pct × (1/30), only for active_slot rows
       split_factor      — split ratio applied (1 = no split detected)
       active_slot       — 1 if this trade occupies one of the 30 book
                           slots that period, 0 if crowded out
       n_signals_period  — how many tickers signaled BUY CALL that period
                           (for transparency — can exceed 30)

Run without S3 / AWS credentials needed.
"""
import csv, math, os
from collections import defaultdict

IN_PATH  = "results/buy_call_trades_detail.csv"
OUT_PATH = "results/buy_call_trades_detail.csv"

SPLIT_RATIOS   = [20, 10, 5, 4, 3, 2]
RATIO_TOL      = 0.15   # ±15% of split ratio
STRIKE_TOL     = 0.15   # strike must be within 15% of stock_entry
WEIGHT         = 1 / 30
MAX_SLOTS      = 30


def sf(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def detect_split(stock_entry, stock_expiry, strike):
    """Return split multiplier if split detected, else 1."""
    if not (stock_entry and stock_expiry and strike):
        return 1
    ratio = stock_entry / stock_expiry
    for r in SPLIT_RATIOS:
        if abs(ratio - r) / r < RATIO_TOL:
            if abs(stock_entry - strike) / stock_entry < STRIKE_TOL:
                return r
    return 1


def main():
    with open(IN_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} rows from {IN_PATH}")

    new_rows  = []
    n_fixed   = 0

    for row in rows:
        se     = sf(row["stock_entry"])
        sx     = sf(row["stock_expiry"])
        strike = sf(row["strike"])
        prem   = sf(row["premium"])

        split_factor = detect_split(se, sx, strike)

        if split_factor > 1:
            sx_adj   = round(sx * split_factor, 4)
            payoff   = round(max(0.0, sx_adj - strike), 2)
            pnl_dol  = round(payoff - prem, 2) if prem is not None else None
            pnl_pct  = round(pnl_dol / se * 100, 4) if (pnl_dol is not None and se) else None
            curr_ret = round((sx_adj / se - 1) * 100, 2) if (se and sx_adj) else None

            print(f"  SPLIT {split_factor}:1  {row['period_start']}  {row['ticker']:<6} "
                  f"entry={se:.2f}  raw_expiry={sx:.2f}  adj_expiry={sx_adj:.2f}  "
                  f"payoff {row['payoff']}→{payoff}  "
                  f"pnl_pct {row['pnl_pct']}→{pnl_pct}%")

            row["stock_expiry"]                  = sx_adj
            row["payoff"]                        = payoff
            row["pnl_dollar"]                    = pnl_dol
            row["pnl_pct"]                       = pnl_pct
            row["current_return_for_underlying"] = curr_ret
            n_fixed += 1

        row["split_factor"] = split_factor
        new_rows.append(row)

    # ── cap at 30 active slots per period, prioritized by premium_pct ───────
    by_period = defaultdict(list)
    for row in new_rows:
        by_period[row["period_start"]].append(row)

    n_capped_periods = 0
    n_crowded_out     = 0

    for period, prows in by_period.items():
        n_signals = len(prows)
        prows.sort(key=lambda r: (sf(r["premium_pct"]) or 0.0), reverse=True)
        active = prows[:MAX_SLOTS]
        crowded = prows[MAX_SLOTS:]

        if n_signals > MAX_SLOTS:
            n_capped_periods += 1
            n_crowded_out    += len(crowded)
            print(f"  CAP  {period}: {n_signals} signals → kept top {MAX_SLOTS} "
                  f"by premium_pct, crowded out {len(crowded)}")

        for r in prows:
            r["n_signals_period"] = n_signals
            r["active_slot"] = 1 if r in active else 0
            pnl_pct_val = sf(r["pnl_pct"])
            r["contribution_pct"] = (round(pnl_pct_val * WEIGHT, 6)
                                     if (pnl_pct_val is not None and r["active_slot"]) else 0.0)

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

    # restore original sort order (period_start, ticker)
    new_rows.sort(key=lambda r: (r["period_start"], r["ticker"]))

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(new_rows)

    print(f"\nDone. {n_fixed} rows split-corrected.")
    print(f"      {n_capped_periods} periods exceeded 30 signals; "
          f"{n_crowded_out} trades crowded out of the book (capital left in cash).")
    print(f"      {len(new_rows)} rows written to {OUT_PATH}")


if __name__ == "__main__":
    main()
