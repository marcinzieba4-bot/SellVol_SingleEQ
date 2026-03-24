"""
patch_buy_call_csv.py
─────────────────────
Post-processes results/buy_call_trades_detail.csv to:

  1. Detect split-affected rows (stock_entry/stock_expiry ≈ known split ratio
     AND strike ≈ stock_entry) and correct stock_expiry, payoff, pnl_dollar,
     pnl_pct, current_return_for_underlying.

  2. Add two new columns:
       contribution_pct  — pnl_pct × (1/30): weighted contribution to the
                           equal-30 portfolio (max loss = −premium_pct / 30)
       split_factor      — split ratio applied (1 = no split detected)

Run without S3 / AWS credentials needed.
"""
import csv, math, os

IN_PATH  = "results/buy_call_trades_detail.csv"
OUT_PATH = "results/buy_call_trades_detail.csv"

SPLIT_RATIOS   = [20, 10, 5, 4, 3, 2]
RATIO_TOL      = 0.15   # ±15% of split ratio
STRIKE_TOL     = 0.15   # strike must be within 15% of stock_entry
WEIGHT         = 1 / 30


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

        # Add new columns
        pnl_pct_val = sf(row["pnl_pct"])
        row["contribution_pct"] = (round(pnl_pct_val * WEIGHT, 6)
                                   if pnl_pct_val is not None else None)
        row["split_factor"] = split_factor
        new_rows.append(row)

    fieldnames = [
        "period_start", "period_end", "ticker",
        "premium", "premium_pct",
        "underlying_ret_in_prev_period",
        "current_return_for_underlying",
        "pnl_dollar", "pnl_pct",
        "contribution_pct",
        "stock_entry", "stock_expiry", "strike", "payoff",
        "atm_correction", "carried", "split_factor",
    ]

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(new_rows)

    print(f"\nDone. {n_fixed} rows split-corrected. {len(new_rows)} rows written to {OUT_PATH}")


if __name__ == "__main__":
    main()
