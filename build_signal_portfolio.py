"""
build_signal_portfolio.py
--------------------------
Builds period-level and monthly portfolio series from the SIGNAL-FILTERED
trade book (results/buy_call_trades_detail.csv), replacing the old fixed
top-30-by-market-cap book used by portfolio_levered_call.py.

Strategy definition used here:
  - Each period, only tickers that actually fired a BUY CALL signal are
    traded (n_signals_period can be < 30 or > 30).
  - At most 30 active slots per period. When more than 30 signal, the
    top 30 by premium_pct (richer/more volatile option) are kept active;
    the rest sit out (contribution_pct = 0, capital stays in cash).
  - Portfolio pnl per period = sum(contribution_pct) across all signaling
    tickers that period (contribution_pct already = pnl_pct / 30 for
    active slots, 0 otherwise) — i.e. unsignaled slots contribute 0, not
    a fixed top-30 roster.
  - Idle capital (not used as option premium) earns plain money-market,
    UNLEVERED. A 4x-leveraged variant is also produced for comparison,
    scaling only the option P&L and capital_used by 4x (same idle-cash
    money-market logic as portfolio_levered_call.py).

Outputs (results/):
  signal_periods_B.csv     — unlevered, period-level
  signal_periods_C.csv     — 4x leveraged, period-level
  signal_monthly_B.csv     — monthly table (unlevered), attributed by period midpoint
  signal_monthly_C.csv     — monthly table (4x leveraged), attributed by period midpoint
"""

import os
import csv
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(BASE, "results")
DETAIL_CSV = os.path.join(RES, "buy_call_trades_detail.csv")

MAX_SLOTS = 30
PPY       = 13.0
LEVERAGE  = 4.0

MMKT_ANNUAL = {
    "2020": 0.0008,
    "2021": 0.0008,
    "2022": 0.0200,
    "2023": 0.0500,
    "2024": 0.0510,
    "2025": 0.0430,
    "2026": 0.0430,
}


def mmkt_period(period_start: str) -> float:
    year = period_start[:4]
    annual = MMKT_ANNUAL.get(year, 0.04)
    return annual / PPY * 100


def build_periods():
    df = pd.read_csv(DETAIL_CSV)

    rows_b, rows_c = [], []
    for ps, grp in df.groupby("period_start"):
        active = grp[grp["active_slot"] == 1]
        n_signals = int(grp["n_signals_period"].iloc[0])

        avg_pnl  = active["pnl_pct"].sum() / MAX_SLOTS
        avg_prem = active["premium_pct"].sum() / MAX_SLOTS

        mm = mmkt_period(ps)
        tickers = "|".join(sorted(active["ticker"].tolist()))

        # B — unlevered, idle cash earns money-market
        idle_b      = max(0.0, 1.0 - avg_prem / 100.0)
        mm_contrib_b = idle_b * mm
        pnl_b        = avg_pnl + mm_contrib_b
        rows_b.append({
            "period_start": ps, "n_signals": n_signals, "n_active": len(active),
            "option_pnl": round(avg_pnl, 4), "mm_contrib": round(mm_contrib_b, 4),
            "total_pnl": round(pnl_b, 4), "avg_premium_pct": round(avg_prem, 4),
            "tickers": tickers,
        })

        # C — 4x leveraged, idle cash earns money-market
        capital_used = LEVERAGE * avg_prem / 100.0
        idle_c       = max(0.0, 1.0 - capital_used)
        mm_contrib_c = idle_c * mm
        pnl_c        = LEVERAGE * avg_pnl + mm_contrib_c
        rows_c.append({
            "period_start": ps, "n_signals": n_signals, "n_active": len(active),
            "option_pnl": round(LEVERAGE * avg_pnl, 4), "mm_contrib": round(mm_contrib_c, 4),
            "total_pnl": round(pnl_c, 4), "avg_premium_pct": round(avg_prem, 4),
            "tickers": tickers,
        })

    rows_b.sort(key=lambda r: r["period_start"])
    rows_c.sort(key=lambda r: r["period_start"])

    cum_b = 0.0
    for r in rows_b:
        cum_b += r["total_pnl"]
        r["cumulative_pct"] = round(cum_b, 4)
    cum_c = 0.0
    for r in rows_c:
        cum_c += r["total_pnl"]
        r["cumulative_pct"] = round(cum_c, 4)

    fieldnames = ["period_start", "n_signals", "n_active", "option_pnl",
                  "mm_contrib", "total_pnl", "cumulative_pct",
                  "avg_premium_pct", "tickers"]

    for label, rows in (("B", rows_b), ("C", rows_c)):
        out = os.path.join(RES, f"signal_periods_{label}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"✓ {out}  ({len(rows)} periods)")

    return rows_b, rows_c


def build_monthly(rows, label):
    """Monthly table using ONLY periods whose period_start falls in the
    midpoint of the period (period_start + 14 days, i.e. the centre of a
    ~4-week cycle). Every period contributes its full P&L to exactly one
    month — no periods are dropped — and the assignment reflects which
    month the bulk of the period actually occurred in, rather than an
    arbitrary 'which month did it start in' rule."""
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    by_year_month = defaultdict(float)
    for r in rows:
        d = datetime.strptime(r["period_start"], "%Y-%m-%d")
        mid = d + timedelta(days=14)
        by_year_month[(mid.year, mid.month)] += r["total_pnl"]

    years = sorted({y for y, m in by_year_month})
    table_rows = []
    for y in years:
        row = {"Year": y}
        annual = 0.0
        any_val = False
        for mi, mname in enumerate(months, 1):
            v = by_year_month.get((y, mi))
            if v is not None:
                row[mname] = round(v, 4)
                annual += v
                any_val = True
            else:
                row[mname] = ""
        row["Annual"] = round(annual, 4) if any_val else ""
        table_rows.append(row)

    out = os.path.join(RES, f"signal_monthly_{label}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Year"] + months + ["Annual"])
        w.writeheader()
        w.writerows(table_rows)
    print(f"✓ {out}  ({len(table_rows)} years; {len(rows)} periods, "
          f"all attributed by midpoint month)")
    return table_rows


if __name__ == "__main__":
    rows_b, rows_c = build_periods()
    build_monthly(rows_b, "B")
    build_monthly(rows_c, "C")
