"""
compute_monthly_returns.py
---------------------------
Reads results/buy_call_trades_detail.csv and computes a monthly return table
using Variant C logic: 4× leverage + rest in money-market.

Period-to-month rule: whichever calendar month holds the MAJORITY of days
in [period_start, period_end] (both endpoints inclusive) gets the full period.
"""

import csv
from datetime import datetime, timedelta
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
LEVERAGE   = 4.0
PPY        = 13.0   # ~4-week periods per year

MMKT_ANNUAL = {
    "2020": 0.0008,
    "2021": 0.0008,
    "2022": 0.0200,
    "2023": 0.0500,
    "2024": 0.0510,
    "2025": 0.0430,
    "2026": 0.0430,
}

CSV_PATH = "results/buy_call_trades_detail.csv"

# ── Helpers ───────────────────────────────────────────────────────────────────

def majority_month(period_start: str, period_end: str) -> str:
    """Return YYYY-MM for the month that contains the most calendar days
    in [period_start … period_end] (both endpoints inclusive)."""
    start = datetime.strptime(period_start, "%Y-%m-%d")
    end   = datetime.strptime(period_end,   "%Y-%m-%d")
    counts: dict[str, int] = defaultdict(int)
    d = start
    while d <= end:
        counts[f"{d.year}-{d.month:02d}"] += 1
        d += timedelta(days=1)
    return max(counts, key=counts.__getitem__)


def mmkt_period(period_start: str) -> float:
    """Per-period money-market return (%), using the year of period_start."""
    year   = period_start[:4]
    annual = MMKT_ANNUAL.get(year, 0.04)
    return annual / PPY * 100.0

# ── Read CSV ──────────────────────────────────────────────────────────────────

rows_by_period: dict[tuple, list] = defaultdict(list)

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["period_start"], row["period_end"])
        rows_by_period[key].append(row)

print(f"Loaded {sum(len(v) for v in rows_by_period.values())} trade rows "
      f"across {len(rows_by_period)} periods.\n")

# ── Variant C per period ──────────────────────────────────────────────────────

period_summary = []

for (ps, pe), trades in sorted(rows_by_period.items()):
    # contribution_pct = pnl_pct / 30  (fixed 30-slot portfolio)
    # sum across active tickers gives the equal-weight portfolio return
    # (inactive slots contribute 0, correctly)
    portfolio_ret = sum(float(t["contribution_pct"]) for t in trades)

    # Premium fraction of total capital: sum(premium_pct) / 30
    # (same 30-slot denominator keeps inactive slots at 0 cost)
    prem_pcts = [float(t["premium_pct"]) if t["premium_pct"].strip() else 0.0
                 for t in trades]
    avg_prem_30 = sum(prem_pcts) / 30.0   # always divide by 30

    mm            = mmkt_period(ps)
    capital_used  = LEVERAGE * avg_prem_30 / 100.0
    idle_fraction = max(0.0, 1.0 - capital_used)
    mm_contrib    = idle_fraction * mm
    period_pnl_C  = LEVERAGE * portfolio_ret + mm_contrib

    maj_month = majority_month(ps, pe)

    period_summary.append({
        "period_start":   ps,
        "period_end":     pe,
        "majority_month": maj_month,
        "n_tickers":      len(trades),
        "portfolio_ret":  portfolio_ret,
        "avg_prem_30":    avg_prem_30,
        "option_pnl":     LEVERAGE * portfolio_ret,
        "mm_contrib":     mm_contrib,
        "period_pnl_C":   period_pnl_C,
    })

# ── Aggregate by majority month ───────────────────────────────────────────────

monthly: dict[str, float] = defaultdict(float)
for p in period_summary:
    monthly[p["majority_month"]] += p["period_pnl_C"]

# ── Print detail (period → month assignments) ─────────────────────────────────

print("Period detail:")
print(f"  {'period_start':<12} {'period_end':<12} {'maj_month':<8} "
      f"{'n':>3} {'opt_pnl':>8} {'mm':>6} {'total_C':>8}")
print("  " + "─" * 66)
for p in period_summary:
    print(f"  {p['period_start']:<12} {p['period_end']:<12} {p['majority_month']:<8} "
          f"{p['n_tickers']:>3} {p['option_pnl']:>+8.2f}% {p['mm_contrib']:>+6.3f}% "
          f"{p['period_pnl_C']:>+8.2f}%")

# ── Monthly return table ──────────────────────────────────────────────────────

MN     = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
months = sorted(monthly.keys())
years  = sorted({m[:4] for m in months})

print()
print("=" * 105)
print("  MONTHLY RETURNS — Variant C  (4× Leverage + Money Market)")
print("  Period → month rule: majority of calendar days")
print("=" * 105)
header = "  Year   " + "".join(f"  {m:>6}" for m in MN) + "  Annual"
print(header)
print("  " + "─" * 101)

for yr in years:
    annual = 0.0
    parts  = [f"  {yr}  "]
    for mi in range(1, 13):
        key = f"{yr}-{mi:02d}"
        val = monthly.get(key)
        if val is not None:
            annual += val
            parts.append(f"  {val:>+6.1f}")
        else:
            parts.append("        ")
    parts.append(f"  {annual:>+6.1f}")
    print("".join(parts))

print("  " + "─" * 101)

# Overall stats
all_monthly_vals = [monthly[k] for k in months]
total   = sum(all_monthly_vals)
n_years = len(years)

print(f"\n  Total periods : {len(period_summary)}")
print(f"  Total months  : {len(months)}")
print(f"  Sum of monthly: {total:+.2f}%")

# Save to CSV
import os
out_csv = "results/monthly_returns_4xlev.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Year"] + MN + ["Annual"])
    for yr in years:
        annual = 0.0
        row = [yr]
        for mi in range(1, 13):
            key = f"{yr}-{mi:02d}"
            val = monthly.get(key)
            if val is not None:
                annual += val
                row.append(f"{val:+.2f}")
            else:
                row.append("")
        row.append(f"{annual:+.2f}")
        w.writerow(row)

print(f"\n  Saved → {out_csv}")
