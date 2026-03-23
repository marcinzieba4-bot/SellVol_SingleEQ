"""
portfolio_levered_call.py
--------------------------
Buy-Call-only simulation with:
  • Margin realism  — only the option premium is "spent"; the remainder sits in
                      money-market and earns a time-varying short-rate.
  • 4× leverage     — each period we control 4× notional; option P&L scales 4×
                      but the margin cost also scales 4×; leftover capital still
                      earns money-market.

Three variants are reported for comparison:

  A — Baseline (original)
        pnl = avg( pnl_dollar / stock_entry )
        (treats the full stock price as capital — no idle-cash credit)

  B — 1× + Money Market
        pnl = avg( pnl_dollar / stock_entry )
              + ( 1 - avg(premium / stock_entry) ) × mmkt_rate_period
        (no extra leverage; idle cash earns money-market)

  C — 4× Leverage + Money Market
        pnl = 4 × avg( pnl_dollar / stock_entry )
              + max(0, 1 - 4 × avg(premium / stock_entry)) × mmkt_rate_period
        (4× notional; option P&L ×4; leftover earns money-market)

Money-market rates (approximate US short-term yield by year):
  2020–2021 : ~0.08%/yr   (COVID ZIRP)
  2022      : ~2.00%/yr   (hiking cycle, average)
  2023      : ~5.00%/yr
  2024      : ~5.10%/yr
  2025      : ~4.30%/yr
  2026      : ~4.30%/yr

Stock selection per period:
  Walk the SPX-50 market-cap ranking (spx50_periods.json) and take the
  first 30 tickers that have S3 options data.
"""

import os, json, csv, math, io, contextlib
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
from strategy_vol_premium import run_strategy

# ── Config ────────────────────────────────────────────────────────────────────
SPX50_JSON = "spx50_periods.json"
OUT_DIR    = "results"
TOP_N      = 30
PPY        = 13.0   # ~4-week periods per year
LEVERAGE   = 4.0    # variant C leverage multiplier

# Annual money-market rates (year string → decimal fraction)
MMKT_ANNUAL = {
    "2020": 0.0008,   # 0.08%  — COVID-era ZIRP
    "2021": 0.0008,   # 0.08%
    "2022": 0.0200,   # 2.00%  — rapid hiking, full-year average
    "2023": 0.0500,   # 5.00%
    "2024": 0.0510,   # 5.10%
    "2025": 0.0430,   # 4.30%
    "2026": 0.0430,   # 4.30%
}

def mmkt_period(period_start: str) -> float:
    """Per-4-week money-market return (as %) for the given period_start date."""
    year = period_start[:4]
    annual = MMKT_ANNUAL.get(year, 0.04)   # fallback 4%
    return annual / PPY * 100              # convert to %

# Tickers confirmed present in S3 optionsData/ (excl. SPY index)
S3_TICKERS = frozenset({
    "AAPL","ABBV","ABT","ACN","ADBE","AMZN","AVGO","BAC","BX","CMCSA",
    "COST","CRM","CVX","DIS","GOOG","GOOGL","HD","IBM","INTU","JNJ",
    "JPM","KO","LLY","MA","META","MRK","MSFT","NFLX","NKE","NOW",
    "NVDA","ORCL","PEP","PG","QCOM","TMO","TSLA","UNH","V","WFC","WMT","XOM",
})

os.makedirs(OUT_DIR, exist_ok=True)


# ── 1. Load SPX50 per-period rankings ────────────────────────────────────────
print("Loading SPX50 period rankings...")
with open(SPX50_JSON) as f:
    spx50: dict = json.load(f)
periods_sorted = sorted(spx50.keys())
print(f"  {len(periods_sorted)} periods: {periods_sorted[0]} → {periods_sorted[-1]}")

needed_tickers = S3_TICKERS & {s["ticker"] for v in spx50.values() for s in v}
print(f"  Tickers needed (S3 ∩ SPX50 universe): {len(needed_tickers)}")


# ── 2. Pre-load buy_call results ──────────────────────────────────────────────
print("\nPre-loading buy_call strategy results (this takes ~1-2 min)...")

# cache[ticker][period_start] = {"pnl_pct": float, "premium_pct": float}
#   pnl_pct     = pnl_dollar / stock_entry × 100   (same as original)
#   premium_pct = premium / stock_entry × 100       (fraction of notional paid as margin)
#               = 0 if no trade was made that period
cache: dict = defaultdict(dict)
failed: list = []

for i, ticker in enumerate(sorted(needed_tickers), 1):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            results = run_strategy(ticker, mode="buy_call")
        for r in results:
            ps = r["period_start"]
            is_trade = r["trade"] not in ("NO TRADE", "NO DATA")
            if is_trade and r.get("premium") and r.get("stock_entry"):
                prem_pct = r["premium"] / r["stock_entry"] * 100
            else:
                prem_pct = 0.0
            cache[ticker][ps] = {
                "pnl_pct":     r["pnl_pct"],
                "premium_pct": prem_pct,
            }
    except Exception as e:
        failed.append((ticker, str(e)))
        cache[ticker] = {}
    print(f"  [{i:>2}/{len(needed_tickers)}] {ticker:<8} loaded", flush=True)

if failed:
    print(f"\n  WARNING — {len(failed)} load failures:")
    for t, e in failed:
        print(f"    {t}: {e}")
print("  Pre-load complete.\n")


# ── 2b. Per-ticker premium table ──────────────────────────────────────────────
print("=" * 72)
print("  PER-TICKER PREMIUM ANALYSIS  (buy_call, premium / stock_entry)")
print("=" * 72)
print(f"  {'Ticker':<8}  {'Periods':>7}  {'Traded':>6}  {'Trade%':>6}  "
      f"{'AvgPrem(traded)':>15}  {'AvgPrem(all)':>12}  {'AvgPnL(traded)':>14}")
print("  " + "─" * 74)

ticker_prem_rows = []
for ticker in sorted(cache.keys()):
    pdata = cache[ticker]
    all_periods  = list(pdata.values())
    traded       = [d for d in all_periods if d["premium_pct"] > 0]
    n_all        = len(all_periods)
    n_traded     = len(traded)
    trade_rate   = n_traded / n_all * 100 if n_all else 0
    avg_prem_tr  = sum(d["premium_pct"] for d in traded) / n_traded if n_traded else 0
    avg_prem_all = sum(d["premium_pct"] for d in all_periods) / n_all if n_all else 0
    avg_pnl_tr   = sum(d["pnl_pct"]     for d in traded) / n_traded if n_traded else 0
    ticker_prem_rows.append((ticker, n_all, n_traded, trade_rate,
                             avg_prem_tr, avg_prem_all, avg_pnl_tr))
    print(f"  {ticker:<8}  {n_all:>7}  {n_traded:>6}  {trade_rate:>5.0f}%  "
          f"{avg_prem_tr:>14.2f}%  {avg_prem_all:>11.2f}%  {avg_pnl_tr:>13.3f}%")

# Summary row
all_tr  = [r for _, _, nt, _, ap, _, _ in ticker_prem_rows for _ in range(nt) for ap in [ap]]
# simpler: weight by n_traded
n_t_tot   = sum(r[2] for r in ticker_prem_rows)
n_all_tot = sum(r[1] for r in ticker_prem_rows)
wavg_tr   = (sum(r[2] * r[4] for r in ticker_prem_rows) / n_t_tot) if n_t_tot else 0
wavg_all  = (sum(r[1] * r[5] for r in ticker_prem_rows) / n_all_tot) if n_all_tot else 0
wavg_pnl  = (sum(r[2] * r[6] for r in ticker_prem_rows) / n_t_tot) if n_t_tot else 0
print("  " + "─" * 74)
print(f"  {'WEIGHTED AVG':<8}  {n_all_tot:>7}  {n_t_tot:>6}  "
      f"{n_t_tot/n_all_tot*100:>5.0f}%  "
      f"{wavg_tr:>14.2f}%  {wavg_all:>11.2f}%  {wavg_pnl:>13.3f}%")
print()

# Save to CSV
prem_csv = os.path.join(OUT_DIR, "per_ticker_premium.csv")
with open(prem_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ticker","n_periods","n_traded","trade_pct",
                "avg_premium_pct_traded","avg_premium_pct_all","avg_pnl_pct_traded"])
    for row in ticker_prem_rows:
        w.writerow([row[0], row[1], row[2], f"{row[3]:.1f}",
                    f"{row[4]:.4f}", f"{row[5]:.4f}", f"{row[6]:.4f}"])
print(f"  Per-ticker premium table saved → {prem_csv}\n")


# ── 3. Top-30 tickers for a period ───────────────────────────────────────────
def top30_for_period(period_start: str) -> list[str]:
    ranking = spx50.get(period_start, [])
    chosen  = []
    for entry in ranking:
        tk = entry["ticker"]
        if tk in S3_TICKERS and tk in cache:
            chosen.append(tk)
            if len(chosen) == TOP_N:
                break
    return chosen


# ── 4. Build per-variant portfolio time-series ────────────────────────────────
def build_variant(label: str) -> list[dict]:
    """
    label = 'A' | 'B' | 'C'
    Returns list of {period_start, n_tickers, pnl_pct, avg_premium_pct, mmkt_contrib}.
    """
    rows = []
    for ps in periods_sorted:
        tickers = top30_for_period(ps)
        if not tickers:
            continue

        slot_data  = [cache[t].get(ps, {"pnl_pct": 0.0, "premium_pct": 0.0})
                      for t in tickers]
        avg_pnl    = sum(d["pnl_pct"]     for d in slot_data) / len(slot_data)
        avg_prem   = sum(d["premium_pct"] for d in slot_data) / len(slot_data)

        mm = mmkt_period(ps)  # money-market return for this period (%)

        if label == "A":
            # Original: P&L as % of notional, no idle-cash credit
            period_pnl = avg_pnl
            mm_contrib = 0.0

        elif label == "B":
            # 1× + money market on idle capital
            idle_fraction = max(0.0, 1.0 - avg_prem / 100.0)
            mm_contrib    = idle_fraction * mm
            period_pnl    = avg_pnl + mm_contrib

        elif label == "C":
            # 4× leverage + money market on remaining capital
            capital_used  = LEVERAGE * avg_prem / 100.0          # fraction, can be >1
            idle_fraction = max(0.0, 1.0 - capital_used)
            mm_contrib    = idle_fraction * mm
            period_pnl    = LEVERAGE * avg_pnl + mm_contrib

        rows.append({
            "period_start":    ps,
            "n_tickers":       len(tickers),
            "tickers":         tickers,
            "pnl_pct":         period_pnl,
            "avg_prem_pct":    avg_prem,
            "mm_contrib":      mm_contrib,
            "option_pnl":      LEVERAGE * avg_pnl if label == "C" else avg_pnl,
        })
    return rows


# ── 5. Risk stats ─────────────────────────────────────────────────────────────
def risk_stats(rows: list[dict]) -> dict:
    rets   = [r["pnl_pct"] for r in rows]
    n      = len(rets)
    total  = sum(rets)
    years  = n / PPY

    terminal = 1.0 + total / 100.0
    cagr     = ((terminal ** (1.0 / years)) - 1.0) * 100.0 if terminal > 0 else -999.0

    mean_r = total / n if n else 0.0
    var    = sum((r - mean_r)**2 for r in rets) / n if n else 0.0
    std_r  = math.sqrt(var)
    sharpe = (mean_r / std_r) * math.sqrt(PPY) if std_r else 0.0

    down_sq  = [r**2 for r in rets if r < 0]
    down_std = math.sqrt(sum(down_sq) / n) if n else 0.0
    sortino  = (mean_r / down_std) * math.sqrt(PPY) if down_std else 0.0

    equity   = [0.0]
    for r in rets:
        equity.append(equity[-1] + r)
    peak = equity[0]; max_dd = 0.0
    dd_start = dd_end = rows[0]["period_start"] if rows else ""; pk_idx = 0
    for i, v in enumerate(equity[1:], 1):
        if v > peak:
            peak = v; pk_idx = i
        dd = peak - v
        if dd > max_dd:
            max_dd   = dd
            dd_start = rows[pk_idx - 1]["period_start"] if pk_idx > 0 else rows[0]["period_start"]
            dd_end   = rows[i - 1]["period_start"]

    wins   = sum(1 for r in rets if r > 0)
    losses = sum(1 for r in rets if r < 0)

    return dict(
        n=n, total=total, cagr=cagr, mean_r=mean_r, std_r=std_r,
        sharpe=sharpe, sortino=sortino, max_dd=max_dd,
        dd_start=dd_start, dd_end=dd_end,
        wins=wins, losses=losses,
        best=max(rows,  key=lambda x: x["pnl_pct"]),
        worst=min(rows, key=lambda x: x["pnl_pct"]),
        equity=equity,
    )


# ── 6. Monthly aggregation ────────────────────────────────────────────────────
def monthly_returns(rows: list[dict]) -> dict:
    m: dict = defaultdict(float)
    for r in rows:
        m[r["period_start"][:7]] += r["pnl_pct"]
    return dict(m)


# ── 7. CSV helpers ────────────────────────────────────────────────────────────
def write_period_csv(rows: list[dict], variant: str) -> str:
    path = os.path.join(OUT_DIR, f"levered_call_periods_{variant}.csv")
    equity = 0.0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period_start","n_tickers","option_pnl","mm_contrib",
                    "total_pnl","cumulative_pct","avg_premium_pct","tickers"])
        for r in rows:
            equity += r["pnl_pct"]
            w.writerow([
                r["period_start"],
                r["n_tickers"],
                f"{r['option_pnl']:.4f}",
                f"{r['mm_contrib']:.4f}",
                f"{r['pnl_pct']:.4f}",
                f"{equity:.4f}",
                f"{r['avg_prem_pct']:.4f}",
                "|".join(r["tickers"]),
            ])
    return path


def write_monthly_csv(rows: list[dict], variant: str) -> str:
    path   = os.path.join(OUT_DIR, f"levered_call_monthly_{variant}.csv")
    m_ret  = monthly_returns(rows)
    months = sorted(m_ret.keys())
    years  = sorted({k[:4] for k in months})
    MN     = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Year"] + MN + ["Annual"])
        for yr in years:
            annual = 0.0
            row    = [yr]
            for mi in range(1, 13):
                key = f"{yr}-{mi:02d}"
                val = m_ret.get(key)
                if val is not None:
                    annual += val
                    row.append(f"{val:.2f}")
                else:
                    row.append("")
            row.append(f"{annual:.2f}")
            w.writerow(row)
    return path


# ── 8. PNG chart ──────────────────────────────────────────────────────────────
def write_chart(rows_a: list[dict], rows_b: list[dict], rows_c: list[dict],
                stats_a: dict, stats_b: dict, stats_c: dict) -> str:
    path   = os.path.join(OUT_DIR, "levered_call_equity.png")
    m_ret  = monthly_returns(rows_c)
    months = sorted(m_ret.keys())
    years  = sorted({k[:4] for k in months})
    MN_ABB = ["J","F","M","A","M","J","J","A","S","O","N","D"]

    eq_dates = [datetime.strptime(r["period_start"], "%Y-%m-%d") for r in rows_c]

    def dd_series(equity):
        peak_v = equity[0]; dds = []
        for v in equity[1:]:
            if v > peak_v: peak_v = v
            dds.append(peak_v - v)
        return dds

    fig, axes = plt.subplots(3, 1, figsize=(15, 14),
                             gridspec_kw={"height_ratios": [3, 2.2, 1.5]})
    fig.suptitle(
        f"Buy Call — Margin + Money-Market + Leverage  |  Dynamic Top-30  |  SPX50\n"
        f"Leverage {LEVERAGE:.0f}× → Total: {stats_c['total']:+.1f}%  "
        f"CAGR: {stats_c['cagr']:+.1f}%/yr  "
        f"Sharpe: {stats_c['sharpe']:.2f}  Sortino: {stats_c['sortino']:.2f}  "
        f"Max DD: -{stats_c['max_dd']:.1f}%",
        fontsize=11, y=0.99)

    # ── Panel 1: equity curve overlay ────────────────────────────────────────
    ax1 = axes[0]
    for eq, st, clr, lbl, lw, ls in [
        (stats_a["equity"], stats_a, "#aaaaaa", f"A — Original        {stats_a['total']:+.1f}%  CAGR {stats_a['cagr']:+.1f}%", 1.2, "--"),
        (stats_b["equity"], stats_b, "#1f77b4", f"B — 1×+MMkt         {stats_b['total']:+.1f}%  CAGR {stats_b['cagr']:+.1f}%", 1.5, "-."),
        (stats_c["equity"], stats_c, "#2ca02c", f"C — {LEVERAGE:.0f}×+MMkt         {stats_c['total']:+.1f}%  CAGR {stats_c['cagr']:+.1f}%", 2.0, "-"),
    ]:
        ax1.plot(eq_dates, eq[1:], color=clr, linewidth=lw, linestyle=ls, label=lbl)
    ax1.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    ax1.set_ylabel("Cumulative P&L (%)")
    ax1.set_title("Equity Curve (additive %)")
    ax1.legend(fontsize=8.5, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: monthly heatmap of variant C ────────────────────────────────
    ax2 = axes[1]
    grid_data = np.full((len(years), 12), np.nan)
    for yi, yr in enumerate(years):
        for mi in range(1, 13):
            val = m_ret.get(f"{yr}-{mi:02d}")
            if val is not None:
                grid_data[yi, mi - 1] = val
    abs_max = np.nanmax(np.abs(grid_data)) or 1.0
    norm    = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    im      = ax2.imshow(grid_data, cmap="RdYlGn", norm=norm, aspect="auto")
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(MN_ABB, fontsize=9)
    ax2.set_yticks(range(len(years)))
    ax2.set_yticklabels(years, fontsize=9)
    ax2.set_title(f"Monthly Returns Heatmap — Variant C ({LEVERAGE:.0f}× Leverage + MMkt)")
    for yi in range(len(years)):
        for mi in range(12):
            v = grid_data[yi, mi]
            if not np.isnan(v):
                ax2.text(mi, yi, f"{v:+.1f}", ha="center", va="center",
                         fontsize=6.5, color="black")
    fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02, label="%")

    # ── Panel 3: drawdown (variant C) ────────────────────────────────────────
    ax3 = axes[2]
    dd_c = dd_series(stats_c["equity"])
    ax3.fill_between(eq_dates, [-d for d in dd_c], 0, alpha=0.5, color="#d62728")
    ax3.plot(eq_dates, [-d for d in dd_c], color="#d62728", linewidth=1.2)
    ax3.set_ylabel("Drawdown (%)")
    ax3.set_title(f"Drawdown from Peak — Variant C ({LEVERAGE:.0f}× Leverage + MMkt)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 9. Main ───────────────────────────────────────────────────────────────────
print("=" * 72)
print(f"  BUY CALL — MARGIN + MONEY-MARKET + {LEVERAGE:.0f}× LEVERAGE  (Dynamic Top-30)")
print("=" * 72)

VARIANTS = [
    ("A", f"Variant A — Buy Call original (P&L as % of notional, no idle-cash credit)"),
    ("B", f"Variant B — Buy Call 1× + Money Market on idle capital"),
    ("C", f"Variant C — Buy Call {LEVERAGE:.0f}× Leverage + Money Market on remaining"),
]

MN = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
txt_lines: list[str] = [
    f"Buy Call — Margin + Money-Market + {LEVERAGE:.0f}× Leverage\n",
    "=" * 72 + "\n",
]
all_rows:  dict = {}
all_stats: dict = {}

for variant, label in VARIANTS:
    rows  = build_variant(variant)
    stats = risk_stats(rows)
    m_ret = monthly_returns(rows)
    years = sorted({k[:4] for k in m_ret.keys()})

    all_rows[variant]  = rows
    all_stats[variant] = stats

    # Average premium fraction (for info) — only meaningful when trade occurred
    trade_prems = [r["avg_prem_pct"] for r in rows if r["avg_prem_pct"] > 0]
    avg_prem    = sum(trade_prems) / len(trade_prems) if trade_prems else 0.0

    block = [
        f"\n{label}",
        "─" * 65,
        f"  Periods : {stats['n']}  (~{stats['n']/PPY:.1f} years, "
        f"{rows[0]['period_start']} → {rows[-1]['period_start']})",
        f"  Avg premium / notional : {avg_prem:.2f}%  "
        f"→ capital in options (1×): {avg_prem:.2f}%  "
        f"(4×): {min(100, 4*avg_prem):.1f}%   "
        f"idle→MMkt: {max(0, 100 - 4*avg_prem):.1f}%",
        f"  Total P&L    : {stats['total']:+.2f}%   |   CAGR       : {stats['cagr']:+.2f}%/yr",
        f"  Max Drawdown : -{stats['max_dd']:.2f}%  ({stats['dd_start']} → {stats['dd_end']})",
        f"  Sharpe (ann) : {stats['sharpe']:.2f}   |   Sortino    : {stats['sortino']:.2f}",
        f"  Win / Loss   : {stats['wins']} / {stats['losses']}  "
        f"({stats['wins']/(stats['n'] or 1)*100:.0f}% win rate)",
        f"  Avg/period   : {stats['mean_r']:+.3f}%   |   Std dev    : {stats['std_r']:.3f}%",
        f"  Best period  : {stats['best']['period_start']}  {stats['best']['pnl_pct']:+.2f}%",
        f"  Worst period : {stats['worst']['period_start']}  {stats['worst']['pnl_pct']:+.2f}%",
        "",
        "  Monthly Returns (%):",
        "  " + "  ".join(f"{y:>5}" for y in [" Year"] + MN + ["Annual"]),
        "  " + "  ".join(["─────"] + ["─────"] * 12 + ["──────"]),
    ]
    for yr in years:
        annual = 0.0
        parts  = [f"  {yr}"]
        for mi in range(1, 13):
            v = m_ret.get(f"{yr}-{mi:02d}")
            if v is not None:
                annual += v
                parts.append(f"{v:>+5.1f}")
            else:
                parts.append("     ")
        parts.append(f"{annual:>+6.1f}")
        block.append("  " + "  ".join(parts))

    for line in block:
        print(line)
    txt_lines.extend(line + "\n" for line in block)

    p1 = write_period_csv(rows, variant)
    p2 = write_monthly_csv(rows, variant)
    print(f"\n  Saved: {p1}\n         {p2}")
    txt_lines.append(f"\n  Saved: {p1}\n         {p2}\n")

# ── Chart (all three variants on one figure) ──────────────────────────────────
chart_path = write_chart(
    all_rows["A"], all_rows["B"], all_rows["C"],
    all_stats["A"], all_stats["B"], all_stats["C"],
)
print(f"\n  Chart saved: {chart_path}")
txt_lines.append(f"\n  Chart saved: {chart_path}\n")

# ── Comparison table ──────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  COMPARISON SUMMARY — Buy Call only")
print("=" * 72)
hdr = f"  {'Variant':<58} {'Total':>7} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} {'Sortino':>8} {'Win%':>5}"
sep = "  " + "─" * 70
print(hdr); print(sep)
cmp_lines = ["\nCOMPARISON SUMMARY — Buy Call only\n", "=" * 72 + "\n", hdr + "\n", sep + "\n"]
for variant, label in VARIANTS:
    s = all_stats[variant]
    win_pct = s['wins'] / s['n'] * 100 if s['n'] else 0
    line = (f"  {label:<58} {s['total']:>+7.1f}% {s['cagr']:>+6.1f}% "
            f"{-s['max_dd']:>+7.1f}% {s['sharpe']:>7.2f} {s['sortino']:>8.2f} {win_pct:>4.0f}%")
    print(line)
    cmp_lines.append(line + "\n")
print()

# Save txt
stats_path = os.path.join(OUT_DIR, "levered_call_risk_stats.txt")
with open(stats_path, "w") as f:
    f.writelines(txt_lines)
    f.writelines(cmp_lines)
print(f"  Risk stats saved → {stats_path}")
