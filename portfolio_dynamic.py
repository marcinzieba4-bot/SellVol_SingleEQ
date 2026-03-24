"""
portfolio_dynamic.py
--------------------
Simulate 4 strategy variants on a dynamically rebalanced equal-weighted
portfolio of the top-30 largest S&P 500 stocks at each ~4-week period.

Stock selection per period:
  Walk down the SPX-50 market-cap ranking (spx50_periods.json) and take the
  first 30 tickers that have options data in S3.  If a ticker is ranked but
  has no S3 data it is skipped and the next-ranked ticker fills the slot.

Four strategy modes:
  1  buy_momentum      — Buy Call (↑ momentum) / Buy Put (↓ momentum)
  2  buy_call          — Buy Call on ↑ momentum only, no trade on ↓
  3  sell_put          — Sell Put on ↑ momentum only, no trade on ↓
  4  sell_spy_momentum — Sell Put (↑ momentum) / Sell Call (↓ momentum)

Outputs (all in results/):
  dynamic_top30_periods_modeN.csv    — period-level detail (one row per period)
  dynamic_top30_monthly_modeN.csv    — monthly aggregated returns table
  dynamic_top30_equity_modeN.png     — 3-panel chart: equity / heatmap / drawdown
  dynamic_top30_risk_stats.txt       — risk summary for all 4 modes
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

MODES = [
    ("buy_momentum",      "Mode 1 — Buy Call(↑) / Buy Put(↓)"),
    ("buy_call",          "Mode 2 — Buy Call only  (↑ momentum)"),
    ("sell_put",          "Mode 3 — Sell Put only  (↑ momentum)"),
    ("sell_spy_momentum", "Mode 4 — Sell Put(↑) / Sell Call(↓)"),
]

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

# Determine which tickers we actually need to load
needed_tickers = S3_TICKERS & {s["ticker"] for v in spx50.values() for s in v}
print(f"  Tickers needed (S3 ∩ SPX50 universe): {len(needed_tickers)}")


# ── 2. Pre-load all strategy results (suppress per-period print noise) ────────
print("\nPre-loading strategy results (this takes ~2-3 min)...")

# cache[ticker][mode] = {period_start: pnl_pct}
cache: dict = defaultdict(dict)
failed: list = []

for i, ticker in enumerate(sorted(needed_tickers), 1):
    for mode, _ in MODES:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                results = run_strategy(ticker, mode=mode)
            cache[ticker][mode] = {r["period_start"]: r["pnl_pct"] for r in results}
        except Exception as e:
            failed.append((ticker, mode, str(e)))
            cache[ticker][mode] = {}
    print(f"  [{i:>2}/{len(needed_tickers)}] {ticker:<8} loaded", flush=True)

if failed:
    print(f"\n  WARNING — {len(failed)} load failures:")
    for t, m, e in failed:
        print(f"    {t} / {m}: {e}")
print("  Pre-load complete.\n")


# ── 3. Helper: top-30 S3 tickers for a given period ──────────────────────────
def top30_for_period(period_start: str) -> list[str]:
    """
    Walk the SPX50 cap-ranking for this period and pick the first 30
    tickers that have S3 options data.  Returns list of up to 30 tickers.
    """
    ranking = spx50.get(period_start, [])
    chosen  = []
    for entry in ranking:
        tk = entry["ticker"]
        if tk in S3_TICKERS and tk in cache:
            chosen.append(tk)
            if len(chosen) == TOP_N:
                break
    return chosen


# ── 4. Build per-mode portfolio time-series ───────────────────────────────────
def build_portfolio(mode: str) -> list[dict]:
    """
    For each spx50 period, average pnl_pct across the top-30 available
    tickers.  Returns list of period dicts.
    """
    rows = []
    for ps in periods_sorted:
        tickers = top30_for_period(ps)
        if not tickers:
            continue
        pnls = [cache[t][mode].get(ps, 0.0) for t in tickers]
        port_pnl = sum(pnls) / len(pnls)
        rows.append({
            "period_start": ps,
            "n_tickers":    len(tickers),
            "tickers":      tickers,
            "pnl_pct":      port_pnl,
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

    # Max drawdown
    equity   = [0.0]
    for r in rets:
        equity.append(equity[-1] + r)
    peak = equity[0]
    max_dd = 0.0
    dd_start = dd_end = rows[0]["period_start"] if rows else ""
    pk_idx = 0
    for i, v in enumerate(equity[1:], 1):
        if v > peak:
            peak   = v
            pk_idx = i
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
        best=max(rows, key=lambda x: x["pnl_pct"]),
        worst=min(rows, key=lambda x: x["pnl_pct"]),
        equity=equity,
    )


# ── 6. Monthly aggregation ───────────────────────────────────────────────────
def monthly_returns(rows: list[dict]) -> dict:
    """Returns {YYYY-MM: sum_pnl_pct}"""
    m: dict = defaultdict(float)
    for r in rows:
        m[r["period_start"][:7]] += r["pnl_pct"]
    return dict(m)


# ── 7. CSV output ──────────────────────────────────────────────────────────────
def write_period_csv(rows: list[dict], mode_idx: int) -> str:
    path = os.path.join(OUT_DIR, f"dynamic_top30_periods_mode{mode_idx}.csv")
    equity = 0.0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period_start", "n_tickers", "pnl_pct", "cumulative_pct", "tickers"])
        for r in rows:
            equity += r["pnl_pct"]
            w.writerow([
                r["period_start"],
                r["n_tickers"],
                f"{r['pnl_pct']:.4f}",
                f"{equity:.4f}",
                "|".join(r["tickers"]),
            ])
    return path


def write_monthly_csv(rows: list[dict], mode_idx: int) -> str:
    path   = os.path.join(OUT_DIR, f"dynamic_top30_monthly_mode{mode_idx}.csv")
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
def write_chart(rows: list[dict], stats: dict, label: str, mode_idx: int) -> str:
    path    = os.path.join(OUT_DIR, f"dynamic_top30_equity_mode{mode_idx}.png")
    m_ret   = monthly_returns(rows)
    months  = sorted(m_ret.keys())
    years   = sorted({k[:4] for k in months})
    MN_ABB  = ["J","F","M","A","M","J","J","A","S","O","N","D"]

    equity  = stats["equity"]
    dates   = [r["period_start"] for r in rows]
    eq_x    = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

    # drawdown series
    peak_val = equity[0]
    drawdowns = []
    for v in equity[1:]:
        if v > peak_val:
            peak_val = v
        drawdowns.append(peak_val - v)

    fig, axes = plt.subplots(3, 1, figsize=(14, 13),
                             gridspec_kw={"height_ratios": [3, 2.2, 1.5]})
    fig.suptitle(f"Dynamic Top-30  |  {label}\n"
                 f"Total: {stats['total']:+.1f}%   CAGR: {stats['cagr']:+.1f}%/yr   "
                 f"Sharpe: {stats['sharpe']:.2f}   Sortino: {stats['sortino']:.2f}   "
                 f"Max DD: -{stats['max_dd']:.1f}%",
                 fontsize=11, y=0.98)

    # ── Panel 1: equity curve ─────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(eq_x, equity[1:], alpha=0.15, color="#1f77b4")
    ax1.plot(eq_x, equity[1:], color="#1f77b4", linewidth=1.5, label="Portfolio equity")
    ax1.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax1.set_ylabel("Cumulative P&L (%)")
    ax1.set_title("Equity Curve (additive %)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # ── Panel 2: monthly heatmap ──────────────────────────────────────────────
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
    ax2.set_title("Monthly Returns Heatmap (%)")
    for yi in range(len(years)):
        for mi in range(12):
            v = grid_data[yi, mi]
            if not np.isnan(v):
                ax2.text(mi, yi, f"{v:+.1f}", ha="center", va="center",
                         fontsize=6.5, color="black")
    fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02, label="%")

    # ── Panel 3: drawdown ─────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(eq_x, [-d for d in drawdowns], 0, alpha=0.5, color="#d62728")
    ax3.plot(eq_x, [-d for d in drawdowns], color="#d62728", linewidth=1.2)
    ax3.set_ylabel("Drawdown (%)")
    ax3.set_title("Drawdown from Peak")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 9. Main loop ──────────────────────────────────────────────────────────────
all_stats: dict = {}
MN = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

print("=" * 72)
print("  DYNAMIC TOP-30 PORTFOLIO — RISK STATS")
print("=" * 72)

txt_lines: list[str] = ["Dynamic Top-30 Portfolio — Risk Stats\n", "=" * 72 + "\n"]

for idx, (mode, label) in enumerate(MODES, 1):
    rows  = build_portfolio(mode)
    stats = risk_stats(rows)
    m_ret = monthly_returns(rows)
    years = sorted({k[:4] for k in m_ret.keys()})

    all_stats[mode] = stats

    # Console + txt output
    block = [
        f"\n{label}",
        "─" * 65,
        f"  Periods : {stats['n']}  (~{stats['n']/PPY:.1f} years, "
        f"{rows[0]['period_start']} → {rows[-1]['period_start']})",
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

    # Write outputs
    p1 = write_period_csv(rows, idx)
    p2 = write_monthly_csv(rows, idx)
    p3 = write_chart(rows, stats, label, idx)
    print(f"\n  Saved: {p1}")
    print(f"         {p2}")
    print(f"         {p3}")
    txt_lines.append(f"\n  Saved: {p1}\n         {p2}\n         {p3}\n")

# ── 10. Comparison table ─────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  COMPARISON SUMMARY")
print("=" * 72)
hdr = f"  {'Mode':<42} {'Total':>7} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} {'Sortino':>8} {'Win%':>5}"
sep = "  " + "─" * 70
print(hdr)
print(sep)
cmp_lines = ["\nCOMPARISON SUMMARY\n" + "=" * 72 + "\n", hdr + "\n", sep + "\n"]
for mode, label in MODES:
    s = all_stats[mode]
    win_pct = s['wins'] / s['n'] * 100 if s['n'] else 0
    line = (f"  {label:<42} {s['total']:>+7.1f}% {s['cagr']:>+6.1f}% "
            f"{-s['max_dd']:>+7.1f}% {s['sharpe']:>7.2f} {s['sortino']:>8.2f} {win_pct:>4.0f}%")
    print(line)
    cmp_lines.append(line + "\n")
print()

# Save risk stats txt
stats_path = os.path.join(OUT_DIR, "dynamic_top30_risk_stats.txt")
with open(stats_path, "w") as f:
    f.writelines(txt_lines)
    f.writelines(cmp_lines)
print(f"  Risk stats saved → {stats_path}")
