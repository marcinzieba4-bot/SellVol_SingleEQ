"""
portfolio_backtest.py
---------------------
Long leg : Buy Call (4W positive momentum) on NVDA, TSLA, XOM, JPM, AAPL, AMZN
           Each 1/6 equal weight.
Short leg: Sell Call (4W positive momentum) on SPY — 100% weight.

Portfolio P&L per period = avg(long leg pnl_pct) + spy_sell_call_pnl_pct
Results aggregated by calendar month, with full risk stats and PNG chart.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime

# ── inject sell_call_positive mode into strategy module ──────────────────────
import importlib, types
import strategy_vol_premium as _strat

_orig_run = _strat.run_strategy

def run_strategy_extended(ticker, mode="sell_put"):
    """Wraps run_strategy, adding 'sell_call_positive' mode."""
    if mode != "sell_call_positive":
        return _orig_run(ticker, mode=mode)

    # sell call when 4W positive — same signal as buy_call but short side
    results = _orig_run(ticker, mode="buy_call")
    for r in results:
        if r["trade"] == "BUY CALL":
            r["trade"] = "SELL CALL"
            payoff        = max(0.0, (r["stock_expiry"] or 0) - (r["strike"] or 0))
            r["payoff"]   = round(payoff, 2)
            r["pnl_dollar"] = round((r["premium"] or 0) - payoff, 2)
            r["pnl_pct"]  = round(r["pnl_dollar"] / r["stock_entry"] * 100, 4) \
                            if r["stock_entry"] else 0.0
    return results

# ── Config ───────────────────────────────────────────────────────────────────
LONG_TICKERS = ["NVDA", "TSLA", "XOM", "JPM", "AAPL", "AMZN"]
LONG_WEIGHT  = 1 / len(LONG_TICKERS)   # 1/6 each
SHORT_TICKER = "SPY"
SHORT_MODE   = "sell_call_positive"     # sell call when 4W up
SHORT_WEIGHT = 1.0                      # 100%

OUTPUT_TABLE = "portfolio_monthly_pnl.csv"
OUTPUT_CHART = "portfolio_backtest.png"

# ── Load all strategies ───────────────────────────────────────────────────────
print("=" * 60)
print("  PORTFOLIO BACKTEST")
print("=" * 60)

all_series = {}   # ticker -> pd.Series(pnl_pct, index=period_start)

print("\n── Long legs (Buy Call momentum) ──")
for tk in LONG_TICKERS:
    res = run_strategy_extended(tk, mode="buy_call")
    s   = pd.Series(
        {r["period_start"]: r["pnl_pct"] for r in res},
        name=tk,
        dtype=float,
    )
    all_series[tk] = s

print(f"\n── Short leg (Sell Call positive momentum on {SHORT_TICKER}) ──")
spy_res = run_strategy_extended(SHORT_TICKER, mode=SHORT_MODE)
spy_s   = pd.Series(
    {r["period_start"]: r["pnl_pct"] for r in spy_res},
    name=SHORT_TICKER,
    dtype=float,
)
all_series[SHORT_TICKER] = spy_s

# ── Align on common period grid ───────────────────────────────────────────────
df = pd.DataFrame(all_series)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Long portfolio = equal-weighted average of 6 tickers
long_pnl  = df[LONG_TICKERS].mean(axis=1)          # NaN periods excluded per ticker
short_pnl = df[SHORT_TICKER]

# Portfolio P&L = weighted long + short leg
portfolio_pnl = long_pnl * 1.0 + short_pnl * SHORT_WEIGHT   # long is already averaged w/ 1/6 each, full 100% short
# Note: long leg total exposure = 100% (6 × 1/6), short leg = 100%

period_df = pd.DataFrame({
    "long_pnl":  long_pnl,
    "short_pnl": short_pnl,
    "port_pnl":  portfolio_pnl,
})
period_df.index.name = "period_start"

# ── Monthly aggregation ───────────────────────────────────────────────────────
period_df["ym"] = period_df.index.to_period("M")
monthly = period_df.groupby("ym").agg(
    long_pnl  = ("long_pnl",  "sum"),
    short_pnl = ("short_pnl", "sum"),
    port_pnl  = ("port_pnl",  "sum"),
    n_periods = ("port_pnl",  "count"),
)
monthly.index = monthly.index.astype(str)

# Cumulative
monthly["cum_port"]  = monthly["port_pnl"].cumsum()
monthly["cum_long"]  = monthly["long_pnl"].cumsum()
monthly["cum_short"] = monthly["short_pnl"].cumsum()

# ── Risk Stats ────────────────────────────────────────────────────────────────
per = period_df["port_pnl"].dropna()
periods_per_year = 13.0   # ~13 × 4-week periods per year

total_return    = per.sum()
ann_return      = per.mean() * periods_per_year
ann_vol         = per.std()  * np.sqrt(periods_per_year)
sharpe          = ann_return / ann_vol if ann_vol else 0
neg             = per[per < 0]
downside_vol    = neg.std() * np.sqrt(periods_per_year) if len(neg) > 1 else 0
sortino         = ann_return / downside_vol if downside_vol else 0
wins            = (per > 0).sum()
losses          = (per < 0).sum()
win_rate        = wins / len(per) * 100 if len(per) else 0
avg_win         = per[per > 0].mean() if wins else 0
avg_loss        = per[per < 0].mean() if losses else 0

# Max drawdown on cumulative per-period P&L
cum = per.cumsum()
roll_max = cum.cummax()
drawdown = cum - roll_max
max_dd   = drawdown.min()

# ── Print monthly table ───────────────────────────────────────────────────────
print("\n" + "═" * 75)
print(f"  MONTHLY P&L TABLE")
print("═" * 75)
hdr = f"{'Month':<10} {'Long%':>8} {'Short%':>8} {'Port%':>8} {'Cum%':>9}  {'Periods':>7}"
print(hdr)
print("─" * 75)
for ym, row in monthly.iterrows():
    marker = " ▼" if row["port_pnl"] < -3 else ("  " if row["port_pnl"] >= 0 else " ·")
    print(f"{ym:<10} {row['long_pnl']:>7.2f}% {row['short_pnl']:>7.2f}% "
          f"{row['port_pnl']:>7.2f}% {row['cum_port']:>8.2f}%  {int(row['n_periods']):>5}{marker}")
print("─" * 75)
print(f"{'TOTAL':<10} {monthly['long_pnl'].sum():>7.2f}% {monthly['short_pnl'].sum():>7.2f}% "
      f"{monthly['port_pnl'].sum():>7.2f}%")
print()

print("═" * 55)
print("  RISK STATISTICS")
print("═" * 55)
print(f"  Total Return          : {total_return:>+8.2f}%")
print(f"  Annualised Return     : {ann_return:>+8.2f}%")
print(f"  Annualised Volatility : {ann_vol:>8.2f}%")
print(f"  Sharpe Ratio          : {sharpe:>8.2f}")
print(f"  Sortino Ratio         : {sortino:>8.2f}")
print(f"  Max Drawdown          : {max_dd:>+8.2f}%")
print(f"  Win Rate (periods)    : {win_rate:>8.1f}%")
print(f"  Avg Win               : {avg_win:>+8.2f}%")
print(f"  Avg Loss              : {avg_loss:>+8.2f}%")
print(f"  Win/Loss Ratio        : {abs(avg_win/avg_loss):>8.2f}x" if avg_loss else "  Win/Loss Ratio        :      N/A")
print(f"  Total Periods         : {len(per):>8}")
print(f"  Winning Periods       : {wins:>8}")
print(f"  Losing Periods        : {losses:>8}")
print("═" * 55)

# ── Save CSV ─────────────────────────────────────────────────────────────────
monthly.to_csv(OUTPUT_TABLE, float_format="%.4f")
print(f"\n  Monthly table saved → {OUTPUT_TABLE}")

# ── Chart ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                          gridspec_kw={"height_ratios": [3, 2, 1.5]})
fig.patch.set_facecolor("#0f0f1a")
for ax in axes:
    ax.set_facecolor("#0f0f1a")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#aaa", labelsize=9)

mo_idx = np.arange(len(monthly))
mo_labels = list(monthly.index)

# ─ Panel 1: Cumulative P&L ────────────────────────────────────────────────
ax1 = axes[0]
ax1.plot(mo_idx, monthly["cum_port"],  color="#00e5ff", lw=2.2, label="Portfolio (long+short)", zorder=3)
ax1.plot(mo_idx, monthly["cum_long"],  color="#76ff03", lw=1.4, ls="--", alpha=0.8, label="Long leg only (avg 6 stocks)")
ax1.plot(mo_idx, monthly["cum_short"], color="#ff6d00", lw=1.4, ls="--", alpha=0.8, label="Short leg only (SPY sell call)")
ax1.axhline(0, color="#555", lw=0.8, ls=":")
ax1.fill_between(mo_idx, monthly["cum_port"], 0,
                 where=(monthly["cum_port"] >= 0), alpha=0.12, color="#00e5ff")
ax1.fill_between(mo_idx, monthly["cum_port"], 0,
                 where=(monthly["cum_port"] < 0),  alpha=0.15, color="#ff1744")
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
ax1.set_title("Portfolio Backtest — Cumulative P&L", color="white", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("Cumulative P&L (%)", color="#aaa", fontsize=9)
ax1.legend(loc="upper left", fontsize=8.5, facecolor="#1a1a2e", edgecolor="#444",
           labelcolor="white", framealpha=0.9)
ax1.set_xlim(-0.5, len(mo_idx) - 0.5)
tick_step = max(1, len(mo_idx) // 16)
ax1.set_xticks(mo_idx[::tick_step])
ax1.set_xticklabels(mo_labels[::tick_step], rotation=40, ha="right", fontsize=8)

# Annotate final value
final_val = monthly["cum_port"].iloc[-1]
ax1.annotate(f"{final_val:+.1f}%",
             xy=(mo_idx[-1], final_val),
             xytext=(-45, 12), textcoords="offset points",
             color="#00e5ff", fontsize=10, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#00e5ff", lw=1.2))

# ─ Panel 2: Monthly bar chart ─────────────────────────────────────────────
ax2 = axes[1]
colors = ["#00e676" if v >= 0 else "#ff1744" for v in monthly["port_pnl"]]
ax2.bar(mo_idx, monthly["port_pnl"], color=colors, alpha=0.85, width=0.7, zorder=2)
ax2.axhline(0, color="#555", lw=0.8)
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
ax2.set_title("Monthly Portfolio P&L", color="white", fontsize=11, fontweight="bold", pad=6)
ax2.set_ylabel("Monthly P&L (%)", color="#aaa", fontsize=9)
ax2.set_xlim(-0.5, len(mo_idx) - 0.5)
ax2.set_xticks(mo_idx[::tick_step])
ax2.set_xticklabels(mo_labels[::tick_step], rotation=40, ha="right", fontsize=8)

# ─ Panel 3: Drawdown ──────────────────────────────────────────────────────
ax3 = axes[2]
cum_per_period = per.reset_index(drop=True).cumsum()
roll_max_p     = cum_per_period.cummax()
dd_series      = cum_per_period - roll_max_p
dd_idx         = np.arange(len(dd_series))
ax3.fill_between(dd_idx, dd_series, 0, color="#ff1744", alpha=0.55)
ax3.plot(dd_idx, dd_series, color="#ff1744", lw=1.2)
ax3.axhline(0, color="#555", lw=0.8)
ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
ax3.set_title("Drawdown (per period)", color="white", fontsize=11, fontweight="bold", pad=6)
ax3.set_ylabel("Drawdown (%)", color="#aaa", fontsize=9)
ax3.set_xlim(-0.5, len(dd_idx) - 0.5)

# Risk stats box
stats_text = (
    f"Total Return: {total_return:+.1f}%\n"
    f"Ann. Return:  {ann_return:+.1f}%\n"
    f"Ann. Vol:     {ann_vol:.1f}%\n"
    f"Sharpe:       {sharpe:.2f}\n"
    f"Sortino:      {sortino:.2f}\n"
    f"Max DD:       {max_dd:.1f}%\n"
    f"Win Rate:     {win_rate:.0f}%"
)
fig.text(0.985, 0.97, stats_text,
         ha="right", va="top", fontsize=8.5, color="white",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                   edgecolor="#00e5ff", alpha=0.9),
         fontfamily="monospace")

# Subtitle
subtitle = (
    f"Long: {', '.join(LONG_TICKERS)} (1/6 each, Buy Call momentum)  |  "
    f"Short: {SHORT_TICKER} 100% (Sell Call positive momentum)  |  Sep 2020 – Mar 2026"
)
fig.text(0.5, 0.985, subtitle, ha="center", va="top",
         fontsize=8.5, color="#aaa", style="italic")

plt.tight_layout(rect=[0, 0, 1, 0.982])
plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"  Chart saved        → {OUTPUT_CHART}")
print()
