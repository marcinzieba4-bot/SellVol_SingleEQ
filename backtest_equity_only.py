"""
backtest_equity_only.py
-----------------------
Same 8-equity strategy as before (buy call ↑ / buy put ↓, equal weighted)
but with NO SPY leg — pure equity momentum book.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, os.path.dirname(__file__))
from strategy_vol_premium import run_strategy

EQUITIES  = ["AAPL", "AMD", "AMZN", "JPM", "LLY", "NVDA", "TSLA", "XOM"]
W         = 1 / len(EQUITIES)
OUT_TABLE = "equity_only_monthly.csv"
OUT_CHART = "equity_only_chart.png"

# ── Load ──────────────────────────────────────────────────────────────────────
print("=" * 58)
print("  EQUITY-ONLY MOMENTUM BACKTEST")
print("  8 stocks · buy call ↑ / buy put ↓ · equal weighted")
print("=" * 58)

results = {}
for tk in EQUITIES:
    results[tk] = run_strategy(tk, mode="buy_momentum")

# ── Per-period portfolio P&L ──────────────────────────────────────────────────
idx         = {tk: {r["period_start"]: r for r in res} for tk, res in results.items()}
all_periods = sorted({r["period_start"] for res in results.values() for r in res})

rows = []
for ps in all_periods:
    port = 0.0
    leg  = {}
    for tk in EQUITIES:
        r = idx[tk].get(ps)
        if r:
            port += W * r["pnl_pct"]
            leg[tk] = {"trade": r["trade"], "pnl": round(r["pnl_pct"], 3)}
    rows.append({"period_start": ps, "port_pnl": round(port, 4), "legs": leg})

period_df           = pd.DataFrame(rows).set_index("period_start")
period_df.index     = pd.to_datetime(period_df.index)
period_df["ym"]     = period_df.index.to_period("M")

monthly             = period_df.groupby("ym").agg(
                        port_pnl = ("port_pnl", "sum"),
                        n        = ("port_pnl", "count"))
monthly.index       = monthly.index.astype(str)
monthly["cum"]      = monthly["port_pnl"].cumsum()

# ── Stats ─────────────────────────────────────────────────────────────────────
per           = period_df["port_pnl"]
PPY           = 13.0
tot           = per.sum()
ann           = per.mean() * PPY
vol           = per.std()  * np.sqrt(PPY)
sh            = ann / vol if vol else 0
neg           = per[per < 0]
dv            = neg.std() * np.sqrt(PPY) if len(neg) > 1 else 0
so            = ann / dv  if dv  else 0
cum           = per.cumsum()
dd_series     = cum - cum.cummax()
maxdd         = dd_series.min()
wins          = (per > 0).sum()
losses        = (per < 0).sum()
wr            = wins / len(per) * 100
aw            = per[per > 0].mean()
al            = per[per < 0].mean()

# ── Per-ticker breakdown ──────────────────────────────────────────────────────
tk_stats = {}
for tk in EQUITIES:
    res  = results[tk]
    pnls = [r["pnl_pct"] for r in res if "NO" not in r["trade"]]
    bc   = [r["pnl_pct"] for r in res if r["trade"] == "BUY CALL"]
    bp   = [r["pnl_pct"] for r in res if r["trade"] == "BUY PUT"]
    tk_stats[tk] = dict(
        n_call  = len(bc), n_put = len(bp),
        avg_all = np.mean(pnls) if pnls else 0,
        avg_call= np.mean(bc)   if bc   else 0,
        avg_put = np.mean(bp)   if bp   else 0,
        total   = sum(pnls),
        wr      = sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0,
    )

# ── Print ─────────────────────────────────────────────────────────────────────
W2 = 68
print(f"\n{'═'*W2}")
print(f"  MONTHLY P&L")
print(f"{'═'*W2}")
print(f"{'Month':<10} {'Port%':>8} {'Cum%':>9}  n")
print(f"{'─'*W2}")
for ym, row in monthly.iterrows():
    flag = " ▼" if row["port_pnl"] < -3 else ("  " if row["port_pnl"] >= 0 else " ·")
    print(f"{ym:<10} {row['port_pnl']:>7.2f}% {row['cum']:>8.2f}%  {int(row['n'])}{flag}")
print(f"{'─'*W2}")
print(f"{'TOTAL':<10} {monthly['port_pnl'].sum():>7.2f}%")

print(f"\n{'═'*58}")
print(f"  RISK STATISTICS  (equity leg only)")
print(f"{'═'*58}")
print(f"  Total Return          : {tot:>+8.2f}%")
print(f"  Ann. Return           : {ann:>+8.2f}%")
print(f"  Ann. Volatility       : {vol:>8.2f}%")
print(f"  Sharpe Ratio          : {sh:>8.2f}")
print(f"  Sortino Ratio         : {so:>8.2f}")
print(f"  Max Drawdown          : {maxdd:>+8.2f}%")
print(f"  Win Rate (periods)    : {wr:>8.1f}%")
print(f"  Avg Win               : {aw:>+8.2f}%")
print(f"  Avg Loss              : {al:>+8.2f}%")
print(f"  Win/Loss ratio        : {abs(aw/al):>8.2f}x")
print(f"  Winning Periods       : {wins:>8}")
print(f"  Losing Periods        : {losses:>8}")
print(f"{'═'*58}")

print(f"\n  PER-TICKER BREAKDOWN")
print(f"  {'Ticker':<7} {'Calls':>6} {'AvgCall':>8} {'Puts':>6} {'AvgPut':>8}  "
      f"{'Total%':>7}  {'WinRate':>7}")
print(f"  {'─'*60}")
for tk, s in tk_stats.items():
    print(f"  {tk:<7} {s['n_call']:>6} {s['avg_call']:>+7.2f}%"
          f" {s['n_put']:>6} {s['avg_put']:>+7.2f}%"
          f"  {s['total']:>+6.2f}%  {s['wr']:>6.1f}%")
print()

# ── Save CSV ──────────────────────────────────────────────────────────────────
monthly.to_csv(OUT_TABLE, float_format="%.4f")
print(f"  Monthly table saved → {OUT_TABLE}")

# ── Chart ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                          gridspec_kw={"height_ratios": [3, 2, 1.5]})
fig.patch.set_facecolor("#0f0f1a")
for ax in axes:
    ax.set_facecolor("#0f0f1a")
    for sp in ["top","right"]:   ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#444")
    ax.tick_params(colors="#aaa", labelsize=9)

mo_idx    = np.arange(len(monthly))
mo_labels = list(monthly.index)
tick_step = max(1, len(mo_idx) // 16)

# — individual ticker cumulative P&L (thin lines behind) ———
ticker_colors = {
    "AAPL":"#f48fb1","AMZN":"#80cbc4","NVDA":"#ffe082","TSLA":"#b39ddb",
    "JPM":"#80deea", "LLY":"#a5d6a7","XOM":"#ffab91","AMD":"#90caf9",
}
ax1 = axes[0]
for tk in EQUITIES:
    res     = results[tk]
    tk_idx  = {r["period_start"]: r["pnl_pct"] for r in res}
    tk_ser  = pd.Series(tk_idx)
    tk_ser.index = pd.to_datetime(tk_ser.index)
    tk_ser  = tk_ser.groupby(tk_ser.index.to_period("M")).sum()
    tk_ser.index = tk_ser.index.astype(str)
    tk_cum  = tk_ser.reindex(monthly.index, fill_value=0).cumsum()
    ax1.plot(mo_idx, tk_cum.values, color=ticker_colors.get(tk,"#aaa"),
             lw=0.9, alpha=0.55, ls=":", label=tk)

ax1.plot(mo_idx, monthly["cum"], color="#00e5ff", lw=2.5,
         label="Portfolio (equal-weighted avg)", zorder=5)
ax1.axhline(0, color="#555", lw=0.8, ls=":")
ax1.fill_between(mo_idx, monthly["cum"], 0,
                 where=(monthly["cum"] >= 0), alpha=0.12, color="#00e5ff")
ax1.fill_between(mo_idx, monthly["cum"], 0,
                 where=(monthly["cum"] < 0),  alpha=0.18, color="#ff1744")
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
ax1.set_title("Equity-Only Momentum — Cumulative P&L (buy call ↑ / buy put ↓)",
              color="white", fontsize=12, fontweight="bold", pad=10)
ax1.set_ylabel("Cumulative P&L (%)", color="#aaa", fontsize=9)
ax1.legend(loc="upper left", fontsize=7.5, facecolor="#1a1a2e",
           edgecolor="#444", labelcolor="white", framealpha=0.9, ncol=3)
ax1.set_xlim(-0.5, len(mo_idx) - 0.5)
ax1.set_xticks(mo_idx[::tick_step])
ax1.set_xticklabels(mo_labels[::tick_step], rotation=40, ha="right", fontsize=8)
final = monthly["cum"].iloc[-1]
ax1.annotate(f"{final:+.1f}%", xy=(mo_idx[-1], final),
             xytext=(-55, 12), textcoords="offset points",
             color="#00e5ff", fontsize=11, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#00e5ff", lw=1.2))

# — monthly bars ———
ax2 = axes[1]
bar_colors = ["#00e676" if v >= 0 else "#ff1744" for v in monthly["port_pnl"]]
ax2.bar(mo_idx, monthly["port_pnl"], color=bar_colors, alpha=0.85, width=0.7, zorder=2)
ax2.axhline(0, color="#555", lw=0.8)
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
ax2.set_title("Monthly Portfolio P&L", color="white", fontsize=11,
              fontweight="bold", pad=6)
ax2.set_ylabel("Monthly P&L (%)", color="#aaa", fontsize=9)
ax2.set_xlim(-0.5, len(mo_idx) - 0.5)
ax2.set_xticks(mo_idx[::tick_step])
ax2.set_xticklabels(mo_labels[::tick_step], rotation=40, ha="right", fontsize=8)

# — drawdown ———
ax3 = axes[2]
dd_idx = np.arange(len(dd_series))
ax3.fill_between(dd_idx, dd_series.values, 0, color="#ff1744", alpha=0.55)
ax3.plot(dd_idx, dd_series.values, color="#ff1744", lw=1.2)
ax3.axhline(0, color="#555", lw=0.8)
ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
ax3.set_title("Drawdown (per period)", color="white", fontsize=11,
              fontweight="bold", pad=6)
ax3.set_ylabel("Drawdown (%)", color="#aaa", fontsize=9)
ax3.set_xlim(-0.5, len(dd_idx) - 0.5)

# stats box
box = (
    f"Total:    {tot:>+6.1f}%\n"
    f"Ann Ret:  {ann:>+6.1f}%\n"
    f"Ann Vol:  {vol:>6.1f}%\n"
    f"Sharpe:   {sh:>6.2f}\n"
    f"Sortino:  {so:>6.2f}\n"
    f"Max DD:   {maxdd:>+6.1f}%\n"
    f"WinRate:  {wr:>5.0f}%\n"
    f"Avg Win:  {aw:>+6.2f}%\n"
    f"Avg Loss: {al:>+6.2f}%"
)
fig.text(0.987, 0.97, box, ha="right", va="top", fontsize=8.5,
         color="white", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                   edgecolor="#00e5ff", alpha=0.92))

sub = (f"AAPL · AMD · AMZN · JPM · LLY · NVDA · TSLA · XOM  |  "
       f"1/8 each  |  Buy Call ↑ / Buy Put ↓  |  Sep 2020 – Feb 2026  |  NO SPY leg")
fig.text(0.5, 0.987, sub, ha="center", va="top",
         fontsize=8.5, color="#aaa", style="italic")

plt.tight_layout(rect=[0, 0, 1, 0.983])
plt.savefig(OUT_CHART, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"  Chart saved        → {OUT_CHART}")
