"""
backtest_aligned_spy.py
------------------------
Same proportional weighting as backtest_proportional_spy.py, but SPY direction
is ALIGNED with equity momentum (not opposed):

  n equities with BUY CALL  →  SPY SELL PUT  at weight n/8
  m equities with BUY PUT   →  SPY SELL CALL at weight m/8

Logic: when stocks trend up, sell puts on SPY (bullish premium collection).
       when stocks trend down, sell calls on SPY (bearish premium collection).
Same philosophy as single equities — trade with momentum, not against it.
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
N         = len(EQUITIES)
OUT_TABLE = "aligned_spy_monthly.csv"
OUT_CHART = "aligned_spy_chart.png"

print("=" * 64)
print("  ALIGNED SPY HEDGE BACKTEST")
print("  Equities : buy call ↑ / buy put ↓  (1/8 each)")
print("  SPY leg  : sell put ↑ (n_call/8)   sell call ↓ (n_put/8)")
print("=" * 64)

# ── Load ──────────────────────────────────────────────────────────────────────
eq_res = {tk: run_strategy(tk, mode="buy_momentum") for tk in EQUITIES}

print("\n── SPY (sell_put_always + sell_call_always) ──")
spy_sp = {r["period_start"]: r for r in run_strategy("SPY", mode="sell_put_always")}
spy_sc = {r["period_start"]: r for r in run_strategy("SPY", mode="sell_call_always")}

# ── Per-period P&L ─────────────────────────────────────────────────────────────
eq_idx      = {tk: {r["period_start"]: r for r in res} for tk, res in eq_res.items()}
all_periods = sorted({r["period_start"] for res in eq_res.values() for r in res})

rows = []
for ps in all_periods:
    n_call = n_put = 0
    eq_pnl = 0.0
    for tk in EQUITIES:
        r = eq_idx[tk].get(ps)
        if not r:
            continue
        eq_pnl += r["pnl_pct"] / N
        if r["trade"] == "BUY CALL":
            n_call += 1
        elif r["trade"] == "BUY PUT":
            n_put  += 1

    # ALIGNED: bullish equities → sell PUT on SPY, bearish → sell CALL on SPY
    w_sp = n_call / N   # sell-put  weight  (aligned with buy-call momentum)
    w_sc = n_put  / N   # sell-call weight  (aligned with buy-put  momentum)

    spy_pnl  = (w_sp * (spy_sp[ps]["pnl_pct"] if ps in spy_sp else 0.0)
              + w_sc * (spy_sc[ps]["pnl_pct"] if ps in spy_sc else 0.0))
    port_pnl = eq_pnl + spy_pnl

    rows.append({
        "period_start": ps,
        "eq_pnl":    round(eq_pnl,   4),
        "spy_pnl":   round(spy_pnl,  4),
        "port_pnl":  round(port_pnl, 4),
        "n_call": n_call, "n_put": n_put,
        "w_sp": round(w_sp, 3), "w_sc": round(w_sc, 3),
    })

period_df       = pd.DataFrame(rows).set_index("period_start")
period_df.index = pd.to_datetime(period_df.index)
period_df["ym"] = period_df.index.to_period("M")

monthly = period_df.groupby("ym").agg(
    eq_pnl   = ("eq_pnl",   "sum"),
    spy_pnl  = ("spy_pnl",  "sum"),
    port_pnl = ("port_pnl", "sum"),
    n        = ("port_pnl", "count"),
    avg_nc   = ("n_call",   "mean"),
    avg_np   = ("n_put",    "mean"),
)
monthly.index     = monthly.index.astype(str)
monthly["cum"]    = monthly["port_pnl"].cumsum()
monthly["cum_eq"] = monthly["eq_pnl"].cumsum()
monthly["cum_spy"]= monthly["spy_pnl"].cumsum()

# ── Stats ─────────────────────────────────────────────────────────────────────
def risk_stats(s, ppY=13.0):
    s = s.dropna()
    tot = s.sum(); ann = s.mean()*ppY; vol = s.std()*np.sqrt(ppY)
    sh  = ann/vol if vol else 0
    dv  = s[s<0].std()*np.sqrt(ppY) if (s<0).sum()>1 else 0
    so  = ann/dv  if dv  else 0
    dd  = (s.cumsum() - s.cumsum().cummax()).min()
    wins = (s>0).sum(); losses = (s<0).sum()
    wr  = wins/len(s)*100 if len(s) else 0
    return dict(tot=tot, ann=ann, vol=vol, sh=sh, so=so, dd=dd,
                wr=wr, wins=int(wins), losses=int(losses),
                aw=s[s>0].mean() if wins else 0,
                al=s[s<0].mean() if losses else 0)

ps_port = risk_stats(period_df["port_pnl"])
ps_eq   = risk_stats(period_df["eq_pnl"])
ps_spy  = risk_stats(period_df["spy_pnl"])

# ── Print ─────────────────────────────────────────────────────────────────────
W = 80
print(f"\n{'═'*W}")
print(f"  MONTHLY P&L")
print(f"{'═'*W}")
print(f"{'Month':<10} {'EqLeg%':>8} {'SPYLeg%':>8} {'Port%':>8} {'Cum%':>9}  "
      f"{'nCall':>5} {'nPut':>5}")
print(f"{'─'*W}")
for ym, row in monthly.iterrows():
    flag = " ▼" if row["port_pnl"] < -3 else ("  " if row["port_pnl"] >= 0 else " ·")
    print(f"{ym:<10} {row['eq_pnl']:>7.2f}% {row['spy_pnl']:>7.2f}% "
          f"{row['port_pnl']:>7.2f}% {row['cum']:>8.2f}%  "
          f"{row['avg_nc']:>5.1f} {row['avg_np']:>5.1f}{flag}")
print(f"{'─'*W}")
print(f"{'TOTAL':<10} {monthly['eq_pnl'].sum():>7.2f}% "
      f"{monthly['spy_pnl'].sum():>7.2f}% {monthly['port_pnl'].sum():>7.2f}%")

print(f"\n{'═'*66}")
print(f"  RISK STATISTICS")
print(f"{'═'*66}")
print(f"  {'Metric':<26} {'Portfolio':>11} {'Eq Leg':>10} {'SPY Leg':>10}")
print(f"  {'─'*64}")
for label, key, fmt in [
    ("Total Return",    "tot",    "{:>+10.2f}%"),
    ("Ann. Return",     "ann",    "{:>+10.2f}%"),
    ("Ann. Volatility", "vol",    "{:>10.2f}%"),
    ("Sharpe Ratio",    "sh",     "{:>10.2f}"),
    ("Sortino Ratio",   "so",     "{:>10.2f}"),
    ("Max Drawdown",    "dd",     "{:>+10.2f}%"),
    ("Win Rate",        "wr",     "{:>10.1f}%"),
    ("Avg Win",         "aw",     "{:>+10.2f}%"),
    ("Avg Loss",        "al",     "{:>+10.2f}%"),
    ("Win Periods",     "wins",   "{:>10}"),
    ("Loss Periods",    "losses", "{:>10}"),
]:
    print(f"  {label:<26} "
          f"{fmt.format(ps_port[key])} "
          f"{fmt.format(ps_eq[key])} "
          f"{fmt.format(ps_spy[key])}")
print(f"{'═'*66}")

# ── Save ──────────────────────────────────────────────────────────────────────
monthly.to_csv(OUT_TABLE, float_format="%.4f")
print(f"\n  Monthly table → {OUT_TABLE}")

# ── Chart ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(16, 17),
                          gridspec_kw={"height_ratios": [3, 1.8, 1.8, 1.2]})
fig.patch.set_facecolor("#0f0f1a")
for ax in axes:
    ax.set_facecolor("#0f0f1a")
    for sp in ["top","right"]:   ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#444")
    ax.tick_params(colors="#aaa", labelsize=9)

mo_idx    = np.arange(len(monthly))
mo_labels = list(monthly.index)
tick_step = max(1, len(mo_idx) // 16)

# P1 — Cumulative
ax1 = axes[0]
ax1.plot(mo_idx, monthly["cum"],     color="#00e5ff", lw=2.4, label="Portfolio (eq + aligned SPY)", zorder=4)
ax1.plot(mo_idx, monthly["cum_eq"],  color="#76ff03", lw=1.4, ls="--", alpha=0.8, label="Equity leg only")
ax1.plot(mo_idx, monthly["cum_spy"], color="#ff9100", lw=1.4, ls="--", alpha=0.8, label="SPY aligned leg")
ax1.axhline(0, color="#555", lw=0.8, ls=":")
ax1.fill_between(mo_idx, monthly["cum"], 0,
                 where=(monthly["cum"] >= 0), alpha=0.12, color="#00e5ff")
ax1.fill_between(mo_idx, monthly["cum"], 0,
                 where=(monthly["cum"] < 0),  alpha=0.18, color="#ff1744")
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
ax1.set_title("Aligned SPY Hedge — Cumulative P&L  (sell put ↑ / sell call ↓)",
              color="white", fontsize=12, fontweight="bold", pad=10)
ax1.set_ylabel("Cumulative P&L (%)", color="#aaa", fontsize=9)
ax1.legend(loc="upper left", fontsize=8.5, facecolor="#1a1a2e",
           edgecolor="#444", labelcolor="white", framealpha=0.9)
ax1.set_xlim(-0.5, len(mo_idx)-.5)
ax1.set_xticks(mo_idx[::tick_step])
ax1.set_xticklabels(mo_labels[::tick_step], rotation=40, ha="right", fontsize=8)
for val, col in [(monthly["cum"].iloc[-1],"#00e5ff"),
                  (monthly["cum_eq"].iloc[-1],"#76ff03"),
                  (monthly["cum_spy"].iloc[-1],"#ff9100")]:
    ax1.annotate(f"{val:+.1f}%", xy=(mo_idx[-1], val),
                 xytext=(-52, 8), textcoords="offset points", color=col,
                 fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=col, lw=1.1))

# P2 — Monthly bars
ax2 = axes[1]
cols = ["#00e676" if v >= 0 else "#ff1744" for v in monthly["port_pnl"]]
ax2.bar(mo_idx, monthly["port_pnl"], color=cols, alpha=0.85, width=0.7, zorder=2)
ax2.axhline(0, color="#555", lw=0.8)
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
ax2.set_title("Monthly Portfolio P&L", color="white", fontsize=11, fontweight="bold", pad=6)
ax2.set_ylabel("Monthly P&L (%)", color="#aaa", fontsize=9)
ax2.set_xlim(-0.5, len(mo_idx)-.5)
ax2.set_xticks(mo_idx[::tick_step])
ax2.set_xticklabels(mo_labels[::tick_step], rotation=40, ha="right", fontsize=8)

# P3 — SPY weights (sell-put in green = aligned with bullish, sell-call in red = aligned with bearish)
ax3 = axes[2]
w_sp_mo = period_df.groupby("ym")["w_sp"].mean().reindex(
            [pd.Period(m) for m in monthly.index], fill_value=0)
w_sc_mo = period_df.groupby("ym")["w_sc"].mean().reindex(
            [pd.Period(m) for m in monthly.index], fill_value=0)
w_sp_arr = w_sp_mo.values
w_sc_arr = w_sc_mo.values
ax3.bar(mo_idx, w_sp_arr, color="#26a69a", alpha=0.85, width=0.7,
        label="Sell PUT weight (n_call/8, bullish)")
ax3.bar(mo_idx, w_sc_arr, bottom=w_sp_arr, color="#ef5350", alpha=0.85, width=0.7,
        label="Sell CALL weight (n_put/8, bearish)")
ax3.set_ylim(0, 1.15)
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax3.set_title("SPY Weights per Period  (avg by month)",
              color="white", fontsize=10, fontweight="bold", pad=6)
ax3.set_ylabel("Weight", color="#aaa", fontsize=9)
ax3.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
ax3.set_xlim(-0.5, len(mo_idx)-.5)
ax3.set_xticks(mo_idx[::tick_step])
ax3.set_xticklabels(mo_labels[::tick_step], rotation=40, ha="right", fontsize=8)

# P4 — Drawdown
ax4 = axes[3]
per = period_df["port_pnl"]
dd  = (per.cumsum() - per.cumsum().cummax()).values
ax4.fill_between(range(len(dd)), dd, 0, color="#ff1744", alpha=0.55)
ax4.plot(range(len(dd)), dd, color="#ff1744", lw=1.2)
ax4.axhline(0, color="#555", lw=0.8)
ax4.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
ax4.set_title("Drawdown (per period)", color="white", fontsize=10, fontweight="bold", pad=4)
ax4.set_ylabel("DD (%)", color="#aaa", fontsize=9)
ax4.set_xlim(-0.5, len(dd)-.5)

s = ps_port
box = (f"Total:    {s['tot']:>+6.1f}%\n"
       f"Ann Ret:  {s['ann']:>+6.1f}%\n"
       f"Ann Vol:  {s['vol']:>6.1f}%\n"
       f"Sharpe:   {s['sh']:>6.2f}\n"
       f"Sortino:  {s['so']:>6.2f}\n"
       f"Max DD:   {s['dd']:>+6.1f}%\n"
       f"WinRate:  {s['wr']:>5.0f}%\n"
       f"Avg Win:  {s['aw']:>+6.2f}%\n"
       f"Avg Loss: {s['al']:>+6.2f}%")
fig.text(0.987, 0.975, box, ha="right", va="top", fontsize=8.5,
         color="white", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                   edgecolor="#00e5ff", alpha=0.92))

sub = ("8 equities buy call↑/put↓ (1/8)  |  "
       "SPY: sell put = n_call/8, sell call = n_put/8  |  Sep 2020 – Feb 2026")
fig.text(0.5, 0.988, sub, ha="center", va="top",
         fontsize=8.5, color="#aaa", style="italic")

plt.tight_layout(rect=[0, 0, 1, 0.984])
plt.savefig(OUT_CHART, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"  Chart        → {OUT_CHART}")
print()
