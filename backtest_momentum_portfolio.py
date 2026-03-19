"""
backtest_momentum_portfolio.py
------------------------------
Long leg : 8 single equities (AAPL, AMD, AMZN, JPM, LLY, NVDA, TSLA, XOM)
           each 1/8 equal weight.
           Signal > 0 → BUY CALL  |  Signal < 0 → BUY PUT  (always in market)

SPY leg  : 100% weight.
           Signal > 0 → SELL PUT  |  Signal < 0 → SELL CALL (always in market)

P&L aggregated by calendar month.  Risk stats + PNG chart saved.
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

EQUITIES   = ["AAPL", "AMD", "AMZN", "JPM", "LLY", "NVDA", "TSLA", "XOM"]
EQ_WEIGHT  = 1 / len(EQUITIES)   # 1/8 each
SPY        = "SPY"

OUT_TABLE  = "momentum_portfolio_monthly.csv"
OUT_CHART  = "momentum_portfolio_chart.png"

# ── 1. Load all strategies ────────────────────────────────────────────────────
print("=" * 62)
print("  MOMENTUM PORTFOLIO BACKTEST")
print("  Equities: buy call / buy put on momentum")
print("  SPY:      sell put / sell call on momentum")
print("=" * 62)

all_results = {}

print("\n── Equities (buy_momentum) ──")
for tk in EQUITIES:
    all_results[tk] = run_strategy(tk, mode="buy_momentum")

print(f"\n── SPY (sell_spy_momentum) ──")
all_results[SPY] = run_strategy(SPY, mode="sell_spy_momentum")

# ── 2. Build per-period portfolio P&L ────────────────────────────────────────
# Index each ticker's results by period_start for fast lookup
def to_series(results):
    return {r["period_start"]: r for r in results}

idx = {tk: to_series(res) for tk, res in all_results.items()}

# Union of all period starts across all tickers
all_periods = sorted({ps for res in all_results.values() for r in res for ps in [r["period_start"]]})

rows = []
for ps in all_periods:
    eq_pnl  = 0.0
    eq_cnt  = 0
    eq_detail = {}

    for tk in EQUITIES:
        r = idx[tk].get(ps)
        if r:
            eq_pnl += EQ_WEIGHT * r["pnl_pct"]
            eq_detail[tk] = {"trade": r["trade"], "pnl": round(r["pnl_pct"], 3)}
            eq_cnt += 1

    spy_r    = idx[SPY].get(ps)
    spy_pnl  = spy_r["pnl_pct"] if spy_r else 0.0
    spy_trade= spy_r["trade"]   if spy_r else "—"

    port_pnl = eq_pnl + spy_pnl

    rows.append({
        "period_start": ps,
        "eq_pnl":       round(eq_pnl,  4),
        "spy_pnl":      round(spy_pnl, 4),
        "port_pnl":     round(port_pnl,4),
        "spy_trade":    spy_trade,
        "eq_detail":    eq_detail,
        "eq_cnt":       eq_cnt,
    })

period_df = pd.DataFrame(rows).set_index("period_start")
period_df.index = pd.to_datetime(period_df.index)

# ── 3. Monthly aggregation ────────────────────────────────────────────────────
period_df["ym"] = period_df.index.to_period("M")
monthly = period_df.groupby("ym").agg(
    eq_pnl   = ("eq_pnl",   "sum"),
    spy_pnl  = ("spy_pnl",  "sum"),
    port_pnl = ("port_pnl", "sum"),
    n_periods= ("port_pnl", "count"),
)
monthly.index = monthly.index.astype(str)
monthly["cum_port"]  = monthly["port_pnl"].cumsum()
monthly["cum_eq"]    = monthly["eq_pnl"].cumsum()
monthly["cum_spy"]   = monthly["spy_pnl"].cumsum()

# ── 4. Per-period risk stats ──────────────────────────────────────────────────
per = period_df["port_pnl"]
per_eq  = period_df["eq_pnl"]
per_spy = period_df["spy_pnl"]

periods_per_year = 13.0

def stats(s, ppY=13.0, label=""):
    s = s.dropna()
    tot  = s.sum()
    ann  = s.mean() * ppY
    vol  = s.std()  * np.sqrt(ppY)
    sh   = ann / vol if vol else 0
    neg  = s[s < 0]
    dv   = neg.std() * np.sqrt(ppY) if len(neg) > 1 else 0
    so   = ann / dv  if dv  else 0
    cum  = s.cumsum()
    rm   = cum.cummax()
    dd   = (cum - rm).min()
    wins = (s > 0).sum()
    loss = (s < 0).sum()
    wr   = wins / len(s) * 100 if len(s) else 0
    aw   = s[s > 0].mean() if wins else 0
    al   = s[s < 0].mean() if loss else 0
    return dict(label=label, total=tot, ann=ann, vol=vol,
                sharpe=sh, sortino=so, maxdd=dd,
                winrate=wr, wins=wins, losses=loss,
                avg_win=aw, avg_loss=al, n=len(s))

ps = stats(per,     label="Portfolio")
es = stats(per_eq,  label="Equity leg")
ss = stats(per_spy, label="SPY leg   ")

# ── 5. Print monthly table ────────────────────────────────────────────────────
W = 72
print(f"\n{'═'*W}")
print(f"  MONTHLY P&L")
print(f"{'═'*W}")
print(f"{'Month':<10} {'EqLeg%':>8} {'SPYLeg%':>8} {'Port%':>8} {'Cum%':>9}  n")
print(f"{'─'*W}")
for ym, row in monthly.iterrows():
    flag = " ▼" if row["port_pnl"] < -3 else ("  " if row["port_pnl"] >= 0 else " ·")
    print(f"{ym:<10} {row['eq_pnl']:>7.2f}% {row['spy_pnl']:>7.2f}% "
          f"{row['port_pnl']:>7.2f}% {row['cum_port']:>8.2f}%  "
          f"{int(row['n_periods'])}{flag}")
print(f"{'─'*W}")
print(f"{'TOTAL':<10} {monthly['eq_pnl'].sum():>7.2f}% "
      f"{monthly['spy_pnl'].sum():>7.2f}% {monthly['port_pnl'].sum():>7.2f}%")

# ── 6. Risk stats table ───────────────────────────────────────────────────────
print(f"\n{'═'*62}")
print(f"  RISK STATISTICS  (per-period returns, {periods_per_year:.0f} periods/year)")
print(f"{'═'*62}")
print(f"  {'Metric':<26} {'Portfolio':>11} {'Eq Leg':>10} {'SPY Leg':>10}")
print(f"  {'─'*60}")
metrics = [
    ("Total Return",        "total",   "{:>+10.2f}%"),
    ("Ann. Return",         "ann",     "{:>+10.2f}%"),
    ("Ann. Volatility",     "vol",     "{:>10.2f}%"),
    ("Sharpe Ratio",        "sharpe",  "{:>10.2f}"),
    ("Sortino Ratio",       "sortino", "{:>10.2f}"),
    ("Max Drawdown",        "maxdd",   "{:>+10.2f}%"),
    ("Win Rate",            "winrate", "{:>10.1f}%"),
    ("Avg Win",             "avg_win", "{:>+10.2f}%"),
    ("Avg Loss",            "avg_loss","{:>+10.2f}%"),
    ("Winning Periods",     "wins",    "{:>10}"),
    ("Losing Periods",      "losses",  "{:>10}"),
]
for label, key, fmt in metrics:
    pv = ps[key]; ev = es[key]; sv = ss[key]
    pf = fmt.format(pv); ef = fmt.format(ev); sf = fmt.format(sv)
    print(f"  {label:<26} {pf} {ef} {sf}")
print(f"{'═'*62}")

# ── 7. Trade direction breakdown ─────────────────────────────────────────────
print(f"\n  TRADE DIRECTION BREAKDOWN (per-period counts)")
print(f"  {'Ticker':<8} {'BuyCall':>8} {'BuyPut':>8} {'NoTrade':>9} {'AvgPnl':>8}")
print(f"  {'─'*50}")
for tk in EQUITIES:
    res = all_results[tk]
    bc = sum(1 for r in res if r["trade"] == "BUY CALL")
    bp = sum(1 for r in res if r["trade"] == "BUY PUT")
    nt = sum(1 for r in res if "NO" in r["trade"])
    pnls = [r["pnl_pct"] for r in res if "NO" not in r["trade"]]
    avg  = np.mean(pnls) if pnls else 0
    print(f"  {tk:<8} {bc:>8} {bp:>8} {nt:>9} {avg:>+7.2f}%")
print(f"  {'─'*50}")
spy_res = all_results[SPY]
sp_cnt  = sum(1 for r in spy_res if r["trade"] == "SELL PUT")
sc_cnt  = sum(1 for r in spy_res if r["trade"] == "SELL CALL")
nt_cnt  = sum(1 for r in spy_res if "NO" in r["trade"])
spy_pnls= [r["pnl_pct"] for r in spy_res if "NO" not in r["trade"]]
spy_avg = np.mean(spy_pnls) if spy_pnls else 0
print(f"  {'SPY':<8} {'SellPut:':>8}{sp_cnt:<3} {'SellCall:':>5}{sc_cnt:<3} "
      f"{nt_cnt:>9} {spy_avg:>+7.2f}%")
print()

# ── 8. Save CSV ───────────────────────────────────────────────────────────────
monthly.to_csv(OUT_TABLE, float_format="%.4f")
print(f"  Monthly table saved → {OUT_TABLE}")

# ── 9. Chart ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                          gridspec_kw={"height_ratios": [3, 2, 1.5]})
fig.patch.set_facecolor("#0f0f1a")
for ax in axes:
    ax.set_facecolor("#0f0f1a")
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]:
        ax.spines[sp].set_color("#444")
    ax.tick_params(colors="#aaa", labelsize=9)

mo_idx    = np.arange(len(monthly))
mo_labels = list(monthly.index)
tick_step = max(1, len(mo_idx) // 16)

# Panel 1 — Cumulative P&L
ax1 = axes[0]
ax1.plot(mo_idx, monthly["cum_port"], color="#00e5ff", lw=2.2,
         label="Portfolio (eq + SPY)", zorder=3)
ax1.plot(mo_idx, monthly["cum_eq"],   color="#76ff03", lw=1.4, ls="--", alpha=0.85,
         label="Equity leg (avg 8 stocks)")
ax1.plot(mo_idx, monthly["cum_spy"],  color="#ff9100", lw=1.4, ls="--", alpha=0.85,
         label="SPY leg")
ax1.axhline(0, color="#555", lw=0.8, ls=":")
ax1.fill_between(mo_idx, monthly["cum_port"], 0,
                 where=(monthly["cum_port"] >= 0), alpha=0.12, color="#00e5ff")
ax1.fill_between(mo_idx, monthly["cum_port"], 0,
                 where=(monthly["cum_port"] < 0),  alpha=0.15, color="#ff1744")
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
ax1.set_title("Momentum Portfolio — Cumulative P&L", color="white",
              fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("Cumulative P&L (%)", color="#aaa", fontsize=9)
ax1.legend(loc="upper left", fontsize=8.5, facecolor="#1a1a2e",
           edgecolor="#444", labelcolor="white", framealpha=0.9)
ax1.set_xlim(-0.5, len(mo_idx) - 0.5)
ax1.set_xticks(mo_idx[::tick_step])
ax1.set_xticklabels(mo_labels[::tick_step], rotation=40, ha="right", fontsize=8)

# Annotate final values
for val, col, xoff in [
    (monthly["cum_port"].iloc[-1], "#00e5ff", -50),
    (monthly["cum_eq"].iloc[-1],   "#76ff03", -50),
    (monthly["cum_spy"].iloc[-1],  "#ff9100", -50),
]:
    ax1.annotate(f"{val:+.1f}%", xy=(mo_idx[-1], val),
                 xytext=(xoff, 8), textcoords="offset points",
                 color=col, fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=col, lw=1.1))

# Panel 2 — Monthly bars (colour by direction)
ax2 = axes[1]
colors = ["#00e676" if v >= 0 else "#ff1744" for v in monthly["port_pnl"]]
ax2.bar(mo_idx, monthly["port_pnl"], color=colors, alpha=0.85, width=0.7, zorder=2)
ax2.axhline(0, color="#555", lw=0.8)
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
ax2.set_title("Monthly Portfolio P&L", color="white", fontsize=11,
              fontweight="bold", pad=6)
ax2.set_ylabel("Monthly P&L (%)", color="#aaa", fontsize=9)
ax2.set_xlim(-0.5, len(mo_idx) - 0.5)
ax2.set_xticks(mo_idx[::tick_step])
ax2.set_xticklabels(mo_labels[::tick_step], rotation=40, ha="right", fontsize=8)

# Panel 3 — Drawdown
ax3 = axes[2]
cum_series  = per.reset_index(drop=True).cumsum()
dd_series   = cum_series - cum_series.cummax()
dd_idx      = np.arange(len(dd_series))
ax3.fill_between(dd_idx, dd_series, 0, color="#ff1744", alpha=0.55)
ax3.plot(dd_idx, dd_series, color="#ff1744", lw=1.2)
ax3.axhline(0, color="#555", lw=0.8)
ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
ax3.set_title("Drawdown (per period)", color="white", fontsize=11,
              fontweight="bold", pad=6)
ax3.set_ylabel("Drawdown (%)", color="#aaa", fontsize=9)
ax3.set_xlim(-0.5, len(dd_idx) - 0.5)

# Stats box
stats_text = (
    f"Total Return:  {ps['total']:>+6.1f}%\n"
    f"Ann. Return:   {ps['ann']:>+6.1f}%\n"
    f"Ann. Vol:      {ps['vol']:>6.1f}%\n"
    f"Sharpe:        {ps['sharpe']:>6.2f}\n"
    f"Sortino:       {ps['sortino']:>6.2f}\n"
    f"Max Drawdown:  {ps['maxdd']:>+6.1f}%\n"
    f"Win Rate:      {ps['winrate']:>5.0f}%\n"
    f"Avg Win:       {ps['avg_win']:>+6.2f}%\n"
    f"Avg Loss:      {ps['avg_loss']:>+6.2f}%"
)
fig.text(0.987, 0.97, stats_text, ha="right", va="top", fontsize=8.5,
         color="white", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                   edgecolor="#00e5ff", alpha=0.92))

subtitle = (
    f"Long: {', '.join(EQUITIES)} (1/8 each, buy call↑ / buy put↓)  |  "
    f"SPY: sell put↑ / sell call↓  |  Sep 2020 – Feb 2026"
)
fig.text(0.5, 0.987, subtitle, ha="center", va="top",
         fontsize=8.5, color="#aaa", style="italic")

plt.tight_layout(rect=[0, 0, 1, 0.983])
plt.savefig(OUT_CHART, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"  Chart saved        → {OUT_CHART}")
print()
