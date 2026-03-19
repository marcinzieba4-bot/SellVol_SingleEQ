"""
backtest_vrp_spy.py
--------------------
Exploiting the SPY Volatility Risk Premium (VRP):

  Equity leg : buy call ↑ / buy put ↓  (1/8 each, always in)
  SPY leg    : SELL PUT only, weight = n_call / 8
               → zero SPY exposure when equities are bearish (never sell calls)
               → full SPY put-sell when all 8 equities are bullish

Rationale:
  • SPY implied vol > realized vol ~75-80% of periods  → structural premium for sellers
  • Equity markets have positive drift                 → selling puts aligns with drift
  • Selling calls fights drift AND pays lower VRP      → avoid completely
  • Scale with bullish conviction: more equities agree → larger SPY put position

Also shows, for reference:
  • SPY sell put ALWAYS (unconditional, full size)     → raw VRP harvest baseline
  • SPY sell put MOMENTUM only (existing sell_put mode)→ momentum-gated VRP
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
OUT_TABLE = "vrp_spy_monthly.csv"
OUT_CHART = "vrp_spy_chart.png"

print("=" * 66)
print("  VRP EXPLOITATION — SPY SELL PUT ONLY (no call selling)")
print("=" * 66)

# ── Load ──────────────────────────────────────────────────────────────────────
eq_res = {tk: run_strategy(tk, mode="buy_momentum") for tk in EQUITIES}

print("\n── SPY variants ──")
spy_sp_always = {r["period_start"]: r for r in run_strategy("SPY", mode="sell_put_always")}
spy_sp_mom    = {r["period_start"]: r for r in run_strategy("SPY", mode="sell_put")}

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

    spy_base = spy_sp_always.get(ps, {}).get("pnl_pct", 0.0)
    spy_mom  = spy_sp_mom.get(ps,    {}).get("pnl_pct", 0.0)

    # Proportional: only sell puts, weight = n_call / N
    w_prop   = n_call / N
    spy_prop = w_prop * spy_base

    rows.append({
        "period_start":    ps,
        "eq_pnl":          round(eq_pnl,              4),
        "spy_always_pnl":  round(spy_base,             4),
        "spy_mom_pnl":     round(spy_mom,              4),
        "spy_prop_pnl":    round(spy_prop,             4),
        "port_prop_pnl":   round(eq_pnl + spy_prop,   4),
        "port_always_pnl": round(eq_pnl + spy_base,   4),
        "port_mom_pnl":    round(eq_pnl + spy_mom,    4),
        "n_call": n_call, "n_put": n_put, "w_prop": round(w_prop, 3),
    })

df       = pd.DataFrame(rows).set_index("period_start")
df.index = pd.to_datetime(df.index)
df["ym"] = df.index.to_period("M")

def monthly_agg(df, col):
    m = df.groupby("ym")[col].sum()
    m.index = m.index.astype(str)
    return m

mo_eq         = monthly_agg(df, "eq_pnl")
mo_spy_always = monthly_agg(df, "spy_always_pnl")
mo_spy_mom    = monthly_agg(df, "spy_mom_pnl")
mo_spy_prop   = monthly_agg(df, "spy_prop_pnl")
mo_prop       = monthly_agg(df, "port_prop_pnl")
mo_always     = monthly_agg(df, "port_always_pnl")
mo_mom_port   = monthly_agg(df, "port_mom_pnl")
mo_nc         = df.groupby("ym")["n_call"].mean(); mo_nc.index = mo_nc.index.astype(str)
mo_np         = df.groupby("ym")["n_put"].mean();  mo_np.index = mo_np.index.astype(str)

idx = mo_prop.index

# ── Stats ─────────────────────────────────────────────────────────────────────
def stats(s, ppY=13.0):
    s = s.dropna()
    tot = s.sum(); ann = s.mean()*ppY; vol = s.std()*np.sqrt(ppY)
    sh  = ann/vol if vol else 0
    dv  = s[s<0].std()*np.sqrt(ppY) if (s<0).sum()>1 else 0
    so  = ann/dv  if dv  else 0
    dd  = (s.cumsum() - s.cumsum().cummax()).min()
    wins = (s>0).sum(); losses = (s<0).sum()
    return dict(tot=tot, ann=ann, vol=vol, sh=sh, so=so, dd=dd,
                wr=wins/len(s)*100 if len(s) else 0,
                wins=int(wins), losses=int(losses),
                aw=s[s>0].mean() if wins else 0,
                al=s[s<0].mean() if losses else 0)

per = df["port_prop_pnl"]

st_prop       = stats(df["port_prop_pnl"])
st_always     = stats(df["port_always_pnl"])
st_eq         = stats(df["eq_pnl"])
st_spy_always = stats(df["spy_always_pnl"])
st_spy_prop   = stats(df["spy_prop_pnl"])
st_spy_mom    = stats(df["spy_mom_pnl"])

# ── Monthly table ─────────────────────────────────────────────────────────────
W = 88
print(f"\n{'═'*W}")
print(f"  MONTHLY P&L  —  Eq + proportional SPY sell-put only")
print(f"{'═'*W}")
print(f"{'Month':<10} {'EqLeg%':>8} {'SPYProp%':>9} {'Port%':>8} {'Cum%':>9}  "
      f"{'nCall':>5} {'SPYwt':>6}")
print(f"{'─'*W}")
cum = 0.0
for m in idx:
    ep  = mo_eq.get(m, 0)
    sp  = mo_spy_prop.get(m, 0)
    pp  = mo_prop.get(m, 0)
    cum += pp
    nc  = mo_nc.get(m, 0)
    wt  = mo_nc.get(m, 0) / N
    flag = " ▼" if pp < -3 else ("  " if pp >= 0 else " ·")
    print(f"{m:<10} {ep:>7.2f}% {sp:>8.2f}% {pp:>7.2f}% {cum:>8.2f}%  "
          f"{nc:>5.1f} {wt:>5.0%}{flag}")
print(f"{'─'*W}")
print(f"{'TOTAL':<10} {mo_eq.sum():>7.2f}% {mo_spy_prop.sum():>8.2f}% "
      f"{mo_prop.sum():>7.2f}%")

# ── Risk stats — full comparison table ────────────────────────────────────────
print(f"\n{'═'*84}")
print(f"  RISK STATISTICS — ALL VARIANTS")
print(f"{'═'*84}")
print(f"  {'Metric':<22} {'Eq only':>10} {'Prop SPY':>10} {'Always SPY':>11} "
      f"{'SPY alone':>10} {'SPY(mom)':>10}")
print(f"  {'─'*82}")
cols = [st_eq, st_prop, st_always, st_spy_always, st_spy_mom]
for label, key, fmt in [
    ("Total Return",    "tot",    "{:>+9.2f}%"),
    ("Ann. Return",     "ann",    "{:>+9.2f}%"),
    ("Ann. Vol",        "vol",    "{:>9.2f}%"),
    ("Sharpe",          "sh",     "{:>9.2f} "),
    ("Sortino",         "so",     "{:>9.2f} "),
    ("Max Drawdown",    "dd",     "{:>+9.2f}%"),
    ("Win Rate",        "wr",     "{:>9.1f}%"),
    ("Avg Win",         "aw",     "{:>+9.2f}%"),
    ("Avg Loss",        "al",     "{:>+9.2f}%"),
    ("Win Periods",     "wins",   "{:>9}  "),
    ("Loss Periods",    "losses", "{:>9}  "),
]:
    row = "  " + f"{label:<22}"
    for c in cols:
        row += " " + fmt.format(c[key])
    print(row)
print(f"{'═'*84}")

# ── SPY-only VRP stats ────────────────────────────────────────────────────────
print(f"\n  SPY SELL PUT — STANDALONE ANALYSIS")
print(f"  {'Mode':<22} {'Total':>9} {'Ann':>8} {'Sharpe':>8} {'MaxDD':>9} {'WinRate':>9}")
print(f"  {'─'*70}")
for label, s in [("Always (full size)", st_spy_always),
                  ("Momentum filter",    st_spy_mom),
                  ("Prop to n_call/8",   st_spy_prop)]:
    print(f"  {label:<22} {s['tot']:>+8.2f}% {s['ann']:>+7.2f}% "
          f"{s['sh']:>8.2f} {s['dd']:>+8.2f}% {s['wr']:>8.1f}%")

# ── Save CSV ──────────────────────────────────────────────────────────────────
out = pd.DataFrame({
    "eq_pnl":       mo_eq,
    "spy_prop_pnl": mo_spy_prop,
    "port_prop_pnl":mo_prop,
    "port_cum":     mo_prop.cumsum(),
    "eq_cum":       mo_eq.cumsum(),
    "n_call":       mo_nc,
    "spy_weight":   mo_nc / N,
})
out.to_csv(OUT_TABLE, float_format="%.4f")
print(f"\n  Monthly table → {OUT_TABLE}")

# ── Chart ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(16, 17),
                          gridspec_kw={"height_ratios": [3, 2, 1.8, 1.2]})
fig.patch.set_facecolor("#0f0f1a")
for ax in axes:
    ax.set_facecolor("#0f0f1a")
    for sp in ["top","right"]:   ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#444")
    ax.tick_params(colors="#aaa", labelsize=9)

mo_arr    = np.arange(len(idx))
tick_step = max(1, len(mo_arr) // 16)
xlabels   = [idx[i] for i in range(0, len(idx), tick_step)]

# P1 — Cumulative (4 curves)
ax1 = axes[0]
ax1.plot(mo_arr, mo_prop.cumsum().values,    color="#00e5ff", lw=2.4, zorder=5,
         label=f"Eq + prop SPY sell-put  ({st_prop['tot']:+.1f}%)")
ax1.plot(mo_arr, mo_eq.cumsum().values,      color="#76ff03", lw=1.5, ls="--", alpha=0.8,
         label=f"Equity only              ({st_eq['tot']:+.1f}%)")
ax1.plot(mo_arr, mo_always.cumsum().values,  color="#ffd740", lw=1.3, ls=":", alpha=0.85,
         label=f"Eq + SPY sell-put always ({st_always['tot']:+.1f}%)")
ax1.plot(mo_arr, mo_spy_always.cumsum().values, color="#ff9100", lw=1.2, ls="--", alpha=0.7,
         label=f"SPY sell-put alone       ({st_spy_always['tot']:+.1f}%)")
ax1.axhline(0, color="#555", lw=0.8, ls=":")
ax1.fill_between(mo_arr, mo_prop.cumsum().values, 0,
                 where=(mo_prop.cumsum().values >= 0), alpha=0.10, color="#00e5ff")
ax1.fill_between(mo_arr, mo_prop.cumsum().values, 0,
                 where=(mo_prop.cumsum().values < 0),  alpha=0.15, color="#ff1744")
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
ax1.set_title("VRP Exploitation — SPY Sell Put Only (proportional to bullish equity count)",
              color="white", fontsize=12, fontweight="bold", pad=10)
ax1.set_ylabel("Cumulative P&L (%)", color="#aaa", fontsize=9)
ax1.legend(loc="upper left", fontsize=8.2, facecolor="#1a1a2e",
           edgecolor="#444", labelcolor="white", framealpha=0.9)
ax1.set_xlim(-0.5, len(mo_arr)-.5)
ax1.set_xticks(range(0, len(mo_arr), tick_step))
ax1.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
final = mo_prop.cumsum().iloc[-1]
ax1.annotate(f"{final:+.1f}%", xy=(mo_arr[-1], final),
             xytext=(-55, 10), textcoords="offset points",
             color="#00e5ff", fontsize=10, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#00e5ff", lw=1.2))

# P2 — Monthly bars split: equity (green/red) + SPY put contribution (teal overlay)
ax2 = axes[1]
eq_arr  = mo_eq.reindex(idx, fill_value=0).values
spy_arr = mo_spy_prop.reindex(idx, fill_value=0).values
bar_col = ["#00e676" if v >= 0 else "#ff1744" for v in (eq_arr + spy_arr)]
ax2.bar(mo_arr, eq_arr,  color=bar_col, alpha=0.75, width=0.7, label="Equity leg")
ax2.bar(mo_arr, spy_arr, bottom=eq_arr, color="#00bcd4", alpha=0.85, width=0.7,
        label="SPY sell-put (prop.)")
ax2.axhline(0, color="#555", lw=0.8)
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
ax2.set_title("Monthly P&L — Equity (green/red) + SPY sell-put contribution (teal)",
              color="white", fontsize=10, fontweight="bold", pad=6)
ax2.set_ylabel("Monthly P&L (%)", color="#aaa", fontsize=9)
ax2.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
ax2.set_xlim(-0.5, len(mo_arr)-.5)
ax2.set_xticks(range(0, len(mo_arr), tick_step))
ax2.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)

# P3 — SPY put weight over time + cumulative SPY-only P&L
ax3  = axes[2]
ax3b = ax3.twinx()
wt_arr = (mo_nc.reindex(idx, fill_value=0) / N).values
ax3.bar(mo_arr, wt_arr, color="#00bcd4", alpha=0.55, width=0.7, label="SPY sell-put weight")
ax3b.plot(mo_arr, mo_spy_always.reindex(idx, fill_value=0).cumsum().values,
          color="#ff9100", lw=1.6, label="SPY sell-put alone (cumulative)")
ax3b.plot(mo_arr, mo_spy_prop.reindex(idx, fill_value=0).cumsum().values,
          color="#00e5ff", lw=1.6, ls="--", label="SPY prop. portion (cumulative)")
ax3b.axhline(0, color="#555", lw=0.8, ls=":")
ax3.set_ylim(0, 1.3); ax3.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax3b.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
ax3b.tick_params(colors="#aaa", labelsize=9)
ax3.set_title("SPY Sell-Put Weight (bars) vs Cumulative SPY P&L (lines)",
              color="white", fontsize=10, fontweight="bold", pad=6)
ax3.set_ylabel("SPY weight", color="#aaa", fontsize=9)
ax3b.set_ylabel("Cum P&L (%)", color="#aaa", fontsize=9)
lines1, labs1 = ax3.get_legend_handles_labels()
lines2, labs2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1+lines2, labs1+labs2, fontsize=7.5, facecolor="#1a1a2e",
           edgecolor="#444", labelcolor="white", loc="lower left")
ax3.set_xlim(-0.5, len(mo_arr)-.5)
ax3.set_xticks(range(0, len(mo_arr), tick_step))
ax3.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)

# P4 — Drawdown comparison
ax4 = axes[3]
for series, col, lbl, lw in [
    (df["port_prop_pnl"],   "#00e5ff", "Eq + prop SPY",  2.2),
    (df["eq_pnl"],          "#76ff03", "Equity only",    1.2),
    (df["port_always_pnl"], "#ffd740", "Eq + always SPY",1.2),
]:
    dd = (series.cumsum() - series.cumsum().cummax()).values
    ax4.plot(range(len(dd)), dd, color=col, lw=lw, alpha=0.85, label=lbl)
ax4.fill_between(range(len(df)), (df["port_prop_pnl"].cumsum() - df["port_prop_pnl"].cumsum().cummax()).values,
                 0, color="#00e5ff", alpha=0.12)
ax4.axhline(0, color="#555", lw=0.8)
ax4.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
ax4.set_title("Drawdown Comparison", color="white", fontsize=10, fontweight="bold", pad=4)
ax4.set_ylabel("DD (%)", color="#aaa", fontsize=9)
ax4.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", loc="lower right")
ax4.set_xlim(-0.5, len(df)-.5)

# Stats box
s = st_prop
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
       "SPY: SELL PUT only at weight n_call/8  |  never sell SPY calls  |  Sep 2020 – Feb 2026")
fig.text(0.5, 0.988, sub, ha="center", va="top",
         fontsize=8.5, color="#aaa", style="italic")

plt.tight_layout(rect=[0, 0, 1, 0.984])
plt.savefig(OUT_CHART, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"  Chart        → {OUT_CHART}")
print()
