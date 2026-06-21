"""
generate_variantC_charts.py
----------------------------
Generates two publication-quality PNGs from pre-computed CSV data.
No backtest re-run needed.

Outputs (written to results/):
  variantC_equity.png   — equity curve + drawdown panels
  variantC_heatmap.png  — monthly returns heatmap (text always legible)
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

BASE   = os.path.dirname(os.path.abspath(__file__))
RES    = os.path.join(BASE, "results")

# ── chart sizes tuned for PDF embedding (≈ 17 cm content width @ 150 dpi) ──
PDF_WIDTH_IN  = 7.2   # inches  ← matches reportlab CONTENT_W
DPI           = 150

# ── colour palette ──────────────────────────────────────────────────────────
DARK_BLUE = "#0A2342"
MID_BLUE  = "#1F4E79"
GREEN     = "#2ca02c"
RED       = "#d62728"
LIGHT_BG  = "#F8F9FA"


# ════════════════════════════════════════════════════════════════════════════
#  Load data
# ════════════════════════════════════════════════════════════════════════════

def load_periods(variant="C"):
    path = os.path.join(RES, f"signal_periods_{variant}.csv")
    df = pd.read_csv(path, parse_dates=["period_start"])
    df = df.sort_values("period_start").reset_index(drop=True)
    return df


def load_monthly(variant="C"):
    path = os.path.join(RES, f"signal_monthly_{variant}.csv")
    df = pd.read_csv(path)
    return df


# ════════════════════════════════════════════════════════════════════════════
#  Chart 1 — Equity curve + Drawdown
# ════════════════════════════════════════════════════════════════════════════

def chart_equity(variant="C"):
    df = load_periods(variant)

    pnl = df["total_pnl"].values
    dates = df["period_start"].values

    # Cumulative equity
    equity = np.cumsum(pnl)

    # Drawdown series
    running_max = np.maximum.accumulate(equity)
    drawdown    = equity - running_max          # ≤ 0

    stats = {
        "total": equity[-1],
        "cagr":  ((1 + equity[-1]/100) ** (1/(len(pnl)/13)) - 1) * 100,
        "max_dd": float(np.min(drawdown)),
        "sharpe": np.mean(pnl) / np.std(pnl) * math.sqrt(13) if np.std(pnl) else 0,
        "sortino": (np.mean(pnl) / math.sqrt(np.mean(pnl[pnl < 0]**2)) * math.sqrt(13)
                    if np.any(pnl < 0) else 0),
        "wins": int(np.sum(pnl > 0)),
        "n":    len(pnl),
    }

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(PDF_WIDTH_IN, PDF_WIDTH_IN * 0.75),
        gridspec_kw={"height_ratios": [3, 1.2]},
        facecolor=LIGHT_BG,
    )
    fig.patch.set_facecolor(LIGHT_BG)

    # ── subtitle stats box ────────────────────────────────────────────────
    subtitle = (
        f"Total: {stats['total']:+.1f}%   "
        f"CAGR: {stats['cagr']:+.1f}%/yr   "
        f"Max DD: {stats['max_dd']:.1f}%   "
        f"Sharpe: {stats['sharpe']:.2f}   "
        f"Sortino: {stats['sortino']:.2f}   "
        f"Win rate: {stats['wins']}/{stats['n']} "
        f"({stats['wins']/stats['n']*100:.0f}%)"
    )
    variant_label = ("Unlevered — Idle Cash Earns Money-Market" if variant == "B"
                      else "4x Leveraged — Idle Cash Earns Money-Market" if variant == "C"
                      else f"Variant {variant}")
    fig.suptitle(
        f"Momentum Call Strategy   |   Signal-Filtered (Cap 30)   |   S&P 500 Universe   |   {variant_label}",
        fontsize=10, fontweight="bold", color=DARK_BLUE, y=0.99,
    )

    # ── Panel 1: Equity curve ─────────────────────────────────────────────
    ax1.set_facecolor(LIGHT_BG)
    ax1.plot(dates, equity, color=GREEN, linewidth=1.8, zorder=3)
    ax1.fill_between(dates, equity, 0, alpha=0.12, color=GREEN, zorder=2)
    ax1.axhline(0, color="grey", linewidth=0.6, linestyle=":")

    # Annotate best and worst periods
    best_idx  = int(np.argmax(pnl))
    worst_idx = int(np.argmin(pnl))
    for idx, color, label in [
        (best_idx,  GREEN, f"Best {pnl[best_idx]:+.1f}%"),
        (worst_idx, RED,   f"Worst {pnl[worst_idx]:+.1f}%"),
    ]:
        ax1.annotate(
            label,
            xy=(dates[idx], equity[idx]),
            xytext=(0, 14),
            textcoords="offset points",
            fontsize=7, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            ha="center",
        )

    # Stats text box inside chart
    ax1.text(
        0.01, 0.97, subtitle,
        transform=ax1.transAxes,
        fontsize=6.8, va="top", ha="left",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=MID_BLUE, alpha=0.85, linewidth=0.7),
    )

    ax1.set_ylabel("Cumulative P&L (%)", fontsize=8)
    ax1.set_title("Equity Curve", fontsize=9, color=MID_BLUE, pad=4)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f%%"))
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    ax1.tick_params(labelsize=7)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    ax2.set_facecolor(LIGHT_BG)
    ax2.fill_between(dates, drawdown, 0, alpha=0.55, color=RED, zorder=2)
    ax2.plot(dates, drawdown, color=RED, linewidth=1.2, zorder=3)
    ax2.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax2.set_ylabel("Drawdown (%)", fontsize=8)
    ax2.set_title("Drawdown from Peak", fontsize=9, color=MID_BLUE, pad=4)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.tick_params(labelsize=7)

    # Shared x-axis formatting
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.6)
    out = os.path.join(RES, f"variant{variant}_equity.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"✓ Equity chart → {out}")
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Chart 2 — Monthly heatmap  (no text overlap)
# ════════════════════════════════════════════════════════════════════════════

def chart_heatmap(variant="C"):
    df = load_monthly(variant)
    months  = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

    years = df["Year"].astype(str).tolist()
    n_rows = len(years)
    n_cols = len(months)          # 12

    # Build numeric grid  (n_rows × 12)
    grid = np.full((n_rows, n_cols), np.nan)
    for ri, row in df.iterrows():
        for ci, m in enumerate(months):
            v = row.get(m)
            try:
                grid[ri, ci] = float(v)
            except (TypeError, ValueError):
                pass

    annual = np.array([
        row.get("Annual", np.nan) for _, row in df.iterrows()
    ], dtype=float)

    # ── figure sizing: give each cell ~0.52 inches wide, ~0.52 tall ──────
    cell_w  = 0.52
    cell_h  = 0.52
    ann_col = 0.72          # annual column is a bit wider
    left_m  = 0.52          # year label column
    cbar_w  = 0.28
    fig_w   = left_m + n_cols * cell_w + ann_col + cbar_w + 0.15
    fig_h   = 0.55 + n_rows * cell_h + 0.45

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)
    variant_label = ("Unlevered" if variant == "B"
                      else "4x Leveraged" if variant == "C"
                      else f"Variant {variant}")
    fig.suptitle(
        f"Monthly Returns — Momentum Call Strategy ({variant_label})",
        fontsize=9.5, fontweight="bold", color=DARK_BLUE, y=0.995,
    )

    abs_max = np.nanmax(np.abs(grid))
    norm    = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    cmap    = plt.cm.RdYlGn

    # Draw main grid cells
    for ri in range(n_rows):
        for ci in range(n_cols):
            v = grid[ri, ci]
            x = left_m + ci * cell_w
            y = (n_rows - 1 - ri) * cell_h   # top-row = first year

            if np.isnan(v):
                face = "#E8E8E8"
                txt  = ""
            else:
                face = cmap(norm(v))
                txt  = f"{v:+.1f}"

            rect = FancyBboxPatch(
                (x + 0.02, y + 0.02),
                cell_w - 0.04, cell_h - 0.04,
                boxstyle="round,pad=0.02",
                facecolor=face, edgecolor="white", linewidth=0.6,
            )
            ax.add_patch(rect)

            if txt:
                # Choose font color for contrast
                r_val, g_val, b_val, _ = cmap(norm(v))
                lum = 0.299*r_val + 0.587*g_val + 0.114*b_val
                txt_color = "#1a1a1a" if lum > 0.45 else "white"
                ax.text(
                    x + cell_w/2, y + cell_h/2, txt,
                    ha="center", va="center",
                    fontsize=7.5, fontweight="bold",
                    color=txt_color,
                )

    # Annual column (right side)
    ann_x = left_m + n_cols * cell_w + 0.06
    ax.text(
        ann_x + ann_col/2, n_rows * cell_h + 0.12,
        "Annual", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=DARK_BLUE,
    )
    for ri, v in enumerate(annual):
        y = (n_rows - 1 - ri) * cell_h
        if np.isnan(v):
            face = "#E8E8E8"; txt = ""
        else:
            face = cmap(norm(v)); txt = f"{v:+.1f}%"
        rect = FancyBboxPatch(
            (ann_x + 0.02, y + 0.02),
            ann_col - 0.04, cell_h - 0.04,
            boxstyle="round,pad=0.02",
            facecolor=face, edgecolor="white", linewidth=0.6,
        )
        ax.add_patch(rect)
        if txt:
            r_val, g_val, b_val, _ = cmap(norm(v))
            lum = 0.299*r_val + 0.587*g_val + 0.114*b_val
            txt_color = "#1a1a1a" if lum > 0.45 else "white"
            ax.text(
                ann_x + ann_col/2, y + cell_h/2, txt,
                ha="center", va="center",
                fontsize=7.5, fontweight="bold",
                color=txt_color,
            )

    # Year labels (left)
    for ri, yr in enumerate(years):
        y = (n_rows - 1 - ri) * cell_h + cell_h/2
        ax.text(
            left_m - 0.06, y, str(yr),
            ha="right", va="center",
            fontsize=8, color=DARK_BLUE, fontweight="bold",
        )

    # Month labels (top)
    for ci, m in enumerate(months):
        x = left_m + ci * cell_w + cell_w/2
        ax.text(
            x, n_rows * cell_h + 0.12, m,
            ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=DARK_BLUE,
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([
        (left_m + n_cols * cell_w + ann_col + 0.12) / fig_w,
        0.1, 0.018, 0.75,
    ])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Return (%)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    ax.set_xlim(0, left_m + n_cols * cell_w + ann_col + 0.05)
    ax.set_ylim(-0.1, n_rows * cell_h + 0.35)
    ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(RES, f"variant{variant}_heatmap.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"✓ Heatmap chart → {out}")
    return out


if __name__ == "__main__":
    for v in ("B", "C"):
        chart_equity(v)
        chart_heatmap(v)
