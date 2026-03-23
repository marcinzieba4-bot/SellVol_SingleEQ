"""
generate_report.py
------------------
Focused PDF report: Buy Call Momentum — Variant C
  (4× Leverage + Money Market, Dynamic Top-30, SPX 50)

Sections
  1. Cover
  2. Strategy Overview
  3. Universe & Signal Rules
  4. Risk Statistics
  5. Equity Curve & Drawdown
  6. Monthly Returns Heatmap
  7. Monthly Returns Table
  8. Key Take-Aways

Output: results/SellVol_Report.pdf
"""

import os
import math
from datetime import date

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

# ── paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(BASE, "results")
OUT  = os.path.join(RES,  "SellVol_Report.pdf")

# ── colours ────────────────────────────────────────────────────────────────
DARK_BLUE  = colors.HexColor("#0A2342")
MID_BLUE   = colors.HexColor("#1F4E79")
ACCENT     = colors.HexColor("#2E86AB")
LIGHT_GRAY = colors.HexColor("#F5F5F5")
MED_GRAY   = colors.HexColor("#CCCCCC")
GREEN_CELL = colors.HexColor("#C6EFCE")
RED_CELL   = colors.HexColor("#FFC7CE")
ZERO_CELL  = colors.HexColor("#FFEB9C")
HEADER_BG  = colors.HexColor("#1F4E79")
ALT_ROW    = colors.HexColor("#EBF3FB")
WHITE      = colors.white
BLACK      = colors.black

# ── page geometry ──────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN         = 1.8 * cm
CONTENT_W      = PAGE_W - 2 * MARGIN


# ══════════════════════════════════════════════════════════════════════════
#  Styles
# ══════════════════════════════════════════════════════════════════════════

def _styles():
    base = getSampleStyleSheet()

    def add(name, **kw):
        if name not in base:
            base.add(ParagraphStyle(name=name, **kw))
        return base[name]

    add("CoverTitle",
        fontName="Helvetica-Bold", fontSize=26, leading=32,
        textColor=WHITE, alignment=TA_CENTER, spaceAfter=8)
    add("CoverSub",
        fontName="Helvetica", fontSize=13, leading=17,
        textColor=colors.HexColor("#BDD7EE"), alignment=TA_CENTER, spaceAfter=5)
    add("CoverMeta",
        fontName="Helvetica-Oblique", fontSize=10, leading=13,
        textColor=colors.HexColor("#9DC3E6"), alignment=TA_CENTER)
    add("SectionTitle",
        fontName="Helvetica-Bold", fontSize=13, leading=17,
        textColor=DARK_BLUE, spaceBefore=12, spaceAfter=5)
    add("SubTitle",
        fontName="Helvetica-Bold", fontSize=10, leading=13,
        textColor=MID_BLUE, spaceBefore=8, spaceAfter=3)
    add("Body",
        fontName="Helvetica", fontSize=9, leading=13,
        textColor=BLACK, spaceAfter=4)
    add("Caption",
        fontName="Helvetica-Oblique", fontSize=8, leading=10,
        textColor=colors.HexColor("#555555"), alignment=TA_CENTER, spaceAfter=4)
    add("Bullet",
        fontName="Helvetica", fontSize=9, leading=13,
        textColor=BLACK, leftIndent=14, bulletIndent=4, spaceAfter=3)
    add("Disclaimer",
        fontName="Helvetica-Oblique", fontSize=7.5, leading=10,
        textColor=colors.HexColor("#888888"), alignment=TA_CENTER)
    return base


# ══════════════════════════════════════════════════════════════════════════
#  Page templates
# ══════════════════════════════════════════════════════════════════════════

def _on_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(DARK_BLUE)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PAGE_H * 0.38, PAGE_W, 3, fill=1, stroke=0)
    canvas.restoreState()


def _on_normal(canvas, doc):
    canvas.saveState()
    # Header bar
    canvas.setFillColor(DARK_BLUE)
    canvas.rect(0, PAGE_H - 22, PAGE_W, 22, fill=1, stroke=0)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(WHITE)
    canvas.drawString(MARGIN, PAGE_H - 14,
                      "Buy Call Momentum — Variant C (4× Leverage + Money Market)")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 14,
                           f"Generated {date.today():%d %b %Y}")
    # Footer
    canvas.setFillColor(LIGHT_GRAY)
    canvas.rect(0, 0, PAGE_W, 18, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawCentredString(PAGE_W / 2, 5,
                             f"Page {doc.page}  |  Confidential — For Internal Use Only")
    canvas.restoreState()


def _build_doc():
    doc = BaseDocTemplate(
        OUT, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN + 22, bottomMargin=MARGIN + 18,
    )
    cover_frame  = Frame(0, 0, PAGE_W, PAGE_H, id="cover")
    normal_frame = Frame(
        MARGIN, MARGIN + 18,
        CONTENT_W, PAGE_H - MARGIN - 18 - MARGIN - 22,
        id="normal",
    )
    doc.addPageTemplates([
        PageTemplate(id="Cover",  frames=[cover_frame],  onPage=_on_cover),
        PageTemplate(id="Normal", frames=[normal_frame], onPage=_on_normal),
    ])
    return doc


# ══════════════════════════════════════════════════════════════════════════
#  Reusable helpers
# ══════════════════════════════════════════════════════════════════════════

def _hr():
    return HRFlowable(width="100%", thickness=1, color=MED_GRAY, spaceAfter=5)


def _section(title, st):
    return [_hr(), Paragraph(title, st["SectionTitle"])]


def _embed_image(path, width, caption, st):
    elems = []
    if not os.path.exists(path):
        return elems
    elems.append(Image(path, width=width, height=width * 0.76, kind="proportional"))
    elems.append(Paragraph(caption, st["Caption"]))
    return elems


# ══════════════════════════════════════════════════════════════════════════
#  Load & compute stats from CSV
# ══════════════════════════════════════════════════════════════════════════

def _compute_stats():
    df = pd.read_csv(os.path.join(RES, "levered_call_periods_C.csv"),
                     parse_dates=["period_start"])
    df = df.sort_values("period_start").reset_index(drop=True)
    pnl = df["total_pnl"].values
    n   = len(pnl)
    PPY = 13.0

    total   = float(np.sum(pnl))
    years   = n / PPY
    cagr    = ((1 + total/100) ** (1/years) - 1) * 100
    mean_r  = float(np.mean(pnl))
    std_r   = float(np.std(pnl))
    sharpe  = mean_r / std_r * math.sqrt(PPY) if std_r else 0
    down_sq = pnl[pnl < 0] ** 2
    sortino = (mean_r / math.sqrt(np.mean(down_sq)) * math.sqrt(PPY)
               if len(down_sq) else 0)
    equity  = np.cumsum(pnl)
    running_max = np.maximum.accumulate(equity)
    max_dd  = float(np.min(equity - running_max))
    wins    = int(np.sum(pnl > 0))
    losses  = int(np.sum(pnl < 0))
    best    = df.loc[df["total_pnl"].idxmax()]
    worst   = df.loc[df["total_pnl"].idxmin()]

    avg_prem = float(df["avg_premium_pct"].mean())

    return dict(
        n=n, total=total, cagr=cagr, mean_r=mean_r, std_r=std_r,
        sharpe=sharpe, sortino=sortino, max_dd=max_dd,
        wins=wins, losses=losses,
        best_date=str(best["period_start"])[:10], best_pnl=float(best["total_pnl"]),
        worst_date=str(worst["period_start"])[:10], worst_pnl=float(worst["total_pnl"]),
        start=str(df["period_start"].iloc[0])[:10],
        end=str(df["period_start"].iloc[-1])[:10],
        avg_prem=avg_prem,
    )


# ══════════════════════════════════════════════════════════════════════════
#  Section builders
# ══════════════════════════════════════════════════════════════════════════

def _cover(st):
    elems = [Spacer(1, PAGE_H * 0.27)]
    elems.append(Paragraph("Buy Call Momentum", st["CoverTitle"]))
    elems.append(Paragraph("Variant C — 4× Leverage + Money Market", st["CoverSub"]))
    elems.append(Spacer(1, 16))
    elems.append(Paragraph(f"Generated {date.today():%B %d, %Y}", st["CoverMeta"]))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph("Backtest Period: Sep 2020 – Feb 2026  (~5.5 years, 72 periods)",
                            st["CoverMeta"]))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph("Dynamic Top-30 High-IV Stocks  |  SPX 50 Universe",
                            st["CoverMeta"]))
    return elems


def _overview(st):
    elems = _section("1. Strategy Overview", st)
    elems.append(Paragraph(
        "This report covers a <b>Buy Call momentum strategy</b> applied to a "
        "dynamically selected universe of the top-30 highest implied-volatility "
        "stocks within the S&amp;P 500 (SPX 50 mega-cap subset). "
        "Capital efficiency is maximised through Variant C: 4× notional leverage on "
        "call options, with all idle capital (≈94% of portfolio) deployed into a "
        "Money Market account earning a time-varying short-term rate.", st["Body"]))

    elems.append(Paragraph("<b>How the trade works each period:</b>", st["SubTitle"]))
    for b in [
        "If a stock's previous monthly candle closed <b>UP</b> (open → close), "
        "it receives a buy-call signal for the current period.",
        "An at-the-money call option is purchased at the previous month's closing price "
        "(strike = last close). The option premium averages <b>~1.52% of stock notional</b>.",
        "At 4× leverage, the strategy controls 4× the underlying notional. "
        "Capital allocated to options ≈ <b>6.1% of portfolio notional</b>. "
        "Remaining <b>~93.9%</b> is placed in Money Market.",
        "Position is held to expiry (~3–4 weeks). P&L = 4 × option gain/loss + MMkt credit.",
        "Portfolio is equally weighted across all active signals (typically 15–25 stocks per period).",
    ]:
        elems.append(Paragraph(f"• {b}", st["Bullet"]))

    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "<b>Why Variant C outperforms:</b> Because only ~6% of capital is at risk "
        "in options at any time, the remaining 94% earns 4–5% p.a. in money market "
        "(post-2022 rate environment). The 4× leverage amplifies option P&L while "
        "the MMkt floor keeps drawdowns contained.", st["Body"]))
    return elems


def _universe_rules(st):
    elems = _section("2. Universe &amp; Signal Rules", st)
    rows = [
        ["Parameter", "Value"],
        ["Base universe",      "S&P 500 — 50 mega-cap tickers (10 per sector)"],
        ["Dynamic filter",     "Top 30 by IV rank each period (~bi-monthly rebalance)"],
        ["Signal condition",   "Previous monthly candle UP (close > open) → Buy Call"],
        ["Option type",        "ATM call, strike = last month close"],
        ["Expiry",             "End of current month (~3–4 weeks hold)"],
        ["Avg call premium",   "~1.52% of stock notional (sourced from Barchart IV data)"],
        ["Leverage (Var. C)",  "4× notional  →  ~6.1% capital in options"],
        ["Idle capital",       "~93.9% in Money Market (time-varying short rate)"],
        ["MMkt rate (2020–21)","~0.08% p.a. (COVID ZIRP)"],
        ["MMkt rate (2022)",   "~2.00% p.a. (hiking cycle average)"],
        ["MMkt rate (2023–26)","~4.30–5.10% p.a."],
        ["Portfolio weight",   "Equal weight across all active signals"],
        ["Rebalance",          "Every period (bi-monthly option expiry cycle)"],
        ["Backtest horizon",   "Sep 2020 – Feb 2026  (72 periods ≈ 5.5 years)"],
    ]
    col_w = [CONTENT_W * 0.36, CONTENT_W * 0.64]
    ts = TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  HEADER_BG),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  8.5),
        ("ALIGN",          (0, 0), (-1, 0),  "CENTER"),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 1), (-1, -1), 8.5),
        ("ALIGN",          (0, 1), (0, -1),  "LEFT"),
        ("ALIGN",          (1, 1), (1, -1),  "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, ALT_ROW]),
        ("GRID",           (0, 0), (-1, -1), 0.4, MED_GRAY),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
    ])
    elems.append(Table(rows, colWidths=col_w, style=ts))
    return elems


def _risk_stats_section(stats, st):
    elems = [PageBreak()]
    elems += _section("3. Risk Statistics", st)

    # Summary sentence
    elems.append(Paragraph(
        f"Over <b>{stats['n']} periods</b> ({stats['start']} → {stats['end']}), "
        f"Variant C delivered a <b>total return of {stats['total']:+.1f}%</b> "
        f"(<b>CAGR {stats['cagr']:+.1f}% p.a.</b>) with a maximum drawdown of only "
        f"<b>{stats['max_dd']:.1f}%</b> — a strong risk-adjusted outcome driven by "
        f"the combined effect of 4× option leverage and persistent Money Market income.",
        st["Body"]))
    elems.append(Spacer(1, 6))

    # Stats table
    rows = [
        ["Metric", "Value", "Metric", "Value"],
        ["Total P&L",       f"{stats['total']:+.2f}%",
         "Sharpe (ann.)",   f"{stats['sharpe']:.2f}"],
        ["CAGR",            f"{stats['cagr']:+.2f}% / yr",
         "Sortino (ann.)",  f"{stats['sortino']:.2f}"],
        ["Max Drawdown",    f"{stats['max_dd']:.2f}%",
         "Win Rate",        f"{stats['wins']}/{stats['n']}  ({stats['wins']/stats['n']*100:.0f}%)"],
        ["Avg / period",    f"{stats['mean_r']:+.3f}%",
         "Wins / Losses",   f"{stats['wins']} / {stats['losses']}"],
        ["Std Dev / period",f"{stats['std_r']:.3f}%",
         "Avg call premium",f"{stats['avg_prem']:.2f}% of notional"],
        ["Best period",     f"{stats['best_date']}  {stats['best_pnl']:+.2f}%",
         "Worst period",    f"{stats['worst_date']}  {stats['worst_pnl']:+.2f}%"],
    ]
    half = CONTENT_W / 2
    col_w = [half * 0.42, half * 0.58, half * 0.42, half * 0.58]
    ts = TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  HEADER_BG),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  8.5),
        ("ALIGN",          (0, 0), (-1, 0),  "CENTER"),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 1), (-1, -1), 8.5),
        ("ALIGN",          (0, 1), (0, -1),  "LEFT"),
        ("ALIGN",          (1, 1), (1, -1),  "RIGHT"),
        ("ALIGN",          (2, 1), (2, -1),  "LEFT"),
        ("ALIGN",          (3, 1), (3, -1),  "RIGHT"),
        ("FONTNAME",       (1, 1), (1, -1),  "Helvetica-Bold"),
        ("FONTNAME",       (3, 1), (3, -1),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, ALT_ROW]),
        ("GRID",           (0, 0), (-1, -1), 0.4, MED_GRAY),
        ("LINEAFTER",      (1, 1), (1, -1),  1.0, MED_GRAY),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 7),
    ])
    # Highlight best/worst rows
    ts.add("TEXTCOLOR", (1, 6), (1, 6), colors.HexColor("#276221"))
    ts.add("TEXTCOLOR", (3, 6), (3, 6), colors.HexColor("#9C0006"))
    elems.append(Table(rows, colWidths=col_w, style=ts))
    return elems


def _equity_section(st):
    elems = _section("4. Equity Curve &amp; Drawdown", st)
    elems.append(Paragraph(
        "The equity curve shows cumulative P&amp;L as a percentage of initial notional. "
        "The drawdown panel tracks the peak-to-trough decline at each point in time. "
        "The strategy's largest drawdown of <b>−8.8%</b> occurred Mar–Aug 2022 "
        "(rising-rate shock), and recovered within a few periods.", st["Body"]))
    elems += _embed_image(
        os.path.join(RES, "variantC_equity.png"),
        width=CONTENT_W,
        caption="Fig. 1 — Cumulative P&L (%) and drawdown from peak (Variant C: 4× Leverage + Money Market)",
        st=st,
    )
    return elems


def _heatmap_section(st):
    elems = [PageBreak()]
    elems += _section("5. Monthly Returns Heatmap", st)
    elems.append(Paragraph(
        "Each cell shows the strategy return for that calendar month. "
        "Colour scale: <font color='#276221'><b>green = positive</b></font>, "
        "<font color='#9C0006'><b>red = negative</b></font>, "
        "yellow = near-zero. The Annual column aggregates all periods within the year.",
        st["Body"]))
    elems += _embed_image(
        os.path.join(RES, "variantC_heatmap.png"),
        width=CONTENT_W,
        caption="Fig. 2 — Monthly returns heatmap (Variant C: 4× Leverage + Money Market, % return per period)",
        st=st,
    )
    return elems


def _monthly_table_section(st):
    elems = [PageBreak()]
    elems += _section("6. Monthly Returns — Numeric Table", st)
    elems.append(Paragraph(
        "Same data as the heatmap, presented as a precise numeric table. "
        "Green = positive, Red = negative, Yellow ≈ zero (±0.1%).", st["Body"]))
    elems.append(Spacer(1, 6))

    df = pd.read_csv(os.path.join(RES, "levered_call_monthly_C.csv"))
    months   = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec","Annual"]
    for m in months:
        if m not in df.columns:
            df[m] = None

    header   = ["Year"] + months
    col_year = CONTENT_W * 0.065
    col_mon  = (CONTENT_W - col_year) / len(months)
    col_widths = [col_year] + [col_mon] * len(months)

    data         = [header]
    cell_styles  = []

    for ri, row in df.iterrows():
        r_data = [str(int(row["Year"]))]
        for ci, m in enumerate(months):
            val = row.get(m)
            try:
                fval = float(val)
                r_data.append(f"{fval:+.1f}")
                tr, tc = ri + 1, ci + 1
                if m == "Annual":
                    bg = (colors.HexColor("#A9D18E") if fval > 0.1
                          else colors.HexColor("#FF9999") if fval < -0.1
                          else ZERO_CELL)
                else:
                    bg = (GREEN_CELL if fval > 0.1
                          else RED_CELL if fval < -0.1
                          else ZERO_CELL)
                cell_styles.append(("BACKGROUND", (tc, tr), (tc, tr), bg))
            except (TypeError, ValueError):
                r_data.append("")
        data.append(r_data)

    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  7),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 7),
        ("FONTNAME",      (-1, 0), (-1, -1), "Helvetica-Bold"),
        ("ALIGN",         (0, 1), (0, -1),  "LEFT"),
        ("GRID",          (0, 0), (-1, -1), 0.3, MED_GRAY),
        ("LINEAFTER",     (-2, 0), (-2, -1), 1.0, MID_BLUE),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 2),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 2),
    ] + cell_styles)

    elems.append(Table(data, colWidths=col_widths, style=ts, repeatRows=1))
    return elems


def _takeaways(st):
    elems = [PageBreak()]
    elems += _section("7. Key Take-Aways", st)

    bullets = [
        "<b>Total return +140% over 5.5 years (CAGR +17.1%)</b> with a "
        "maximum drawdown of only −8.8%. This ratio (return / drawdown ≈ 16×) "
        "reflects the structural advantage of the strategy.",

        "<b>4× leverage on call options amplifies momentum gains</b> while the "
        "limited-loss nature of long options caps the downside to the premium paid "
        "(≈ 6.1% of portfolio per period at maximum).",

        "<b>Money Market allocation (~94% of portfolio)</b> acts as a performance floor. "
        "Post-2022, the 4–5% annual MMkt yield contributes ≈ 0.33% per period "
        "before any option trade, making the strategy profitable even in many "
        "sideways markets.",

        "<b>Sharpe 1.14 / Sortino 3.30</b> — high Sortino relative to Sharpe indicates "
        "that volatility is predominantly upward; downside episodes are shallow and short.",

        "<b>Win rate 60% (43/72 periods)</b>. Positive skew: average win +4.4%, "
        "average loss −2.7%. The asymmetry arises from buying options "
        "(limited loss, unlimited upside).",

        "<b>2023 was the standout year (+47.3%)</b>: broad tech/AI-driven rally combined "
        "with elevated IV premiums created ideal conditions for ATM call buying.",

        "<b>Operational implementation:</b> signals are generated once per period by the "
        "existing Lambda function (lambda_function.py). Option data is sourced from "
        "Barchart via S3. No intraday monitoring required.",
    ]
    for b in bullets:
        elems.append(Paragraph(f"• {b}", st["Bullet"]))
        elems.append(Spacer(1, 3))

    elems.append(Spacer(1, 14))
    elems.append(_hr())
    elems.append(Paragraph(
        "Disclaimer: This document is for internal research purposes only. "
        "Past backtest performance is not indicative of future results. "
        "Options trading involves significant risk of loss.",
        st["Disclaimer"]))
    return elems


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def build():
    st    = _styles()
    doc   = _build_doc()
    stats = _compute_stats()

    story = []
    story += _cover(st)
    story.append(NextPageTemplate("Normal"))
    story.append(PageBreak())

    story += _overview(st)
    story += _universe_rules(st)
    story += _risk_stats_section(stats, st)
    story += _equity_section(st)
    story += _heatmap_section(st)
    story += _monthly_table_section(st)
    story += _takeaways(st)

    doc.build(story)
    print(f"✓ Report written → {OUT}")


if __name__ == "__main__":
    build()
