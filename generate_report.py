"""
generate_report.py
------------------
Client-facing pitch report: Momentum Call Strategy

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
                      "Momentum Call Strategy — Equity Options")
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
    elems.append(Paragraph("Momentum Call Strategy", st["CoverTitle"]))
    elems.append(Paragraph("Systematic Equity Options — Buy Call Approach", st["CoverSub"]))
    elems.append(Spacer(1, 16))
    elems.append(Paragraph(f"Prepared {date.today():%B %d, %Y}", st["CoverMeta"]))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph("Backtest Period: September 2020 – February 2026  (~5.5 years)",
                            st["CoverMeta"]))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph("Dynamic Universe — Top 30 Stocks  |  S&amp;P 500",
                            st["CoverMeta"]))
    return elems


def _overview(st):
    elems = _section("1. Strategy Overview", st)
    elems.append(Paragraph(
        "The Momentum Call Strategy is a systematic, rules-based approach that "
        "uses call options to capture upward price momentum in large-cap U.S. equities. "
        "The strategy monitors a dynamic universe of 30 stocks — selected each period "
        "from the S&amp;P 500 for their strong options activity and liquidity — and "
        "enters call positions only on names showing a confirmed positive price trend. "
        "Capital not deployed in options is kept in short-term instruments, generating "
        "a steady income base independent of market direction.", st["Body"]))

    elems.append(Paragraph("<b>How it works:</b>", st["SubTitle"]))
    for b in [
        "<b>Momentum filter:</b> each period, a stock qualifies for a call position only "
        "if its previous monthly candle closed higher than it opened — a simple, "
        "objective confirmation of upward trend.",
        "<b>Call option entry:</b> an at-the-money call is purchased at the prior "
        "month's closing price as the strike. The option captures the next move up "
        "with a defined, limited cost.",
        "<b>Average option spend:</b> ~1.52% of notional per period across the portfolio. "
        "This figure is naturally low because not all 30 stocks pass the momentum filter "
        "simultaneously — on average 15–20 names are active each period, so only a "
        "fraction of capital is committed to options at any one time.",
        "<b>Cash reserve:</b> the substantial portion of capital not deployed in options "
        "is held in short-term instruments, earning prevailing market interest rates "
        "(0.08% p.a. during 2020–21 near-zero rates; 4.3–5.1% p.a. post-2022).",
        "<b>Equal weighting:</b> all active positions carry the same weight, "
        "avoiding concentration risk in any single name.",
        "<b>Rebalance cadence:</b> the entire portfolio is reviewed and reset every "
        "~3–4 weeks, aligned with monthly option expiry cycles.",
    ]:
        elems.append(Paragraph(f"• {b}", st["Bullet"]))

    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "<b>Structural edge:</b> because call options offer unlimited upside with "
        "strictly capped downside, the strategy benefits from positive return "
        "asymmetry — winning periods tend to be significantly larger than losing ones. "
        "The income from the cash reserve provides a meaningful buffer, "
        "reducing the net cost of holding options and dampening portfolio drawdowns.",
        st["Body"]))
    return elems


def _universe_rules(st):
    elems = _section("2. Strategy Parameters", st)
    rows = [
        ["Parameter", "Value"],
        ["Investment universe",   "S&P 500 — 50 large-cap stocks across 10 sectors"],
        ["Active portfolio size", "Top 30 most liquid, high-activity names per period"],
        ["Entry condition",       "Stock's prior monthly candle closes UP (close > open)"],
        ["Instrument",           "At-the-money call option, strike = prior month close"],
        ["Holding period",        "~3–4 weeks (one monthly expiry cycle)"],
        ["Avg option cost",       "~1.52% of notional (reflects partial signal activation —\n"
                                  "typically 15–20 of 30 stocks qualify each period)"],
        ["Cash reserve",          "Balance of capital held in short-term instruments"],
        ["Cash yield (2020–21)",  "~0.08% p.a.  (low-rate environment)"],
        ["Cash yield (2022)",     "~2.00% p.a.  (rate normalisation)"],
        ["Cash yield (2023–26)",  "~4.30–5.10% p.a.  (current rate environment)"],
        ["Position sizing",       "Equal weight across all qualifying positions"],
        ["Rebalance frequency",   "Monthly — full portfolio reset each expiry"],
        ["Backtest period",       "September 2020 – February 2026  (72 periods, ~5.5 years)"],
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
    elems += _section("3. Performance &amp; Risk Statistics", st)

    elems.append(Paragraph(
        f"Over <b>{stats['n']} trading periods</b> ({stats['start']} to {stats['end']}), "
        f"the strategy delivered a <b>total return of {stats['total']:+.1f}%</b> "
        f"— equivalent to a <b>compound annual growth rate of {stats['cagr']:+.1f}%</b> — "
        f"while the largest peak-to-trough decline across the entire period was only "
        f"<b>{stats['max_dd']:.1f}%</b>. This combination of strong returns and "
        f"limited drawdown reflects the asymmetric payoff structure built into the approach.",
        st["Body"]))
    elems.append(Spacer(1, 6))

    # Stats table
    rows = [
        ["Metric", "Value", "Metric", "Value"],
        ["Total Return",    f"{stats['total']:+.2f}%",
         "Sharpe Ratio",    f"{stats['sharpe']:.2f}"],
        ["CAGR",            f"{stats['cagr']:+.2f}% p.a.",
         "Sortino Ratio",   f"{stats['sortino']:.2f}"],
        ["Max Drawdown",    f"{stats['max_dd']:.2f}%",
         "Win Rate",        f"{stats['wins']}/{stats['n']}  ({stats['wins']/stats['n']*100:.0f}%)"],
        ["Avg return / period", f"{stats['mean_r']:+.3f}%",
         "Periods win / loss",  f"{stats['wins']} / {stats['losses']}"],
        ["Volatility / period", f"{stats['std_r']:.3f}%",
         "Avg option cost",     f"{stats['avg_prem']:.2f}% of notional"],
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
    elems = _section("4. Growth of Capital", st)
    elems.append(Paragraph(
        "The chart below shows how $100 of initial capital would have grown over the "
        "backtest period, expressed as cumulative percentage return. "
        "The lower panel tracks the maximum decline from any prior peak — "
        "the strategy's deepest drawdown of <b>8.8%</b> occurred during the "
        "2022 rate-shock period and recovered within a small number of periods, "
        "illustrating the resilience of the approach even in adverse conditions.", st["Body"]))
    elems += _embed_image(
        os.path.join(RES, "variantC_equity.png"),
        width=CONTENT_W,
        caption="Fig. 1 — Cumulative return (%) and peak-to-trough drawdown, Sep 2020 – Feb 2026",
        st=st,
    )
    return elems


def _heatmap_section(st):
    elems = [PageBreak()]
    elems += _section("5. Monthly Returns at a Glance", st)
    elems.append(Paragraph(
        "The heatmap below presents strategy returns for each calendar month across "
        "the full backtest. Each cell shows the return for that period. "
        "<font color='#276221'><b>Green</b></font> cells represent positive months, "
        "<font color='#9C0006'><b>red</b></font> cells represent negative months, "
        "and yellow indicates near-zero. The Annual column shows the full-year total. "
        "The pattern highlights the strategy's consistent positive bias and the "
        "concentration of strong gains in momentum-driven market regimes.",
        st["Body"]))
    elems += _embed_image(
        os.path.join(RES, "variantC_heatmap.png"),
        width=CONTENT_W,
        caption="Fig. 2 — Monthly return heatmap (%), Sep 2020 – Feb 2026",
        st=st,
    )
    return elems


def _monthly_table_section(st):
    elems = [PageBreak()]
    elems += _section("6. Monthly Returns — Detail Table", st)
    elems.append(Paragraph(
        "Full numeric breakdown of returns by month and year. "
        "Colour coding matches the heatmap: "
        "green = positive, red = negative, yellow ≈ zero.", st["Body"]))
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
    elems += _section("7. Investment Case", st)

    bullets = [
        "<b>+140% total return over 5.5 years — CAGR +17.1% p.a.</b> with a peak "
        "drawdown of only 8.8%. The ratio of total return to maximum drawdown exceeds "
        "16:1, a compelling risk-adjusted result.",

        "<b>Call options provide built-in downside protection.</b> Unlike direct equity "
        "exposure, the maximum loss on any single position is limited to the option cost. "
        "This structural feature caps the damage in adverse periods while preserving "
        "full participation in strong moves.",

        "<b>The cash reserve is a silent contributor to returns.</b> With the large "
        "majority of capital held in short-term instruments, the portfolio earns "
        "ongoing income regardless of market conditions — particularly meaningful "
        "in the current interest rate environment (4–5% p.a.).",

        "<b>Sharpe Ratio 1.14 — Sortino Ratio 3.30.</b> The large gap between Sortino "
        "and Sharpe confirms that the strategy's volatility is predominantly positive: "
        "upswings are frequent and sizeable, while drawdowns are shallow and brief.",

        "<b>Win rate 60% with positive skew</b> (43 profitable periods out of 72). "
        "The average gain in winning periods substantially exceeds the average loss "
        "in losing periods — the hallmark of an asymmetric return profile.",

        "<b>2023 delivered +47.3%</b>, driven by broad market momentum and "
        "elevated option activity. Conversely, even the challenging 2022 "
        "rate-shock environment produced a positive full-year result (+3.2%).",

        "<b>Fully systematic and low-maintenance.</b> Signals are generated once "
        "per monthly cycle with no intraday monitoring required. The approach is "
        "transparent, rule-based, and straightforward to audit.",
    ]
    for b in bullets:
        elems.append(Paragraph(f"• {b}", st["Bullet"]))
        elems.append(Spacer(1, 3))

    elems.append(Spacer(1, 14))
    elems.append(_hr())
    elems.append(Paragraph(
        "Disclaimer: This document is prepared for informational purposes only and "
        "does not constitute investment advice or an offer to buy or sell securities. "
        "Past backtest performance is hypothetical and not indicative of future results. "
        "Options trading involves significant risk and may not be suitable for all investors.",
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
