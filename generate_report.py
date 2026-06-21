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

def _compute_stats(variant="C"):
    df = pd.read_csv(os.path.join(RES, f"signal_periods_{variant}.csv"),
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
    elems.append(Paragraph("Backtest Period: September 2020 – June 2026  (~5.7 years)",
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


def _risk_stats_section(stats_b, stats_c, st):
    elems = [PageBreak()]
    elems += _section("3. Performance &amp; Risk Statistics", st)

    elems.append(Paragraph(
        f"Two capital-deployment variants are shown side by side. "
        f"<b>Unlevered</b> deploys 1x notional in the options book; idle, "
        f"undeployed capital earns a plain money-market rate (no leverage on the "
        f"cash leg). <b>4x Leveraged</b> applies 4x notional to the same option "
        f"signals, scaling both the option P&amp;L and the option cost by 4x, with "
        f"any capital still idle after the 4x option allocation earning money-market. "
        f"Over <b>{stats_c['n']} trading periods</b> ({stats_c['start']} to {stats_c['end']}), "
        f"Unlevered returned <b>{stats_b['total']:+.1f}%</b> total "
        f"(CAGR {stats_b['cagr']:+.1f}%, max drawdown {stats_b['max_dd']:.1f}%), while "
        f"4x Leveraged returned <b>{stats_c['total']:+.1f}%</b> total "
        f"(CAGR {stats_c['cagr']:+.1f}%, max drawdown {stats_c['max_dd']:.1f}%). "
        f"Leverage amplifies both the gains and the drawdowns roughly proportionally — "
        f"it is not a free source of additional return.",
        st["Body"]))
    elems.append(Spacer(1, 6))

    def fmt_row(label, key, fmt, b, c):
        return [label, fmt.format(b[key]), fmt.format(c[key])]

    rows = [
        ["Metric", "Unlevered (1x)", "4x Leveraged"],
        ["Total Return",        f"{stats_b['total']:+.2f}%",  f"{stats_c['total']:+.2f}%"],
        ["CAGR",                f"{stats_b['cagr']:+.2f}% p.a.", f"{stats_c['cagr']:+.2f}% p.a."],
        ["Max Drawdown",        f"{stats_b['max_dd']:.2f}%",  f"{stats_c['max_dd']:.2f}%"],
        ["Sharpe Ratio",        f"{stats_b['sharpe']:.2f}",   f"{stats_c['sharpe']:.2f}"],
        ["Sortino Ratio",       f"{stats_b['sortino']:.2f}",  f"{stats_c['sortino']:.2f}"],
        ["Win Rate",            f"{stats_b['wins']}/{stats_b['n']} ({stats_b['wins']/stats_b['n']*100:.0f}%)",
                                 f"{stats_c['wins']}/{stats_c['n']} ({stats_c['wins']/stats_c['n']*100:.0f}%)"],
        ["Avg return / period", f"{stats_b['mean_r']:+.3f}%", f"{stats_c['mean_r']:+.3f}%"],
        ["Volatility / period", f"{stats_b['std_r']:.3f}%",   f"{stats_c['std_r']:.3f}%"],
        ["Best period",         f"{stats_b['best_date']}  {stats_b['best_pnl']:+.2f}%",
                                 f"{stats_c['best_date']}  {stats_c['best_pnl']:+.2f}%"],
        ["Worst period",        f"{stats_b['worst_date']}  {stats_b['worst_pnl']:+.2f}%",
                                 f"{stats_c['worst_date']}  {stats_c['worst_pnl']:+.2f}%"],
        ["Avg option cost",     f"{stats_b['avg_prem']:.2f}% of notional",
                                 f"{stats_c['avg_prem']:.2f}% of notional"],
    ]
    col_w = [CONTENT_W * 0.34, CONTENT_W * 0.33, CONTENT_W * 0.33]
    ts = TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  HEADER_BG),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  8.5),
        ("ALIGN",          (0, 0), (-1, 0),  "CENTER"),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 1), (-1, -1), 8.5),
        ("ALIGN",          (0, 1), (0, -1),  "LEFT"),
        ("ALIGN",          (1, 1), (-1, -1), "RIGHT"),
        ("FONTNAME",       (1, 1), (-1, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, ALT_ROW]),
        ("GRID",           (0, 0), (-1, -1), 0.4, MED_GRAY),
        ("LINEAFTER",      (0, 1), (0, -1),  1.0, MED_GRAY),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 7),
    ])
    elems.append(Table(rows, colWidths=col_w, style=ts))
    return elems


def _equity_section(st):
    elems = _section("4. Growth of Capital", st)
    elems.append(Paragraph(
        "The charts below show how $100 of initial capital would have grown over the "
        "backtest period under each variant, expressed as cumulative percentage return. "
        "The lower panel of each tracks the maximum decline from any prior peak. "
        "Leverage scales both the upside and the drawdowns — it does not change the "
        "underlying win rate or signal quality, only the size of each swing.", st["Body"]))
    elems += _embed_image(
        os.path.join(RES, "variantB_equity.png"),
        width=CONTENT_W,
        caption="Fig. 1a — Unlevered: cumulative return (%) and drawdown, Sep 2020 – Feb 2026",
        st=st,
    )
    elems.append(Spacer(1, 8))
    elems += _embed_image(
        os.path.join(RES, "variantC_equity.png"),
        width=CONTENT_W,
        caption="Fig. 1b — 4x Leveraged: cumulative return (%) and drawdown, Sep 2020 – Feb 2026",
        st=st,
    )
    return elems


def _heatmap_section(st):
    elems = [PageBreak()]
    elems += _section("5. Monthly Returns at a Glance", st)
    elems.append(Paragraph(
        "The heatmaps below present strategy returns for each calendar month across "
        "the full backtest, for both variants. Each cell shows the return for that "
        "period. <font color='#276221'><b>Green</b></font> cells represent positive "
        "months, <font color='#9C0006'><b>red</b></font> cells represent negative "
        "months, and yellow indicates near-zero. The Annual column shows the "
        "full-year total.",
        st["Body"]))
    elems += _embed_image(
        os.path.join(RES, "variantB_heatmap.png"),
        width=CONTENT_W,
        caption="Fig. 2a — Unlevered monthly return heatmap (%), Sep 2020 – Feb 2026",
        st=st,
    )
    elems.append(Spacer(1, 8))
    elems += _embed_image(
        os.path.join(RES, "variantC_heatmap.png"),
        width=CONTENT_W,
        caption="Fig. 2b — 4x Leveraged monthly return heatmap (%), Sep 2020 – Feb 2026",
        st=st,
    )
    return elems


def _recent_history_section(st):
    elems = [PageBreak()]
    elems += _section("6. Recent History — Last 6 Periods (Unlevered)", st)
    elems.append(Paragraph(
        "Detail view of the most recent monthly cycles under the unlevered "
        "(1x) variant, showing how many tickers actually signaled BUY CALL "
        "that period, how many were active (capped at 30, prioritized by "
        "premium %), the average option premium paid, the option P&amp;L, "
        "the money-market contribution from idle cash, and the resulting "
        "period and cumulative returns. This is the same mechanic that "
        "drives every period in the backtest — shown here at full "
        "granularity as a worked example. (A 4x leveraged variant exists "
        "— see Section 3 — and scales the option P&amp;L and option cost "
        "columns by 4x; idle-cash credit shrinks accordingly.)",
        st["Body"]))
    elems.append(Spacer(1, 6))

    df = pd.read_csv(os.path.join(RES, "signal_periods_B.csv"),
                     parse_dates=["period_start"])
    df = df.sort_values("period_start").tail(6)

    header = ["Period", "Signals", "Active", "Avg Premium", "Option P&L", "MM Contrib",
              "Total P&L", "Cumulative"]
    rows = [header]
    for _, r in df.iterrows():
        rows.append([
            str(r["period_start"])[:10],
            f"{int(r['n_signals'])}",
            f"{int(r['n_active'])}/30",
            f"{r['avg_premium_pct']:.2f}%",
            f"{r['option_pnl']:+.2f}%",
            f"{r['mm_contrib']:+.2f}%",
            f"{r['total_pnl']:+.2f}%",
            f"{r['cumulative_pct']:+.1f}%",
        ])

    col_w = [CONTENT_W * 0.13, CONTENT_W * 0.10, CONTENT_W * 0.11, CONTENT_W * 0.13,
              CONTENT_W * 0.14, CONTENT_W * 0.13, CONTENT_W * 0.13, CONTENT_W * 0.13]
    ts = TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  HEADER_BG),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  8.5),
        ("ALIGN",          (0, 0), (-1, 0),  "CENTER"),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 1), (-1, -1), 8.5),
        ("ALIGN",          (0, 1), (0, -1),  "LEFT"),
        ("ALIGN",          (1, 1), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, ALT_ROW]),
        ("GRID",           (0, 0), (-1, -1), 0.4, MED_GRAY),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
    ])
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        color = colors.HexColor("#276221") if r["total_pnl"] >= 0 else colors.HexColor("#9C0006")
        ts.add("TEXTCOLOR", (6, i), (6, i), color)
        ts.add("FONTNAME", (6, i), (6, i), "Helvetica-Bold")
    elems.append(Table(rows, colWidths=col_w, style=ts))

    elems.append(Spacer(1, 10))
    elems.append(Paragraph("<b>Most recent period — tickers held:</b>", st["SubTitle"]))
    last = df.iloc[-1]
    tickers_str = ", ".join(last["tickers"].split("|"))
    elems.append(Paragraph(tickers_str, st["Body"]))
    return elems


def _takeaways(st):
    elems = [PageBreak()]
    elems += _section("7. Investment Case", st)

    bullets = [
        "<b>Unlevered: +52.7% total return over ~5.7 years — CAGR +7.4% p.a.</b>, "
        "with a peak drawdown of 3.3%. This is the realistic base case: only "
        "tickers that actually signal BUY CALL are traded (capped at 30 active "
        "slots, prioritized by option premium %), idle capital earns plain "
        "money-market, no leverage on either leg.",

        "<b>4x Leveraged: +150.1% total return — CAGR +16.7% p.a.</b>, with a peak "
        "drawdown of 14.0%. Leverage scales both the return and the drawdown "
        "versus the unlevered case — it amplifies the existing edge, it does "
        "not create a new one.",

        "<b>Call options provide built-in downside protection.</b> Unlike direct equity "
        "exposure, the maximum loss on any single position is limited to the option cost "
        "(scaled by leverage, if used). This structural feature caps the damage in "
        "adverse periods while preserving full participation in strong moves.",

        "<b>The cash reserve is a real contributor to returns, not leveraged.</b> "
        "Money-market income only accrues on capital that is genuinely idle after "
        "the options allocation — never on borrowed or notional exposure — and "
        "uses period-appropriate short-term rates (0.08% in 2020–21 ZIRP, rising to "
        "~5% in 2023–24).",

        "<b>Sharpe / Sortino — Unlevered 1.76 / 2.27, 4x Leveraged 1.27 / 1.57.</b> "
        "The unlevered variant has the better risk-adjusted profile on both measures; "
        "leverage increases the absolute return but at a worse Sharpe/Sortino ratio, "
        "as expected when scaling a fixed signal with a fixed-cost cash leg.",

        "<b>Win rate: 71% unlevered (55/77 periods), 66% with 4x leverage (51/77)</b> "
        "— the same underlying trade decisions; leverage changes the magnitude of "
        "wins and losses, not which periods are profitable.",

        "<b>The three most recent periods (Mar–Jun 2026) use simulated option "
        "premiums.</b> Live option-chain data for this window was not available, "
        "so premiums were approximated from the VIX level at each period's start "
        "(ATM call ≈ 0.4 × stock price × VIX/100 × √(28/365), the standard "
        "at-the-money Black-Scholes approximation), applied uniformly across all "
        "names that period. This is a market-wide volatility proxy, not a "
        "per-ticker priced option, and should be treated as indicative rather "
        "than a precise historical fill.",

        "<b>By calendar year (unlevered / 4x leveraged): 2022 −0.7% / −7.9%, "
        "2023 +16.0% / +46.4%, 2024 +17.4% / +53.1%, 2026 YTD +6.3% / +17.4% "
        "(through early June).</b> 2023–24 broad market momentum drove the "
        "bulk of cumulative return; 2022's rate-shock year was the one "
        "calendar-year loss in this backtest, amplified under 4x leverage. "
        "(Each ~4-week period is attributed to the calendar month containing "
        "its midpoint — see Section 5 for the full monthly breakdown, which "
        "sums exactly to these yearly figures. The three most recent periods, "
        "March–June 2026, use simulated option premiums — see note below.)",

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
    st      = _styles()
    doc     = _build_doc()
    stats_b = _compute_stats("B")
    stats_c = _compute_stats("C")

    story = []
    story += _cover(st)
    story.append(NextPageTemplate("Normal"))
    story.append(PageBreak())

    story += _overview(st)
    story += _universe_rules(st)
    story += _risk_stats_section(stats_b, stats_c, st)
    story += _equity_section(st)
    story += _heatmap_section(st)
    story += _recent_history_section(st)
    story += _takeaways(st)

    doc.build(story)
    print(f"✓ Report written → {OUT}")


if __name__ == "__main__":
    build()
