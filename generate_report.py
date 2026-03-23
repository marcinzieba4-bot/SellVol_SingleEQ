"""
generate_report.py
------------------
Builds a professional PDF backtest report for the SellVol / BuyCall strategy.

Sections
  1. Cover
  2. Strategy Overview
  3. Universe & signal rules
  4. Dynamic Top-30 — Mode comparison (equity curves + risk table)
  5. Dynamic Top-30 — Monthly return tables (all 4 modes, colour-coded)
  6. Buy-Call with Leverage — Variant comparison (equity curve + risk table)
  7. Buy-Call — Monthly return tables (Variants A / B / C, colour-coded)
  8. Key take-aways

Output: results/SellVol_Report.pdf
"""

import os
import re
from datetime import date

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
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
BASE   = os.path.dirname(os.path.abspath(__file__))
RES    = os.path.join(BASE, "results")
OUT    = os.path.join(RES,  "SellVol_Report.pdf")

# ── colour palette ─────────────────────────────────────────────────────────
DARK_BLUE   = colors.HexColor("#0A2342")
MID_BLUE    = colors.HexColor("#1F4E79")
ACCENT      = colors.HexColor("#2E86AB")
LIGHT_GRAY  = colors.HexColor("#F5F5F5")
MED_GRAY    = colors.HexColor("#CCCCCC")
GREEN_CELL  = colors.HexColor("#C6EFCE")
RED_CELL    = colors.HexColor("#FFC7CE")
ZERO_CELL   = colors.HexColor("#FFEB9C")
HEADER_BG   = colors.HexColor("#1F4E79")
ALT_ROW     = colors.HexColor("#EBF3FB")
WHITE       = colors.white
BLACK       = colors.black

# ── page layout ────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4          # 595 × 842 pt
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

    add("Cover_Title",
        fontName="Helvetica-Bold", fontSize=28, leading=34,
        textColor=WHITE, alignment=TA_CENTER, spaceAfter=10)
    add("Cover_Sub",
        fontName="Helvetica", fontSize=14, leading=18,
        textColor=colors.HexColor("#BDD7EE"), alignment=TA_CENTER, spaceAfter=6)
    add("Cover_Date",
        fontName="Helvetica-Oblique", fontSize=11, leading=14,
        textColor=colors.HexColor("#9DC3E6"), alignment=TA_CENTER)
    add("SectionTitle",
        fontName="Helvetica-Bold", fontSize=14, leading=18,
        textColor=DARK_BLUE, spaceBefore=14, spaceAfter=6,
        borderPad=4)
    add("SubTitle",
        fontName="Helvetica-Bold", fontSize=11, leading=14,
        textColor=MID_BLUE, spaceBefore=10, spaceAfter=4)
    add("Body",
        fontName="Helvetica", fontSize=9, leading=13,
        textColor=BLACK, spaceAfter=4)
    add("Caption",
        fontName="Helvetica-Oblique", fontSize=8, leading=10,
        textColor=colors.HexColor("#555555"), alignment=TA_CENTER, spaceAfter=6)
    add("TableHeader",
        fontName="Helvetica-Bold", fontSize=8, leading=10,
        textColor=WHITE, alignment=TA_CENTER)
    add("TableCell",
        fontName="Helvetica", fontSize=7.5, leading=9,
        textColor=BLACK, alignment=TA_RIGHT)
    add("BulletBody",
        fontName="Helvetica", fontSize=9, leading=13,
        textColor=BLACK, leftIndent=14, spaceAfter=3,
        bulletIndent=4)
    return base


# ══════════════════════════════════════════════════════════════════════════
#  Page templates (cover vs. normal)
# ══════════════════════════════════════════════════════════════════════════

def _on_cover_page(canvas, doc):
    canvas.saveState()
    # Full-page dark background
    canvas.setFillColor(DARK_BLUE)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # Accent stripe
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PAGE_H * 0.38, PAGE_W, 4, fill=1, stroke=0)
    canvas.restoreState()


def _on_normal_page(canvas, doc):
    canvas.saveState()
    # Subtle top bar
    canvas.setFillColor(DARK_BLUE)
    canvas.rect(0, PAGE_H - 22, PAGE_W, 22, fill=1, stroke=0)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(WHITE)
    canvas.drawString(MARGIN, PAGE_H - 14, "SellVol / BuyCall — Backtest Report")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 14, f"Generated {date.today():%B %d, %Y}")
    # Footer
    canvas.setFillColor(LIGHT_GRAY)
    canvas.rect(0, 0, PAGE_W, 18, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawCentredString(PAGE_W / 2, 5, f"Page {doc.page} | Confidential — For Internal Use Only")
    canvas.restoreState()


def _build_doc():
    doc = BaseDocTemplate(
        OUT,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN + 22,
        bottomMargin=MARGIN + 18,
    )
    cover_frame  = Frame(0, 0, PAGE_W, PAGE_H, id="cover")
    normal_frame = Frame(
        MARGIN, MARGIN + 18,
        CONTENT_W, PAGE_H - MARGIN - 18 - MARGIN - 22,
        id="normal"
    )
    doc.addPageTemplates([
        PageTemplate(id="Cover",  frames=[cover_frame],  onPage=_on_cover_page),
        PageTemplate(id="Normal", frames=[normal_frame], onPage=_on_normal_page),
    ])
    return doc


# ══════════════════════════════════════════════════════════════════════════
#  Helper builders
# ══════════════════════════════════════════════════════════════════════════

def _hr():
    return HRFlowable(width="100%", thickness=1, color=MED_GRAY, spaceAfter=6)


def _section(title, st):
    return [
        _hr(),
        Paragraph(title, st["SectionTitle"]),
    ]


def _image(path, width=None, caption=None, st=None):
    elems = []
    if not os.path.exists(path):
        return elems
    w = width or CONTENT_W
    img = Image(path, width=w, height=w * 0.45, kind="proportional")
    elems.append(img)
    if caption and st:
        elems.append(Paragraph(caption, st["Caption"]))
    return elems


def _risk_table(rows, st):
    """
    rows: list of dicts with keys:
      name, total, cagr, maxdd, sharpe, sortino, win
    """
    headers = ["Strategy / Variant", "Total P&L", "CAGR", "Max DD", "Sharpe", "Sortino", "Win %"]
    col_w = [CONTENT_W * f for f in [0.33, 0.11, 0.10, 0.10, 0.09, 0.10, 0.08]]

    data = [headers]
    for r in rows:
        data.append([
            r["name"], r["total"], r["cagr"], r["maxdd"],
            r["sharpe"], r["sortino"], r["win"],
        ])

    ts = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  HEADER_BG),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  8),
        ("ALIGN",        (0, 0), (-1, 0),  "CENTER"),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 8),
        ("ALIGN",        (1, 1), (-1, -1), "CENTER"),
        ("ALIGN",        (0, 1), (0, -1),  "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, ALT_ROW]),
        ("GRID",         (0, 0), (-1, -1), 0.4, MED_GRAY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
    ])

    # Highlight best CAGR row (skip header)
    try:
        cagr_vals = [float(r["cagr"].replace("%", "").replace("+", "")) for r in rows]
        best_idx  = cagr_vals.index(max(cagr_vals)) + 1
        ts.add("BACKGROUND", (0, best_idx), (-1, best_idx), colors.HexColor("#D9F0FF"))
        ts.add("FONTNAME",   (0, best_idx), (-1, best_idx), "Helvetica-Bold")
    except Exception:
        pass

    return [Table(data, colWidths=col_w, style=ts, repeatRows=1)]


def _monthly_table(csv_path, label, st):
    """Colour-coded monthly returns table from CSV."""
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Annual"]
    # ensure all month cols present
    for m in months:
        if m not in df.columns:
            df[m] = None

    header_row = ["Year"] + months
    col_w_year = CONTENT_W * 0.07
    col_w_mon  = (CONTENT_W - col_w_year) / len(months)
    col_widths = [col_w_year] + [col_w_mon] * len(months)

    data = [header_row]
    cell_styles = []

    for ri, row in df.iterrows():
        r_data = [str(int(row["Year"]))]
        for ci, m in enumerate(months):
            val = row.get(m)
            try:
                fval = float(val)
                r_data.append(f"{fval:+.1f}")
                # colour cell
                table_row = ri + 1
                table_col = ci + 1
                if fval > 0.05:
                    cell_styles.append(("BACKGROUND", (table_col, table_row), (table_col, table_row), GREEN_CELL))
                elif fval < -0.05:
                    cell_styles.append(("BACKGROUND", (table_col, table_row), (table_col, table_row), RED_CELL))
                else:
                    cell_styles.append(("BACKGROUND", (table_col, table_row), (table_col, table_row), ZERO_CELL))
            except (TypeError, ValueError):
                r_data.append("")

        # Annual column colour
        try:
            ann = float(row.get("Annual"))
            table_row = ri + 1
            if ann > 0.05:
                cell_styles.append(("BACKGROUND", (len(months), table_row), (len(months), table_row),
                                    colors.HexColor("#A9D18E")))
            elif ann < -0.05:
                cell_styles.append(("BACKGROUND", (len(months), table_row), (len(months), table_row),
                                    colors.HexColor("#FF9999")))
        except (TypeError, ValueError):
            pass

        data.append(r_data)

    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  7),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 7),
        ("GRID",          (0, 0), (-1, -1), 0.3, MED_GRAY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 2),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 2),
        # Bold Annual column header
        ("FONTNAME",      (-1, 0), (-1, -1), "Helvetica-Bold"),
        # Year col left-align
        ("ALIGN",         (0, 1), (0, -1),  "LEFT"),
    ] + cell_styles)

    elems = [Paragraph(label, st["SubTitle"])]
    elems.append(Table(data, colWidths=col_widths, style=ts, repeatRows=1))
    elems.append(Spacer(1, 8))
    return elems


# ══════════════════════════════════════════════════════════════════════════
#  Parse risk-stats text files
# ══════════════════════════════════════════════════════════════════════════

def _parse_stat(text, key):
    m = re.search(rf"{key}\s*[:\|]\s*([^\s|]+)", text)
    return m.group(1) if m else "—"


def _extract_mode_stats(txt_path):
    """Return list of dicts for each Mode block."""
    with open(txt_path) as f:
        content = f.read()

    blocks = re.split(r"Mode \d+", content)
    headers = re.findall(r"Mode \d+ — (.+)", content)
    rows = []
    for i, (hdr, blk) in enumerate(zip(headers, blocks[1:]), 1):
        rows.append({
            "name":    f"Mode {i}: {hdr.strip()}",
            "total":   _parse_stat(blk, "Total P&L"),
            "cagr":    _parse_stat(blk, "CAGR"),
            "maxdd":   _parse_stat(blk, "Max Drawdown"),
            "sharpe":  _parse_stat(blk, "Sharpe"),
            "sortino": _parse_stat(blk, "Sortino"),
            "win":     _parse_stat(blk, "Win / Loss").split("(")[-1].rstrip(")") if "(" in _parse_stat(blk, "Win / Loss") else "—",
        })
    return rows


def _extract_variant_stats(txt_path):
    """Return list of dicts for each Variant block."""
    with open(txt_path) as f:
        content = f.read()

    blocks = re.split(r"Variant [A-C]", content)
    headers = re.findall(r"Variant ([A-C]) — (.+)", content)
    rows = []
    for (letter, hdr), blk in zip(headers, blocks[1:]):
        rows.append({
            "name":    f"Variant {letter}: {hdr.strip()}",
            "total":   _parse_stat(blk, "Total P&L"),
            "cagr":    _parse_stat(blk, "CAGR"),
            "maxdd":   _parse_stat(blk, "Max Drawdown"),
            "sharpe":  _parse_stat(blk, "Sharpe"),
            "sortino": _parse_stat(blk, "Sortino"),
            "win":     _parse_stat(blk, "Win / Loss").split("(")[-1].rstrip(")") if "(" in _parse_stat(blk, "Win / Loss") else "—",
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════
#  Content builders
# ══════════════════════════════════════════════════════════════════════════

def _cover(st):
    elems = [Spacer(1, PAGE_H * 0.28)]
    elems.append(Paragraph("SellVol / BuyCall", st["Cover_Title"]))
    elems.append(Paragraph("Options-Based Equity Strategy — Backtest Report", st["Cover_Sub"]))
    elems.append(Spacer(1, 18))
    elems.append(Paragraph(f"Generated {date.today():%B %d, %Y}", st["Cover_Date"]))
    elems.append(Spacer(1, 8))
    elems.append(Paragraph("Period: Sep 2020 – Feb 2026 (~5.5 years)", st["Cover_Date"]))
    return elems


def _strategy_overview(st):
    elems = _section("1. Strategy Overview", st)
    elems.append(Paragraph(
        "This report evaluates an options-based long/short equity strategy applied to a "
        "dynamic universe of the top-30 high-IV stocks drawn from the S&amp;P 500. "
        "Two broad families of trades are tested:", st["Body"]))

    bullets = [
        "<b>Sell Put (↑ momentum)</b> — collect premium by selling at-the-money puts on "
        "stocks whose previous monthly candle closed <i>up</i>. Premium is fixed at ~3.3% "
        "of notional.",
        "<b>Buy Call (↑ momentum)</b> — buy ATM calls on the same up-candle stocks. "
        "Leverage variants layer on a Money-Market allocation for idle capital.",
        "Signals are refreshed every ~3 weeks (bi-monthly option expiry cycle). "
        "Portfolio is equally weighted across all active signals.",
    ]
    for b in bullets:
        elems.append(Paragraph(f"• {b}", st["BulletBody"]))

    elems.append(Spacer(1, 6))
    elems.append(Paragraph(
        "<b>Capital model:</b> $1 notional invested at inception. P&amp;L expressed as "
        "cumulative % return. Options cost is modelled as the quoted premium "
        "(entry_premium or last price from Barchart data stored in S3).", st["Body"]))
    return elems


def _universe_rules(st):
    elems = _section("2. Universe &amp; Signal Rules", st)

    rows = [
        ["Parameter", "Value"],
        ["Base universe",       "S&P 500 — 50 mega-cap tickers (10 per sector)"],
        ["Dynamic filter",      "Top 30 by realised IV rank each period"],
        ["Signal direction",    "Monthly candle: UP → sell put / buy call; DOWN → no trade"],
        ["Option type",         "ATM put (Sell Put) or ATM call (Buy Call)"],
        ["Strike",              "Last month closing price"],
        ["Expiry",              "End of current calendar month (~3-week hold)"],
        ["Premium (Sell Put)",  "3.3% of stock notional (fixed model)"],
        ["Premium (Buy Call)",  "Avg 1.52% of notional (sourced from Barchart IV)"],
        ["Portfolio weight",    "Equal weight across all active signals"],
        ["Rebalance",           "Every period (~bi-monthly expiry cycle)"],
        ["Backtest horizon",    "Sep 2020 – Feb 2026  (72 periods ≈ 5.5 years)"],
    ]
    col_w = [CONTENT_W * 0.35, CONTENT_W * 0.65]
    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8.5),
        ("ALIGN",         (0, 0), (-1, 0),  "CENTER"),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("ALIGN",         (0, 1), (0, -1),  "LEFT"),
        ("ALIGN",         (1, 1), (1, -1),  "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, ALT_ROW]),
        ("GRID",          (0, 0), (-1, -1), 0.4, MED_GRAY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ])
    elems.append(Table(rows, colWidths=col_w, style=ts))
    return elems


def _dynamic_top30_section(st):
    elems = [PageBreak()]
    elems += _section("3. Dynamic Top-30 — Mode Comparison", st)

    elems.append(Paragraph(
        "Four trading modes are compared on the same universe and signal. "
        "Mode 2 (Buy Call only) shows the best risk-adjusted return (Sharpe 1.00, "
        "MaxDD −2.4%). Mode 4 (Sell Put ↑ / Sell Call ↓ — direction-neutral) "
        "disappoints with a −16.5% drawdown over the full period.", st["Body"]))

    # Risk stats table
    stats = _extract_mode_stats(os.path.join(RES, "dynamic_top30_risk_stats.txt"))
    elems += _risk_table(stats, st)
    elems.append(Spacer(1, 10))

    # Equity curve charts — 2×2 grid
    elems.append(Paragraph("Equity Curves — All Modes", st["SubTitle"]))
    chart_files = [
        (os.path.join(RES, "dynamic_top30_equity_mode1.png"), "Mode 1: Buy Call(↑) / Buy Put(↓)"),
        (os.path.join(RES, "dynamic_top30_equity_mode2.png"), "Mode 2: Buy Call only (↑ momentum)"),
        (os.path.join(RES, "dynamic_top30_equity_mode3.png"), "Mode 3: Sell Put only (↑ momentum)"),
        (os.path.join(RES, "dynamic_top30_equity_mode4.png"), "Mode 4: Sell Put(↑) / Sell Call(↓)"),
    ]
    half_w = (CONTENT_W - 8) / 2
    img_rows = []
    pair_buf = []
    for path, cap in chart_files:
        if os.path.exists(path):
            cell_content = [Image(path, width=half_w, height=half_w * 0.5, kind="proportional"),
                            Paragraph(cap, st["Caption"])]
            pair_buf.append(cell_content)
        if len(pair_buf) == 2:
            img_rows.append(pair_buf)
            pair_buf = []
    if pair_buf:
        img_rows.append(pair_buf + [[]])

    for row_pair in img_rows:
        row_data = [[Table([[r] for r in cell], colWidths=[half_w]) for cell in row_pair]]
        t = Table(row_data, colWidths=[half_w + 4, half_w + 4])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("LEFTPADDING",  (0,0),(-1,-1), 0),
                                ("RIGHTPADDING", (0,0),(-1,-1), 0),
                                ("TOPPADDING",   (0,0),(-1,-1), 0),
                                ("BOTTOMPADDING",(0,0),(-1,-1), 0)]))
        elems.append(t)
    return elems


def _dynamic_monthly_tables(st):
    elems = [PageBreak()]
    elems += _section("4. Dynamic Top-30 — Monthly Returns (colour-coded)", st)
    elems.append(Paragraph(
        "Green = positive, Red = negative, Yellow = near-zero (±0.05%). "
        "Annual column in bold.", st["Body"]))
    elems.append(Spacer(1, 6))

    configs = [
        ("dynamic_top30_monthly_mode1.csv", "Mode 1 — Buy Call(↑) / Buy Put(↓)"),
        ("dynamic_top30_monthly_mode2.csv", "Mode 2 — Buy Call only (↑ momentum)"),
        ("dynamic_top30_monthly_mode3.csv", "Mode 3 — Sell Put only (↑ momentum)"),
        ("dynamic_top30_monthly_mode4.csv", "Mode 4 — Sell Put(↑) / Sell Call(↓)"),
    ]
    for fname, label in configs:
        elems += _monthly_table(os.path.join(RES, fname), label, st)

    return elems


def _levered_call_section(st):
    elems = [PageBreak()]
    elems += _section("5. Buy Call with Leverage — Variant Comparison", st)

    elems.append(Paragraph(
        "Three capital-efficiency structures are tested on top of Mode 2 (Buy Call only). "
        "Variant C (4× leverage + Money Market) achieves a CAGR of <b>+17.1%</b> "
        "with a MaxDD of only −8.8% and Sharpe 1.14. "
        "Variant B (1× + Money Market) offers the cleanest risk-adjusted profile "
        "(Sharpe 1.55, Sortino 5.06).", st["Body"]))

    stats = _extract_variant_stats(os.path.join(RES, "levered_call_risk_stats.txt"))
    elems += _risk_table(stats, st)
    elems.append(Spacer(1, 10))

    elems.append(Paragraph("Equity Curves — All Variants", st["SubTitle"]))
    elems += _image(
        os.path.join(RES, "levered_call_equity.png"),
        width=CONTENT_W,
        caption="Cumulative P&L (%) — Variants A, B, C vs. SPY benchmark",
        st=st,
    )
    return elems


def _levered_monthly_tables(st):
    elems = [PageBreak()]
    elems += _section("6. Buy Call (Leveraged) — Monthly Returns", st)
    elems.append(Paragraph(
        "Same colour convention as Section 4.", st["Body"]))
    elems.append(Spacer(1, 6))

    configs = [
        ("levered_call_monthly_A.csv", "Variant A — Buy Call 1× (no idle-cash credit)"),
        ("levered_call_monthly_B.csv", "Variant B — Buy Call 1× + Money Market on idle capital"),
        ("levered_call_monthly_C.csv", "Variant C — Buy Call 4× + Money Market on remaining"),
    ]
    for fname, label in configs:
        elems += _monthly_table(os.path.join(RES, fname), label, st)

    return elems


def _takeaways(st):
    elems = [PageBreak()]
    elems += _section("7. Key Take-Aways", st)

    bullets = [
        "<b>Buy Call (Mode 2) is the structural winner</b> among the four trading modes: "
        "CAGR +5.0%, MaxDD −2.4%, Sharpe 1.00. The strategy profits from positive momentum "
        "while capping downside to the premium paid.",

        "<b>Sell Put (Mode 3) has lower CAGR (+2.9%) and larger drawdown (−8.0%)</b> "
        "driven by a concentrated loss period (Jul–Aug 2024). Despite a 65% win rate, "
        "the tail risk of short puts is evident.",

        "<b>Direction-neutral short premium (Mode 4) destroys value</b>: CAGR +0.7%, "
        "MaxDD −16.5%, Sharpe 0.10. Selling calls on down-momentum stocks adds "
        "directional risk in rallying markets.",

        "<b>Money Market allocation (Variant B) transforms the risk profile</b>: "
        "idle capital (~94% of notional) earns ~4–5% p.a., lifting CAGR from +5.0% to "
        "+7.3% and Sharpe from 1.00 to 1.55 with a <i>lower</i> MaxDD (−1.85%).",

        "<b>4× leverage (Variant C) delivers +17.1% CAGR</b> — the best absolute return — "
        "at a manageable MaxDD of −8.8%. Suitable for risk-tolerant allocators "
        "who can absorb short-term volatility (monthly std dev ~6.2%).",

        "<b>Dynamic IV-rank filtering</b> (Top 30 from SPX 50) concentrates exposure "
        "in the highest-premium names, improving expected premium collection "
        "relative to a static universe.",

        "<b>Operational fit:</b> the strategy is fully systematic, option-expiry driven "
        "(bi-monthly rebalance), and requires no intraday monitoring. "
        "Lambda-based automation handles signal generation and S3 storage.",
    ]

    for b in bullets:
        elems.append(Paragraph(f"• {b}", st["BulletBody"]))
        elems.append(Spacer(1, 4))

    elems.append(Spacer(1, 12))
    elems.append(_hr())
    elems.append(Paragraph(
        "<i>Disclaimer: This document is for internal research purposes only. "
        "Past backtest performance is not indicative of future results. "
        "Options trading involves significant risk of loss.</i>",
        ParagraphStyle("Disclaimer", fontName="Helvetica-Oblique", fontSize=7.5,
                       textColor=colors.HexColor("#888888"), alignment=TA_CENTER)))
    return elems


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def build():
    st   = _styles()
    doc  = _build_doc()

    story = []

    # --- Cover (no page number) ---
    story += _cover(st)
    story.append(NextPageTemplate("Normal"))
    story.append(PageBreak())

    # --- Content sections ---
    story += _strategy_overview(st)
    story += _universe_rules(st)
    story += _dynamic_top30_section(st)
    story += _dynamic_monthly_tables(st)
    story += _levered_call_section(st)
    story += _levered_monthly_tables(st)
    story += _takeaways(st)

    doc.build(story)
    print(f"✓ Report written → {OUT}")


if __name__ == "__main__":
    build()
