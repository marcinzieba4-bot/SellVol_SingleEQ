#!/usr/bin/env python3
"""
Generate current_play_list.json for the next buy-call period.

Signal logic (mirrors backtest):
  - Measure return of each stock over the MOST RECENTLY EXPIRED period
    (2026-02-13 → 2026-03-13)
  - If return >= MIN_RETURN_PCT  → BUY CALL next period
  - Strike = today's closing price (ATM call)
  - Premium = Black-Scholes call estimate using VIX-based IV model

Output: current_play_list.json
"""

import json
import math
import warnings
from datetime import date, datetime

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── shared constants ──────────────────────────────────────────────────────────
from backtest import TICKERS, STOCK_RV_RATIOS, IV_SCALE, _norm_cdf

# ── config ────────────────────────────────────────────────────────────────────
# Period whose return drives the signal for the NEXT play
SIGNAL_PERIOD_START = "2026-02-13"
SIGNAL_PERIOD_END   = "2026-03-13"

# Next play window (approximate — last Friday of March → last Friday of April)
NEXT_ENTRY          = "2026-03-28"   # last Friday of March 2026
NEXT_EXPIRY         = "2026-04-24"   # last Friday of April 2026
DTE                 = 27             # days

MIN_RETURN_PCT      = 2.0            # minimum prev-period return to qualify


# ── Black-Scholes call (full, not ATM-only) ───────────────────────────────────
def bs_call(S, K, T_years, iv_pct, r_pct):
    """Call price in dollars."""
    if T_years <= 0:
        return max(S - K, 0.0)
    sigma = iv_pct / 100.0
    r     = r_pct  / 100.0
    sqrtT = math.sqrt(T_years)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T_years) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    call = S * _norm_cdf(d1) - K * math.exp(-r * T_years) * _norm_cdf(d2)
    return max(call, 0.0)


def iv_estimate(vix, ticker):
    rv = STOCK_RV_RATIOS.get(ticker, sum(STOCK_RV_RATIOS.values()) / len(STOCK_RV_RATIOS))
    return vix * rv * IV_SCALE


# ── step 1: fetch signal-period prices for all tickers ───────────────────────
print(f"Fetching signal-period returns ({SIGNAL_PERIOD_START} → {SIGNAL_PERIOD_END}) ...")
raw = yf.download(
    TICKERS,
    start=SIGNAL_PERIOD_START,
    end=(pd.Timestamp(SIGNAL_PERIOD_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    auto_adjust=True,
    progress=False,
)
if isinstance(raw.columns, pd.MultiIndex):
    closes = raw["Close"]
else:
    closes = raw

signal_returns = {}
for t in TICKERS:
    try:
        ser = closes[t].dropna() if t in closes.columns else closes.dropna()
        if len(ser) >= 2:
            entry_px = float(ser.iloc[0])
            exit_px  = float(ser.iloc[-1])
            signal_returns[t] = (exit_px - entry_px) / entry_px * 100.0
    except Exception as e:
        print(f"  WARNING: {t} signal return failed: {e}")

print(f"  Returns computed for {len(signal_returns)}/{len(TICKERS)} tickers")


# ── step 2: fetch VIX & RFR ───────────────────────────────────────────────────
print("Fetching VIX & risk-free rate ...")
vix_raw = yf.download("^VIX", period="5d", auto_adjust=False, progress=False)
rfr_raw = yf.download("^IRX", period="5d", auto_adjust=False, progress=False)
for df in (vix_raw, rfr_raw):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
vix = float(vix_raw["Close"].dropna().iloc[-1]) if not vix_raw.empty else 20.0
rfr = float(rfr_raw["Close"].dropna().iloc[-1]) if not rfr_raw.empty else 4.0
print(f"  VIX={vix:.1f}  RFR={rfr:.2f}%")


# ── step 3: fetch today's prices for qualifying tickers ──────────────────────
qualifiers = sorted(
    [t for t, r in signal_returns.items() if r >= MIN_RETURN_PCT],
    key=lambda t: -signal_returns[t],
)
print(f"\nQualifying tickers (prev return >= {MIN_RETURN_PCT}%): {len(qualifiers)}")

print(f"Fetching today's prices for {len(qualifiers)} tickers ...")
px_raw = yf.download(qualifiers, period="5d", auto_adjust=True, progress=False)
if isinstance(px_raw.columns, pd.MultiIndex):
    px_closes = px_raw["Close"]
else:
    px_closes = px_raw

today_prices = {}
for t in qualifiers:
    try:
        col = px_closes[t] if t in px_closes.columns else px_closes
        today_prices[t] = float(col.dropna().iloc[-1])
    except Exception:
        pass


# ── step 4: build play list ───────────────────────────────────────────────────
T_years = DTE / 365.0
weight  = 1.0 / len(qualifiers) if qualifiers else 0.0

plays = []
for t in qualifiers:
    price = today_prices.get(t)
    if price is None:
        print(f"  WARNING: no live price for {t}, skipping")
        continue

    iv_pct   = iv_estimate(vix, t)
    call_px  = bs_call(price, price, T_years, iv_pct, rfr)   # ATM call
    prem_pct = call_px / price * 100.0

    plays.append({
        "ticker":            t,
        "signal":            "buy_call",
        "prev_period_return": round(signal_returns[t], 2),
        "signal_period":     f"{SIGNAL_PERIOD_START} → {SIGNAL_PERIOD_END}",
        "entry_date":        NEXT_ENTRY,
        "expiry_date":       NEXT_EXPIRY,
        "dte":               DTE,
        "current_price":     round(price, 2),
        "strike":            round(price, 2),     # ATM = current price
        "iv_pct":            round(iv_pct, 2),
        "premium_per_share": round(call_px, 2),
        "premium_pct":       round(prem_pct, 3),  # % of notional
        "weight":            round(weight, 4),
    })

# ── step 5: save ──────────────────────────────────────────────────────────────
output = {
    "generated_at":      datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "signal_period":     f"{SIGNAL_PERIOD_START} → {SIGNAL_PERIOD_END}",
    "next_entry":        NEXT_ENTRY,
    "next_expiry":       NEXT_EXPIRY,
    "dte":               DTE,
    "vix":               round(vix, 2),
    "rfr_pct":           round(rfr, 2),
    "min_return_filter": MIN_RETURN_PCT,
    "n_plays":           len(plays),
    "equal_weight":      round(weight, 4),
    "plays":             plays,
}

import os
out_path = os.path.join(os.path.dirname(__file__), "current_play_list.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

# ── print summary ─────────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"  BUY-CALL PLAY LIST  —  entry {NEXT_ENTRY}  expiry {NEXT_EXPIRY}  ({DTE} DTE)")
print(f"  Signal: prev-period return >= {MIN_RETURN_PCT}%  |  VIX={vix:.1f}  RFR={rfr:.2f}%")
print(f"  {len(plays)} stocks  |  equal weight {weight*100:.1f}% each")
print(f"{'═'*65}")
print(f"  {'TICKER':<7} {'PREV_RET%':>9} {'PRICE':>8} {'STRIKE':>8} "
      f"{'IV%':>6} {'PREM$':>7} {'PREM%':>7}")
print(f"  {'─'*7} {'─'*9} {'─'*8} {'─'*8} {'─'*6} {'─'*7} {'─'*7}")
for p in plays:
    print(f"  {p['ticker']:<7} {p['prev_period_return']:>+9.2f} "
          f"{p['current_price']:>8.2f} {p['strike']:>8.2f} "
          f"{p['iv_pct']:>6.1f} {p['premium_per_share']:>7.2f} "
          f"{p['premium_pct']:>7.3f}%")

avg_prem = sum(p["premium_pct"] for p in plays) / len(plays) if plays else 0
print(f"\n  Avg premium: {avg_prem:.3f}%  of notional per position")
print(f"\n  Saved → {out_path}")
