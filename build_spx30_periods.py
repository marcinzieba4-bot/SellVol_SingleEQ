"""
build_spx30_periods.py
----------------------
For each ~4-week option period (using SPY's S3 period grid as reference),
determine the 30 largest S&P 500 stocks by market cap.

Market-cap estimation method:
  historical_mktcap_i(t) = current_mktcap_i * (adj_hist_price_i(t) / current_price_i)

This equals shares_i * hist_price_i (the canonical formula) when shares are constant.
For the TOP-30 ranking, a ±20% error in mktcap at any point is inconsequential.

Outputs:
  spx30_periods.json        – per-period top-30 list with cap weights
  spx30_required_tickers.txt – unique ticker set across all periods (options data needed)
"""

import os, sys, json, time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
from strategy_vol_premium import run_strategy

OUT_JSON  = "spx30_periods.json"
OUT_TICKS = "spx30_required_tickers.txt"

# ── Candidate universe: all stocks that could be in top 50 over 2020-2026 ─────
# ~126 tickers covering current S&P 500 top-100 + historical top-50 entrants
_UNIVERSE_RAW = [
    # Mega-cap tech & growth
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","AVGO","ORCL",
    "CRM","ADBE","QCOM","AMD","TXN","AMAT","KLAC","LRCX","ADI","MU",
    "INTC","NOW","PANW","SNPS","CDNS","INTU","FTNT","MRVL","NFLX","UBER",
    # Healthcare
    "LLY","UNH","JNJ","MRK","ABBV","TMO","ABT","ISRG","MDT","GILD",
    "BMY","AMGN","VRTX","REGN","SYK","BSX","ELV","CI","CVS","HCA",
    # Finance
    "BRK-B","JPM","V","MA","BAC","WFC","GS","MS","BLK","AXP",
    "COF","USB","SCHW","SPGI","CB","CME","ICE","MCO","BX","PGR",
    # Consumer disc & staples
    "WMT","HD","COST","MCD","SBUX","NKE","LOW","TGT","PG","KO",
    "PEP","PM","MO","CL","MDLZ",
    # Energy
    "XOM","CVX","COP","EOG","SLB","PSX","MPC","OXY","VLO",
    # Industrials & defense
    "GE","CAT","HON","UNP","DE","RTX","GD","LMT","NOC","BA",
    "MMM","ETN","ITW","PH","EMR",
    # Communication & media
    "T","VZ","DIS","CMCSA","CHTR",
    # Utilities & REITs
    "NEE","SO","DUK","AMT","PLD",
    # Other large-caps often in top 50
    "ACN","IBM","CSCO","WM","ECL","APD","SHW",
]
# deduplicate preserving order
_seen = set()
UNIVERSE = [t for t in _UNIVERSE_RAW if t not in _seen and not _seen.add(t)]
print(f"Universe: {len(UNIVERSE)} candidate tickers (selecting top 30 each period)")

# ── Current market cap table ($B, approximate March 2026) ─────────────────────
# Used to anchor the historical market cap scaling.
# Values accurate ±30% — sufficient for stable top-50 ranking.
MKTCAP_BN = {
    "AAPL":3700,"MSFT":2900,"NVDA":3100,"AMZN":2400,"GOOGL":2000,"GOOG":350,
    "META":1600,"TSLA":1000,"AVGO":900, "LLY": 680, "WMT": 780, "JPM": 700,
    "ORCL":500, "V":   580, "UNH": 530, "MA":  450, "NFLX":400, "XOM": 510,
    "COST":420, "CRM": 310, "JNJ": 380, "ABBV":330, "PG":  390, "HD":  380,
    "BAC": 330, "KO":  270, "PEP": 200, "MRK": 230, "TMO": 195, "CVX": 290,
    "ABT": 210, "AMGN":160, "ISRG":180, "MDT": 120, "IBM": 230, "INTC":93,
    "QCOM":165, "AMD": 175, "TXN": 180, "AMAT":145, "KLAC":100, "LRCX":95,
    "ADI": 90,  "MU":  95,  "NOW": 195, "PANW":125, "SNPS":80,  "CDNS":70,
    "INTU":165, "FTNT":62,  "MRVL":75,  "UBER":160, "ADBE":160, "BMY": 120,
    "GILD":120, "REGN":60,  "VRTX":100, "SYK": 135, "BSX":115,
    "ELV": 95,  "CI":  70,  "CVS": 65,  "HCA": 60,  "BRK-B":930,"GS":  195,
    "MS":  185, "BLK": 155, "AXP": 180, "COF": 65,  "USB": 65,  "SCHW":140,
    "SPGI":130, "CB":  95,  "CME": 75,  "ICE": 80,  "MCO": 75,  "BX":  175,
    "PGR": 145, "WFC": 220, "MCD": 215, "SBUX":90,  "NKE": 95,  "LOW": 150,
    "TGT": 55,  "PM":  210, "MO":  87,  "CL":  65,  "MDLZ":80,  "COP": 150,
    "EOG": 65,  "SLB": 55,  "PSX": 50,  "MPC": 55,  "OXY": 45,  "VLO": 45,
    "GE":  190, "CAT": 195, "HON": 140, "UNP": 130, "DE":  100, "RTX": 150,
    "GD":  80,  "LMT": 105, "NOC": 65,  "BA":  120, "MMM": 70,  "ETN": 95,
    "ITW": 80,  "PH":  80,  "EMR": 65,  "T":   160, "VZ":  165, "DIS": 190,
    "CMCSA":150,"CHTR":45,  "NEE": 110, "SO":  80,  "DUK": 70,  "AMT": 90,
    "PLD": 95,  "ACN": 220, "CSCO":230, "WM":  88,  "ECL": 60,  "APD": 45,
    "SHW": 85,
}

# ── SPY period grid ────────────────────────────────────────────────────────────
print("\nLoading SPY periods from S3...", flush=True)
spy_res      = run_strategy("SPY", mode="buy_call")
period_dates = sorted({r["period_start"] for r in spy_res})
print(f"  {len(period_dates)} rebalance periods  ({period_dates[0]} -> {period_dates[-1]})")

# ── Yahoo Finance session ─────────────────────────────────────────────────────
_YF_CHART = "https://query{n}.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS  = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
}
print("\nInitialising Yahoo Finance session...", flush=True)
_SESS = requests.Session()
_SESS.headers.update(_HEADERS)
try:
    _SESS.get("https://finance.yahoo.com", timeout=15)
    print("  Session ready")
except Exception as e:
    print(f"  Session init failed: {e}")
time.sleep(1)


def _yf_get(ticker, params, max_retries=4):
    for attempt in range(max_retries):
        n   = 1 + (attempt % 2)
        url = _YF_CHART.format(ticker=ticker, n=n)
        try:
            r = _SESS.get(url, params=params, timeout=20)
            if r.status_code == 429:
                wait = 20 * (attempt + 1)
                print(f"\n  429 on {ticker}, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            if r.status_code != 200 or not r.text.strip():
                return None
            return r.json()
        except Exception:
            time.sleep(5)
    return None


def fetch_price_series(ticker, start, end):
    """
    Returns (pd.Series adj-close, current_price_float).
    current_price = regularMarketPrice from chart meta.
    """
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    end_ts   = int(datetime.strptime(end,   "%Y-%m-%d").timestamp()) + 86400
    j = _yf_get(ticker, {"period1": start_ts, "period2": end_ts,
                          "interval": "1d", "events": "splits"})
    if not j:
        return None, None
    try:
        result = j["chart"]["result"]
        if not result:
            return None, None
        res           = result[0]
        current_price = res["meta"].get("regularMarketPrice")
        ts_list       = res.get("timestamp", [])
        closes        = res["indicators"]["adjclose"][0]["adjclose"]
        if not ts_list or not closes:
            return None, current_price
        dates  = [datetime.utcfromtimestamp(t).strftime("%Y-%m-%d") for t in ts_list]
        series = pd.Series(closes, index=pd.to_datetime(dates), dtype=float).dropna()
        return series, current_price
    except Exception:
        return None, None


# ── Download price histories ────────────────────────────────────────────────
START_DATE = "2020-08-01"
END_DATE   = (datetime.today() + timedelta(days=2)).strftime("%Y-%m-%d")

print(f"\nDownloading price histories {START_DATE} -> {END_DATE}...")
all_prices     = {}   # ticker -> pd.Series (adj-close)
current_prices = {}   # ticker -> float (most recent price)

for i, tk in enumerate(UNIVERSE):
    series, cur_px = fetch_price_series(tk, START_DATE, END_DATE)
    ok = series is not None and len(series) > 0
    if ok:
        all_prices[tk] = series
    if cur_px and cur_px > 0:
        current_prices[tk] = float(cur_px)
    print(f"  [{i+1}/{len(UNIVERSE)}] {tk:<8} "
          f"{'OK ' if ok else 'SKP'} ({len(series) if series is not None else 0} days)",
          end="\r", flush=True)
    time.sleep(0.4)

print(f"\n  Price series: {len(all_prices)} tickers loaded, "
      f"{len(current_prices)} current prices")

# ── Build price DataFrame and market cap matrix ───────────────────────────────
# Only include tickers with both price history and a known market cap
valid = [t for t in UNIVERSE
         if t in all_prices and t in current_prices and MKTCAP_BN.get(t, 0) > 0]
print(f"  Valid tickers (price + mktcap data): {len(valid)}")

price_df = pd.DataFrame({t: all_prices[t] for t in valid}).sort_index()
price_df.index = pd.to_datetime(price_df.index).tz_localize(None)

# Precompute scaling factor: current_mktcap / current_price = "shares proxy"
# historical_mktcap(t) = shares_proxy * adj_hist_price(t)
shares_proxy = {t: MKTCAP_BN[t] * 1e9 / current_prices[t] for t in valid}

# ── Build top-30 per period ────────────────────────────────────────────────────
print("\nBuilding top-30 ranking for each period...")
spx50_periods = {}

for period_start in period_dates:
    pd_date = pd.Timestamp(period_start)
    avail   = price_df.index[price_df.index <= pd_date]
    if len(avail) == 0:
        print(f"  WARNING: no price data before {period_start}")
        continue
    row = price_df.loc[avail[-1]]

    mktcaps = {}
    for tk in valid:
        px = row.get(tk)
        if px and not np.isnan(float(px)) and float(px) > 0:
            mktcaps[tk] = shares_proxy[tk] * float(px)

    ranked    = sorted(mktcaps.items(), key=lambda x: x[1], reverse=True)[:30]
    total_cap = sum(v for _, v in ranked)

    spx50_periods[period_start] = [
        {
            "rank":     rank,
            "ticker":   tk,
            "mktcap_bn": round(cap / 1e9, 1),
            "weight":   round(cap / total_cap, 6),
        }
        for rank, (tk, cap) in enumerate(ranked, 1)
    ]

print(f"  Built top-30 for {len(spx50_periods)} periods")

# ── Save JSON ──────────────────────────────────────────────────────────────────
with open(OUT_JSON, "w") as f:
    json.dump(spx50_periods, f, indent=2)
print(f"\n  Saved -> {OUT_JSON}")

# ── Required tickers ──────────────────────────────────────────────────────────
all_tickers = sorted({s["ticker"] for stocks in spx50_periods.values() for s in stocks})
with open(OUT_TICKS, "w") as f:
    f.write(f"# Unique tickers in SPX Top-30 across all periods\n")
    f.write(f"# ({period_dates[0]} -> {period_dates[-1]}, {len(period_dates)} periods)\n")
    f.write(f"# Total unique: {len(all_tickers)}\n\n")
    for tk in all_tickers:
        f.write(tk + "\n")
print(f"  Saved -> {OUT_TICKS}  ({len(all_tickers)} unique tickers)")

# ── Composition snapshots ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  COMPOSITION SNAPSHOTS (first 10 of 30 shown)")
print("=" * 65)
samples = [period_dates[0], period_dates[len(period_dates)//3],
           period_dates[2*len(period_dates)//3], period_dates[-1]]
for p in samples:
    stocks = spx50_periods.get(p, [])
    if not stocks:
        continue
    print(f"\n  Period: {p}")
    print(f"  {'#':<4} {'Ticker':<8} {'MktCap $B':>10}  {'Weight':>7}")
    print(f"  {'-'*38}")
    for s in stocks[:10]:
        print(f"  {s['rank']:<4} {s['ticker']:<8} {s['mktcap_bn']:>9.0f}B  {s['weight']*100:>6.2f}%")
    print(f"  ... ({len(stocks)} total)")

print(f"\n{'='*65}")
print(f"  UNIQUE TICKERS NEEDED ({len(all_tickers)} total)")
print(f"  You need to provide options data for all these stocks:")
print(f"{'─'*65}")
for i in range(0, len(all_tickers), 10):
    print("  " + "  ".join(f"{t:<8}" for t in all_tickers[i:i+10]))
print(f"{'='*65}")
