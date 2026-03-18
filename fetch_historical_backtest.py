"""
fetch_historical_backtest.py

Build a monthly ATM-put history for a single ticker from 2017-06-01 to today.

For each month:
  observation_date = first business day of that month
  stock_price      = closing price on observation_date  (yfinance)
  target_expiry    = standard monthly expiry ~30-35 DTE from observation_date
                     (3rd Friday of the following month)
  ATM strike       = option strike closest to stock_price
  data downloaded  = full put chain from Barchart for that expiry,
                     filtered to the ATM strike row

Two Barchart download strategies are tried in order:
  1. date-snapshot URL  — adds &tradeDate=YYYY-MM-DD to the download request.
       If Barchart supports this (premium historical feature), the CSV will
       contain prices as of that specific date (~30 DTE at entry).
  2. plain-expiry URL   — no date parameter; Barchart returns prices as of the
       last trading session for that expiry (settlement / expiry-day data).
       Useful for P&L-at-expiry analysis even if entry price is unknown.

Usage:
    python fetch_historical_backtest.py
    python fetch_historical_backtest.py --ticker AAPL --start 2020-01-01
    python fetch_historical_backtest.py --ticker NVDA --show-browser

Output:
    historical_options/
        NVDA_2017-06-01_2017-07-21.csv   ← ATM put row, observation + expiry in name
        ...
        NVDA_history_summary.csv         ← all months combined
"""

import os
import sys
import json
import math
import time
import argparse
import tempfile
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException,
    ElementNotInteractableException, ElementClickInterceptedException,
)

# ---------------------------------------------------------------------------
# Re-use helpers from fetch_options_barchart where possible
# ---------------------------------------------------------------------------
from fetch_options_barchart import (
    setup_driver,
    login,
    extract_atm_put,
    _dismiss_cmp_overlay,
    _third_friday,
)

# ---------------------------------------------------------------------------
OPTIONS_PAGE = "https://www.barchart.com/stocks/quotes/{symbol}/options"
LOGIN_URL    = "https://www.barchart.com/login"

WAIT  = 20   # seconds to wait for page elements
DELAY = 2.5  # polite pause between downloads


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _monthly_schedule(start: date, end: date) -> list[tuple[date, date]]:
    """
    Generate non-overlapping (observation_date, target_expiry) pairs.

    observation_date = start, then each subsequent obs = first business day
                       after the previous expiry (no overlap between trades)
    target_expiry    = nearest standard monthly (3rd Friday) with >= 25 DTE
                       from the observation date
    """
    pairs = []

    # Move start to first business day
    obs = start
    while obs.weekday() >= 5:
        obs += timedelta(days=1)

    while obs <= end:
        y, m = obs.year, obs.month

        # Find the nearest monthly expiry with >= 25 DTE
        expiry = _third_friday(y, m)
        if (expiry - obs).days < 25:
            ny, nm = (y, m + 1) if m < 12 else (y + 1, 1)
            expiry = _third_friday(ny, nm)

        if expiry > end:
            break

        pairs.append((obs, expiry))

        # Next observation: first business day after expiry (no overlap)
        obs = expiry + timedelta(days=1)
        while obs.weekday() >= 5:
            obs += timedelta(days=1)

    return pairs


# ---------------------------------------------------------------------------
# Historical stock price
# ---------------------------------------------------------------------------

def get_historical_price(ticker: str, target_date: date) -> float | None:
    """
    Return the ACTUAL (unadjusted) closing price of ticker on target_date.

    yfinance always returns split-adjusted prices. To recover the actual
    historical price (needed to match against Barchart's option strikes),
    we multiply the adjusted price by the cumulative factor of all splits
    that occurred AFTER target_date.

    Example for NVDA: 4:1 split 2021-07-20, 10:1 split 2024-06-10.
    yfinance shows ~$3.61 for 2017-06-01; actual was ~$144.36 ($3.61 × 40).
    """
    start = target_date - timedelta(days=7)
    end   = target_date + timedelta(days=1)   # yfinance end is exclusive
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start.isoformat(), end=end.isoformat(), auto_adjust=True)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).date
        df = df[df.index <= target_date]
        if df.empty:
            return None
        adj_price = float(df["Close"].iloc[-1])

        # Un-apply splits that happened after target_date
        splits = t.splits
        if not splits.empty:
            splits.index = pd.to_datetime(splits.index).date
            future_splits = splits[splits.index > target_date]
            for factor in future_splits:
                adj_price *= float(factor)

        return adj_price
    except Exception as e:
        print(f"    yfinance error for {ticker} on {target_date}: {e}")
        return None


# ---------------------------------------------------------------------------
# Wait for downloaded file
# ---------------------------------------------------------------------------

def _wait_for_download(download_dir: Path, timeout: int = 30) -> Path | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        files = [
            f for f in download_dir.iterdir()
            if f.suffix.lower() == ".csv" and not f.name.endswith(".crdownload")
        ]
        if files:
            return max(files, key=lambda f: f.stat().st_mtime)
        time.sleep(0.5)
    return None


# ---------------------------------------------------------------------------
# Historical option price lookup via Barchart's price-history endpoint
# ---------------------------------------------------------------------------

_JS_FETCH = """
    var done = arguments[arguments.length - 1];
    var url  = arguments[0];
    var m    = document.cookie.match(/XSRF-TOKEN=([^;]+)/);
    var xsrf = m ? decodeURIComponent(m[1]) : '';
    fetch(url, {
        credentials: 'include',
        headers: {'Accept': 'application/json', 'X-XSRF-TOKEN': xsrf}
    })
    .then(function(r) { return r.json(); })
    .then(function(d) { done(JSON.stringify(d)); })
    .catch(function(e) { done(null); });
"""


def _bc_fetch(driver: webdriver.Chrome, url: str) -> dict | None:
    """Run a credentialed GET inside the browser and return parsed JSON."""
    try:
        driver.set_script_timeout(20)
        raw = driver.execute_async_script(_JS_FETCH, url)
        return json.loads(raw) if raw else None
    except Exception as e:
        print(f"\n    [diag] fetch error: {e}")
        return None


def _build_occ(ticker: str, expiry: date, strike: float) -> str:
    """
    Build a standard OCC option symbol for a put.
    Format: TICKER + YYMMDD + P + 8-digit-strike*1000
    e.g.  AAPL170721P00142000  (AAPL $142 put expiring 2017-07-21)
    """
    return f"{ticker}{expiry.strftime('%y%m%d')}P{int(round(strike * 1000)):08d}"


def _candidate_strikes(price: float, n: int = 15) -> list[float]:
    """
    Return up to n put strikes closest to price, covering all common
    increment conventions ($0.50, $1, $2.50, $5, $10).
    """
    seen: set[float] = set()
    out: list[float] = []
    for step in [1.0, 2.5, 5.0, 0.5, 10.0]:
        base = math.floor(price / step) * step
        for k in range(-8, 9):
            s = round(base + k * step, 2)
            if s > 0 and s not in seen:
                seen.add(s)
                out.append(s)
    return sorted(out, key=lambda s: abs(s - price))[:n]


def _fetch_option_price(
    driver: webdriver.Chrome,
    ticker: str,
    expiry: date,
    strike: float,
    obs_date: date,
) -> dict | None:
    """
    Fetch OHLC history for one specific put option on obs_date using
    Barchart's price-history API (the same data shown on
    barchart.com/options/price-history?symbol=...&expirationDate=...&
    symbolType=P&strikePrice=...).

    Returns a dict with keys like 'open','high','low','close','volume',
    'openInterest', or None if no data exists for this contract/date.
    """
    # Strategy A: historical/get with OCC symbol
    occ = _build_occ(ticker, expiry, strike)
    url = (
        f"/proxies/core-api/v1/historical/get"
        f"?symbol={occ}"
        f"&startDate={obs_date.isoformat()}"
        f"&endDate={obs_date.isoformat()}"
        f"&type=daily&raw=1"
        f"&fields=tradeTime,open,high,low,close,volume,openInterest,impliedVolatility"
    )
    data = _bc_fetch(driver, url)
    if data and data.get('data'):
        rows = data['data']
        if rows:
            return rows[0].get('raw', rows[0])

    # Strategy B: price-history endpoint with separate params
    url2 = (
        f"/proxies/core-api/v1/options/historical.json"
        f"?symbol={ticker}"
        f"&expirationDate={expiry.isoformat()}"
        f"&symbolType=P"
        f"&strikePrice={strike:.2f}"
        f"&startDate={obs_date.isoformat()}"
        f"&endDate={obs_date.isoformat()}"
        f"&raw=1"
    )
    data2 = _bc_fetch(driver, url2)
    if data2 and data2.get('data'):
        rows2 = data2['data']
        if rows2:
            return rows2[0].get('raw', rows2[0])

    return None


def download_historical_chain(
    driver: webdriver.Chrome,
    download_dir: Path,
    ticker: str,
    expiry: date,
    observation_date: date,
    stock_price: float,
) -> tuple[Path | None, str]:
    """
    Find the ATM put for (ticker, expiry) as of observation_date and return
    its closing price on that date.

    Iterates over candidate strikes (sorted by distance from stock_price)
    and calls Barchart's option price-history API for each until one returns
    data.  Uses the browser's own fetch() so the authenticated session is
    inherited automatically.

    Returns (csv_path, "entry_snapshot") or (None, "failed").
    """
    # Ensure we are on a barchart.com page so relative fetch() URLs work
    if 'barchart.com' not in driver.current_url:
        driver.get(OPTIONS_PAGE.format(symbol=ticker))
        time.sleep(2)

    for f in download_dir.glob("*.csv"):
        f.unlink()

    candidates = _candidate_strikes(stock_price)

    for strike in candidates:
        row = _fetch_option_price(driver, ticker, expiry, strike, observation_date)
        if not row:
            continue

        occ = _build_occ(ticker, expiry, strike)
        df = pd.DataFrame([{
            'Symbol':        occ,
            'Strike':        strike,
            'Last':          row.get('close') or row.get('lastPrice'),
            'Open':          row.get('open'),
            'High':          row.get('high'),
            'Low':           row.get('low'),
            'Volume':        row.get('volume'),
            'Open Interest': row.get('openInterest'),
            'IV':            row.get('impliedVolatility') or row.get('volatility'),
        }])
        out = download_dir / f'{ticker}_{expiry}_{observation_date}_hist.csv'
        df.to_csv(out, index=False)
        return out, "entry_snapshot"

    print(
        f"\n    [diag] no price history for {ticker} {expiry} "
        f"near ${stock_price:.2f} on {observation_date} "
        f"(tried {len(candidates)} strikes)"
    )
    return None, "failed"


# ---------------------------------------------------------------------------
# Process one (observation_date, expiry) pair
# ---------------------------------------------------------------------------

def process_month(
    driver: webdriver.Chrome,
    ticker: str,
    observation_date: date,
    expiry: date,
    download_dir: Path,
    out_dir: Path,
) -> dict | None:
    """
    Download and parse one monthly ATM put data point.
    Returns a dict of key fields, or None on failure.
    """
    dte = (expiry - observation_date).days
    print(f"  {observation_date}  expiry={expiry}  ({dte} DTE)", end="  ")

    # 1. Historical stock price
    price = get_historical_price(ticker, observation_date)
    if price is None:
        print("SKIP (no price data)")
        return None
    print(f"${price:.2f}", end="  ")

    # 2. Download chain
    csv_path, source = download_historical_chain(
        driver, download_dir, ticker, expiry, observation_date, price
    )
    if csv_path is None:
        print("FAILED (no download)")
        return None

    print(f"[{source}]", end="  ")

    # 3. Extract ATM put
    atm = extract_atm_put(csv_path, price, ticker, expiry)
    if atm is None or atm.empty:
        print("FAILED (no ATM row)")
        return None

    row = atm.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # Validate the CSV is for the right ticker + expiry.
    # Barchart option symbols are OCC-format: TICKER+YYMMDD+P+strike
    # e.g. AAPL170721P00140000  →  expiry 2017-07-21, put, strike $140
    # If the symbol shows a different date (redirect gave us wrong data),
    # discard the row entirely.
    # ------------------------------------------------------------------
    sym_col = next((c for c in atm.columns
                    if c.lower() in ("symbol", "contract", "option_symbol")), None)
    if sym_col:
        sym_val  = str(atm.iloc[0].get(sym_col, ""))
        exp_ymd  = expiry.strftime("%y%m%d")          # "170721"
        tkr_up   = ticker.upper()
        if exp_ymd not in sym_val:
            print(f"INVALID(symbol={sym_val!r} ≠ expiry {exp_ymd}) — skipping")
            return None
        if tkr_up not in sym_val.upper():
            print(f"INVALID(symbol={sym_val!r} ≠ ticker {tkr_up}) — skipping")
            return None

    # ------------------------------------------------------------------
    # 4a. Compute entry_premium = mid(bid, ask) at observation_date.
    #     The "Last" column is the last TRADED price and may be stale by
    #     days or weeks.  Bid/ask as-of the tradeDate are always fresh.
    # ------------------------------------------------------------------
    def _num(val) -> float | None:
        try:
            v = float(str(val).replace(",", "").replace("$", "").strip())
            return v if v > 0 else None
        except (ValueError, TypeError):
            return None

    bid_col  = next((c for c in atm.columns if c.lower() in ("bid", "bid_price", "bid price")), None)
    ask_col  = next((c for c in atm.columns if c.lower() in ("ask", "ask_price", "ask price")), None)
    last_col = next((c for c in atm.columns if c.lower() in ("last", "last_price", "last price", "close")), None)

    bid  = _num(row.get(bid_col))  if bid_col  else None
    ask  = _num(row.get(ask_col))  if ask_col  else None
    last = _num(row.get(last_col)) if last_col else None

    if bid is not None and ask is not None:
        entry_premium = round((bid + ask) / 2, 4)
        price_method  = "mid"
    elif last is not None:
        entry_premium = last
        price_method  = "last"
    else:
        entry_premium = None
        price_method  = "n/a"

    if source == "settlement":
        print("WARN:settlement-prices ", end="")

    # ------------------------------------------------------------------
    # 4b. Enrich row with observation context
    # ------------------------------------------------------------------
    row["ticker"]           = ticker
    row["observation_date"] = observation_date.isoformat()
    row["expiry"]           = expiry.isoformat()
    row["dte_at_obs"]       = dte
    row["stock_price"]      = price
    row["entry_premium"]    = entry_premium
    row["price_method"]     = price_method
    row["data_source"]      = source

    # Log option price and IV
    if entry_premium is not None:
        print(f"premium=${entry_premium:.2f}({price_method})", end="  ")
    else:
        print("premium=? ", end="  ")

    iv_col = next((c for c in atm.columns if "iv" in c.lower() or "implied" in c.lower()), None)
    if iv_col:
        try:
            print(f"IV={float(str(row.get(iv_col,'?')).replace('%','')):.1f}%", end="")
        except (ValueError, TypeError):
            pass
    print()

    # 5. Save per-month CSV
    fname = f"{ticker}_{observation_date}_{expiry}.csv"
    pd.DataFrame([row]).to_csv(out_dir / fname, index=False)

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download monthly historical ATM put data from Barchart"
    )
    parser.add_argument("--ticker",      default="NVDA",       help="Ticker symbol (default: NVDA)")
    parser.add_argument("--start",       default="2017-06-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",         default="",           help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--show-browser", action="store_true", help="Show Chrome window")
    parser.add_argument("--out-dir",     default="historical_options", help="Output directory")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip months that already have a CSV file in out-dir")
    args = parser.parse_args()

    username = os.getenv("BARCHART_USER", "")
    password = os.getenv("BARCHART_PASS", "")
    if not username or not password:
        print("ERROR: Set BARCHART_USER and BARCHART_PASS environment variables")
        sys.exit(1)

    ticker     = args.ticker.upper()
    start_date = date.fromisoformat(args.start)
    end_date   = date.fromisoformat(args.end) if args.end else date.today()

    out_dir = Path(args.out_dir) / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    download_dir = Path(tempfile.mkdtemp(prefix="barchart_hist_"))

    schedule = _monthly_schedule(start_date, end_date)

    print(f"Historical ATM Put Downloader")
    print(f"Ticker  : {ticker}")
    print(f"Range   : {start_date} → {end_date}")
    print(f"Months  : {len(schedule)}")
    print(f"Output  : {out_dir}")
    print(f"Browser : {'visible' if args.show_browser else 'headless'}\n")
    print("Note: 'entry_snapshot' = prices as of observation date (30 DTE)")
    print("      'settlement'     = prices at/near expiry (last traded)\n")

    driver = setup_driver(str(download_dir), headless=not args.show_browser)
    results = []
    failed  = []
    skipped = 0

    try:
        login(driver, username, password)
        print()

        for obs_date, expiry in schedule:
            # Skip-existing check
            if args.skip_existing:
                fname = f"{ticker}_{obs_date}_{expiry}.csv"
                if (out_dir / fname).exists():
                    print(f"  {obs_date}  expiry={expiry}  SKIP (file exists)")
                    skipped += 1
                    continue

            row = process_month(driver, ticker, obs_date, expiry, download_dir, out_dir)
            if row:
                results.append(row)
            else:
                failed.append((obs_date, expiry))

            time.sleep(DELAY)

    finally:
        driver.quit()

    # Summary CSV
    print(f"\n{'='*60}")
    print(f"Done.  {len(results)} succeeded  |  {len(failed)} failed  |  {skipped} skipped")

    if results:
        summary = pd.DataFrame(results)
        # Put key columns first
        front = ["ticker", "observation_date", "expiry", "dte_at_obs",
                 "stock_price", "data_source"]
        other = [c for c in summary.columns if c not in front]
        summary = summary[front + other]
        summary = summary.sort_values("observation_date")
        summary_path = out_dir / f"{ticker}_history_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Summary → {summary_path}  ({len(summary)} rows)")

        # Print IV column if available
        iv_col = next((c for c in summary.columns
                       if "iv" in c.lower() or "implied" in c.lower()), None)
        if iv_col:
            print(f"\nMonthly ATM IV ({iv_col}):")
            for _, r in summary[["observation_date", "stock_price", iv_col, "data_source"]].iterrows():
                src = "E" if r["data_source"] == "entry_snapshot" else "S"
                try:
                    iv_str = f"{float(str(r[iv_col]).replace('%','')):.1f}%"
                except (ValueError, TypeError):
                    iv_str = str(r[iv_col])
                print(f"  {r['observation_date']}  ${r['stock_price']:7.2f}  {iv_str:6s}  [{src}]")
        print("\n[E]=entry snapshot at obs date, [S]=settlement price at expiry")

    if failed:
        print(f"\nFailed months ({len(failed)}):")
        for obs, exp in failed:
            print(f"  {obs} → expiry {exp}")


if __name__ == "__main__":
    main()
