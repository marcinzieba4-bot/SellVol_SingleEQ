"""
fetch_historical_backtest.py

Build a monthly ATM-put history for a single ticker from 2020-09-01 to today.

For each month:
  observation_date = first business day of that month
  stock_price      = closing price on observation_date  (yfinance)
  target_expiry    = nearest Friday with >= 25 DTE from observation_date
                     (covers weekly and standard monthly expiries)
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

def _next_friday(from_date: date, min_dte: int = 25) -> date:
    """Return the nearest Friday at least min_dte days after from_date."""
    earliest = from_date + timedelta(days=min_dte)
    days_ahead = (4 - earliest.weekday()) % 7   # 4 = Friday
    return earliest + timedelta(days=days_ahead)


def _monthly_schedule(start: date, end: date) -> list[tuple[date, date]]:
    """
    Generate non-overlapping (observation_date, target_expiry) pairs.

    observation_date = start, then each subsequent obs = first business day
                       after the previous expiry (no overlap between trades)
    target_expiry    = nearest Friday with >= 25 DTE from observation date
                       (covers both weekly and standard monthly expiries)
    """
    pairs = []

    # Move start to first business day
    obs = start
    while obs.weekday() >= 5:
        obs += timedelta(days=1)

    while obs <= end:
        expiry = _next_friday(obs, min_dte=25)

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


def _build_bc_symbol(ticker: str, expiry: date, strike: float) -> str:
    """
    Barchart's own pipe-delimited option symbol (confirmed from XHR intercept).
    Format: TICKER|YYYYMMDD|STRIKE.XXP
    e.g.  AAPL|20201002|130.50P
    """
    return f"{ticker}|{expiry.strftime('%Y%m%d')}|{strike:.2f}P"


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


def _fetch_option_price_api(
    driver: webdriver.Chrome,
    ticker: str,
    expiry: date,
    strike: float,
    obs_date: date,
    diag: bool = False,
) -> dict | None:
    """
    Try multiple Barchart API endpoints for one specific put contract.
    Returns an OHLC dict or None.  Set diag=True to print raw responses.
    """
    bc_sym = _build_bc_symbol(ticker, expiry, strike)
    occ    = _build_occ(ticker, expiry, strike)

    # Use a small window around obs_date so minor calendar offsets don't miss the row
    d0 = (obs_date - timedelta(days=3)).isoformat()
    d1 = (obs_date + timedelta(days=1)).isoformat()
    obs_str = obs_date.isoformat()   # "2020-09-01"

    # Barchart returns dates as "m/d/Y" (e.g. "9/1/2020") — build a match set
    obs_date_variants = {
        obs_str,
        obs_date.strftime('%m/%d/%Y'),              # "09/01/2020"
        f"{obs_date.month}/{obs_date.day}/{obs_date.year}",  # "9/1/2020"
        obs_date.strftime('%m/%d/%y'),              # "09/01/20"
        f"{obs_date.month}/{obs_date.day}/{obs_date.year % 100:02d}",  # "9/1/20"
    }

    def _is_valid_row(raw: dict) -> bool:
        """Reject Barchart placeholder rows (epoch-zero date or all N/A prices)."""
        trade_time = str(raw.get('tradeTime', raw.get('date', '')))
        if '01/01/70' in trade_time or '1970' in trade_time:
            return False
        last = raw.get('lastPrice') or raw.get('close') or raw.get('last', '')
        return str(last).strip() not in ('', 'N/A', 'n/a', 'null', 'None')

    def _parse_trade_date(raw: dict) -> date | None:
        """Parse Barchart's tradeTime string to a date, or None on failure."""
        s = str(raw.get('tradeTime', raw.get('date', ''))).strip()
        for fmt in ('%m/%d/%y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
                    '%m/%d/%y', '%m/%d/%Y', '%Y-%m-%d'):
            try:
                return datetime.strptime(s.split()[0], fmt.split()[0]).date()
            except ValueError:
                continue
        return None

    def _rows_for_date(data: dict | None) -> dict | None:
        """Return the valid row whose tradeTime is closest to obs_date (≤ 2 days prior)."""
        if not data or data.get('error'):
            return None
        rows = [r.get('raw', r) for r in data.get('data', [])]
        rows = [r for r in rows if _is_valid_row(r)]
        if not rows:
            return None

        # First pass: exact match on obs_date
        for raw in rows:
            trade_time = str(raw.get('tradeTime', raw.get('date', '')))
            if any(v in trade_time for v in obs_date_variants):
                return raw

        # Second pass: pick the row whose date is closest to obs_date but not after it
        # (handles weekends, T+1 settlement, and sparse data)
        best, best_delta = None, timedelta(days=3)  # accept up to 2 trading days back
        for raw in rows:
            d = _parse_trade_date(raw)
            if d and d <= obs_date:
                delta = obs_date - d
                if delta < best_delta:
                    best, best_delta = raw, delta
        return best

    # Correct field names confirmed from XHR intercept:
    #   openPrice, highPrice, lowPrice, lastPrice  (NOT open/high/low/close)
    bc_fields = "tradeTime,openPrice,highPrice,lowPrice,lastPrice,volume,openInterest"

    endpoints = [
        # PRIMARY — Barchart's actual pipe-symbol format (confirmed from XHR)
        (f"/proxies/core-api/v1/historical/get"
         f"?symbol={bc_sym}&startDate={d0}&endDate={d1}&raw=1"
         f"&fields={bc_fields}"),
        # A — OCC symbol (compact, no spaces)
        (f"/proxies/core-api/v1/historical/get"
         f"?symbol={occ}&startDate={d0}&endDate={d1}&raw=1"
         f"&fields={bc_fields}"),
        # B — space-padded OCC (OSI 6-char root form)
        (f"/proxies/core-api/v1/historical/get"
         f"?symbol={ticker:<6s}{expiry.strftime('%y%m%d')}P{int(round(strike*1000)):08d}"
         f"&startDate={d0}&endDate={d1}&raw=1&fields={bc_fields}"),
        # C — options-specific historical endpoint with separate params
        (f"/proxies/core-api/v1/options/historical.json"
         f"?symbol={ticker}&expirationDate={expiry.isoformat()}"
         f"&symbolType=P&strikePrice={strike:.2f}"
         f"&startDate={d0}&endDate={d1}&raw=1"),
    ]

    for i, url in enumerate(endpoints):
        data = _bc_fetch(driver, url)
        if diag:
            preview = str(data)[:180] if data else 'None'
            print(f"\n    [diag-{chr(65+i)}] {url[url.find('?')-20:url.find('?')+60]}... → {preview}")
        row = _rows_for_date(data)
        if row:
            return row

    # ── Wide-window retry ────────────────────────────────────────────────────
    # All narrow-window queries returned count:0, total:N — the symbol exists
    # but Barchart only stored data near expiry (common for 2020–2021 options).
    # Fetch the full lifetime of the contract and pick the closest valid row.
    wide_start = (expiry - timedelta(days=90)).isoformat()
    wide_end   = expiry.isoformat()
    wide_url   = (f"/proxies/core-api/v1/historical/get"
                  f"?symbol={bc_sym}&startDate={wide_start}&endDate={wide_end}"
                  f"&raw=1&fields={bc_fields}")
    wide_data  = _bc_fetch(driver, wide_url)
    if diag:
        preview = str(wide_data)[:180] if wide_data else 'None'
        print(f"\n    [diag-wide] {wide_url[wide_url.find('?')-20:wide_url.find('?')+60]}... → {preview}")
    row = _rows_for_date(wide_data)
    if row:
        d_found = _parse_trade_date(row)
        if diag:
            print(f"    [diag-wide] accepted row dated {d_found} for obs_date {obs_date}")
        return row

    return None


def _scrape_price_history_page(
    driver: webdriver.Chrome,
    ticker: str,
    expiry: date,
    strike: float,
    obs_date: date,
    download_dir: Path,
) -> dict | None:
    """
    Navigate to Barchart's price-history page for a specific put contract
    and extract the OHLC row for obs_date by scraping the rendered table.

    This is the guaranteed fallback — the page the user confirmed shows
    historical data at:
      barchart.com/options/price-history?symbol=AAPL&expirationDate=...
      &symbolType=P&strikePrice=130.00
    """
    page_url = (
        f"https://www.barchart.com/options/price-history"
        f"?symbol={ticker}"
        f"&expirationDate={expiry.isoformat()}"
        f"&symbolType=P"
        f"&strikePrice={strike:.2f}"
    )
    driver.get(page_url)
    _dismiss_cmp_overlay(driver)

    # Wait for the data table to appear (up to 8 s)
    for _ in range(16):
        time.sleep(0.5)
        try:
            tbl = driver.find_element(By.CSS_SELECTOR, 'table')
            if tbl:
                break
        except Exception:
            pass

    # Try to intercept the XHR Barchart made while loading the page
    # (captures whatever API endpoint it actually uses)
    try:
        xhrdata = driver.execute_script("""
            var entries = performance.getEntriesByType('resource');
            for (var e of entries) {
                if (e.name.includes('proxies') || e.name.includes('historical')) {
                    return e.name;
                }
            }
            return null;
        """)
        if xhrdata:
            print(f"\n    [diag-page-xhr] {xhrdata[:200]}")
    except Exception:
        pass

    # Date variants Barchart might display (no %-m — not portable on Windows)
    obs_variants = {
        obs_date.strftime('%m/%d/%y'),    # 09/01/20
        obs_date.strftime('%m/%d/%Y'),    # 09/01/2020
        obs_date.isoformat(),             # 2020-09-01
        f"{obs_date.month}/{obs_date.day}/{obs_date.year % 100:02d}",  # 9/1/20
        f"{obs_date.month}/{obs_date.day}/{obs_date.year}",            # 9/1/2020
    }

    # Scrape all table rows, find the one matching obs_date
    try:
        raw_rows = driver.execute_script("""
            var result = [];
            var rows = document.querySelectorAll('table tr');
            for (var r of rows) {
                var cells = r.querySelectorAll('td');
                if (cells.length >= 4) {
                    result.push(Array.from(cells).map(function(c){return c.textContent.trim();}));
                }
            }
            return result;
        """)
        if raw_rows:
            for row in raw_rows:
                if any(v in row[0] for v in obs_variants):
                    # Typical column order: Date, Open, High, Low, Last, Volume, OI
                    def _f(x):
                        try:
                            return float(str(x).replace(',', ''))
                        except Exception:
                            return None
                    return {
                        'close':         _f(row[4]) if len(row) > 4 else _f(row[-1]),
                        'open':          _f(row[1]) if len(row) > 1 else None,
                        'high':          _f(row[2]) if len(row) > 2 else None,
                        'low':           _f(row[3]) if len(row) > 3 else None,
                        'volume':        _f(row[5]) if len(row) > 5 else None,
                        'openInterest':  _f(row[6]) if len(row) > 6 else None,
                    }
            # If date not found, log the first few rows so we can see the format
            print(f"\n    [diag-page] table has {len(raw_rows)} rows; "
                  f"first row[0]={raw_rows[0][0]!r} (looking for {obs_variants})")
    except Exception as e:
        print(f"\n    [diag-page] scrape error: {e}")

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

    Phase 1 (fast): try multiple Barchart API endpoints for each candidate
                    strike — no page navigation required.
    Phase 2 (slow): navigate to the actual price-history page and scrape
                    the rendered table for the top-5 closest strikes.

    Returns (csv_path, "entry_snapshot") or (None, "failed").
    """
    # Ensure we are on a barchart.com page so relative fetch() URLs work
    if 'barchart.com' not in driver.current_url:
        driver.get(OPTIONS_PAGE.format(symbol=ticker))
        time.sleep(2)

    for f in download_dir.glob("*.csv"):
        f.unlink()

    candidates = _candidate_strikes(stock_price)

    def _save(strike: float, row: dict) -> tuple[Path, str]:
        occ = _build_occ(ticker, expiry, strike)
        # Accept both naming conventions (Barchart XHR uses *Price suffixes)
        df = pd.DataFrame([{
            'Symbol':        occ,
            'Strike':        strike,
            'Last':          row.get('lastPrice') or row.get('close') or row.get('last'),
            'Open':          row.get('openPrice') or row.get('open'),
            'High':          row.get('highPrice') or row.get('high'),
            'Low':           row.get('lowPrice')  or row.get('low'),
            'Volume':        row.get('volume'),
            'Open Interest': row.get('openInterest'),
            'IV':            row.get('impliedVolatility') or row.get('volatility'),
        }])
        out = download_dir / f'{ticker}_{expiry}_{observation_date}_hist.csv'
        df.to_csv(out, index=False)
        return out, "entry_snapshot"

    # ── Phase 1: API (fast, no navigation) ──────────────────────────────
    for i, strike in enumerate(candidates):
        # Print full diagnostics only for the single closest candidate
        row = _fetch_option_price_api(
            driver, ticker, expiry, strike, observation_date, diag=(i == 0)
        )
        if row:
            return _save(strike, row)

    # ── Phase 2: page scrape (guaranteed if page has data) ───────────────
    print(f"\n    [diag] Phase-1 API failed — scraping price-history page")
    for strike in candidates[:5]:
        row = _scrape_price_history_page(
            driver, ticker, expiry, strike, observation_date, download_dir
        )
        if row:
            return _save(strike, row)

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
    parser.add_argument("--start",       default="2020-09-01", help="Start date YYYY-MM-DD")
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
