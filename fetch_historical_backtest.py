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
DOWNLOAD_BASE = (
    "https://www.barchart.com/stocks/quotes/{symbol}/options/download"
    "?expiration={expiry_value}&type=put&moneyness=allRows"
)
LOGIN_URL = "https://www.barchart.com/login"

WAIT  = 20   # seconds to wait for page elements
DELAY = 2.5  # polite pause between downloads


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _monthly_schedule(start: date, end: date) -> list[tuple[date, date]]:
    """
    Generate (observation_date, target_expiry) pairs for every month in
    [start, end].

    observation_date = first business day (Mon-Fri) of the month
    target_expiry    = standard monthly expiry closest to 30 DTE from obs date:
                       - try current month's 3rd Friday if DTE >= 25
                       - otherwise use following month's 3rd Friday
    """
    pairs = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        # First business day of month
        obs = date(y, m, 1)
        while obs.weekday() >= 5:
            obs += timedelta(days=1)

        # Try current month's 3rd Friday first
        expiry = _third_friday(y, m)
        dte    = (expiry - obs).days
        if dte < 25:
            # Roll to next month (gives ~44-50 DTE)
            ny, nm = (y, m + 1) if m < 12 else (y + 1, 1)
            expiry = _third_friday(ny, nm)

        pairs.append((obs, expiry))

        m += 1
        if m > 12:
            m, y = 1, y + 1

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
# Download one historical options chain from Barchart
# ---------------------------------------------------------------------------

def download_historical_chain(
    driver: webdriver.Chrome,
    download_dir: Path,
    ticker: str,
    expiry: date,
    observation_date: date,
) -> tuple[Path | None, str]:
    """
    Try to download the put chain for (ticker, expiry) from Barchart.

    Tries two URLs:
      1. With &tradeDate=YYYY-MM-DD  →  prices as of observation_date (entry snapshot)
      2. Without tradeDate           →  prices at expiry (settlement data)

    Returns (csv_path_or_None, source_label).
    source_label is "entry_snapshot", "settlement", or "failed".
    """
    # Clear stale CSVs
    for f in download_dir.glob("*.csv"):
        f.unlink()

    expiry_value = f"{expiry.isoformat()}-m"   # e.g. "2017-07-21-m"
    base_url = DOWNLOAD_BASE.format(symbol=ticker, expiry_value=expiry_value)

    # ------------------------------------------------------------------
    # Strategy 1: historical snapshot (tradeDate parameter)
    # ------------------------------------------------------------------
    snap_url = base_url + f"&tradeDate={observation_date.isoformat()}"
    driver.get(snap_url)
    result = _wait_for_download(download_dir, timeout=20)
    if result and result.stat().st_size > 200:
        return result, "entry_snapshot"

    # Clear any partial download
    for f in download_dir.glob("*.csv"):
        f.unlink()

    # ------------------------------------------------------------------
    # Strategy 2: settlement / last-traded prices (no date filter)
    # ------------------------------------------------------------------
    driver.get(base_url)
    result = _wait_for_download(download_dir, timeout=20)
    if result and result.stat().st_size > 200:
        return result, "settlement"

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
        driver, download_dir, ticker, expiry, observation_date
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

    # 4. Enrich with observation context
    row["ticker"]           = ticker
    row["observation_date"] = observation_date.isoformat()
    row["expiry"]           = expiry.isoformat()
    row["dte_at_obs"]       = dte
    row["stock_price"]      = price
    row["data_source"]      = source

    # Log IV if present
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
