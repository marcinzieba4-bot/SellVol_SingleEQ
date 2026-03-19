"""
fetch_options_barchart.py — Download ATM put data from Barchart.com (premium)

For each of the 50 strategy stocks this script:
  1. Logs in with your Barchart premium account
  2. Gets the current stock price
  3. Selects the front monthly expiry (nearest standard monthly with >= 14 DTE)
  4. Downloads the options chain CSV for that expiry
  5. Extracts the single ATM put (strike closest to current price)
  6. Saves per-ticker CSV + a combined summary

Usage:
    pip install selenium pandas
    export BARCHART_USER=your@email.com
    export BARCHART_PASS=yourpassword
    python fetch_options_barchart.py

    # Specific tickers only:
    python fetch_options_barchart.py --tickers AAPL NVDA MSFT

    # Show browser window (useful for debugging):
    python fetch_options_barchart.py --show-browser

Output:
    options_data/YYYY-MM-DD/
        AAPL_2025-03-21.csv       ← ATM put row for AAPL
        NVDA_2025-03-21.csv       ← ATM put row for NVDA
        ...
        summary_2025-01-17.csv    ← all ATM puts combined (iv_30d compatible)
"""

import os
import sys
import math
import time
import shutil
import argparse
import calendar
import tempfile
import pandas as pd
from datetime import datetime, date, timedelta
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
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "AMD", "INTC", "CRM",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "USB", "C", "SCHW",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "CVS", "MDT",
    "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW", "PG", "KO",
    "XOM", "CVX", "CAT", "GE", "HON", "LMT", "UPS", "BA", "RTX", "NEE",
]

LOGIN_URL   = "https://www.barchart.com/login"
OPTIONS_URL = "https://www.barchart.com/stocks/quotes/{symbol}/options"

MIN_DTE     = 14    # minimum days-to-expiry; if less, roll to next monthly
WAIT        = 15    # max seconds to wait for any element
DELAY       = 3.0   # polite pause between tickers (seconds)

# ---------------------------------------------------------------------------
# Expiry helpers
# ---------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of the given month (standard monthly expiry)."""
    # Find the first Friday
    first_day = date(year, month, 1)
    first_fri = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    return first_fri + timedelta(weeks=2)


def _all_monthly_expiries(start: date, n_months: int = 4) -> list[date]:
    """Return the next n_months standard monthly expiry dates from start."""
    expiries = []
    y, m = start.year, start.month
    for _ in range(n_months):
        expiries.append(_third_friday(y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return expiries


def pick_front_monthly(available_dates: list[date], today: date) -> date | None:
    """
    From available_dates, pick the nearest monthly expiry with >= MIN_DTE days left.
    Falls back to any expiry >= MIN_DTE if no standard monthly found.
    """
    candidates = [d for d in available_dates if (d - today).days >= MIN_DTE]
    if not candidates:
        return None

    standard = _all_monthly_expiries(today, n_months=4)

    # Prefer a date that matches a standard monthly expiry
    for exp in standard:
        if exp in candidates:
            return exp

    # If no exact match (e.g. weekly-heavy broker), take the nearest candidate
    return min(candidates, key=lambda d: abs((d - today).days - 30))


# ---------------------------------------------------------------------------
# Driver setup
# ---------------------------------------------------------------------------

def setup_driver(download_dir: str, headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    # Tell Chrome where to save downloaded files
    prefs = {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    opts.add_experimental_option("prefs", prefs)

    # Selenium 4.6+ includes its own driver manager — no webdriver-manager needed
    return webdriver.Chrome(options=opts)


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def _dismiss_consent_overlay(driver: webdriver.Chrome) -> None:
    """Dismiss cookie/GDPR consent banners that block clicks (e.g. #cmpwrapper)."""
    # Try clicking known accept buttons inside consent iframes / overlays
    for sel in [
        '#cmpwrapper button[title*="Accept"]',
        '#cmpwrapper button[class*="accept"]',
        '#cmpwrapper button',
        'button[title*="Accept"]',
        'button[id*="accept"]',
        'button[class*="accept-all"]',
        '[aria-label*="Accept"]',
    ]:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, sel)
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(0.5)
            return
        except NoSuchElementException:
            continue

    # If no button found, just hide the overlay via JS so it stops intercepting clicks
    driver.execute_script("""
        const el = document.getElementById('cmpwrapper');
        if (el) el.style.display = 'none';
    """)


def login(driver: webdriver.Chrome, username: str, password: str) -> None:
    print("  Logging in to Barchart …")
    driver.get(LOGIN_URL)
    wait = WebDriverWait(driver, WAIT)

    # Dismiss any consent/cookie overlay before interacting with the form
    time.sleep(1)
    _dismiss_consent_overlay(driver)

    # Email field — try multiple selectors
    for sel in ['input[name="email"]', 'input[type="email"]', '#email']:
        try:
            el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
            el.clear()
            el.send_keys(username)
            break
        except TimeoutException:
            continue
    else:
        raise RuntimeError("Could not find email input on Barchart login page.")

    # Password field
    for sel in ['input[name="password"]', 'input[type="password"]', '#password']:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            el.clear()
            el.send_keys(password)
            break
        except NoSuchElementException:
            continue
    else:
        raise RuntimeError("Could not find password input on Barchart login page.")

    # Submit button — use JS click to bypass any remaining overlay interception
    for sel in ['button[type="submit"]', 'input[type="submit"]', 'button.login-button', 'button.login']:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, sel)
            try:
                btn.click()
            except (ElementNotInteractableException, ElementClickInterceptedException):
                driver.execute_script("arguments[0].click();", btn)
            break
        except NoSuchElementException:
            continue
    else:
        raise RuntimeError("Could not find submit button on Barchart login page.")

    # Wait until we're no longer on the login page
    try:
        wait.until(EC.url_changes(LOGIN_URL))
    except TimeoutException:
        pass  # some flows redirect to same URL with params

    time.sleep(2)

    if "login" in driver.current_url.lower():
        raise RuntimeError(
            "Still on login page after submitting credentials. "
            "Check BARCHART_USER / BARCHART_PASS and that your account is active."
        )
    print("  Logged in successfully.")


# ---------------------------------------------------------------------------
# Get current stock price
# ---------------------------------------------------------------------------

def get_current_price(driver: webdriver.Chrome, ticker: str) -> float:
    """Navigate to the stock quote page and extract the last price."""
    url = f"https://www.barchart.com/stocks/quotes/{ticker}/overview"
    driver.get(url)
    wait = WebDriverWait(driver, WAIT)

    price_selectors = [
        'span[data-ng-bind="quote.lastPrice | number:2"]',
        'span.price-value',
        'div.price-last span',
        '[class*="last-price"]',
        '[data-field="lastPrice"]',
    ]
    for sel in price_selectors:
        try:
            el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
            text = el.text.replace(",", "").replace("$", "").strip()
            if text:
                return float(text)
        except (TimeoutException, ValueError):
            continue

    # Fallback: look for any element whose text looks like a price
    try:
        elements = driver.find_elements(By.XPATH,
            '//*[contains(@class,"last") or contains(@class,"price")]')
        for el in elements:
            t = el.text.replace(",", "").replace("$", "").strip()
            try:
                v = float(t)
                if 1 < v < 100_000:
                    return v
            except ValueError:
                continue
    except Exception:
        pass

    raise RuntimeError(f"Could not parse current price for {ticker}")


# ---------------------------------------------------------------------------
# Get available expiry dates from the options page dropdown
# ---------------------------------------------------------------------------

def get_available_expiries(driver: webdriver.Chrome, ticker: str) -> list[date]:
    """
    Navigate to the options page and read the expiry dropdown.
    Returns a sorted list of date objects.

    Barchart uses an AngularJS select: <select data-ng-model="selected">
    Options have value="2026-04-17-m" (monthly) or "2026-03-18-w" (weekly).
    The visible text is "2026-04-17 (m)" — NOT parseable directly by pd.to_datetime.
    Always parse from the value attribute, taking the first 10 characters (YYYY-MM-DD).
    """
    url = OPTIONS_URL.format(symbol=ticker)
    driver.get(url)
    wait = WebDriverWait(driver, WAIT)

    expiries: list[date] = []

    # Strategy 1: Barchart's AngularJS select (confirmed selector from live page)
    for sel in [
        'select[data-ng-model="selected"]',   # confirmed from page HTML
        'select[aria-label*="expir"]',         # aria-label="set expiration date"
        'select[data-ng-model]',               # any ng-model select
    ]:
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
            # Wait for Angular to inject <option> elements (async rendering)
            WebDriverWait(driver, WAIT).until(
                lambda d, s=sel: len(Select(d.find_element(By.CSS_SELECTOR, s)).options) > 1
            )
            select = Select(driver.find_element(By.CSS_SELECTOR, sel))
            for opt in select.options:
                # value="2026-04-17-m"  →  take first 10 chars = "2026-04-17"
                val = opt.get_attribute("value") or ""
                date_str = val[:10]
                try:
                    expiries.append(pd.to_datetime(date_str).date())
                except (ValueError, TypeError):
                    continue
            if expiries:
                break
        except (TimeoutException, NoSuchElementException):
            continue

    # Strategy 2: JavaScript — read value attributes directly (bypasses text parsing)
    if not expiries:
        try:
            raw = driver.execute_script("""
                const dates = [];
                document.querySelectorAll('select[data-ng-model] option, select option').forEach(opt => {
                    const v = (opt.value || '').trim();
                    if (v.length >= 10) dates.push(v.substring(0, 10));
                });
                return dates;
            """)
            for date_str in (raw or []):
                try:
                    expiries.append(pd.to_datetime(date_str).date())
                except (ValueError, TypeError):
                    continue
        except Exception:
            pass

    return sorted(set(expiries))


# ---------------------------------------------------------------------------
# Select the chosen expiry in the dropdown / UI
# ---------------------------------------------------------------------------

def select_expiry(driver: webdriver.Chrome, expiry: date) -> str:
    """
    Select the given expiry in the Barchart options page dropdown.
    Returns the full option value string (e.g. "2026-04-17-m") on success,
    or the bare ISO date string if selection failed.
    """
    expiry_prefix = expiry.isoformat()  # "2026-04-17"

    # Barchart option values: "2026-04-17-m" or "2026-03-18-w"
    # Match by value prefix so we handle both monthly and weekly suffixes.
    for sel in [
        'select[data-ng-model="selected"]',
        'select[aria-label*="expir"]',
        'select[data-ng-model]',
    ]:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            select = Select(el)
            for opt in select.options:
                val = opt.get_attribute("value") or ""
                if val.startswith(expiry_prefix):
                    select.select_by_value(val)
                    time.sleep(1.5)  # let the chain reload
                    return val          # e.g. "2026-04-17-m"
        except (NoSuchElementException, ElementNotInteractableException):
            continue

    print(f"    WARNING: Could not select expiry {expiry_prefix} in UI — using default")
    return expiry_prefix


# ---------------------------------------------------------------------------
# Download the CSV
# ---------------------------------------------------------------------------

def _dismiss_cmp_overlay(driver: webdriver.Chrome) -> None:
    """
    Dismiss Barchart's GDPR/cookie consent overlay (CMP wrapper).
    The overlay intercepts clicks when present; we must remove it first.
    """
    try:
        # Hide via JS — works regardless of which button/text the CMP shows
        driver.execute_script("""
            const ids = ['cmpwrapper', 'cmp-wrapper', 'gdpr-banner', 'cookie-banner',
                         'consent-banner', 'onetrust-banner-sdk'];
            ids.forEach(id => {
                const el = document.getElementById(id);
                if (el) el.style.display = 'none';
            });
            // Also hide by common class patterns
            document.querySelectorAll(
                '[class*="cmp"], [class*="gdpr"], [class*="cookie-consent"], ' +
                '[class*="consent-modal"], [id*="cmp"]'
            ).forEach(el => { el.style.display = 'none'; });
        """)
    except Exception:
        pass

    # Also try clicking an explicit Accept / Close button if visible
    for sel in [
        'button#cmpbntyestxt',
        'button[title*="Accept"]',
        'button[aria-label*="Accept"]',
        'button[aria-label*="Close"]',
        '.cmp-intro_acceptAll',
        '#cmp-btn-accept',
        'button.accept',
    ]:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, sel)
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(0.5)
            break
        except (NoSuchElementException, ElementNotInteractableException):
            continue

def _wait_for_download(download_dir: Path, timeout: int = 30) -> Path | None:
    """Poll download_dir until a new .csv file appears (not still downloading)."""
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


def download_options_csv(
    driver: webdriver.Chrome,
    download_dir: Path,
    ticker: str = "",
    expiry: date | None = None,
    expiry_value: str = "",
) -> Path | None:
    """Download the options CSV for the given ticker/expiry."""

    # Clear any existing CSVs in download_dir so we can detect the new one
    for f in download_dir.glob("*.csv"):
        f.unlink()

    # ------------------------------------------------------------------ #
    # Strategy 1: direct download URL — most reliable when logged in.     #
    # Barchart honours session cookies; no JS rendering issues.            #
    # ------------------------------------------------------------------ #
    if ticker and expiry:
        expiry_param = expiry_value if expiry_value else expiry.strftime("%Y-%m-%d")
        dl_url = (
            f"https://www.barchart.com/stocks/quotes/{ticker}"
            f"/options/download?expiration={expiry_param}&type=put&moneyness=allRows"
        )
        print(f"    Downloading via direct URL (expiry={expiry_param}) …")
        driver.get(dl_url)
        result = _wait_for_download(download_dir, timeout=30)
        if result:
            return result
        print(f"    Direct URL produced no file — trying UI button …")

    # ------------------------------------------------------------------ #
    # Strategy 2: click the Download button in the page toolbar.          #
    # Must be back on the options page; re-navigate if needed.            #
    # ------------------------------------------------------------------ #
    if ticker and expiry:
        options_url = OPTIONS_URL.format(symbol=ticker)
        if driver.current_url.rstrip("/").split("?")[0] != options_url.rstrip("/"):
            driver.get(options_url)
            time.sleep(2)
            # Re-select the expiry after navigating back
            try:
                select_expiry(driver, expiry)
            except Exception:
                pass

    # Dismiss any consent overlay before clicking
    _dismiss_cmp_overlay(driver)

    def _click_el(el) -> bool:
        """Try native click, then JS click. Returns True if click fired."""
        try:
            el.click()
            return True
        except (ElementNotInteractableException, ElementClickInterceptedException):
            pass
        try:
            driver.execute_script("arguments[0].click();", el)
            return True
        except Exception:
            return False

    download_clicked = False

    # Confirmed from live page HTML: <a class="toolbar-button download" data-bc-download-button="...">
    for sel in [
        'a.toolbar-button.download',
        'a[data-bc-download-button]',
        'button#download',
        'button.download',
        'a#download',
        'a.download',
        'button[class*="download"]',
        'a[class*="download"]',
        'a[href*="download"]',
        'a[href*=".csv"]',
    ]:
        for el in driver.find_elements(By.CSS_SELECTOR, sel):
            if _click_el(el):
                download_clicked = True
                break
        if download_clicked:
            break

    if not download_clicked:
        for el in driver.find_elements(By.XPATH, '//*[contains(text(),"Download")]'):
            if _click_el(el):
                download_clicked = True
                break

    if download_clicked:
        result = _wait_for_download(download_dir, timeout=30)
        if result:
            return result
        print("    Button clicked but no file appeared in download dir")
        print(f"    (download_dir={download_dir})")

    print("    WARNING: All download strategies failed")
    return None


# ---------------------------------------------------------------------------
# Parse downloaded CSV and extract ATM put
# ---------------------------------------------------------------------------

def extract_atm_put(
    csv_path: Path,
    current_price: float,
    ticker: str,
    expiry: date,
) -> pd.DataFrame | None:
    """
    Load the Barchart options CSV and return a 1-row DataFrame with the
    ATM put (strike closest to current underlying price).

    The underlying price is read directly from the CSV's "Price~" column
    when present — this is Barchart's own authoritative stock price and is
    always correct.  The `current_price` argument is only used as a fallback
    when "Price~" is absent (e.g. in legacy / hand-built CSVs).
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"    ERROR reading CSV: {e}")
        return None

    if df.empty:
        return None

    # Normalise column names: lowercase, strip spaces
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ── Underlying price ─────────────────────────────────────────────────────
    # Barchart CSVs include a "Price~" column (normalised to "price~") which
    # is the underlying stock price as recorded by Barchart.  Use it when
    # available; it is always correct and avoids any yfinance adjustment issues.
    price_col = next(
        (c for c in df.columns if c.rstrip("~") in ("price", "underlying", "stock_price")),
        None,
    )
    if price_col:
        try:
            p = pd.to_numeric(
                df[price_col].astype(str).str.replace(",", "").str.replace("$", ""),
                errors="coerce",
            ).dropna()
            if not p.empty and float(p.iloc[0]) > 0:
                current_price = float(p.iloc[0])
        except Exception:
            pass  # fall back to the passed-in current_price

    # Identify strike column
    strike_col = next(
        (c for c in df.columns if "strike" in c), None
    )
    if strike_col is None:
        print(f"    WARNING: No strike column found. Columns: {list(df.columns)}")
        return None

    # Filter to puts — try multiple strategies in order of reliability

    # Strategy A: explicit type/call-put column
    type_col = next(
        (c for c in df.columns if c in ("type", "option_type", "call/put", "putcall")), None
    )
    if type_col:
        put_mask = df[type_col].str.strip().str.lower().str.startswith("p")
        df = df[put_mask]

    # Strategy B: option symbol encodes type — OCC format TICKER+YYMMDD+P+strike
    # e.g. "NVDA250221P00138000"  (P = put, C = call)
    if df.empty or type_col is None:
        sym_col = next(
            (c for c in df.columns if c in ("symbol", "contract", "option_symbol", "ticker")), None
        )
        if sym_col is not None:
            # Look for a capital P (or lowercase p) immediately after the date digits
            put_mask = df[sym_col].astype(str).str.contains(r'\d{6}[Pp]', regex=True)
            if put_mask.any():
                df = df[put_mask]

    if df.empty:
        print("    WARNING: No put rows found in CSV")
        return None

    # Convert strike to numeric
    df[strike_col] = pd.to_numeric(
        df[strike_col].astype(str).str.replace(",", "").str.replace("$", ""),
        errors="coerce",
    )
    df = df.dropna(subset=[strike_col])

    # Find closest strike to current price
    df["_dist"] = (df[strike_col] - current_price).abs()
    atm_row = df.loc[df["_dist"].idxmin()].drop("_dist")

    result = atm_row.to_frame().T.copy()
    result.insert(0, "ticker", ticker)
    result.insert(1, "price", current_price)
    result.insert(2, "expiry", expiry.isoformat())
    result.insert(3, "dte", (expiry - date.today()).days)

    return result


# ---------------------------------------------------------------------------
# Process one ticker
# ---------------------------------------------------------------------------

def process_ticker(
    driver: webdriver.Chrome,
    ticker: str,
    output_dir: Path,
    download_dir: Path,
    today: date,
) -> pd.DataFrame | None:
    print(f"\n{'─'*55}")
    print(f"  {ticker}")

    # 1. Current price
    try:
        price = get_current_price(driver, ticker)
        print(f"    price = ${price:,.2f}")
    except Exception as e:
        print(f"    ERROR getting price: {e}")
        return None

    # 2. Available expiries
    try:
        available = get_available_expiries(driver, ticker)
        if not available:
            print("    ERROR: No expiries found on options page")
            return None
        print(f"    Found {len(available)} expiry dates: "
              f"{available[0]} … {available[-1]}")
    except Exception as e:
        print(f"    ERROR getting expiries: {e}")
        return None

    # 3. Choose front monthly
    expiry = pick_front_monthly(available, today)
    if expiry is None:
        print(f"    ERROR: No expiry with >= {MIN_DTE} DTE found")
        return None
    dte = (expiry - today).days
    print(f"    Selected expiry: {expiry}  ({dte} DTE)")

    # 4. Select expiry in UI (driver is already on options page)
    expiry_value = expiry.isoformat()  # fallback; select_expiry returns full value e.g. "2026-04-17-m"
    try:
        expiry_value = select_expiry(driver, expiry)
    except Exception as e:
        print(f"    WARNING: expiry selection failed: {e}")

    # 5. Download CSV
    csv_path = download_options_csv(driver, download_dir, ticker=ticker, expiry=expiry,
                                    expiry_value=expiry_value)
    if csv_path is None:
        print("    ERROR: Download failed or timed out")
        return None
    print(f"    Downloaded: {csv_path.name}  ({csv_path.stat().st_size:,} bytes)")

    # 6. Extract ATM put
    atm = extract_atm_put(csv_path, price, ticker, expiry)
    if atm is None:
        return None

    # Find IV column to log
    iv_col = next((c for c in atm.columns if "iv" in c or "implied" in c), None)
    if iv_col:
        iv_val = atm[iv_col].iloc[0]
        try:
            print(f"    ATM put  strike={atm.iloc[0].get('strike', atm.iloc[0].get('strikeprice','?'))}  "
                  f"IV={float(str(iv_val).replace('%','')):.1f}%")
        except (ValueError, TypeError):
            pass

    # Save per-ticker CSV
    out_file = output_dir / f"{ticker}_{expiry}.csv"
    atm.to_csv(out_file, index=False)

    return atm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download ATM put data from Barchart")
    parser.add_argument(
        "--tickers", nargs="+", default=TICKERS,
        help="Space-separated list of tickers (default: all 50)",
    )
    parser.add_argument(
        "--show-browser", action="store_true",
        help="Show Chrome window (useful for debugging)",
    )
    parser.add_argument(
        "--out-dir", default="options_data",
        help="Root output directory (default: options_data/)",
    )
    args = parser.parse_args()

    username = os.getenv("BARCHART_USER", "")
    password = os.getenv("BARCHART_PASS", "")
    if not username or not password:
        print(
            "ERROR: Set environment variables BARCHART_USER and BARCHART_PASS\n"
            "  export BARCHART_USER=your@email.com\n"
            "  export BARCHART_PASS=yourpassword"
        )
        sys.exit(1)

    today = date.today()
    run_dir = Path(args.out_dir) / today.isoformat()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Temp dir for raw downloads (Barchart names them automatically)
    download_dir = Path(tempfile.mkdtemp(prefix="barchart_dl_"))

    print(f"Barchart ATM Put Downloader")
    print(f"Tickers : {len(args.tickers)}")
    print(f"Output  : {run_dir}")
    print(f"Date    : {today}")
    print(f"Browser : {'visible' if args.show_browser else 'headless'}\n")

    driver = setup_driver(str(download_dir), headless=not args.show_browser)
    results: list[pd.DataFrame] = []

    try:
        login(driver, username, password)

        for i, ticker in enumerate(args.tickers):
            try:
                row = process_ticker(driver, ticker, run_dir, download_dir, today)
                if row is not None:
                    results.append(row)
            except Exception as e:
                print(f"  FAILED {ticker}: {e}")

            if i < len(args.tickers) - 1:
                time.sleep(DELAY)

    finally:
        driver.quit()
        shutil.rmtree(download_dir, ignore_errors=True)

    # Summary CSV
    if results:
        summary = pd.concat(results, ignore_index=True)
        summary_path = run_dir / f"summary_{today}.csv"
        summary.to_csv(summary_path, index=False)

        print(f"\n{'='*55}")
        print(f"Done. {len(results)}/{len(args.tickers)} tickers succeeded.")
        print(f"Summary → {summary_path}")

        # Print IV table if column found
        iv_col = next((c for c in summary.columns if "iv" in c or "implied" in c), None)
        if iv_col and "ticker" in summary.columns:
            print("\nATM put IVs:")
            for _, r in summary[["ticker", iv_col]].iterrows():
                print(f"  {r['ticker']:6s}  {r[iv_col]}")
    else:
        print("\nNo data collected. Run with --show-browser to debug.")


if __name__ == "__main__":
    main()
