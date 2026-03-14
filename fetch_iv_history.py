"""
Fetch historical 30-day ATM implied volatility from AlphaQuery.com
for all 50 backtest stocks, then save to iv_history.csv.

Run this locally (not in a restricted environment).
AlphaQuery serves HTML tables with ~5 years of daily IV history.

Usage:
    pip install requests beautifulsoup4 pandas
    python fetch_iv_history.py

Output: iv_history.csv  (columns: date, ticker, iv_30d)
"""

import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "AMD", "INTC", "CRM",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "USB", "C", "SCHW",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "CVS", "MDT",
    "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW", "PG", "KO",
    "XOM", "CVX", "CAT", "GE", "HON", "LMT", "UPS", "BA", "RTX", "NEE",
]

BASE_URL = "https://www.alphaquery.com/stock/{ticker}/volatility-option-statistics/30-day/iv-mean"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.alphaquery.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def parse_iv_table(html: str, ticker: str) -> pd.DataFrame:
    """
    Parse AlphaQuery HTML page and extract the historical IV table.
    Returns DataFrame with columns [date, ticker, iv_30d].
    """
    soup = BeautifulSoup(html, "html.parser")

    # Find all tables — AlphaQuery puts the history in the first/main table
    tables = soup.find_all("table")
    if not tables:
        return pd.DataFrame()

    records = []
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            date_text = cells[0].text.strip()
            iv_text   = cells[1].text.strip().replace("%", "")
            try:
                date = pd.to_datetime(date_text).date()
                iv   = float(iv_text)
                # AlphaQuery shows IV as decimal (e.g. 0.27) or percent (27.0)
                # normalise to decimal fraction
                if iv > 2:
                    iv /= 100.0
                records.append({"date": str(date), "ticker": ticker, "iv_30d": iv})
            except (ValueError, TypeError):
                continue

    return pd.DataFrame(records)


def fetch_ticker(session: requests.Session, ticker: str, retries: int = 3) -> pd.DataFrame:
    url = BASE_URL.format(ticker=ticker)
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                df = parse_iv_table(r.text, ticker)
                if not df.empty:
                    print(f"  {ticker:6s}  {len(df):4d} rows  "
                          f"{df['date'].min()} → {df['date'].max()}  "
                          f"avg_iv={df['iv_30d'].mean()*100:.1f}%")
                    return df
                else:
                    print(f"  {ticker:6s}  parsed 0 rows (page structure may have changed)")
                    return pd.DataFrame()
            else:
                print(f"  {ticker:6s}  HTTP {r.status_code} (attempt {attempt+1})")
        except Exception as e:
            print(f"  {ticker:6s}  ERROR: {e} (attempt {attempt+1})")
        time.sleep(2 ** attempt)   # 1s, 2s, 4s backoff
    return pd.DataFrame()


def main():
    print(f"Fetching 30-day ATM IV history from AlphaQuery for {len(TICKERS)} stocks")
    print("URL: " + BASE_URL.format(ticker="TICKER") + "\n")

    session = requests.Session()
    all_dfs = []

    for i, tkr in enumerate(TICKERS):
        df = fetch_ticker(session, tkr)
        if not df.empty:
            all_dfs.append(df)
        # Polite delay: 1.5s between requests to avoid rate-limiting
        if i < len(TICKERS) - 1:
            time.sleep(1.5)

    if not all_dfs:
        print("\nNo data fetched. Check your network / AlphaQuery access.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["ticker", "date"])

    out_file = "iv_history.csv"
    combined.to_csv(out_file, index=False)

    print(f"\n{'='*50}")
    print(f"Saved {len(combined):,} rows to {out_file}")
    print(f"Date range: {combined['date'].min().date()} → {combined['date'].max().date()}")
    print(f"Tickers: {combined['ticker'].nunique()}")
    print(f"\nPer-stock avg IV:")
    avg = combined.groupby("ticker")["iv_30d"].mean().sort_values(ascending=False)
    for tkr, iv in avg.items():
        print(f"  {tkr:6s}  {iv*100:.1f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
