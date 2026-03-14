"""
AWS Lambda — Sell-Put Monthly Signal Generator
Triggered once per month (e.g. via EventBridge on 1st of each month).

Returns for each of the 50 tickers:
  - signal: "sell_put" | "no_trade"
  - strike: last month close price
  - premium_pct: 4.0
  - expiry: end of current month (ISO date)

Response is also stored in S3 as signals/YYYY-MM.json
"""

import json
import os
import boto3
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

# ------------------------------------------------------------------
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "AMD", "INTC", "CRM",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "USB", "C", "SCHW",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "CVS", "MDT",
    "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW", "PG", "KO",
    "XOM", "CVX", "CAT", "GE", "HON", "LMT", "UPS", "BA", "RTX", "NEE",
]

PREMIUM_PCT = 3.3   # % of notional collected as premium
S3_BUCKET   = os.environ.get("SIGNALS_BUCKET", "sellvol-signals")
WEIGHT_PCT  = round(100.0 / len(TICKERS), 2)   # equal weight %
# ------------------------------------------------------------------


def last_month_range() -> tuple[str, str]:
    """Return (start, end) ISO dates for the previous full calendar month."""
    today      = date.today()
    first_this = today.replace(day=1)
    last_prev  = first_this - timedelta(days=1)
    first_prev = last_prev.replace(day=1)
    return str(first_prev), str(last_prev)


def end_of_current_month() -> str:
    today = date.today()
    next_month = today.replace(day=1) + relativedelta(months=1)
    return str(next_month - timedelta(days=1))


def get_signal(ticker: str, prev_month_start: str, prev_month_end: str) -> dict:
    """
    Download last month's daily data, compute open/close of the month,
    and return the trade signal.
    """
    df = yf.download(ticker, start=prev_month_start, end=prev_month_end,
                     auto_adjust=True, progress=False)

    if df.empty or len(df) < 1:
        return {
            "ticker":      ticker,
            "signal":      "no_data",
            "reason":      "failed to fetch price data",
            "weight_pct":  WEIGHT_PCT,
        }

    month_open  = float(df["Open"].iloc[0])
    month_close = float(df["Close"].iloc[-1])
    candle_up   = month_close > month_open

    if candle_up:
        return {
            "ticker":      ticker,
            "signal":      "sell_put",
            "strike":      round(month_close, 2),
            "premium_pct": PREMIUM_PCT,
            "expiry":      end_of_current_month(),
            "prev_open":   round(month_open,  2),
            "prev_close":  round(month_close, 2),
            "candle_pct":  round((month_close / month_open - 1) * 100, 2),
            "weight_pct":  WEIGHT_PCT,
        }
    else:
        return {
            "ticker":      ticker,
            "signal":      "no_trade",
            "reason":      "last monthly candle was DOWN",
            "prev_open":   round(month_open,  2),
            "prev_close":  round(month_close, 2),
            "candle_pct":  round((month_close / month_open - 1) * 100, 2),
            "weight_pct":  WEIGHT_PCT,
        }


def lambda_handler(event, context):
    prev_start, prev_end = last_month_range()
    run_date = str(date.today())
    month_key = run_date[:7]   # YYYY-MM

    print(f"Generating signals for {month_key} | prev month: {prev_start} → {prev_end}")

    signals = []
    sell_count = 0

    for tkr in TICKERS:
        sig = get_signal(tkr, prev_start, prev_end)
        signals.append(sig)
        if sig["signal"] == "sell_put":
            sell_count += 1

    summary = {
        "run_date":        run_date,
        "signal_month":    month_key,
        "prev_month":      f"{prev_start} to {prev_end}",
        "total_tickers":   len(TICKERS),
        "sell_put_count":  sell_count,
        "no_trade_count":  len(TICKERS) - sell_count,
        "premium_pct":     PREMIUM_PCT,
        "equal_weight_pct": WEIGHT_PCT,
        "signals":         signals,
    }

    # Store result in S3
    try:
        s3 = boto3.client("s3")
        s3_key = f"signals/{month_key}.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(summary, indent=2),
            ContentType="application/json",
        )
        print(f"Saved signals to s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"S3 write failed (continuing): {e}")

    print(f"Signals: {sell_count} sell_put, {len(TICKERS) - sell_count} no_trade")

    return {
        "statusCode": 200,
        "body": json.dumps(summary),
    }


# ------------------------------------------------------------------
# Local test entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    result = lambda_handler({}, None)
    body = json.loads(result["body"])
    print(f"\nSummary: {body['sell_put_count']}/{body['total_tickers']} sell signals")
    for sig in body["signals"]:
        print(f"  {sig['ticker']:6s}  {sig['signal']:10s}  "
              f"candle={sig.get('candle_pct', 'N/A'):>+6}%"
              f"  strike={sig.get('strike', '-')}")
