"""
lambda_vol_premium.py
---------------------
Extracts put volatility premium from options data stored in S3.

S3 source layout:
  s3://{OPTIONS_BUCKET}/optionsData/{TICKER}/{TICKER}/{TICKER}_{start}_{end}.csv

Key fields used:
  - last / entry_premium : put option price at period start
  - iv                   : implied volatility of the put
  - stock_price          : underlying price at observation date
  - strike               : put strike

Output written to:
  s3://{OPTIONS_BUCKET}/vol_premium/vol_premium_summary.json   (full history)
  s3://{OPTIONS_BUCKET}/vol_premium/latest.json               (most recent period per ticker)
"""

import json
import io
import os
import re
import logging
from datetime import datetime, timezone

import boto3
import csv

logger = logging.getLogger()
logger.setLevel(logging.INFO)

OPTIONS_BUCKET = os.environ.get("OPTIONS_BUCKET", "s3bucketmz")
OPTIONS_PREFIX = os.environ.get("OPTIONS_PREFIX", "optionsData/")

s3 = boto3.client("s3")


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def list_tickers(bucket: str, prefix: str) -> list[str]:
    """Return all ticker folder names under prefix."""
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    tickers = []
    for p in resp.get("CommonPrefixes", []):
        # prefix looks like  optionsData/AAPL/
        parts = p["Prefix"].rstrip("/").split("/")
        if parts[-1]:
            tickers.append(parts[-1])
    return tickers


def list_period_files(bucket: str, ticker: str, prefix: str) -> list[str]:
    """Return S3 keys for all period CSVs for a ticker (excludes summary/lock files)."""
    ticker_prefix = f"{prefix}{ticker}/{ticker}/"
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=ticker_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]
            # Skip lock files, summary files, directories
            if filename.startswith(".~lock") or filename.endswith("history_summary.csv"):
                continue
            if filename.endswith(".csv"):
                keys.append(key)
    return keys


def read_csv_from_s3(bucket: str, key: str) -> list[dict]:
    """Download a CSV from S3 and return list of row dicts."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    content = obj["Body"].read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    return list(reader)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_FILENAME_RE = re.compile(r"^(.+?)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv$")


def parse_period_from_key(key: str) -> tuple[str, str] | tuple[None, None]:
    """Extract (period_start, period_end) from a key like .../AAPL_2024-01-22_2024-02-16.csv"""
    filename = key.split("/")[-1]
    m = _FILENAME_RE.match(filename)
    if m:
        return m.group(2), m.group(3)
    return None, None


# ---------------------------------------------------------------------------
# Vol premium extraction
# ---------------------------------------------------------------------------

def safe_float(value) -> float | None:
    """Convert a value to float, returning None on failure."""
    try:
        f = float(value)
        return f if f == f else None  # reject NaN
    except (TypeError, ValueError):
        return None


def extract_premium_record(row: dict, period_start: str, period_end: str) -> dict | None:
    """
    Build a vol-premium record from one CSV row.

    Vol premium metrics:
      premium_pct  = last / stock_price * 100   (option price as % of stock)
      iv           = implied volatility (as decimal, e.g. 0.25 = 25%)
    """
    ticker = row.get("ticker", "").strip()
    if not ticker:
        return None

    # Option price at period start — prefer entry_premium, fall back to last
    last = safe_float(row.get("entry_premium") or row.get("last"))
    stock_price = safe_float(row.get("stock_price") or row.get("price"))
    strike = safe_float(row.get("strike"))
    iv = safe_float(row.get("iv"))

    if last is None or stock_price is None or stock_price == 0:
        return None

    premium_pct = round(last / stock_price * 100, 4)

    record: dict = {
        "ticker": ticker,
        "period_start": period_start,
        "period_end": period_end,
        "strike": strike,
        "stock_price": stock_price,
        "put_price": last,          # option price at period start ("last")
        "premium_pct": premium_pct, # put price as % of stock price
        "iv": iv,                   # implied vol (None if unavailable)
    }

    # Annualised IV premium: iv relative to period length
    if iv is not None:
        try:
            start_dt = datetime.strptime(period_start, "%Y-%m-%d")
            end_dt = datetime.strptime(period_end, "%Y-%m-%d")
            period_days = (end_dt - start_dt).days
            if period_days > 0:
                # Realised period vol expressed as annualised: IV * sqrt(T) / sqrt(T_annual)
                # For now store IV directly; annualised period vol requires price series
                record["period_days"] = period_days
                # IV as % string for readability
                record["iv_pct"] = round(iv * 100, 2) if iv < 10 else round(iv, 2)
        except ValueError:
            pass

    return record


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_all_tickers(bucket: str, prefix: str) -> list[dict]:
    """Scan all ticker folders, read every period CSV, return list of premium records."""
    tickers = list_tickers(bucket, prefix)
    logger.info("Found %d tickers: %s", len(tickers), tickers)

    all_records: list[dict] = []

    for ticker in tickers:
        keys = list_period_files(bucket, ticker, prefix)
        logger.info("  %s: %d period files", ticker, len(keys))

        for key in keys:
            period_start, period_end = parse_period_from_key(key)
            if period_start is None:
                logger.warning("Could not parse dates from key: %s", key)
                continue

            try:
                rows = read_csv_from_s3(bucket, key)
            except Exception as exc:
                logger.error("Failed to read %s: %s", key, exc)
                continue

            for row in rows:
                record = extract_premium_record(row, period_start, period_end)
                if record:
                    all_records.append(record)

    # Sort chronologically
    all_records.sort(key=lambda r: (r["ticker"], r["period_start"]))
    return all_records


def build_latest(records: list[dict]) -> dict[str, dict]:
    """Return the most recent record per ticker."""
    latest: dict[str, dict] = {}
    for rec in records:
        ticker = rec["ticker"]
        if ticker not in latest or rec["period_start"] > latest[ticker]["period_start"]:
            latest[ticker] = rec
    return latest


def write_json_to_s3(bucket: str, key: str, data) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2),
        ContentType="application/json",
    )
    logger.info("Written: s3://%s/%s", bucket, key)


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def lambda_handler(event, context):
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info("Vol premium extraction started at %s", run_ts)

    records = process_all_tickers(OPTIONS_BUCKET, OPTIONS_PREFIX)
    logger.info("Total records extracted: %d", len(records))

    latest = build_latest(records)

    summary = {
        "generated_at": run_ts,
        "total_records": len(records),
        "tickers": sorted(latest.keys()),
        "records": records,
    }

    latest_output = {
        "generated_at": run_ts,
        "tickers": sorted(latest.keys()),
        "latest_per_ticker": latest,
    }

    # Write full history
    write_json_to_s3(OPTIONS_BUCKET, "vol_premium/vol_premium_summary.json", summary)
    # Write latest-only snapshot
    write_json_to_s3(OPTIONS_BUCKET, "vol_premium/latest.json", latest_output)

    return {
        "statusCode": 200,
        "body": {
            "generated_at": run_ts,
            "tickers_processed": len(latest),
            "total_records": len(records),
            "output_keys": [
                "vol_premium/vol_premium_summary.json",
                "vol_premium/latest.json",
            ],
        },
    }
