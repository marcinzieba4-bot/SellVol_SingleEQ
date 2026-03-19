"""
strategy_vol_premium.py
-----------------------
Put-selling strategy with 4-week momentum filter.

Signal rule:
  At period_start: if stock_price > stock_price 4 weeks earlier → SELL PUT.
  Otherwise → NO TRADE.

Price source:
  All stock prices come from the S3 options CSV files (consistent pre/post-split
  within the dataset). Yahoo Finance is NOT used — it returns split-adjusted
  prices that would be incompatible with pre-split strikes in older S3 records.

  • Signal  : current period stock_price vs previous period stock_price
              (periods are ~4 weeks apart, so this IS the 4-week comparison)
  • Expiry  : next period's stock_price (next period starts 3 business days
              after current expiry — close enough as expiry proxy)
  • Last period expiry: fetch from Yahoo using the ratio detected from nearby
              S3 prices (handles split-adjusted vs raw discrepancy).

Missing period rule:
  Gaps > 10 calendar days between consecutive period_ends and period_starts
  are filled with one synthetic period. Synthetic periods carry forward the
  last entry_premium and interpolate stock_price between the surrounding S3
  records.
"""

import os
import boto3
import csv
import io
import re
import requests
from datetime import datetime, timedelta, date

# ── Config ────────────────────────────────────────────────────────────────────
# Credentials are read from environment variables (never hardcode secrets).
# Set before running:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_DEFAULT_REGION=eu-north-1          (or set OPTIONS_REGION)
#   export OPTIONS_BUCKET=s3bucketmz              (optional override)
# Minimum credible premium (dollars). Entries below this are treated as
# data artefacts and replaced by the previous period's premium.
MIN_PREMIUM_FLOOR = 0.05

S3_BUCKET = os.environ.get("OPTIONS_BUCKET", "s3bucketmz")
S3_PREFIX = os.environ.get("OPTIONS_PREFIX", "optionsData/")
AWS_REGION = os.environ.get("OPTIONS_REGION", os.environ.get("AWS_DEFAULT_REGION", "eu-north-1"))

s3 = boto3.client("s3", region_name=AWS_REGION)  # uses env AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY

_FILENAME_RE = re.compile(r"^.+?_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv$")


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def date_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def sf(v) -> float | None:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def next_business_day(d: date, n: int = 1) -> date:
    for _ in range(n):
        d += timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
    return d


# ── S3 loading ────────────────────────────────────────────────────────────────

def load_s3_periods(ticker: str) -> list[dict]:
    """Load all real period CSVs for `ticker`, sorted by period_start."""
    prefix = f"{S3_PREFIX}{ticker}/{ticker}/"
    paginator = s3.get_paginator("list_objects_v2")
    records = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key   = obj["Key"]
            fname = key.split("/")[-1]
            if fname.startswith(".~lock") or "history_summary" in fname:
                continue
            if not fname.endswith(".csv"):
                continue
            m = _FILENAME_RE.match(fname)
            if not m:
                continue
            period_start, period_end = m.group(1), m.group(2)

            raw  = s3.get_object(Bucket=S3_BUCKET, Key=key)
            rows = list(csv.DictReader(io.StringIO(raw["Body"].read().decode())))
            if not rows:
                continue
            row = rows[0]

            records.append({
                "period_start":  period_start,
                "period_end":    period_end,
                "strike":        sf(row.get("strike")),
                "entry_premium": sf(row.get("entry_premium") or row.get("last")),
                "stock_price":   sf(row.get("stock_price") or row.get("price")),
                "synthetic":     False,
            })

    records.sort(key=lambda r: r["period_start"])
    return records


# ── Gap filling ───────────────────────────────────────────────────────────────

def fill_gaps(records: list[dict]) -> list[dict]:
    """
    Insert a synthetic placeholder record for any gap > 10 calendar days
    between period_end[i] and period_start[i+1].
    Stock price is linearly interpolated; premium/strike are carried forward.
    """
    filled = []
    for i, rec in enumerate(records):
        filled.append(rec)
        if i + 1 >= len(records):
            break

        end_i    = to_date(rec["period_end"])
        start_i1 = to_date(records[i + 1]["period_start"])
        gap      = (start_i1 - end_i).days

        if gap <= 10:
            continue

        # One synthetic period per ~26-day block in the gap
        synth_start = next_business_day(end_i)
        while (start_i1 - synth_start).days > 5:
            synth_end = min(synth_start + timedelta(days=25),
                            start_i1 - timedelta(days=1))

            # Interpolate stock price between surrounding real records
            price_before = rec["stock_price"]
            price_after  = records[i + 1]["stock_price"]
            t_total = (start_i1 - end_i).days
            t_synth = (synth_start - end_i).days
            if price_before and price_after and t_total > 0:
                frac        = t_synth / t_total
                synth_price = price_before + frac * (price_after - price_before)
            else:
                synth_price = price_before

            # Carry forward last known strike offset (% of stock)
            last_strike = rec.get("strike")
            last_stock  = rec.get("stock_price") or synth_price
            if last_strike and last_stock and synth_price:
                offset = last_strike - last_stock
                synth_strike = round((synth_price + offset) / 0.5) * 0.5
            else:
                synth_strike = (round(synth_price / 0.5) * 0.5
                                if synth_price else None)

            filled.append({
                "period_start":  date_str(synth_start),
                "period_end":    date_str(synth_end),
                "strike":        synth_strike,
                "entry_premium": rec["entry_premium"],   # carry forward
                "stock_price":   round(synth_price, 4) if synth_price else None,
                "synthetic":     True,
            })
            synth_start = next_business_day(synth_end)

    return filled


# ── Yahoo Finance fallback (last-period expiry only) ─────────────────────────

def get_yahoo_close_on(ticker: str, target: date, scale: float = 1.0) -> float | None:
    """Fetch unadjusted close for `ticker` near `target`, scaled by `scale`."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    start_ts = int(datetime(target.year, target.month, target.day).timestamp()) - 86400 * 5
    end_ts   = int(datetime(target.year, target.month, target.day).timestamp()) + 86400 * 5
    try:
        r = requests.get(
            url,
            params={"period1": start_ts, "period2": end_ts, "interval": "1d"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        result = r.json()["chart"]["result"][0]
        ts_list = result["timestamp"]
        closes  = result["indicators"]["quote"][0]["close"]
        # Find closest date on or after target
        best_price, best_delta = None, 999
        for ts, c in zip(ts_list, closes):
            if c is None:
                continue
            d_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            delta = abs((to_date(d_str) - target).days)
            if delta < best_delta:
                best_delta, best_price = delta, c * scale
        return round(best_price, 4) if best_price else None
    except Exception:
        return None


def detect_scale(ticker: str, s3_price: float, s3_date: str) -> float:
    """
    Detect the Yahoo→S3 price scale factor from one known S3 price point.
    Returns 1.0 if no discrepancy detected.
    """
    yahoo = get_yahoo_close_on(ticker, to_date(s3_date), scale=1.0)
    if yahoo and yahoo > 0 and s3_price and s3_price > 0:
        ratio = s3_price / yahoo
        # Round to nearest common split ratio
        for r in [20, 10, 4, 2, 1]:
            if abs(ratio - r) / r < 0.05:
                return float(r)
    return 1.0


# ── Strategy ──────────────────────────────────────────────────────────────────

def run_strategy(ticker: str, mode: str = "sell_put") -> list[dict]:
    """
    mode = 'sell_put'          : sell put when last 4W positive (momentum)
    mode = 'buy_call'          : buy call when last 4W positive (momentum)
    mode = 'buy_put'           : buy put when last 4W NEGATIVE (counter-momentum hedge)
    mode = 'buy_momentum'      : buy call on positive, buy put on negative (always in)
    mode = 'sell_spy_momentum' : sell put on positive, sell call on negative (always in)
    mode = 'sell_put_always'   : sell put every period, no signal filter
    mode = 'sell_call_always'  : sell call every period, no signal filter
    mode = 'sell_call_negative': sell call only when last 4W was NEGATIVE
    """
    print(f"\nLoading S3 data for {ticker}...")
    real_periods = load_s3_periods(ticker)
    print(f"  {len(real_periods)} period files found")

    periods = fill_gaps(real_periods)
    print(f"  {len(periods)} periods after gap-fill")

    # Detect Yahoo scale factor from the most recent S3 price
    # (recent prices are always post-split, so scale should be 1.0 for lookups)
    last_real = real_periods[-1]
    yahoo_scale = detect_scale(ticker, last_real["stock_price"],
                               last_real["period_start"])
    print(f"  Yahoo price scale factor: {yahoo_scale}x")

    results    = []
    last_prem  = None

    for i, rec in enumerate(periods):
        # ── Stock prices ──────────────────────────────────────────────────────
        stock_entry = rec["stock_price"]

        # Signal price: previous period's stock_price (~4 weeks ago)
        stock_4w_ago = periods[i - 1]["stock_price"] if i > 0 else None

        # Expiry price: next period's stock_price (starts ~3 days after expiry)
        if i + 1 < len(periods):
            stock_expiry = periods[i + 1]["stock_price"]
        else:
            # Last period — fetch from Yahoo (post-split, scale=1.0 for recent)
            stock_expiry = get_yahoo_close_on(ticker, to_date(rec["period_end"]),
                                              scale=yahoo_scale)

        # ── Signal ────────────────────────────────────────────────────────────
        if stock_entry and stock_4w_ago:
            up = stock_entry > stock_4w_ago
            chg_pct = (stock_entry / stock_4w_ago - 1) * 100
            signal_detail = f"{stock_4w_ago:.2f}→{stock_entry:.2f} ({chg_pct:+.1f}%)"
        else:
            up            = False
            signal_detail = "n/a (no prev price)"

        have_signal_data = stock_entry is not None and stock_4w_ago is not None
        if mode in ("sell_call_always", "sell_put_always"):
            signal = stock_entry is not None
        elif mode in ("sell_call_negative", "buy_put"):
            signal = have_signal_data and not up
        elif mode in ("buy_momentum", "sell_spy_momentum"):
            signal = have_signal_data          # always in — direction encoded in trade
        else:  # sell_put / buy_call: trade when last 4W positive
            signal = up

        # Option type for this period (call vs put) — needed for intrinsic correction
        if mode in ("buy_call", "sell_call_always", "sell_call_negative"):
            _opt_type = "call"
        elif mode in ("sell_put", "sell_put_always", "buy_put"):
            _opt_type = "put"
        elif mode in ("buy_momentum", "sell_spy_momentum"):
            _opt_type = "call" if up else "put"
        else:
            _opt_type = "put"

        # ── Premium ───────────────────────────────────────────────────────────
        premium = rec["entry_premium"]
        carried = False
        if premium is None and last_prem is not None:
            premium = last_prem
            carried = True
        elif rec["synthetic"] and premium == last_prem:
            carried = True

        # Replace near-zero premiums with the previous period's value — these
        # are data artefacts (stale/zero quotes).
        if premium is not None and premium < MIN_PREMIUM_FLOOR:
            print(f"  [{rec['period_start']}] {ticker} WARNING: raw premium={premium:.4f} below floor "
                  f"— replacing with prev={last_prem}")
            premium = last_prem   # may still be None if no previous valid premium
            carried = True

        if premium is not None:
            last_prem = premium

        # ITM intrinsic correction: if the option is in-the-money at entry the
        # data source may only store extrinsic (time) value.  Add the intrinsic
        # component so the premium reflects the true market cost.
        #   call intrinsic = max(0, stock_entry − strike)
        #   put  intrinsic = max(0, strike − stock_entry)
        strike = rec["strike"]
        itm_correction = 0.0
        if premium is not None and strike is not None and stock_entry is not None:
            if _opt_type == "call":
                itm_correction = max(0.0, stock_entry - strike)
            else:
                itm_correction = max(0.0, strike - stock_entry)
            if itm_correction > 0.01:
                print(f"  [{rec['period_start']}] {ticker} ITM {_opt_type}: "
                      f"stock={stock_entry:.2f} strike={strike:.2f} "
                      f"→ adding intrinsic {itm_correction:.2f} to premium {premium:.2f}")
                premium = premium + itm_correction

        # ── P&L ───────────────────────────────────────────────────────────────
        if (signal and premium is not None
                and stock_expiry is not None and strike is not None
                and stock_entry is not None):
            if mode == "buy_call":
                payoff     = max(0.0, stock_expiry - strike)
                pnl_dollar = payoff - premium
                trade      = "BUY CALL"
            elif mode == "buy_put":
                payoff     = max(0.0, strike - stock_expiry)
                pnl_dollar = payoff - premium
                trade      = "BUY PUT"
            elif mode == "buy_momentum":
                # up → buy call, down → buy put
                if up:
                    payoff     = max(0.0, stock_expiry - strike)
                    pnl_dollar = payoff - premium
                    trade      = "BUY CALL"
                else:
                    payoff     = max(0.0, strike - stock_expiry)
                    pnl_dollar = payoff - premium
                    trade      = "BUY PUT"
            elif mode == "sell_spy_momentum":
                # up → sell put, down → sell call
                if up:
                    payoff     = max(0.0, strike - stock_expiry)
                    pnl_dollar = premium - payoff
                    trade      = "SELL PUT"
                else:
                    payoff     = max(0.0, stock_expiry - strike)
                    pnl_dollar = premium - payoff
                    trade      = "SELL CALL"
            elif mode in ("sell_call_always", "sell_call_negative"):
                payoff     = max(0.0, stock_expiry - strike)
                pnl_dollar = premium - payoff
                trade      = "SELL CALL"
            elif mode == "sell_put_always":
                payoff     = max(0.0, strike - stock_expiry)
                pnl_dollar = premium - payoff
                trade      = "SELL PUT"
            else:  # sell_put
                payoff     = max(0.0, strike - stock_expiry)
                pnl_dollar = premium - payoff
                trade      = "SELL PUT"
            pnl_pct = pnl_dollar / stock_entry * 100
        else:
            payoff     = 0.0
            pnl_dollar = 0.0
            pnl_pct    = 0.0
            trade      = "NO TRADE" if not signal else "NO DATA"

        results.append({
            "period_start":  rec["period_start"],
            "period_end":    rec["period_end"],
            "synthetic":     rec["synthetic"],
            "signal":        signal,
            "signal_detail": signal_detail,
            "trade":         trade,
            "stock_entry":   round(stock_entry, 2) if stock_entry else None,
            "stock_expiry":  round(stock_expiry, 2) if stock_expiry else None,
            "strike":        strike,
            "premium":       round(premium, 2) if premium else None,
            "itm_correction": round(itm_correction, 2),
            "carried":       carried,
            "payoff":        round(payoff, 2),
            "pnl_dollar":    round(pnl_dollar, 2),
            "pnl_pct":       round(pnl_pct, 4),
        })

    return results


# ── Display ───────────────────────────────────────────────────────────────────

def print_results(ticker: str, results: list[dict], mode: str = "sell_put") -> None:
    W = 135
    payoff_lbl = "Payoff" if mode == "buy_call" else "Intr  "
    header = (
        f"{'Start':<13} {'End':<12} {'Sig':<4} {'4W Move':<24} "
        f"{'Trade':<10} {'Entry':>7} {'Expiry':>7} {'Strike':>7} "
        f"{'Prem':>6} {'C':>1} {payoff_lbl:>6} {'PnL $':>8} {'PnL%':>7} {'Cum%':>8}"
    )

    labels = {
        "sell_put":          "Sell Put  (signal: last 4W positive)",
        "buy_call":          "Buy Call  (signal: last 4W positive)",
        "sell_call_always":  "Sell Call (every period, no filter)",
        "sell_call_negative":"Sell Call (signal: last 4W negative)",
    }
    strategy_label = labels.get(mode, mode)
    print(f"\n{'═'*W}")
    print(f"  {ticker} — {strategy_label}  (4-week momentum filter)")
    print(f"{'═'*W}")
    print(header)
    print(f"{'─'*W}")

    cumul        = 0.0
    total_trades = 0
    no_trades    = 0
    total_pnl    = 0.0
    wins = losses = 0
    active_trade = {"sell_put": "SELL PUT", "buy_call": "BUY CALL",
                    "sell_call_always": "SELL CALL", "sell_call_negative": "SELL CALL"}.get(mode, "SELL PUT")

    for r in results:
        cumul += r["pnl_pct"]
        flag   = "*" if r["synthetic"] else " "
        carry  = "C" if r.get("carried") else " "
        sig    = "YES" if r["signal"] else "no "

        if r["trade"] == active_trade:
            total_trades += 1
            total_pnl    += r["pnl_pct"]
            wins   += 1 if r["pnl_pct"] >= 0 else 0
            losses += 1 if r["pnl_pct"] <  0 else 0
        else:
            no_trades += 1

        e_str  = f"{r['stock_entry']:.2f}"  if r["stock_entry"]  else "  n/a"
        ex_str = f"{r['stock_expiry']:.2f}" if r["stock_expiry"] else "  n/a"
        st_str = f"{r['strike']:.1f}"       if r["strike"]       else "  n/a"
        pr_str = f"{r['premium']:.2f}"      if r["premium"]      else " n/a"

        print(
            f"{r['period_start']:<13} {r['period_end']:<12} {sig:<4} "
            f"{r['signal_detail']:<24} "
            f"{r['trade']:<10} "
            f"{e_str:>7} {ex_str:>7} {st_str:>7} "
            f"{pr_str:>6} {carry:>1} "
            f"{r['payoff']:>6.2f} "
            f"{r['pnl_dollar']:>8.2f} "
            f"{r['pnl_pct']:>6.2f}% "
            f"{cumul:>7.2f}%"
            f"{flag}"
        )

    print(f"{'─'*W}")
    win_rate = wins / total_trades * 100 if total_trades else 0
    avg_pnl  = total_pnl / total_trades   if total_trades else 0

    print(f"\n  Periods : {len(results)}  |  Trades: {total_trades}  No-trade: {no_trades}")
    print(f"  Win/Loss: {wins}/{losses}  ({win_rate:.0f}% win rate)  |  Avg P&L/trade: {avg_pnl:+.2f}%")
    print(f"  Cumulative P&L: {cumul:+.2f}%")
    print(f"\n  Legend: * synthetic (gap) period  |  C premium carried forward")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "AMZN", "TSLA"]
    modes   = ["sell_put", "buy_call"]
    for ticker in tickers:
        for mode in modes:
            results = run_strategy(ticker, mode=mode)
            print_results(ticker, results, mode=mode)
