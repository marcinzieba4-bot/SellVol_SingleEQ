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
  • Expiry  : next real period's stock_price (~1 business day after expiry —
              good proxy for standard non-gap cycles, S3-consistent for splits).
              When the next record is synthetic (gap follows), Yahoo Finance is
              used instead to get the actual expiry-date close.
  • Last period expiry: fetch from Yahoo using the ratio detected from nearby
              S3 prices (handles split-adjusted vs raw discrepancy).

Missing period rule:
  Gaps > 10 calendar days between consecutive period_ends and period_starts
  are filled with synthetic placeholder records. Synthetic periods carry
  forward the last entry_premium and stock_price (forward-fill, no look-ahead).
  No trades are taken during synthetic periods.
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

# Minimum credible premium as a fraction of stock price.
# A 30-day ATM put at 20% IV is ~2.3% of stock; settlement-era prices
# (2-3 DTE) look like ~0.8% and are almost certainly from the wide-window
# fallback in fetch_historical_backtest.py storing near-expiry data instead
# of entry-date data.  Flag anything below 1.0% as suspicious.
MIN_PREMIUM_PCT = 0.010   # 1.0% of stock price

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

            # Forward-fill stock price from the last real record.
            # Interpolating toward price_after would use future data (look-ahead).
            price_before = rec["stock_price"]
            synth_price  = price_before

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

        # Expiry price: next real period's stock_price (starts ~1 business day
        # after current expiry — good proxy for non-gap periods).
        # When the immediately next record is synthetic (gap follows current
        # period), the forward-filled price would equal stock_entry and
        # misrepresent the actual settlement.  Fetch from Yahoo instead.
        if i + 1 < len(periods) and not periods[i + 1]["synthetic"]:
            stock_expiry = periods[i + 1]["stock_price"]
        else:
            # Last period OR gap period follows — use actual expiry close from Yahoo.
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
        elif mode in ("buy_momentum", "sell_spy_momentum", "call_momentum"):
            signal = have_signal_data          # always in — direction encoded in trade
        else:  # sell_put / buy_call: trade when last 4W positive
            signal = up

        # Synthetic (gap-fill) periods use forward-filled prices — never trade
        # on them to avoid any residual look-ahead from the surrounding gap context.
        if rec["synthetic"]:
            signal = False

        # Option type for this period (call vs put) — needed for intrinsic correction
        if mode in ("buy_call", "sell_call_always", "sell_call_negative", "call_momentum"):
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

        # Warn when premium looks like a settlement-era (near-expiry) price rather
        # than a proper ~30-DTE entry price.  An ATM 30-day put at 20% IV is ~2.3%
        # of stock; anything under MIN_PREMIUM_PCT (1%) is a red flag.
        # This does NOT auto-replace — it just prints a warning so you know which
        # S3 periods need to be re-fetched with a proper entry-date snapshot.
        if (premium is not None and stock_entry is not None
                and stock_entry > 0 and not rec["synthetic"]
                and premium / stock_entry < MIN_PREMIUM_PCT):
            print(f"  [{rec['period_start']}] {ticker} LOW-PREMIUM WARNING: "
                  f"premium=${premium:.2f} is only {premium/stock_entry*100:.2f}% of stock "
                  f"${stock_entry:.2f} — likely settlement-era data (re-fetch recommended)")

        if premium is not None:
            last_prem = premium

        # ── ATM normalization ──────────────────────────────────────────────────
        # Puts are intended to be ATM (strike ≈ stock price), but available
        # strikes are at fixed intervals so there is always some distance.
        # Normalize the stored market premium to an ATM-equivalent value using
        # delta ≈ 0.5 (true ATM delta) applied to the moneyness gap:
        #
        #   adjusted_premium = premium + (stock_entry − strike) × 0.5
        #
        # • OTM put (strike < stock): stock_entry − strike > 0  → premium rises
        #   (the market premium is too low vs ATM; add back the deficit)
        # • ITM put (strike > stock): stock_entry − strike < 0  → premium falls
        #   (the market premium includes intrinsic on top of ATM time value;
        #    subtract the excess — note: Barchart stores the FULL market price
        #    so the old "add intrinsic" correction was double-counting)
        # • ATM (strike = stock): no change
        #
        # The adjustment is signed and applied in both directions every period.
        strike = rec["strike"]
        atm_correction = 0.0
        if premium is not None and strike is not None and stock_entry is not None:
            if _opt_type == "call":
                # Call: OTM when strike > stock, ITM when strike < stock
                atm_correction = (strike - stock_entry) * 0.5
            else:
                # Put: OTM when strike < stock, ITM when strike > stock
                atm_correction = (stock_entry - strike) * 0.5
            if abs(atm_correction) > 0.01:
                print(f"  [{rec['period_start']}] {ticker} ATM adj ({_opt_type}): "
                      f"stock={stock_entry:.2f} strike={strike:.2f} "
                      f"→ {atm_correction:+.2f}  premium {premium:.2f} → {premium + atm_correction:.2f}")
            premium = max(0.0, premium + atm_correction)

        # ── Split correction on expiry price ──────────────────────────────────
        # When consecutive S3 period files cross a stock split the next
        # period's stock_price is on a different (post-split) scale while
        # stock_entry and strike are still pre-split.  Detect: ratio
        # stock_entry/stock_expiry ≈ known split AND strike ≈ stock_entry.
        split_factor = 1
        if stock_expiry and stock_entry and strike:
            _ratio = stock_entry / stock_expiry
            for _split_r in [20, 10, 5, 4, 3, 2]:
                if abs(_ratio - _split_r) / _split_r < 0.15:
                    # Confirm: strike is on the same scale as entry (within 15%)
                    if abs(stock_entry - strike) / stock_entry < 0.15:
                        raw_expiry = stock_expiry
                        stock_expiry = round(stock_expiry * _split_r, 4)
                        split_factor = _split_r
                        print(f"  [{rec['period_start']}] {ticker} "
                              f"SPLIT {_split_r}:1 detected "
                              f"(entry={stock_entry:.2f} raw_expiry={raw_expiry:.2f}) "
                              f"→ expiry adjusted to {stock_expiry:.2f}")
                        break

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
            elif mode == "call_momentum":
                # up → buy call, down → sell call (calls only)
                payoff = max(0.0, stock_expiry - strike)
                if up:
                    pnl_dollar = payoff - premium
                    trade      = "BUY CALL"
                else:
                    pnl_dollar = premium - payoff
                    trade      = "SELL CALL"
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
            "premium":        round(premium, 2) if premium else None,
            "atm_correction": round(atm_correction, 2),
            "carried":        carried,
            "payoff":        round(payoff, 2),
            "pnl_dollar":    round(pnl_dollar, 2),
            "pnl_pct":       round(pnl_pct, 4),
            "split_factor":  split_factor,
        })

    return results


# ── Display ───────────────────────────────────────────────────────────────────

def print_results(ticker: str, results: list[dict], mode: str = "sell_put") -> None:
    W = 135
    payoff_lbl = "Payoff" if mode == "buy_call" else "Payoff"
    header = (
        f"{'Start':<13} {'End':<12} {'Sig':<4} {'4W Move':<24} "
        f"{'Trade':<10} {'Entry':>7} {'Expiry':>7} {'Strike':>7} "
        f"{'Prem':>6} {'C':>1} {payoff_lbl:>6} {'PnL $':>8} {'PnL%':>7} {'Cum%':>8}"
    )

    labels = {
        "sell_put":           "Sell Put  (signal: last 4W positive)",
        "buy_call":           "Buy Call  (signal: last 4W positive)",
        "sell_call_always":   "Sell Call (every period, no filter)",
        "sell_call_negative": "Sell Call (signal: last 4W negative)",
        "buy_momentum":       "Buy Call (↑) / Buy Put (↓)  — always in",
        "sell_spy_momentum":  "Sell Put (↑) / Sell Call (↓) — always in",
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
    for r in results:
        cumul += r["pnl_pct"]
        flag   = "*" if r["synthetic"] else " "
        carry  = "C" if r.get("carried") else " "
        sig    = "YES" if r["signal"] else "no "

        is_trade = r["trade"] not in ("NO TRADE", "NO DATA")

        if is_trade:
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


# ── Portfolio stats ───────────────────────────────────────────────────────────

def portfolio_stats(
    results_by_ticker: dict,
    label: str = "Equal-Weighted Portfolio",
) -> None:
    """
    Print portfolio-level risk stats for an equal-weighted mix of per-ticker
    strategy results (output of run_strategy).

    Parameters
    ----------
    results_by_ticker : {ticker: list[dict]}
    label             : header title
    """
    import math
    from collections import defaultdict

    tickers   = list(results_by_ticker.keys())
    n_tickers = len(tickers)

    # ── 1. Align periods by period_start date ────────────────────────────────
    by_date: dict = defaultdict(dict)
    for ticker, res in results_by_ticker.items():
        for r in res:
            by_date[r["period_start"]][ticker] = r["pnl_pct"]

    dates = sorted(by_date.keys())

    # ── 2. Equal-weighted portfolio return per period ────────────────────────
    port_returns = []   # [(date_str, pct)]
    for d in dates:
        row  = by_date[d]
        pnls = [row.get(t, 0.0) for t in tickers]
        port_returns.append((d, sum(pnls) / n_tickers))

    # ── 3. Cumulative equity curve (additive %) ───────────────────────────────
    equity = [0.0]
    for _, r in port_returns:
        equity.append(equity[-1] + r)

    # ── 4. Max drawdown ───────────────────────────────────────────────────────
    peak_val     = equity[0]
    peak_idx     = 0
    max_dd       = 0.0
    dd_start     = dates[0]
    dd_end       = dates[0]
    for i, v in enumerate(equity[1:], 1):
        if v > peak_val:
            peak_val = v
            peak_idx = i
        dd = peak_val - v
        if dd > max_dd:
            max_dd   = dd
            dd_start = dates[peak_idx - 1] if peak_idx > 0 else dates[0]
            dd_end   = dates[i - 1]        if i - 1 < len(dates) else dates[-1]

    # ── 5. Monthly returns ────────────────────────────────────────────────────
    monthly: dict = defaultdict(float)
    for d, r in port_returns:
        monthly[d[:7]] += r          # key = "YYYY-MM"

    # ── 6. Summary stats ─────────────────────────────────────────────────────
    rets  = [r for _, r in port_returns]
    n     = len(rets)
    total = equity[-1]

    # ~13 four-week periods per year
    PPY   = 13.0
    years = n / PPY

    # CAGR from additive % curve — approximate geometric compounding
    terminal = 1.0 + total / 100.0
    cagr     = ((terminal ** (1.0 / years)) - 1.0) * 100.0 if terminal > 0 else float("-inf")

    mean_r = sum(rets) / n if n else 0.0
    var    = sum((r - mean_r) ** 2 for r in rets) / n if n else 0.0
    std_r  = math.sqrt(var)
    sharpe = (mean_r / std_r) * math.sqrt(PPY) if std_r else 0.0

    # Sortino uses downside deviation (vs 0 target)
    down_sq   = [r ** 2 for r in rets if r < 0]
    down_std  = math.sqrt(sum(down_sq) / n) if n else 0.0
    sortino   = (mean_r / down_std) * math.sqrt(PPY) if down_std else 0.0

    best_period  = max(port_returns, key=lambda x: x[1])
    worst_period = min(port_returns, key=lambda x: x[1])

    wins   = sum(1 for r in rets if r > 0)
    losses = sum(1 for r in rets if r < 0)
    flat   = n - wins - losses

    # ── 7. Header + summary ───────────────────────────────────────────────────
    W = 92
    print(f"\n{'═'*W}")
    print(f"  {label}  ({'  +  '.join(tickers)}  |  equal-weighted)")
    print(f"{'═'*W}")
    print(f"  Periods : {n}  (~{years:.1f} years, {dates[0]} → {dates[-1]})")
    print(f"  Total P&L : {total:+.2f}%   |   CAGR : {cagr:+.2f}%/yr")
    print(f"  Max Drawdown : -{max_dd:.2f}%  ({dd_start} → {dd_end})")
    print(f"  Sharpe (ann) : {sharpe:.2f}   |   Sortino (ann) : {sortino:.2f}")
    print(f"  Win / Loss / Flat : {wins} / {losses} / {flat}   ({wins/n*100:.0f}% win rate)")
    print(f"  Avg return/period : {mean_r:+.3f}%   |   Std dev : {std_r:.3f}%")
    print(f"  Best period  : {best_period[0]}  {best_period[1]:+.2f}%")
    print(f"  Worst period : {worst_period[0]}  {worst_period[1]:+.2f}%")

    # ── 8. Monthly returns table ──────────────────────────────────────────────
    MN = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    yrs = sorted({k[:4] for k in monthly})
    print(f"\n  Monthly Returns (%):")
    print(f"  {'Year':>5}  " + "  ".join(f"{m:>6}" for m in MN) + f"  {'Annual':>7}")
    print(f"  {'─'*5}  " + "  ".join("─"*6 for _ in MN) + f"  {'─'*7}")
    for yr in yrs:
        annual = 0.0
        row_parts = []
        for mi in range(1, 13):
            key = f"{yr}-{mi:02d}"
            val = monthly.get(key)
            if val is not None:
                annual += val
                row_parts.append(f"{val:>+6.2f}")
            else:
                row_parts.append(f"{'':>6}")
        print(f"  {yr:>5}  " + "  ".join(row_parts) + f"  {annual:>+7.2f}")

    # ── 9. ASCII equity curve ─────────────────────────────────────────────────
    H, CW = 12, 68
    vals   = equity
    y_min  = min(vals)
    y_max  = max(vals)
    y_rng  = y_max - y_min or 1.0

    step    = max(1, (len(vals) - 1) // CW)
    sampled = [vals[i] for i in range(0, len(vals), step)]
    if sampled[-1] != vals[-1]:
        sampled.append(vals[-1])

    # Place a block at every sampled point
    grid = [[" "] * len(sampled) for _ in range(H)]
    for xi, v in enumerate(sampled):
        yi  = int((v - y_min) / y_rng * (H - 1))
        row = H - 1 - min(H - 1, max(0, yi))
        grid[row][xi] = "█"

    # Fill below the curve to make it a solid area chart
    for xi in range(len(sampled)):
        filled = False
        for ri in range(H):
            if grid[ri][xi] == "█":
                filled = True
            if filled and grid[ri][xi] == " ":
                grid[ri][xi] = "▒"

    print(f"\n  Equity Curve  (cumulative % P&L, additive):")
    for ri, row in enumerate(grid):
        lv = y_max - (ri / (H - 1)) * y_rng
        print(f"  {lv:>7.1f}% │{''.join(row)}")
    # x-axis
    bar = "─" * len(sampled)
    print(f"  {'':>9}└{bar}")
    n_sam   = len(sampled)
    l_date  = dates[0][:7]
    m_date  = dates[len(dates) // 2][:7]
    e_date  = dates[-1][:7]
    pad_mid = (n_sam // 2) - len(l_date)
    pad_end = n_sam - (n_sam // 2) - len(m_date) - 2
    print(f"  {'':>10}{l_date}{' '*max(1,pad_mid)}{m_date}{' '*max(1,pad_end)}{e_date}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "AMZN", "TSLA"]
    modes   = ["sell_put", "buy_call"]
    for ticker in tickers:
        for mode in modes:
            results = run_strategy(ticker, mode=mode)
            print_results(ticker, results, mode=mode)

    # ── Portfolio stats for BX + MSFT ─────────────────────────────────────────
    PORTFOLIO_TICKERS = ["BX", "MSFT"]
    PORTFOLIO_MODES   = [
        ("buy_momentum",      "V1 — Buy Directional (Buy Call ↑ / Buy Put ↓)"),
        ("sell_spy_momentum", "V2 — Sell Vol        (Sell Put ↑ / Sell Call ↓)"),
    ]
    for mode, label in PORTFOLIO_MODES:
        res = {t: run_strategy(t, mode=mode) for t in PORTFOLIO_TICKERS}
        portfolio_stats(res, label=label)
