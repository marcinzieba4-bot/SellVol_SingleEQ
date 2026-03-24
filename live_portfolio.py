#!/usr/bin/env python3
"""
Live monthly portfolio tracker for SellVol SingleEQ.

Strategy : Sell ATM puts on stocks with a positive previous-month candle.
Schedule : Enter on last Friday of month M; expire last Friday of month M+1 (~30 DTE).
Universe : 50 large-cap US equities (same as backtest).

Commands
--------
  python live_portfolio.py schedule        Show upcoming entry/expiry calendar
  python live_portfolio.py entry           Generate this month's positions (saves JSON)
  python live_portfolio.py snapshot        Daily moneyness & P&L snapshot
  python live_portfolio.py settle          Mark final P&L for expired positions
  python live_portfolio.py history         Print full track record

Data stored in ./live/
-----------------------
  positions/YYYY-MM.json    Positions entered in month YYYY-MM
  snapshots/YYYY-MM-DD.csv  Daily position snapshots
  track_record.csv          Settled trade log (one row per ticker per period)
"""

import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ── shared backtest constants ──────────────────────────────────────────────────
from backtest import TICKERS, STOCK_RV_RATIOS, IV_SCALE, _norm_cdf

LIVE_DIR       = Path(__file__).parent / "live"
POSITIONS_DIR  = LIVE_DIR / "positions"
SNAPSHOTS_DIR  = LIVE_DIR / "snapshots"
TRACK_RECORD   = LIVE_DIR / "track_record.csv"

WEIGHT = 1.0 / len(TICKERS)   # equal notional weight per stock


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def last_friday(year: int, month: int) -> date:
    """Last Friday of a calendar month."""
    # find last day of the month
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    # weekday(): Mon=0 … Fri=4 → roll back to nearest Friday
    days_back = (last.weekday() - 4) % 7
    return last - timedelta(days=days_back)


def entry_expiry(year: int, month: int) -> tuple[date, date]:
    """Return (entry, expiry) for a given entry month."""
    entry = last_friday(year, month)
    ey, em = (year, month + 1) if month < 12 else (year + 1, 1)
    expiry = last_friday(ey, em)
    return entry, expiry


def upcoming_schedule(n: int = 6, from_date: date | None = None) -> list[tuple[date, date]]:
    """Next n (entry, expiry) pairs starting from from_date."""
    today = from_date or date.today()
    periods, y, m = [], today.year, today.month
    while len(periods) < n:
        e, x = entry_expiry(y, m)
        if x >= today:
            periods.append((e, x))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return periods


def active_positions_key(as_of: date | None = None) -> str | None:
    """
    Return YYYY-MM key for the most recently entered positions file,
    or None if no entry has occurred yet this cycle.
    """
    today = as_of or date.today()
    # Walk back up to 3 months to find an entry that has already passed
    y, m = today.year, today.month
    for _ in range(4):
        e, x = entry_expiry(y, m)
        if e <= today:
            return f"{y:04d}-{m:02d}"
        # step back one month
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MARKET DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_current_vix_rfr() -> tuple[float, float]:
    """Latest VIX close and 13-week T-bill yield (%)."""
    vix_raw = yf.download("^VIX", period="5d", auto_adjust=False, progress=False)
    rfr_raw = yf.download("^IRX", period="5d", auto_adjust=False, progress=False)
    for df in (vix_raw, rfr_raw):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    vix = float(vix_raw["Close"].iloc[-1]) if not vix_raw.empty else 20.0
    rfr = float(rfr_raw["Close"].iloc[-1]) if not rfr_raw.empty else 4.0
    return vix, rfr


def fetch_prev_month_candle(ticker: str, signal_year: int, signal_month: int
                            ) -> dict | None:
    """
    Download daily OHLC for signal_month and return aggregated monthly candle.
    Returns dict with open, close, return_pct or None on failure.
    """
    if signal_month == 12:
        end_y, end_m = signal_year + 1, 1
    else:
        end_y, end_m = signal_year, signal_month + 1

    start_str = f"{signal_year:04d}-{signal_month:02d}-01"
    end_str   = f"{end_y:04d}-{end_m:02d}-01"

    raw = yf.download(ticker, start=start_str, end=end_str,
                      auto_adjust=True, progress=False)
    if raw.empty or len(raw) < 2:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    open_px  = float(raw["Open"].iloc[0])
    close_px = float(raw["Close"].iloc[-1])
    ret_pct  = (close_px - open_px) / open_px * 100
    return {"open": open_px, "close": close_px, "return_pct": ret_pct}


def fetch_current_prices(tickers: list[str]) -> dict[str, float]:
    """Batch-download latest closes from yfinance."""
    raw = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]] if "Close" in raw else raw

    prices = {}
    for t in tickers:
        try:
            col = closes[t] if t in closes.columns else closes
            prices[t] = float(col.dropna().iloc[-1])
        except Exception:
            pass
    return prices


def fetch_price_on_date(ticker: str, target: date) -> float | None:
    """Closing price on or just before target date."""
    start = (target - timedelta(days=7)).strftime("%Y-%m-%d")
    end   = (target + timedelta(days=1)).strftime("%Y-%m-%d")
    raw   = yf.download(ticker, start=start, end=end,
                        auto_adjust=True, progress=False)
    if raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    # find closest row on or before target
    raw.index = pd.to_datetime(raw.index)
    subset = raw[raw.index.date <= target]
    if subset.empty:
        return None
    return float(subset["Close"].iloc[-1])


# ══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def bs_put_value(S: float, K: float, T_years: float,
                 iv_pct: float, r_pct: float) -> float:
    """
    Full Black-Scholes put value in dollars per share.
    Returns intrinsic value when T_years <= 0.
    """
    if T_years <= 0:
        return max(K - S, 0.0)
    sigma = iv_pct / 100.0
    r     = r_pct  / 100.0
    sqrtT = math.sqrt(T_years)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T_years) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    put = K * math.exp(-r * T_years) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    return max(put, 0.0)


def iv_estimate(vix: float, ticker: str) -> float:
    """Annualised IV estimate (%) using VIX + per-stock RV ratio."""
    rv = STOCK_RV_RATIOS.get(ticker, sum(STOCK_RV_RATIOS.values()) / len(STOCK_RV_RATIOS))
    return vix * rv * IV_SCALE


def atm_premium_pct(iv_pct: float, r_pct: float, T_months: float = 1.0) -> float:
    """ATM put premium as fraction of spot (for entry sizing)."""
    T = T_months / 12.0
    return bs_put_value(1.0, 1.0, T, iv_pct, r_pct)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_signals(entry_year: int, entry_month: int,
                     vix: float, rfr: float) -> list[dict]:
    """
    For entry in month M, signal is based on month M-1 candle.
    Returns list of position dicts (sell_put or no_trade).
    """
    # signal month = month before entry
    if entry_month == 1:
        sig_y, sig_m = entry_year - 1, 12
    else:
        sig_y, sig_m = entry_year, entry_month - 1

    import calendar
    sig_month_name = f"{calendar.month_abbr[sig_m]} {sig_y}"

    entry_dt, expiry_dt = entry_expiry(entry_year, entry_month)
    dte = (expiry_dt - entry_dt).days

    positions = []
    for ticker in TICKERS:
        candle = fetch_prev_month_candle(ticker, sig_y, sig_m)
        if candle is None:
            print(f"  WARNING: no data for {ticker} {sig_month_name}, skipping")
            continue

        candle_up = candle["close"] > candle["open"]

        if candle_up:
            strike     = candle["close"]          # ATM = previous month close
            iv_pct     = iv_estimate(vix, ticker)
            T_months   = dte / 30.0               # actual tenor in months
            prem_pct   = atm_premium_pct(iv_pct, rfr, T_months)
            reason = (
                f"{sig_month_name} candle UP  "
                f"open={candle['open']:.2f}  "
                f"close={candle['close']:.2f}  "
                f"ret={candle['return_pct']:+.1f}%"
            )
            positions.append({
                "ticker":          ticker,
                "signal":          "sell_put",
                "reason":          reason,
                "sig_month":       sig_month_name,
                "sig_open":        round(candle["open"],  2),
                "sig_close":       round(candle["close"], 2),
                "sig_return_pct":  round(candle["return_pct"], 2),
                "strike":          round(strike, 2),
                "iv_at_entry":     round(iv_pct, 2),
                "premium_pct":     round(prem_pct * 100, 3),
                "premium_source":  "bs_model",
                "expiry":          expiry_dt.isoformat(),
                "dte_at_entry":    dte,
                "weight":          round(WEIGHT, 4),
            })
        else:
            reason = (
                f"{sig_month_name} candle DOWN "
                f"open={candle['open']:.2f}  "
                f"close={candle['close']:.2f}  "
                f"ret={candle['return_pct']:+.1f}%"
            )
            positions.append({
                "ticker":         ticker,
                "signal":         "no_trade",
                "reason":         reason,
                "sig_month":      sig_month_name,
                "sig_open":       round(candle["open"],  2),
                "sig_close":      round(candle["close"], 2),
                "sig_return_pct": round(candle["return_pct"], 2),
            })

    return positions


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_entry(year: int | None = None, month: int | None = None) -> Path:
    """
    Generate signals for the given entry month (defaults to current month)
    and save positions to live/positions/YYYY-MM.json.
    """
    today = date.today()
    y = year  or today.year
    m = month or today.month

    entry_dt, expiry_dt = entry_expiry(y, m)
    key = f"{y:04d}-{m:02d}"

    print(f"\n{'─'*60}")
    print(f"  ENTRY: {entry_dt}  →  EXPIRY: {expiry_dt}  ({(expiry_dt-entry_dt).days} DTE)")
    print(f"{'─'*60}")

    print("  Fetching VIX & risk-free rate …")
    vix, rfr = fetch_current_vix_rfr()
    print(f"  VIX={vix:.1f}  RFR={rfr:.2f}%")

    print(f"  Generating signals for {len(TICKERS)} tickers …")
    positions = generate_signals(y, m, vix, rfr)

    sell_puts = [p for p in positions if p["signal"] == "sell_put"]
    no_trades = [p for p in positions if p["signal"] == "no_trade"]

    data = {
        "entry_date":      entry_dt.isoformat(),
        "expiry_date":     expiry_dt.isoformat(),
        "created_at":      datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "vix_at_entry":    round(vix, 2),
        "rfr_at_entry":    round(rfr, 2),
        "total_tickers":   len(positions),
        "sell_put_count":  len(sell_puts),
        "no_trade_count":  len(no_trades),
        "settled":         False,
        "positions":       positions,
    }

    POSITIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = POSITIONS_DIR / f"{key}.json"
    out.write_text(json.dumps(data, indent=2))

    print(f"\n  Signals: {len(sell_puts)} SELL_PUT  |  {len(no_trades)} NO_TRADE")
    print(f"\n  {'TICKER':<8} {'SIGNAL':<10} {'REASON'}")
    print(f"  {'──────':<8} {'──────':<10} {'──────'}")
    for p in sorted(positions, key=lambda x: x["ticker"]):
        sig_label = "SELL PUT" if p["signal"] == "sell_put" else "no trade"
        print(f"  {p['ticker']:<8} {sig_label:<10} {p['reason']}")

    if sell_puts:
        avg_prem = sum(p["premium_pct"] for p in sell_puts) / len(sell_puts)
        avg_iv   = sum(p["iv_at_entry"]  for p in sell_puts) / len(sell_puts)
        print(f"\n  Avg IV (sell_put):      {avg_iv:.1f}%")
        print(f"  Avg premium (sell_put): {avg_prem:.3f}%")

    print(f"\n  Saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# DAILY SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════

def _load_positions(key: str) -> dict | None:
    path = POSITIONS_DIR / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _position_status(pos: dict, current_price: float,
                     vix: float, rfr: float, as_of: date) -> dict:
    """Compute moneyness, estimated current put value, unrealized P&L."""
    strike   = pos["strike"]
    prem_pct = pos["premium_pct"] / 100.0  # fraction of notional
    iv_entry = pos.get("iv_at_entry", iv_estimate(vix, pos["ticker"]))
    expiry   = date.fromisoformat(pos["expiry"])
    dte      = max((expiry - as_of).days, 0)
    T_years  = dte / 365.0

    # moneyness: positive = OTM (good for put seller), negative = ITM
    moneyness_pct = (current_price - strike) / strike * 100.0

    # current IV estimate (use fresh VIX-based estimate for daily tracking)
    iv_now = iv_estimate(vix, pos["ticker"])

    # current BS put value as % of original strike (= original entry price)
    put_val     = bs_put_value(current_price, strike, T_years, iv_now, rfr)
    put_val_pct = put_val / strike  # fraction

    # unrealized P&L for short put: premium received minus current put value
    unreal_pnl_pct = prem_pct - put_val_pct

    return {
        "ticker":          pos["ticker"],
        "strike":          strike,
        "expiry":          pos["expiry"],
        "dte":             dte,
        "current_price":   round(current_price, 2),
        "moneyness_pct":   round(moneyness_pct, 2),
        "status":          "OTM" if moneyness_pct >= 0 else "ITM",
        "iv_entry":        round(iv_entry, 1),
        "iv_now":          round(iv_now, 1),
        "premium_pct":     round(prem_pct * 100, 3),
        "put_value_pct":   round(put_val_pct * 100, 3),
        "unreal_pnl_pct":  round(unreal_pnl_pct * 100, 3),
        "weight":          pos["weight"],
    }


def run_snapshot(as_of: date | None = None) -> pd.DataFrame:
    """
    Print and save daily moneyness snapshot for the active positions.
    Returns DataFrame of per-position status.
    """
    today = as_of or date.today()
    key = active_positions_key(today)
    if key is None:
        print("No active positions found. Run 'entry' first.")
        return pd.DataFrame()

    data = _load_positions(key)
    if data is None:
        print(f"Positions file not found: live/positions/{key}.json")
        return pd.DataFrame()

    sell_puts = [p for p in data["positions"] if p["signal"] == "sell_put"]
    if not sell_puts:
        print("No sell_put positions in this period.")
        return pd.DataFrame()

    entry_dt  = date.fromisoformat(data["entry_date"])
    expiry_dt = date.fromisoformat(data["expiry_date"])
    days_elapsed = (today - entry_dt).days
    dte_remain   = max((expiry_dt - today).days, 0)

    print(f"\n{'═'*70}")
    print(f"  PORTFOLIO SNAPSHOT  {today}  "
          f"(day {days_elapsed} of {(expiry_dt-entry_dt).days}  |  {dte_remain} DTE)")
    print(f"  Entry: {entry_dt}   Expiry: {expiry_dt}")
    print(f"{'═'*70}")

    print("  Fetching current prices & VIX …")
    vix, rfr = fetch_current_vix_rfr()
    tickers   = [p["ticker"] for p in sell_puts]
    prices    = fetch_current_prices(tickers)

    rows = []
    for pos in sell_puts:
        price = prices.get(pos["ticker"])
        if price is None:
            print(f"  WARNING: no price for {pos['ticker']}, skipping")
            continue
        rows.append(_position_status(pos, price, vix, rfr, today))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── print table ───────────────────────────────────────────────────────────
    hdr = (f"  {'TICKER':<7} {'STRIKE':>8} {'PRICE':>8} "
           f"{'MNESS%':>7} {'ST':>4} {'DTE':>4} "
           f"{'PREM%':>7} {'PUTVAL%':>7} {'PNL%':>7}")
    print(f"\n{hdr}")
    print(f"  {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*4} {'─'*4} {'─'*7} {'─'*7} {'─'*7}")

    for r in sorted(rows, key=lambda x: x["moneyness_pct"]):
        st_flag = " ITM" if r["status"] == "ITM" else "  OTM"
        print(
            f"  {r['ticker']:<7} {r['strike']:>8.2f} {r['current_price']:>8.2f} "
            f"{r['moneyness_pct']:>+7.2f} {st_flag:>4} {r['dte']:>4} "
            f"{r['premium_pct']:>7.3f} {r['put_value_pct']:>7.3f} "
            f"{r['unreal_pnl_pct']:>+7.3f}"
        )

    # ── portfolio summary ─────────────────────────────────────────────────────
    n_otm = sum(1 for r in rows if r["status"] == "OTM")
    n_itm = sum(1 for r in rows if r["status"] == "ITM")
    # weighted P&L
    total_weight   = sum(r["weight"] for r in rows)
    wtd_pnl        = sum(r["unreal_pnl_pct"] * r["weight"] for r in rows) / total_weight
    avg_moneyness  = sum(r["moneyness_pct"]   * r["weight"] for r in rows) / total_weight

    print(f"\n  ── Summary ───────────────────────────────────────────────────────")
    print(f"  Positions : {len(rows)} active  ({n_otm} OTM  {n_itm} ITM)")
    print(f"  VIX       : {vix:.1f}    RFR: {rfr:.2f}%")
    print(f"  Avg moneyness (wt) : {avg_moneyness:+.2f}%")
    print(f"  Unrealised P&L (wt): {wtd_pnl:+.3f}%  "
          f"(of {int(total_weight*100)}% notional deployed)")

    # ── save snapshot CSV ─────────────────────────────────────────────────────
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    snap_path = SNAPSHOTS_DIR / f"{today.isoformat()}.csv"
    df_out = df.copy()
    df_out.insert(0, "snapshot_date", today.isoformat())
    df_out.insert(1, "period_key", key)
    df_out.to_csv(snap_path, index=False)
    print(f"\n  Snapshot saved → {snap_path}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SETTLEMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_settle(key: str | None = None) -> pd.DataFrame:
    """
    Mark final P&L for an expired period.
    Fetches closing price on expiry date, computes final outcome,
    appends to track_record.csv and marks positions file as settled.
    """
    if key is None:
        key = active_positions_key()
    if key is None:
        print("No positions to settle.")
        return pd.DataFrame()

    data = _load_positions(key)
    if data is None:
        print(f"Positions file not found: live/positions/{key}.json")
        return pd.DataFrame()

    if data.get("settled"):
        print(f"Period {key} already settled.")
        return pd.DataFrame()

    expiry_dt = date.fromisoformat(data["expiry_date"])
    today     = date.today()
    if today < expiry_dt:
        print(f"Expiry {expiry_dt} has not passed yet (today={today}). "
              "Run 'settle' on or after expiry date.")
        return pd.DataFrame()

    sell_puts = [p for p in data["positions"] if p["signal"] == "sell_put"]
    print(f"\nSettling period {key}  (expiry {expiry_dt})  — {len(sell_puts)} positions")

    records = []
    for pos in sell_puts:
        ticker  = pos["ticker"]
        strike  = pos["strike"]
        prem    = pos["premium_pct"] / 100.0

        expiry_price = fetch_price_on_date(ticker, expiry_dt)
        if expiry_price is None:
            print(f"  WARNING: no expiry price for {ticker}, skipping")
            continue

        if expiry_price >= strike:
            outcome  = "expired"
            pnl_pct  = prem
        else:
            loss     = (strike - expiry_price) / strike
            pnl_pct  = prem - loss
            outcome  = "assigned"

        records.append({
            "entry_date":    data["entry_date"],
            "expiry_date":   data["expiry_date"],
            "ticker":        ticker,
            "reason":        pos["reason"],
            "strike":        strike,
            "premium_pct":   round(prem * 100, 3),
            "iv_at_entry":   pos.get("iv_at_entry", ""),
            "expiry_price":  round(expiry_price, 2),
            "pnl_pct":       round(pnl_pct * 100, 3),
            "outcome":       outcome,
            "weight":        pos["weight"],
        })

        flag = "✓" if outcome == "expired" else "✗"
        print(f"  {flag} {ticker:<7}  strike={strike:.2f}  "
              f"expiry_px={expiry_price:.2f}  "
              f"pnl={pnl_pct*100:+.3f}%  [{outcome}]")

    if not records:
        return pd.DataFrame()

    # weighted portfolio P&L
    total_w = sum(r["weight"] for r in records)
    wtd_pnl = sum(r["pnl_pct"] * r["weight"] for r in records) / total_w
    n_exp   = sum(1 for r in records if r["outcome"] == "expired")
    n_ass   = sum(1 for r in records if r["outcome"] == "assigned")
    print(f"\n  Expired: {n_exp}  Assigned: {n_ass}")
    print(f"  Weighted portfolio P&L: {wtd_pnl:+.3f}%")

    # append to track record
    df = pd.DataFrame(records)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    if TRACK_RECORD.exists():
        df.to_csv(TRACK_RECORD, mode="a", header=False, index=False)
    else:
        df.to_csv(TRACK_RECORD, index=False)
    print(f"\n  Appended {len(df)} rows → {TRACK_RECORD}")

    # mark positions as settled
    data["settled"]            = True
    data["settled_at"]         = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    data["portfolio_pnl_pct"]  = round(wtd_pnl, 4)
    (POSITIONS_DIR / f"{key}.json").write_text(json.dumps(data, indent=2))

    return df


# ══════════════════════════════════════════════════════════════════════════════
# TRACK RECORD / HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def print_history() -> None:
    """Print aggregate track record from settled trades."""
    if not TRACK_RECORD.exists():
        print("No settled trades yet. Run 'settle' after expiry.")
        return

    df = pd.read_csv(TRACK_RECORD)
    if df.empty:
        print("Track record is empty.")
        return

    # per-period weighted P&L
    df["wt_pnl"] = df["pnl_pct"] * df["weight"]
    period_pnl = (
        df.groupby("entry_date")
          .apply(lambda g: g["wt_pnl"].sum() / g["weight"].sum(), include_groups=False)
          .rename("port_pnl_pct")
    )

    print(f"\n{'═'*65}")
    print("  TRACK RECORD — settled periods")
    print(f"{'═'*65}")
    print(f"  {'ENTRY':<12} {'EXPIRY':<12} {'TRADES':>6} "
          f"{'EXPRD':>6} {'ASGND':>6} {'PORT PNL%':>10}")
    print(f"  {'─'*12} {'─'*12} {'─'*6} {'─'*6} {'─'*6} {'─'*10}")

    for entry, grp in df.groupby("entry_date"):
        expiry  = grp["expiry_date"].iloc[0]
        n       = len(grp)
        n_exp   = (grp["outcome"] == "expired").sum()
        n_ass   = (grp["outcome"] == "assigned").sum()
        pnl     = period_pnl.get(entry, float("nan"))
        print(f"  {entry:<12} {expiry:<12} {n:>6} {n_exp:>6} {n_ass:>6} {pnl:>+10.3f}%")

    # overall stats
    pnl_vals = period_pnl.values
    cum_ret  = sum(pnl_vals)
    avg_ret  = cum_ret / len(pnl_vals)
    n_pos    = sum(1 for v in pnl_vals if v > 0)
    n_neg    = sum(1 for v in pnl_vals if v <= 0)

    print(f"\n  Periods   : {len(pnl_vals)}")
    print(f"  Win/Loss  : {n_pos}/{n_neg}")
    print(f"  Avg P&L   : {avg_ret:+.3f}% per period")
    print(f"  Cum P&L   : {cum_ret:+.3f}%")
    print(f"  Tickers   : {df['ticker'].nunique()} unique  ({len(df)} total trades)")


def print_schedule(n: int = 8) -> None:
    """Print upcoming entry/expiry schedule."""
    print(f"\n  {'ENTRY':<14} {'EXPIRY':<14} {'DTE':>4}  STATUS")
    print(f"  {'─'*14} {'─'*14} {'─'*4}  {'─'*12}")
    today = date.today()
    for entry, expiry in upcoming_schedule(n):
        dte    = (expiry - entry).days
        status = ("  ← ACTIVE" if entry <= today <= expiry
                  else ("  ← UPCOMING" if entry > today else "  past"))
        print(f"  {entry.isoformat():<14} {expiry.isoformat():<14} {dte:>4}  {status}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _usage() -> None:
    print(__doc__)


def main() -> None:
    args = sys.argv[1:]
    cmd  = args[0].lower() if args else "help"

    if cmd in ("schedule", "cal", "calendar"):
        print_schedule()

    elif cmd in ("entry", "enter"):
        # optional: --year YYYY --month MM
        y, m = None, None
        if "--year" in args:
            y = int(args[args.index("--year") + 1])
        if "--month" in args:
            m = int(args[args.index("--month") + 1])
        run_entry(y, m)

    elif cmd in ("snapshot", "snap", "status"):
        run_snapshot()

    elif cmd in ("settle", "settlement"):
        k = args[1] if len(args) > 1 and not args[1].startswith("--") else None
        run_settle(k)

    elif cmd in ("history", "hist", "track"):
        print_history()

    else:
        _usage()


if __name__ == "__main__":
    main()
