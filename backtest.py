"""
Sell-Put Monthly Backtest — 50 large SPX stocks
Rule:
  - If last monthly candle CLOSED UP  → sell ATM put next month
  - If last monthly candle CLOSED DOWN → do nothing
  - Strike = previous month close (ATM put)
  - Each stock has equal notional weight = 1/50 of portfolio

=============================================================
PREMIUM MODEL  (user-calibrated, per-stock)
=============================================================

Observed real market premiums at VIX=27.19 (user-provided, March 2026):
  AAPL 3.10%  MSFT 2.90%  NVDA 4.80%  GOOGL 3.70%  AMZN 3.75%
  GS   4.10%  JPM  3.75%  META 3.60%  JNJ   2.70%

These imply per-stock true_calib = premium / (VIX × RV_ratio × BS_COEFF):
  AAPL=0.500  MSFT=0.523  NVDA=0.473  GOOGL=0.615  AMZN=0.512
  GS=0.672    JPM=0.665   META=0.412  JNJ=0.743

Key finding: IV/VIX = 0.608 + 0.257 × RV_ratio  (R²=0.59, 9 data points)
  → High-RV stocks (NVDA 3.24x, META 2.79x) don't scale proportionally in IV.
  → There's a base "floor" (~0.61 × VIX) that all stocks share regardless of beta.
  → Avg calibration factor: 0.568 (well below our previous 0.75 assumption).

Why IV/VIX < RV_ratio (and hence calib < 1):
  (Carr & Wu 2009, Driessen et al. 2005)
  VIX embeds a CORRELATION RISK PREMIUM — implied correlation (39.5%) exceeds
  realized correlation (32.5%) by ~7 pts.  This inflates VIX relative to single
  stocks, which are fairly priced (VRP ≈ 0).

For the 9 calibrated stocks: use exact user-observed calibration.
For the remaining 41 stocks: use the fitted regression:
  IV/VIX = 0.6083 + 0.2571 × RV_ratio
  → calib[stock] = (0.6083 + 0.2571 × RV_ratio) / RV_ratio

1-month ATM premium:  premium = stock_IV / 100 × sqrt(1/(12×2π))
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

IV_HISTORY_FILE = "iv_history.csv"   # produced by fetch_iv_history.py

# ------------------------------------------------------------------
TICKERS = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "AMD", "INTC", "CRM",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "USB", "C", "SCHW",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "CVS", "MDT",
    # Consumer / Retail
    "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW", "PG", "KO",
    # Energy / Industrials / Other
    "XOM", "CVX", "CAT", "GE", "HON", "LMT", "UPS", "BA", "RTX", "NEE",
]

START_DATE = "2010-01-01"
END_DATE   = "2024-12-31"
WEIGHT     = 1.0 / len(TICKERS)

# Black-Scholes ATM premium coefficient for 1-month expiry
# premium/S = IV × sqrt(T/(2π)),  T=1/12
_BS_COEFF = np.sqrt(1.0 / (12.0 * 2.0 * np.pi))   # ≈ 0.1151

# Per-stock realized vol / SPX realized vol ratios (from 15yr price history)
STOCK_RV_RATIOS = {
    "AAPL": 1.98, "MSFT": 1.77, "NVDA": 3.24, "GOOGL": 1.92, "META": 2.79,
    "AMZN": 2.34, "AVGO": 2.66, "AMD":  4.27, "INTC":  2.20, "CRM":  2.54,
    "JPM":  1.80, "BAC":  2.21, "WFC":  1.94, "GS":    1.95, "MS":   2.24,
    "BLK":  1.80, "AXP":  1.79, "USB":  1.73, "C":     2.17, "SCHW": 2.34,
    "UNH":  1.79, "JNJ":  1.16, "LLY":  1.74, "ABBV":  1.99, "MRK":  1.54,
    "PFE":  1.53, "TMO":  1.74, "ABT":  1.50, "CVS":   1.84, "MDT":  1.56,
    "WMT":  1.35, "HD":   1.54, "MCD":  1.25, "NKE":   1.96, "SBUX": 1.82,
    "TGT":  2.03, "COST": 1.45, "LOW":  1.92, "PG":    1.17, "KO":   1.15,
    "XOM":  1.61, "CVX":  1.68, "CAT":  2.06, "GE":    2.09, "HON":  1.47,
    "LMT":  1.42, "UPS":  1.60, "BA":   2.16, "RTX":   1.55, "NEE":  1.47,
}

# User-observed premiums at VIX=27.19 → per-stock true calibration factors
# true_calib = observed_premium / (VIX × RV_ratio × BS_COEFF / 100)
_USER_CALIB_VIX = 27.19
_USER_OBSERVED = {   # raw observed premium fractions
    "AAPL": 0.031, "MSFT": 0.029, "NVDA": 0.048, "GOOGL": 0.037,
    "AMZN": 0.0375,"GS":   0.041, "JPM":  0.0375,"META":  0.036, "JNJ": 0.027,
}
_USER_CALIBS = {
    tkr: prem / (_USER_CALIB_VIX * STOCK_RV_RATIOS[tkr] * _BS_COEFF / 100)
    for tkr, prem in _USER_OBSERVED.items()
}

# Regression for non-calibrated stocks: IV/VIX = A + B × RV_ratio
# Fitted from the 9 user-calibrated stocks (R²=0.59)
_REG_A = 0.6083
_REG_B = 0.2571
_AVG_CALIB = np.mean(list(_USER_CALIBS.values()))   # ≈ 0.568


def get_calib(ticker: str) -> float:
    """
    Return per-stock calibration factor.
    Known stocks: exact user-observed value.
    Others: fitted from IV/VIX = A + B × RV_ratio regression.
    """
    if ticker in _USER_CALIBS:
        return _USER_CALIBS[ticker]
    rv = STOCK_RV_RATIOS.get(ticker, np.mean(list(STOCK_RV_RATIOS.values())))
    return (_REG_A + _REG_B * rv) / rv


def stock_iv_estimate(vix: float, ticker: str) -> float:
    return vix * STOCK_RV_RATIOS.get(ticker, 1.9) * get_calib(ticker)


def vix_to_premium(vix: float, ticker: str) -> float:
    return (stock_iv_estimate(vix, ticker) / 100.0) * _BS_COEFF


def load_iv_history() -> pd.DataFrame | None:
    """
    Load real per-stock IV history from AlphaQuery CSV (if available).
    Returns a DataFrame indexed by (date, ticker) with column iv_30d,
    resampled to month-start, or None if file not found.
    """
    if not os.path.exists(IV_HISTORY_FILE):
        return None
    df = pd.read_csv(IV_HISTORY_FILE, parse_dates=["date"])
    df = df.set_index("date").groupby("ticker")["iv_30d"].resample("MS").mean()
    df = df.reset_index().set_index(["date", "ticker"])["iv_30d"]
    print(f"Loaded real IV history: {len(df):,} rows from {IV_HISTORY_FILE}")
    return df


def get_iv_for_month(iv_history: pd.DataFrame | None,
                     ticker: str, month_start: pd.Timestamp,
                     vix: float) -> float:
    """
    Return 30-day IV for ticker in a given month.
    Priority:
      1. Real AlphaQuery data (if iv_history loaded and has the observation)
      2. Regression model: IV = VIX × (A + B×RV_ratio)
    """
    if iv_history is not None:
        try:
            iv = float(iv_history.loc[(month_start, ticker)])
            if not np.isnan(iv):
                return iv
        except KeyError:
            pass
    # Fallback: regression model
    rv = STOCK_RV_RATIOS.get(ticker, 1.9)
    return vix * rv * get_calib(ticker) / 100   # return as fraction


def fetch_vix_monthly() -> pd.Series:
    raw = yf.download("^VIX", start=START_DATE, end=END_DATE,
                      auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw["Close"].resample("MS").first().dropna()


def fetch_monthly(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open", "Close"]].resample("ME").agg(
        Open=("Open",  "first"),
        Close=("Close", "last"),
    ).dropna()


def backtest_single(ticker: str, vix_ms: pd.Series,
                    iv_history: pd.DataFrame | None = None) -> pd.DataFrame:
    df = fetch_monthly(ticker)
    if len(df) < 2:
        return pd.DataFrame()

    records = []
    for i in range(len(df) - 1):
        signal_open  = float(df["Open"].iloc[i])
        signal_close = float(df["Close"].iloc[i])
        expiry_close = float(df["Close"].iloc[i + 1])
        expiry_date  = df.index[i + 1]
        trade_month_start = expiry_date.replace(day=1)

        candle_up = signal_close > signal_open

        if candle_up:
            vix_val = vix_ms.get(trade_month_start, None)
            if vix_val is None:
                idx = min(vix_ms.index.searchsorted(trade_month_start), len(vix_ms) - 1)
                vix_val = float(vix_ms.iloc[idx])
            else:
                vix_val = float(vix_val)

            # Use real IV if available, else regression model
            iv_frac = get_iv_for_month(iv_history, ticker, trade_month_start, vix_val)
            premium = iv_frac * _BS_COEFF
            strike  = signal_close

            if expiry_close >= strike:
                pnl_pct = premium
                outcome = "expired"
            else:
                loss_pct = (strike - expiry_close) / strike
                pnl_pct  = premium - loss_pct
                outcome  = "assigned"
            trade = True
        else:
            vix_val = float("nan")
            premium = 0.0
            pnl_pct = 0.0
            outcome = "no_trade"
            trade   = False

        records.append({
            "date": expiry_date, "ticker": ticker, "trade": trade,
            "outcome": outcome, "vix": vix_val, "premium": premium, "pnl_pct": pnl_pct,
        })

    return pd.DataFrame(records).set_index("date")


def current_month_estimate():
    raw = yf.download("^VIX", period="5d", auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    latest_vix = float(raw["Close"].iloc[-1])

    print("\n" + "="*72)
    print(f"  CURRENT MONTH PREMIUM ESTIMATES  (VIX={latest_vix:.2f})")
    print(f"  Model: IV/VIX = {_REG_A:.4f} + {_REG_B:.4f}×RV_ratio  (user-calibrated for 9 stocks)")
    print("="*72)
    print(f"  {'Ticker':6s}  {'Calib':>7s}  {'Est IV':>7s}  {'Est prem':>9s}  {'Observed':>9s}  {'Source':>10s}")
    print("  " + "-"*60)
    for tkr in TICKERS:
        calib  = get_calib(tkr)
        iv     = stock_iv_estimate(latest_vix, tkr)
        prem   = vix_to_premium(latest_vix, tkr) * 100
        obs    = _USER_OBSERVED.get(tkr)
        source = "user obs" if tkr in _USER_CALIBS else "regression"
        obs_str = f"{obs*100:>8.2f}%" if obs else "        -"
        print(f"  {tkr:6s}  {calib:>7.4f}  {iv:>6.1f}%  {prem:>8.2f}%  {obs_str}  {source:>10s}")
    avg_prem = np.mean([vix_to_premium(latest_vix, t)*100 for t in TICKERS])
    print("  " + "-"*60)
    print(f"  {'AVG':6s}  {'':>7s}  {'':>7s}  {avg_prem:>8.2f}%")
    print("="*72)
    return latest_vix


def run_backtest() -> dict:
    iv_history = load_iv_history()
    using_real_iv = iv_history is not None
    print(f"\nIV source: {'AlphaQuery real data (' + IV_HISTORY_FILE + ')' if using_real_iv else 'regression model (run fetch_iv_history.py to get real data)'}")

    print("Fetching VIX data ...")
    vix_ms = fetch_vix_monthly()

    print(f"\nAvg premium at various VIX levels (avg calib={_AVG_CALIB:.3f}):")
    avg_rv = np.mean(list(STOCK_RV_RATIOS.values()))
    for v in [12, 15, 18, 20, 25, 30, 35, 40]:
        avg_iv   = np.mean([stock_iv_estimate(v, t) for t in TICKERS])
        avg_prem = np.mean([vix_to_premium(v, t)*100 for t in TICKERS])
        print(f"  VIX={v:2d}  avg_IV={avg_iv:.1f}%  avg_premium={avg_prem:.2f}%")

    print(f"\nRunning backtest on {len(TICKERS)} tickers  [{START_DATE} → {END_DATE}]\n")

    all_pnl = []
    for tkr in TICKERS:
        try:
            result = backtest_single(tkr, vix_ms, iv_history)
            if result.empty:
                print(f"  {tkr:6s}  — no data, skipped")
                continue
            result["weighted_pnl"] = result["pnl_pct"] * WEIGHT
            all_pnl.append(result[["weighted_pnl", "vix", "premium"]])

            trades   = int(result["trade"].sum())
            wins     = int((result["outcome"] == "expired").sum())
            assigned = int((result["outcome"] == "assigned").sum())
            avg_prem = result.loc[result["trade"], "premium"].mean() * 100
            tot_pnl  = result["pnl_pct"].sum() * 100
            print(f"  {tkr:6s}  trades={trades:3d}  expired={wins:3d}  assigned={assigned:3d}"
                  f"  avg_prem={avg_prem:.2f}%  total_pnl={tot_pnl:+.1f}%")
        except Exception as e:
            print(f"  {tkr:6s}  ERROR: {e}")

    if not all_pnl:
        print("No data returned.")
        return {}

    combined    = pd.concat(all_pnl)
    monthly_pnl = combined.groupby(combined.index)["weighted_pnl"].sum().sort_index()
    monthly_pnl.index.name = "date"
    avg_vix_monthly = combined.groupby(combined.index)["vix"].mean()
    avg_prem_pct    = combined.loc[combined["premium"] > 0, "premium"].mean() * 100

    equity = (1 + monthly_pnl).cumprod()

    total_return = float(equity.iloc[-1] - 1)
    n_months     = len(monthly_pnl)
    ann_return   = (1 + total_return) ** (12 / n_months) - 1
    ann_vol      = float(monthly_pnl.std() * np.sqrt(12))
    sharpe       = ann_return / ann_vol if ann_vol > 0 else float("nan")
    max_dd       = float((equity / equity.cummax() - 1).min())
    win_months   = int((monthly_pnl > 0).sum())
    flat_months  = int((monthly_pnl == 0).sum())
    loss_months  = int((monthly_pnl < 0).sum())
    avg_vix      = float(avg_vix_monthly.mean())

    iv_src = "AlphaQuery real IV" if using_real_iv else "regression model (VIX-calibrated)"
    print("\n" + "="*65)
    print("  PORTFOLIO AGGREGATE RESULTS")
    print(f"  IV source: {iv_src}")
    print("="*65)
    print(f"  Period          : {monthly_pnl.index[0].date()} → {monthly_pnl.index[-1].date()}")
    print(f"  Months          : {n_months}")
    print(f"  Avg VIX         : {avg_vix:.1f}")
    print(f"  Avg Premium     : {avg_prem_pct:.2f}%")
    print(f"  Total Return    : {total_return*100:+.2f}%")
    print(f"  Ann. Return     : {ann_return*100:+.2f}%")
    print(f"  Ann. Volatility : {ann_vol*100:.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:.2f}")
    print(f"  Max Drawdown    : {max_dd*100:.2f}%")
    print(f"  Win months      : {win_months}  ({win_months/n_months*100:.0f}%)")
    print(f"  Flat months     : {flat_months}  ({flat_months/n_months*100:.0f}%)")
    print(f"  Loss months     : {loss_months}  ({loss_months/n_months*100:.0f}%)")
    print("="*65)

    return {
        "monthly_pnl":     monthly_pnl,
        "equity_curve":    equity,
        "avg_vix_monthly": avg_vix_monthly,
        "total_return":    total_return,
        "ann_return":      ann_return,
        "ann_vol":         ann_vol,
        "sharpe":          sharpe,
        "max_drawdown":    max_dd,
        "win_months":      win_months,
        "flat_months":     flat_months,
        "loss_months":     loss_months,
        "n_months":        n_months,
        "avg_vix":         avg_vix,
        "avg_premium_pct": avg_prem_pct,
    }


if __name__ == "__main__":
    current_month_estimate()
    results = run_backtest()

    if results:
        results["equity_curve"].to_csv("equity_curve.csv")
        pd.DataFrame({
            "monthly_pnl": results["monthly_pnl"],
            "avg_vix":     results["avg_vix_monthly"],
        }).to_csv("monthly_pnl_with_vix.csv")
        print("\nSaved: equity_curve.csv  |  monthly_pnl_with_vix.csv")
