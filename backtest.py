"""
Sell-Put Monthly Backtest — 50 large SPX stocks
Rule:
  - If last monthly candle CLOSED UP  → sell ATM put next month
  - If last monthly candle CLOSED DOWN → do nothing
  - Premium is derived from VIX at the START of the trade month via
    the Black-Scholes ATM approximation:
        premium_pct = VIX/100 * sqrt(1/12) / sqrt(2*pi)
    e.g. VIX=20 → ~2.3%   VIX=30 → ~3.5%   VIX=40 → ~4.6%
  - Strike = previous month close (ATM put)
  - Each stock has equal notional weight = 1/50 of portfolio
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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

# Black-Scholes ATM put approximation coefficient for 1-month expiry
# premium/S = IV * sqrt(T / (2*pi))  where T = 1/12
_BS_ATM_COEFF = np.sqrt(1 / (12 * 2 * np.pi))   # ≈ 0.1151


def vix_to_premium(vix: float) -> float:
    """Convert annualised VIX (e.g. 20.0) to 1-month ATM put premium fraction."""
    return (vix / 100.0) * _BS_ATM_COEFF


def fetch_vix_monthly() -> pd.Series:
    """
    Download ^VIX daily and return a Series of month-start VIX values.
    Index = month-start date (first trading day of each month).
    We use the VIX on the FIRST day of the trade month as the premium estimate,
    because that's when you'd actually be selling the put.
    """
    raw = yf.download("^VIX", start=START_DATE, end=END_DATE,
                      auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Take month-start (first trading day of each month) closing VIX
    vix_monthly = raw["Close"].resample("MS").first().dropna()
    return vix_monthly


def fetch_monthly(ticker: str) -> pd.DataFrame:
    """Download daily data and resample to monthly OHLC."""
    raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    monthly = raw[["Open", "Close"]].resample("ME").agg(
        Open=("Open",  "first"),
        Close=("Close", "last"),
    ).dropna()
    return monthly


def backtest_single(ticker: str, vix_ms: pd.Series) -> pd.DataFrame:
    """
    Returns monthly P&L (% of notional) for one ticker.
    Index = expiry month-end (the month AFTER the signal candle).
    """
    df = fetch_monthly(ticker)
    if len(df) < 2:
        return pd.DataFrame()

    records = []
    for i in range(len(df) - 1):
        signal_open   = float(df["Open"].iloc[i])
        signal_close  = float(df["Close"].iloc[i])
        expiry_close  = float(df["Close"].iloc[i + 1])
        expiry_date   = df.index[i + 1]          # month-end of trade month
        trade_month_start = expiry_date.replace(day=1)  # first of trade month

        candle_up = signal_close > signal_open

        if candle_up:
            # Look up VIX on first trading day of the trade month
            vix_val = vix_ms.get(trade_month_start, None)
            if vix_val is None:
                # fallback: nearest available VIX
                idx = vix_ms.index.searchsorted(trade_month_start)
                idx = min(idx, len(vix_ms) - 1)
                vix_val = float(vix_ms.iloc[idx])
            else:
                vix_val = float(vix_val)

            premium  = vix_to_premium(vix_val)
            strike   = signal_close

            if expiry_close >= strike:
                pnl_pct = premium
                outcome = "expired"
            else:
                loss_pct = (strike - expiry_close) / strike
                pnl_pct  = premium - loss_pct
                outcome  = "assigned"
            trade = True
        else:
            pnl_pct = 0.0
            vix_val = float("nan")
            premium = 0.0
            outcome = "no_trade"
            trade   = False

        records.append({
            "date":    expiry_date,
            "ticker":  ticker,
            "trade":   trade,
            "outcome": outcome,
            "vix":     vix_val,
            "premium": premium,
            "pnl_pct": pnl_pct,
        })

    return pd.DataFrame(records).set_index("date")


def current_month_estimate():
    """Print VIX-based premium estimate for the current month."""
    raw = yf.download("^VIX", period="5d", auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    latest_vix = float(raw["Close"].iloc[-1])
    premium    = vix_to_premium(latest_vix)
    print("\n" + "="*45)
    print("  CURRENT MONTH PREMIUM ESTIMATE")
    print("="*45)
    print(f"  Latest VIX     : {latest_vix:.2f}")
    print(f"  Formula        : VIX/100 * sqrt(1/(12*2π))")
    print(f"  ATM Put Premium: {premium*100:.2f}%")
    print("="*45)
    return latest_vix, premium


def run_backtest() -> dict:
    print("Fetching VIX data ...")
    vix_ms = fetch_vix_monthly()

    # Show VIX premium table
    print(f"\nVIX → 1-month ATM put premium (BS approximation):")
    for v in [12, 15, 18, 20, 25, 30, 35, 40, 50, 60]:
        print(f"  VIX={v:2d}  →  premium={vix_to_premium(v)*100:.2f}%")

    print(f"\nRunning backtest on {len(TICKERS)} tickers  [{START_DATE} → {END_DATE}]")
    print("Rule: sell ATM put when monthly candle is UP  |  premium derived from VIX\n")

    all_pnl = []
    all_details = []

    for tkr in TICKERS:
        try:
            result = backtest_single(tkr, vix_ms)
            if result.empty:
                print(f"  {tkr:6s}  — no data, skipped")
                continue
            result["weighted_pnl"] = result["pnl_pct"] * WEIGHT
            all_pnl.append(result[["weighted_pnl", "vix", "premium"]])
            all_details.append(result)

            trades   = int(result["trade"].sum())
            wins     = int((result["outcome"] == "expired").sum())
            assigned = int((result["outcome"] == "assigned").sum())
            avg_prem = result.loc[result["trade"], "premium"].mean() * 100
            total_pnl = result["pnl_pct"].sum() * 100
            print(f"  {tkr:6s}  trades={trades:3d}  expired={wins:3d}  assigned={assigned:3d}"
                  f"  avg_prem={avg_prem:.2f}%  total_pnl={total_pnl:+.1f}%")
        except Exception as e:
            print(f"  {tkr:6s}  ERROR: {e}")

    if not all_pnl:
        print("No data returned.")
        return {}

    combined    = pd.concat(all_pnl)
    monthly_pnl = combined.groupby(combined.index)["weighted_pnl"].sum().sort_index()
    monthly_pnl.index.name = "date"

    # Average VIX per month (across all traded positions)
    avg_vix_monthly = combined.groupby(combined.index)["vix"].mean()

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
    avg_premium  = vix_to_premium(avg_vix) * 100

    print("\n" + "="*58)
    print("  PORTFOLIO AGGREGATE RESULTS  (VIX-based premium)")
    print("="*58)
    print(f"  Period          : {monthly_pnl.index[0].date()} → {monthly_pnl.index[-1].date()}")
    print(f"  Months          : {n_months}")
    print(f"  Avg VIX         : {avg_vix:.1f}")
    print(f"  Avg Premium     : {avg_premium:.2f}%")
    print(f"  Total Return    : {total_return*100:+.2f}%")
    print(f"  Ann. Return     : {ann_return*100:+.2f}%")
    print(f"  Ann. Volatility : {ann_vol*100:.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:.2f}")
    print(f"  Max Drawdown    : {max_dd*100:.2f}%")
    print(f"  Win months      : {win_months}  ({win_months/n_months*100:.0f}%)")
    print(f"  Flat months     : {flat_months}  ({flat_months/n_months*100:.0f}%)")
    print(f"  Loss months     : {loss_months}  ({loss_months/n_months*100:.0f}%)")
    print("="*58)

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
        "avg_premium_pct": avg_premium,
    }


if __name__ == "__main__":
    current_month_estimate()
    results = run_backtest()

    if results:
        results["equity_curve"].to_csv("equity_curve.csv")
        results["monthly_pnl"].to_csv("monthly_pnl.csv")
        vix_pnl = pd.DataFrame({
            "monthly_pnl": results["monthly_pnl"],
            "avg_vix":     results["avg_vix_monthly"],
        })
        vix_pnl.to_csv("monthly_pnl_with_vix.csv")
        print("\nSaved: equity_curve.csv  |  monthly_pnl_with_vix.csv")
