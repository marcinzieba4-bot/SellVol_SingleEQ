"""
Sell-Put Monthly Backtest — 50 large SPX stocks
Rule:
  - If last monthly candle CLOSED UP  → sell ATM put next month (collect 4% premium)
  - If last monthly candle CLOSED DOWN → do nothing
  - Put expires worthless if stock closes ABOVE strike at expiry
  - Put is assigned (loss) if stock closes BELOW strike
    loss = (strike - expiry_close) / strike  — net P&L = premium - loss
  - Strike = previous month close (ATM put)
  - Each stock has equal notional weight = 1/50 of portfolio
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 50 large-cap SPX stocks (diversified across sectors)
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

PREMIUM_PCT = 0.04   # 4% premium received when selling put
START_DATE  = "2010-01-01"
END_DATE    = "2024-12-31"
WEIGHT      = 1.0 / len(TICKERS)   # equal weight per stock


def fetch_monthly(ticker: str) -> pd.DataFrame:
    """Download daily data and resample to monthly OHLC."""
    raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()

    # yfinance ≥1.0 returns a MultiIndex; flatten to single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close"]].copy()

    # Resample to month-end
    monthly = df.resample("ME").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
    ).dropna()

    return monthly


def backtest_single(ticker: str) -> pd.DataFrame:
    """
    Returns monthly P&L (% of notional) for one ticker.
    Index = expiry month (the month AFTER the signal candle).
    """
    df = fetch_monthly(ticker)
    if len(df) < 2:
        return pd.DataFrame()

    records = []
    for i in range(len(df) - 1):
        signal_open   = float(df["Open"].iloc[i])
        signal_close  = float(df["Close"].iloc[i])
        expiry_close  = float(df["Close"].iloc[i + 1])
        expiry_date   = df.index[i + 1]

        candle_up = signal_close > signal_open

        if candle_up:
            strike   = signal_close
            premium  = PREMIUM_PCT

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
            outcome = "no_trade"
            trade   = False

        records.append({
            "date":    expiry_date,
            "ticker":  ticker,
            "trade":   trade,
            "outcome": outcome,
            "pnl_pct": pnl_pct,
        })

    return pd.DataFrame(records).set_index("date")


def run_backtest() -> dict:
    print(f"Running backtest on {len(TICKERS)} tickers  [{START_DATE} → {END_DATE}]")
    print(f"Rule: sell ATM put when monthly candle is UP  |  premium = {PREMIUM_PCT*100:.0f}%\n")

    all_pnl = []

    for tkr in TICKERS:
        try:
            result = backtest_single(tkr)
            if result.empty:
                print(f"  {tkr:6s}  — no data, skipped")
                continue
            result["weighted_pnl"] = result["pnl_pct"] * WEIGHT
            all_pnl.append(result[["weighted_pnl", "trade", "outcome", "ticker"]])

            trades      = int(result["trade"].sum())
            wins        = int((result["outcome"] == "expired").sum())
            assigned    = int((result["outcome"] == "assigned").sum())
            total_pnl   = result["pnl_pct"].sum() * 100
            print(f"  {tkr:6s}  trades={trades:3d}  expired={wins:3d}  assigned={assigned:3d}  "
                  f"total_pnl={total_pnl:+.1f}%")
        except Exception as e:
            print(f"  {tkr:6s}  ERROR: {e}")

    if not all_pnl:
        print("No data returned.")
        return {}

    combined = pd.concat(all_pnl)

    # Sum weighted P&L across all stocks per month
    monthly_pnl = combined.groupby(combined.index)["weighted_pnl"].sum().sort_index()
    monthly_pnl.index.name = "date"

    # Portfolio equity curve (cumulative compound)
    equity = (1 + monthly_pnl).cumprod()

    # ---- Summary stats ----
    total_return = float(equity.iloc[-1] - 1)
    n_months     = len(monthly_pnl)
    ann_return   = (1 + total_return) ** (12 / n_months) - 1
    ann_vol      = float(monthly_pnl.std() * np.sqrt(12))
    sharpe       = ann_return / ann_vol if ann_vol > 0 else float("nan")
    max_dd       = float((equity / equity.cummax() - 1).min())
    win_months   = int((monthly_pnl > 0).sum())
    flat_months  = int((monthly_pnl == 0).sum())
    loss_months  = int((monthly_pnl < 0).sum())

    print("\n" + "="*58)
    print("  PORTFOLIO AGGREGATE RESULTS  (equal weight, 50 stocks)")
    print("="*58)
    print(f"  Period          : {monthly_pnl.index[0].date()} → {monthly_pnl.index[-1].date()}")
    print(f"  Months          : {n_months}")
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
        "monthly_pnl":   monthly_pnl,
        "equity_curve":  equity,
        "total_return":  total_return,
        "ann_return":    ann_return,
        "ann_vol":       ann_vol,
        "sharpe":        sharpe,
        "max_drawdown":  max_dd,
        "win_months":    win_months,
        "flat_months":   flat_months,
        "loss_months":   loss_months,
        "n_months":      n_months,
    }


if __name__ == "__main__":
    results = run_backtest()

    if results:
        results["equity_curve"].to_csv("equity_curve.csv")
        results["monthly_pnl"].to_csv("monthly_pnl.csv")
        print("\nSaved: equity_curve.csv  |  monthly_pnl.csv")
