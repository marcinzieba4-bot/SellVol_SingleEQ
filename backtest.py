"""
Sell-Put Monthly Backtest — 50 large SPX stocks

Rule:
  - If last monthly candle CLOSED UP  → sell ATM put next month
  - If last monthly candle CLOSED DOWN → do nothing
  - Strike = previous month close (ATM put)
  - Each stock has equal notional weight = 1/50 of portfolio

Premium model:
  1. IV estimate   = VIX × RV_ratio × IV_SCALE
       RV_ratio: per-stock realized vol / SPX realized vol (from 15yr history)
       IV_SCALE:  0.55 — aligns model IVs with observed single-stock option markets
  2. Premium = full Black-Scholes ATM put price using IV + 1-month T-bill rate

  If iv_history.csv is present (from fetch_iv_history.py / AlphaQuery),
  real per-stock IV is used instead of the VIX model.
"""

import os
import math
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

IV_HISTORY_FILE = "iv_history.csv"   # optional — produced by fetch_iv_history.py

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

# ---- Premium model parameters ----
IV_SCALE = 0.55   # VIX → single-stock IV scaling factor
                  # (CBOE stock vol indices trade ~1.0-1.6x VIX depending on stock;
                  #  0.55 × RV_ratio gives realistic per-stock IVs across the universe)

T_MONTHS = 1.0 / 12.0   # option tenor: 1 calendar month

# Per-stock realized vol / SPX realized vol ratios (computed from 2010–2024 daily prices)
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


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put_premium(iv_pct: float, r_pct: float = 4.0) -> float:
    """
    Full Black-Scholes ATM put premium as a fraction of spot/strike.
    ATM → S = K, so the formula simplifies to:
        P/S = e^(-rT) N(-d2) - N(-d1)
    where
        d1 = (r/σ + σ/2) √T
        d2 = d1 - σ √T
    """
    sigma = iv_pct / 100.0
    r     = r_pct  / 100.0
    sqrtT = math.sqrt(T_MONTHS)
    d1    = (r / sigma + sigma / 2.0) * sqrtT
    d2    = d1 - sigma * sqrtT
    prem  = math.exp(-r * T_MONTHS) * _norm_cdf(-d2) - _norm_cdf(-d1)
    return max(prem, 0.0)


def iv_estimate(vix: float, ticker: str) -> float:
    """Annualised 30-day IV estimate (%) from VIX + per-stock RV ratio."""
    rv = STOCK_RV_RATIOS.get(ticker, np.mean(list(STOCK_RV_RATIOS.values())))
    return vix * rv * IV_SCALE


# ------------------------------------------------------------------
def load_iv_history() -> pd.DataFrame | None:
    """Load AlphaQuery IV history CSV if available."""
    if not os.path.exists(IV_HISTORY_FILE):
        return None
    df = pd.read_csv(IV_HISTORY_FILE, parse_dates=["date"])
    df = df.set_index("date").groupby("ticker")["iv_30d"].resample("MS").mean()
    df = df.reset_index().set_index(["date", "ticker"])["iv_30d"]
    print(f"Loaded real IV history: {len(df):,} rows → {IV_HISTORY_FILE}")
    return df


def get_iv_pct(iv_history, ticker: str, month_start: pd.Timestamp, vix: float) -> float:
    """IV % for a given stock/month — real data first, model fallback."""
    if iv_history is not None:
        try:
            v = float(iv_history.loc[(month_start, ticker)])
            if not np.isnan(v):
                return v * 100  # stored as decimal fraction
        except KeyError:
            pass
    return iv_estimate(vix, ticker)


def fetch_vix_monthly() -> pd.Series:
    raw = yf.download("^VIX", start=START_DATE, end=END_DATE,
                      auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw["Close"].resample("MS").first().dropna()


def fetch_rfr_monthly() -> pd.Series:
    """13-week T-bill annualised yield (%) resampled to month-start."""
    raw = yf.download("^IRX", start=START_DATE, end=END_DATE,
                      auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = raw["Close"].resample("MS").first().dropna()
    return s  # already in % annualised (e.g. 5.25)


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


def _lookup(series: pd.Series, month_start: pd.Timestamp, default: float) -> float:
    val = series.get(month_start, None)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        idx = min(series.index.searchsorted(month_start), len(series) - 1)
        val = float(series.iloc[idx])
    return float(val)


def backtest_single(ticker: str, vix_ms: pd.Series, rfr_ms: pd.Series,
                    iv_history=None) -> pd.DataFrame:
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
            vix_val = _lookup(vix_ms, trade_month_start, 20.0)
            rfr_val = _lookup(rfr_ms, trade_month_start, 4.0)

            iv_pct  = get_iv_pct(iv_history, ticker, trade_month_start, vix_val)
            prem    = bs_put_premium(iv_pct, rfr_val)
            strike  = signal_close

            if expiry_close >= strike:
                pnl_pct = prem
                outcome = "expired"
            else:
                loss_pct = (strike - expiry_close) / strike
                pnl_pct  = prem - loss_pct
                outcome  = "assigned"
            trade = True
        else:
            vix_val = float("nan")
            rfr_val = float("nan")
            iv_pct  = float("nan")
            prem    = 0.0
            pnl_pct = 0.0
            outcome = "no_trade"
            trade   = False

        records.append({
            "date": expiry_date, "ticker": ticker, "trade": trade,
            "outcome": outcome, "vix": vix_val, "rfr": rfr_val,
            "iv_pct": iv_pct, "premium": prem, "pnl_pct": pnl_pct,
        })

    return pd.DataFrame(records).set_index("date")


def current_month_estimate():
    raw_vix = yf.download("^VIX", period="5d", auto_adjust=False, progress=False)
    raw_rfr = yf.download("^IRX", period="5d", auto_adjust=False, progress=False)
    if isinstance(raw_vix.columns, pd.MultiIndex):
        raw_vix.columns = raw_vix.columns.get_level_values(0)
    if isinstance(raw_rfr.columns, pd.MultiIndex):
        raw_rfr.columns = raw_rfr.columns.get_level_values(0)
    vix = float(raw_vix["Close"].iloc[-1])
    rfr = float(raw_rfr["Close"].iloc[-1]) if not raw_rfr.empty else 4.0

    print("\n" + "="*68)
    print(f"  CURRENT MONTH ESTIMATES  (VIX={vix:.2f}, r={rfr:.2f}%)")
    print(f"  IV = VIX × RV_ratio × {IV_SCALE}  |  premium = full Black-Scholes ATM put")
    print("="*68)
    print(f"  {'Ticker':6s}  {'RV ratio':>8s}  {'IV est':>7s}  {'BS put':>8s}")
    print("  " + "-"*40)
    prems = []
    for tkr in TICKERS:
        iv   = iv_estimate(vix, tkr)
        prem = bs_put_premium(iv, rfr) * 100
        rv   = STOCK_RV_RATIOS.get(tkr, 0)
        prems.append(prem)
        print(f"  {tkr:6s}  {rv:>8.2f}x  {iv:>6.1f}%  {prem:>7.2f}%")
    print("  " + "-"*40)
    print(f"  {'AVG':6s}  {'':>8s}  {'':>7s}  {np.mean(prems):>7.2f}%")
    print("="*68)

    print(f"\n  VIX-to-BS-premium table (avg portfolio, r={rfr:.1f}%):")
    print(f"  {'VIX':>5s}  {'Avg IV':>8s}  {'BS put':>8s}")
    for v in [12, 15, 18, 20, 25, 30, 35, 40]:
        ivs = [iv_estimate(v, t) for t in TICKERS]
        avg_iv   = np.mean(ivs)
        avg_bs   = np.mean([bs_put_premium(iv, rfr)*100 for iv in ivs])
        print(f"  {v:>5d}  {avg_iv:>7.1f}%  {avg_bs:>7.2f}%")

    return vix, rfr


def run_backtest() -> dict:
    iv_history   = load_iv_history()
    using_real_iv = iv_history is not None

    print("Fetching VIX and risk-free rate ...")
    vix_ms = fetch_vix_monthly()
    rfr_ms = fetch_rfr_monthly()
    print(f"Running backtest [{START_DATE} → {END_DATE}]  "
          f"IV source: {'AlphaQuery CSV' if using_real_iv else f'VIX × RV × {IV_SCALE}'}  "
          f"premium: full Black-Scholes\n")

    all_pnl = []
    for tkr in TICKERS:
        try:
            result = backtest_single(tkr, vix_ms, rfr_ms, iv_history)
            if result.empty:
                print(f"  {tkr:6s}  — skipped")
                continue
            result["weighted_pnl"] = result["pnl_pct"] * WEIGHT
            all_pnl.append(result[["weighted_pnl", "vix", "rfr", "iv_pct", "premium"]])

            trades   = int(result["trade"].sum())
            wins     = int((result["outcome"] == "expired").sum())
            assigned = int((result["outcome"] == "assigned").sum())
            avg_iv   = result.loc[result["trade"], "iv_pct"].mean()
            avg_prem = result.loc[result["trade"], "premium"].mean() * 100
            tot_pnl  = result["pnl_pct"].sum() * 100
            print(f"  {tkr:6s}  trades={trades:3d}  expired={wins:3d}  assigned={assigned:3d}"
                  f"  avg_IV={avg_iv:.0f}%  avg_prem={avg_prem:.2f}%  total_pnl={tot_pnl:+.1f}%")
        except Exception as e:
            print(f"  {tkr:6s}  ERROR: {e}")

    if not all_pnl:
        return {}

    combined    = pd.concat(all_pnl)
    monthly_pnl = combined.groupby(combined.index)["weighted_pnl"].sum().sort_index()
    monthly_pnl.index.name = "date"
    avg_vix_monthly = combined.groupby(combined.index)["vix"].mean()
    avg_prem_pct    = combined.loc[combined["premium"] > 0, "premium"].mean() * 100
    avg_iv_pct      = combined.loc[combined["iv_pct"] > 0, "iv_pct"].mean()

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

    print("\n" + "="*60)
    print("  PORTFOLIO AGGREGATE RESULTS")
    print(f"  IV model: VIX × RV_ratio × {IV_SCALE}  |  premium: full Black-Scholes")
    print("="*60)
    print(f"  Period          : {monthly_pnl.index[0].date()} → {monthly_pnl.index[-1].date()}")
    print(f"  Months          : {n_months}")
    print(f"  Avg VIX         : {avg_vix:.1f}")
    print(f"  Avg IV (traded) : {avg_iv_pct:.1f}%")
    print(f"  Avg BS premium  : {avg_prem_pct:.2f}%")
    avg_rfr = combined.loc[combined["premium"] > 0, "rfr"].mean() if "rfr" in combined.columns else float("nan")
    print(f"  Avg risk-free   : {avg_rfr:.2f}%")
    print(f"  Total Return    : {total_return*100:+.2f}%")
    print(f"  Ann. Return     : {ann_return*100:+.2f}%")
    print(f"  Ann. Volatility : {ann_vol*100:.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:.2f}")
    print(f"  Max Drawdown    : {max_dd*100:.2f}%")
    print(f"  Win months      : {win_months}  ({win_months/n_months*100:.0f}%)")
    print(f"  Flat months     : {flat_months}  ({flat_months/n_months*100:.0f}%)")
    print(f"  Loss months     : {loss_months}  ({loss_months/n_months*100:.0f}%)")
    print("="*60)

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
        "avg_iv_pct":      avg_iv_pct,
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
