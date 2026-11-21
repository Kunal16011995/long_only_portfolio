"""
NIFTY100 Long-Only Momentum Strategy Backtest
===============================================
Based on: Rajan Raju & Abhijit Chandrasekaran (2019)
          "Implementing a Systematic Long-only Momentum Strategy: Evidence From India"

Strategy:
  - Universe: NIFTY100 constituents
  - Signal: 12-month return skipping the most recent month (t-12 to t-1)
  - Selection: Top decile (default 10 stocks) by momentum score
  - Weighting: Equal weight
  - Rebalancing: Monthly (rank at month-end close, execute next month open)
  - Holding: 1 month (open-to-open returns)

Data source: Yahoo Finance via yfinance
Run: python nifty100_momentum_backtest.py
"""

import os
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

START_DATE = "2010-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")

# In-sample / Out-of-sample split
IS_START, IS_END = "2011-01-01", "2019-12-31"   # matches paper period roughly
OOS_START, OOS_END = "2020-01-01", END_DATE

# Strategy parameters
MOMENTUM_LOOKBACK_MONTHS = 12    # 12-month lookback
SKIP_RECENT_MONTHS = 1           # skip most recent month (12-1 momentum)
TOP_N_STOCKS = 10                # top decile of NIFTY100
COST_PER_TRADE_PCT = 0.001       # ~0.1% one-way cost (STT + charges + slippage)
INITIAL_CAPITAL = 1_000_000.0    # INR

# Benchmark
BENCHMARK_TICKER = "^CNX100"     # NIFTY100 index on Yahoo Finance

# Rate limit safety for Yahoo downloads
SLEEP_BETWEEN_DOWNLOADS = 0.3

# ---------------------------------------------------------------------------
# NIFTY100 CONSTITUENTS (current list; update as needed)
# Note: Using current constituents introduces survivorship bias in long backtests.
# ---------------------------------------------------------------------------

NIFTY100_TICKERS = [
    # Nifty 50
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "BPCL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
    "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS",
    "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    # Nifty Next 50 (sample; update to full current list for live use)
    "ABCAPITAL.NS", "ABB.NS", "ACC.NS", "ADANIENSOL.NS", "AMBUJACEM.NS",
    "ASHOKLEY.NS", "AUBANK.NS", "AUROPHARMA.NS", "BAJAJHLDNG.NS", "BANDHANBNK.NS",
    "BANKBARODA.NS", "BERGEPAINT.NS", "BHEL.NS", "BOSCHLTD.NS", "CANBK.NS",
    "CHOLAFIN.NS", "COLPAL.NS", "DABUR.NS", "DLF.NS", "DMART.NS",
    "GAIL.NS", "GODREJCP.NS", "HAL.NS", "HAVELLS.NS", "HINDPETRO.NS",
    "HINDZINC.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDBI.NS", "INDIGO.NS",
    "IOC.NS", "IRCTC.NS", "JINDALSTEL.NS", "JIOFIN.NS", "LICI.NS",
    "LODHA.NS", "LTIM.NS", "MARICO.NS", "MCDOWELL-N.NS", "MOTHERSON.NS",
    "MUTHOOTFIN.NS", "NAUKRI.NS", "NESTLEIND.NS", "NHPC.NS", "NMDC.NS",
    "OBEROIRLTY.NS", "OFSS.NS", "PAYTM.NS", "PEL.NS", "PFC.NS",
    "PIDILITIND.NS", "POLYCAB.NS", "RECLTD.NS", "SAIL.NS", "SHREECEM.NS",
    "SIEMENS.NS", "SRF.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TVSMOTOR.NS",
    "UNIONBANK.NS", "VEDL.NS", "VOLTAS.NS", "ZOMATO.NS", "ZYDUSLIFE.NS"
]

# ---------------------------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_data(ticker: str, start: str, end: str, interval: str = "1d",
                  cache_dir: str = "data_cache_momentum") -> Optional[pd.DataFrame]:
    """Download daily OHLCV from Yahoo Finance with local caching."""
    os.makedirs(cache_dir, exist_ok=True)
    safe_ticker = ticker.replace("/", "_").replace("^", "_")
    cache_file = os.path.join(cache_dir, f"{safe_ticker}_{start}_{end}_{interval}.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if not df.empty:
            return df
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            return None
        df = _flatten_columns(df)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
        df.to_csv(cache_file)
        return df
    except Exception as e:
        print(f"  [!] Failed to download {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# MONTHLY MOMENTUM ENGINE
# ---------------------------------------------------------------------------

def compute_monthly_open_close(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to monthly open (first day) and close (last day)."""
    # Group by month, take first Open and last Close/High/Low/Volume
    monthly = df.resample("ME").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })
    # pandas version compatibility: PeriodIndex needs conversion
    if hasattr(monthly.index, "to_timestamp"):
        monthly.index = monthly.index.to_timestamp()
    return monthly


def compute_momentum_score(monthly_close: pd.Series, lookback: int = 12, skip: int = 1) -> Optional[float]:
    """
    12-1 momentum: return from month (t - lookback) to month (t - skip).
    Returns None if not enough history.
    """
    if len(monthly_close) < lookback + 1:
        return None
    old_price = monthly_close.iloc[-(lookback + 1)]  # t - 12
    recent_price = monthly_close.iloc[-(skip + 1)]   # t - 1
    if pd.isna(old_price) or pd.isna(recent_price) or old_price <= 0:
        return None
    return (recent_price / old_price) - 1.0


def select_top_momentum_stocks(momentum_scores: Dict[str, float], n: int) -> List[str]:
    """Select top N stocks by momentum score. Exclude negative/zero scores."""
    valid = {k: v for k, v in momentum_scores.items() if v is not None and v > 0}
    sorted_stocks = sorted(valid.items(), key=lambda x: x[1], reverse=True)
    return [ticker for ticker, _ in sorted_stocks[:n]]


# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------

@dataclass
class Rebalance:
    date: pd.Timestamp
    selected_stocks: List[str]
    momentum_scores: Dict[str, float]
    prev_stocks: List[str]
    turnover: float


def run_backtest(stock_data: Dict[str, pd.DataFrame],
                 benchmark_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[Rebalance]]:
    """
    Run monthly momentum backtest.
    Returns equity curve DataFrame and list of rebalances.
    """
    # Build monthly close/open panels
    monthly_panels = {}
    for ticker, df in stock_data.items():
        if df is None or df.empty:
            continue
        monthly = compute_monthly_open_close(df)
        if len(monthly) < MOMENTUM_LOOKBACK_MONTHS + 2:
            continue
        monthly_panels[ticker] = monthly

    if not monthly_panels:
        raise ValueError("No valid stock data available.")

    # Align all monthly close series into a DataFrame
    close_panel = pd.DataFrame({t: m["Close"] for t, m in monthly_panels.items()})
    open_panel = pd.DataFrame({t: m["Open"] for t, m in monthly_panels.items()})

    # Common month-end dates where we have enough data
    common_months = close_panel.index

    portfolio_value = INITIAL_CAPITAL
    equity_curve = []
    rebalances = []
    prev_selected = []

    # We need lookback months to compute the first signal, then one more month for first return
    start_idx = MOMENTUM_LOOKBACK_MONTHS

    for i in range(start_idx, len(common_months) - 1):
        signal_date = common_months[i]

        # Momentum: using close up to month i-1 (i.e., skip most recent month i)
        # Actually 12-1 momentum at month i means compare close[i-12] to close[i-1]
        lookback_close = close_panel.iloc[:i]  # up to month i-1
        momentum_scores = {}
        for ticker in lookback_close.columns:
            series = lookback_close[ticker].dropna()
            score = compute_momentum_score(series, MOMENTUM_LOOKBACK_MONTHS, SKIP_RECENT_MONTHS)
            if score is not None:
                momentum_scores[ticker] = score

        selected = select_top_momentum_stocks(momentum_scores, TOP_N_STOCKS)

        # Entry: open of month i+1, Exit: open of month i+2
        entry_date = common_months[i + 1]
        exit_date = common_months[i + 2] if i + 2 < len(common_months) else None

        if exit_date is None:
            break

        # Compute open-to-open returns for selected stocks
        entry_prices = open_panel.loc[entry_date, selected]
        exit_prices = open_panel.loc[exit_date, selected]
        valid = entry_prices.notna() & exit_prices.notna() & (entry_prices > 0)
        if valid.sum() == 0:
            monthly_return = 0.0
        else:
            stock_returns = (exit_prices[valid] / entry_prices[valid]) - 1.0
            monthly_return = stock_returns.mean()

        # Apply trading costs proportional to turnover
        if prev_selected:
            old_set = set(prev_selected)
            new_set = set(selected)
            turnover = len(old_set.symmetric_difference(new_set)) / max(len(old_set), len(new_set))
        else:
            turnover = 1.0
        cost = turnover * COST_PER_TRADE_PCT * 2  # round-trip cost
        net_monthly_return = monthly_return - cost

        portfolio_value *= (1 + net_monthly_return)

        equity_curve.append({
            "Date": exit_date,
            "PortfolioValue": portfolio_value,
            "GrossReturn": monthly_return,
            "NetReturn": net_monthly_return,
            "Turnover": turnover,
            "Cost": cost,
            "NumStocks": valid.sum()
        })

        rebalances.append(Rebalance(
            date=signal_date,
            selected_stocks=selected,
            momentum_scores=momentum_scores,
            prev_stocks=prev_selected,
            turnover=turnover
        ))

        prev_selected = selected

    equity_df = pd.DataFrame(equity_curve).set_index("Date")
    return equity_df, rebalances


# ---------------------------------------------------------------------------
# BENCHMARK
# ---------------------------------------------------------------------------

def compute_benchmark_return(benchmark_data: pd.DataFrame, equity_dates: pd.DatetimeIndex) -> pd.Series:
    """Compute open-to-open monthly returns for the benchmark aligned with strategy dates."""
    monthly = compute_monthly_open_close(benchmark_data)
    monthly["Return"] = monthly["Open"].shift(-1) / monthly["Open"] - 1
    returns = monthly["Return"].reindex(equity_dates)
    return returns.dropna()


# ---------------------------------------------------------------------------
# PERFORMANCE METRICS
# ---------------------------------------------------------------------------

def calculate_metrics(equity: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict:
    rets = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 0.25)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1

    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    vol = rets.std() * np.sqrt(12)
    sharpe = (rets.mean() * 12) / vol if vol > 0 else 0

    metrics = {
        "start": equity.index[0].strftime("%Y-%m-%d"),
        "end": equity.index[-1].strftime("%Y-%m-%d"),
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "max_dd_pct": max_dd * 100,
        "monthly_vol_annualized": vol * 100,
        "sharpe_annualized": sharpe,
        "avg_monthly_return_pct": rets.mean() * 100,
        "avg_turnover_pct": 0,  # filled later
    }

    if benchmark is not None and not benchmark.empty:
        bench_rets = benchmark.dropna()
        bench_equity = (1 + bench_rets).cumprod() * INITIAL_CAPITAL
        bench_total = bench_equity.iloc[-1] / bench_equity.iloc[0] - 1
        bench_years = max((bench_equity.index[-1] - bench_equity.index[0]).days / 365.25, 0.25)
        bench_cagr = (bench_equity.iloc[-1] / bench_equity.iloc[0]) ** (1 / bench_years) - 1
        bench_peak = bench_equity.cummax()
        bench_dd = (bench_equity - bench_peak) / bench_peak
        bench_max_dd = bench_dd.min()
        bench_vol = bench_rets.std() * np.sqrt(12)
        bench_sharpe = (bench_rets.mean() * 12) / bench_vol if bench_vol > 0 else 0

        metrics.update({
            "bench_total_return_pct": bench_total * 100,
            "bench_cagr_pct": bench_cagr * 100,
            "bench_max_dd_pct": bench_max_dd * 100,
            "bench_sharpe_annualized": bench_sharpe,
            "excess_cagr_pct": (cagr - bench_cagr) * 100,
        })

    return metrics


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("NIFTY100 Long-Only Momentum Strategy Backtest")
    print("=" * 70)
    print(f"Full period: {START_DATE} to {END_DATE}")
    print(f"In-Sample : {IS_START} to {IS_END}")
    print(f"Out-of-Sample : {OOS_START} to {OOS_END}")
    print(f"Momentum lookback: {MOMENTUM_LOOKBACK_MONTHS} months, skip {SKIP_RECENT_MONTHS} month")
    print(f"Top N stocks: {TOP_N_STOCKS}")
    print(f"Initial capital: Rs {INITIAL_CAPITAL:,.0f}")
    print("=" * 70)

    print("\nDownloading stock data...")
    stock_data = {}
    for idx, ticker in enumerate(NIFTY100_TICKERS, 1):
        print(f"  [{idx}/{len(NIFTY100_TICKERS)}] {ticker}")
        df = download_data(ticker, START_DATE, END_DATE)
        if df is not None and not df.empty:
            stock_data[ticker] = df
        time.sleep(SLEEP_BETWEEN_DOWNLOADS)

    print(f"\nValid stocks with data: {len(stock_data)}")

    print("\nDownloading benchmark data...")
    benchmark_data = download_data(BENCHMARK_TICKER, START_DATE, END_DATE,
                                   cache_dir="data_cache_momentum")
    if benchmark_data is not None:
        print("  Benchmark downloaded.")
    else:
        print("  [!] Benchmark download failed.")

    print("\nRunning backtest...")
    equity_df, rebalances = run_backtest(stock_data, benchmark_data)

    # Benchmark returns aligned to strategy dates
    bench_rets = None
    if benchmark_data is not None:
        bench_rets = compute_benchmark_return(benchmark_data, equity_df.index)

    metrics_full = calculate_metrics(equity_df["PortfolioValue"], bench_rets)
    metrics_full["avg_turnover_pct"] = equity_df["Turnover"].mean() * 100

    # IS / OOS split
    is_mask_eq = (equity_df.index >= IS_START) & (equity_df.index <= IS_END)
    oos_mask_eq = (equity_df.index >= OOS_START) & (equity_df.index <= OOS_END)

    is_mask_bench = (bench_rets.index >= IS_START) & (bench_rets.index <= IS_END) if bench_rets is not None else None
    oos_mask_bench = (bench_rets.index >= OOS_START) & (bench_rets.index <= OOS_END) if bench_rets is not None else None

    metrics_is = calculate_metrics(equity_df.loc[is_mask_eq, "PortfolioValue"],
                                   bench_rets.loc[is_mask_bench] if bench_rets is not None else None)
    metrics_is["avg_turnover_pct"] = equity_df.loc[is_mask_eq, "Turnover"].mean() * 100

    metrics_oos = calculate_metrics(equity_df.loc[oos_mask_eq, "PortfolioValue"],
                                    bench_rets.loc[oos_mask_bench] if bench_rets is not None else None)
    metrics_oos["avg_turnover_pct"] = equity_df.loc[oos_mask_eq, "Turnover"].mean() * 100

    # Save outputs
    equity_df.to_csv("momentum_equity_curve.csv")
    print("\n[+] Saved momentum_equity_curve.csv")

    # Rebalance log
    rebalance_records = []
    for rb in rebalances:
        for stock in rb.selected_stocks:
            rebalance_records.append({
                "Date": rb.date,
                "Stock": stock,
                "MomentumScore": rb.momentum_scores.get(stock),
                "Turnover": rb.turnover
            })
    rebalance_df = pd.DataFrame(rebalance_records)
    rebalance_df.to_csv("momentum_rebalance_log.csv", index=False)
    print("[+] Saved momentum_rebalance_log.csv")

    # Report
    report_lines = [
        "NIFTY100 LONG-ONLY MOMENTUM STRATEGY REPORT",
        "=" * 50,
        f"Full period: {metrics_full['start']} to {metrics_full['end']}",
        f"In-Sample: {IS_START} to {IS_END}",
        f"Out-of-Sample: {OOS_START} to {OOS_END}",
        f"Momentum signal: {MOMENTUM_LOOKBACK_MONTHS}-month return skipping {SKIP_RECENT_MONTHS} month",
        f"Top N stocks: {TOP_N_STOCKS} (equal weight)",
        f"Rebalancing: Monthly (open-to-open)",
        f"Trading cost: {COST_PER_TRADE_PCT * 100:.2f}% one-way",
        "",
        "FULL PERIOD PERFORMANCE",
        "-" * 50,
        f"Total return        : {metrics_full['total_return_pct']:.2f}%",
        f"CAGR                : {metrics_full['cagr_pct']:.2f}%",
        f"Max drawdown        : {metrics_full['max_dd_pct']:.2f}%",
        f"Monthly vol (ann.)  : {metrics_full['monthly_vol_annualized']:.2f}%",
        f"Sharpe (ann.)       : {metrics_full['sharpe_annualized']:.2f}",
        f"Avg monthly return  : {metrics_full['avg_monthly_return_pct']:.2f}%",
        f"Avg turnover        : {metrics_full['avg_turnover_pct']:.2f}%",
        "",
        "IN-SAMPLE PERFORMANCE (2011-2019)",
        "-" * 50,
        f"Total return        : {metrics_is['total_return_pct']:.2f}%",
        f"CAGR                : {metrics_is['cagr_pct']:.2f}%",
        f"Max drawdown        : {metrics_is['max_dd_pct']:.2f}%",
        f"Monthly vol (ann.)  : {metrics_is['monthly_vol_annualized']:.2f}%",
        f"Sharpe (ann.)       : {metrics_is['sharpe_annualized']:.2f}",
        f"Avg monthly return  : {metrics_is['avg_monthly_return_pct']:.2f}%",
        f"Avg turnover        : {metrics_is['avg_turnover_pct']:.2f}%",
    ]

    if "bench_cagr_pct" in metrics_is:
        report_lines.extend([
            "",
            "IN-SAMPLE BENCHMARK (^CNX100)",
            "-" * 50,
            f"Total return        : {metrics_is['bench_total_return_pct']:.2f}%",
            f"CAGR                : {metrics_is['bench_cagr_pct']:.2f}%",
            f"Max drawdown        : {metrics_is['bench_max_dd_pct']:.2f}%",
            f"Sharpe (ann.)       : {metrics_is['bench_sharpe_annualized']:.2f}",
            f"Excess CAGR         : {metrics_is['excess_cagr_pct']:.2f}%",
        ])

    report_lines.extend([
        "",
        "OUT-OF-SAMPLE PERFORMANCE (2020-today)",
        "-" * 50,
        f"Total return        : {metrics_oos['total_return_pct']:.2f}%",
        f"CAGR                : {metrics_oos['cagr_pct']:.2f}%",
        f"Max drawdown        : {metrics_oos['max_dd_pct']:.2f}%",
        f"Monthly vol (ann.)  : {metrics_oos['monthly_vol_annualized']:.2f}%",
        f"Sharpe (ann.)       : {metrics_oos['sharpe_annualized']:.2f}",
        f"Avg monthly return  : {metrics_oos['avg_monthly_return_pct']:.2f}%",
        f"Avg turnover        : {metrics_oos['avg_turnover_pct']:.2f}%",
    ])

    if "bench_cagr_pct" in metrics_oos:
        report_lines.extend([
            "",
            "OUT-OF-SAMPLE BENCHMARK (^CNX100)",
            "-" * 50,
            f"Total return        : {metrics_oos['bench_total_return_pct']:.2f}%",
            f"CAGR                : {metrics_oos['bench_cagr_pct']:.2f}%",
            f"Max drawdown        : {metrics_oos['bench_max_dd_pct']:.2f}%",
            f"Sharpe (ann.)       : {metrics_oos['bench_sharpe_annualized']:.2f}",
            f"Excess CAGR         : {metrics_oos['excess_cagr_pct']:.2f}%",
        ])

    report = "\n".join(report_lines)
    print("\n" + report)
    with open("momentum_summary_report.txt", "w") as f:
        f.write(report)
    print("\n[+] Saved momentum_summary_report.txt")


if __name__ == "__main__":
    main()
