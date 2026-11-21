"""
Indian Long-Only Quality Strategy Backtest
============================================
Adaptation of Rajan Raju (2019):
"Implementing a Systematic Long-only Quality Strategy in the Hong Kong Market"

Strategy:
  - Universe: NIFTY100 stocks
  - Signal: Composite quality score (Profitability + Growth + Safety)
  - Selection: Top decile (10 stocks) by quality score
  - Weighting: Equal weight
  - Rebalancing: Quarterly

Quality components:
  - Profitability: ROE, ROA, Operating Margin
  - Growth: 3-year revenue CAGR, 3-year earnings CAGR, operating margin stability
  - Safety: Beta (vs Nifty), volatility, debt-to-equity

Note: This uses free yfinance fundamental data which has limitations.
      ESG is excluded due to lack of free point-in-time ESG data.

Run: python indian_quality_strategy_backtest.py
"""

import os
import time
import warnings
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

START_DATE = "2012-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")

IS_START, IS_END = "2015-01-01", "2019-12-31"
OOS_START, OOS_END = "2020-01-01", END_DATE

INITIAL_CAPITAL = 1_000_000.0
TOP_N_STOCKS = 10                  # top decile of ~100 stocks
MIN_HISTORY_DAYS = 252             # ~1 year of price history
COST_PER_TRADE_PCT = 0.001         # ~0.1% one-way cost
BENCHMARK_TICKER = "^CNX100"       # NIFTY100 index

SLEEP_BETWEEN_DOWNLOADS = 0.3

# ---------------------------------------------------------------------------
# NIFTY100 STOCK UNIVERSE (current constituents; update as needed)
# ---------------------------------------------------------------------------

NIFTY100_STOCKS = [
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
    "ABCAPITAL.NS", "ABB.NS", "ACC.NS", "ADANIENSOL.NS", "AMBUJACEM.NS",
    "ASHOKLEY.NS", "AUBANK.NS", "AUROPHARMA.NS", "BAJAJHLDNG.NS", "BANDHANBNK.NS",
    "BANKBARODA.NS", "BERGEPAINT.NS", "BHEL.NS", "BOSCHLTD.NS", "CANBK.NS",
    "CHOLAFIN.NS", "COLPAL.NS", "DABUR.NS", "DLF.NS", "DMART.NS",
    "GAIL.NS", "GODREJCP.NS", "HAL.NS", "HAVELLS.NS", "HINDPETRO.NS",
    "HINDZINC.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDBI.NS", "INDIGO.NS",
    "IOC.NS", "IRCTC.NS", "JINDALSTEL.NS", "JIOFIN.NS", "LICI.NS",
    "LODHA.NS", "LTIM.NS", "MARICO.NS", "MCDOWELL-N.NS", "MOTHERSON.NS",
    "MUTHOOTFIN.NS", "NAUKRI.NS", "NHPC.NS", "NMDC.NS", "OBEROIRLTY.NS",
    "OFSS.NS", "PEL.NS", "PFC.NS", "PIDILITIND.NS", "POLYCAB.NS",
    "RECLTD.NS", "SAIL.NS", "SHREECEM.NS", "SIEMENS.NS", "SRF.NS",
    "TATAPOWER.NS", "TORNTPHARM.NS", "TVSMOTOR.NS", "UNIONBANK.NS", "VEDL.NS",
    "VOLTAS.NS", "ZYDUSLIFE.NS"
]

# ---------------------------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_price_data(ticker: str, start: str, end: str,
                        cache_dir: str = "data_cache_prices") -> Optional[pd.DataFrame]:
    os.makedirs(cache_dir, exist_ok=True)
    safe_ticker = ticker.replace("/", "_").replace("^", "_")
    cache_file = os.path.join(cache_dir, f"{safe_ticker}_{start}_{end}.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if not df.empty:
            return df
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < MIN_HISTORY_DAYS:
            return None
        df = _flatten_columns(df)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
        df.to_csv(cache_file)
        return df
    except Exception as e:
        print(f"  [!] Price download failed {ticker}: {e}")
        return None


def download_fundamentals(ticker: str,
                          cache_dir: str = "data_cache_fundamentals") -> Optional[Dict]:
    """Download annual financial statements from yfinance."""
    os.makedirs(cache_dir, exist_ok=True)
    safe_ticker = ticker.replace("/", "_").replace("^", "_")
    cache_file = os.path.join(cache_dir, f"{safe_ticker}_fundamentals.pkl")
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)
    try:
        stock = yf.Ticker(ticker)
        data = {
            "income_stmt": stock.income_stmt,
            "balance_sheet": stock.balance_sheet,
            "cash_flow": stock.cash_flow,
            "info": stock.info,
        }
        pd.to_pickle(data, cache_file)
        return data
    except Exception as e:
        print(f"  [!] Fundamentals download failed {ticker}: {e}")
        return None


def compute_monthly_open_close(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to quarterly open (first day) and close (last day)."""
    quarterly = df.resample("QE").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })
    if hasattr(quarterly.index, "to_timestamp"):
        quarterly.index = quarterly.index.to_timestamp()
    return quarterly


# ---------------------------------------------------------------------------
# QUALITY SCORE COMPONENTS
# ---------------------------------------------------------------------------

def safe_get(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
    """Get a row from financial statement if it exists."""
    if df is None or df.empty:
        return None
    # Try exact match and case-insensitive match
    for idx in df.index:
        if key.lower() in idx.lower():
            return df.loc[idx]
    return None


def compute_profitability_score(fund: Dict) -> Optional[float]:
    """Compute profitability z-score from ROE, ROA, operating margin."""
    income = fund.get("income_stmt")
    balance = fund.get("balance_sheet")
    info = fund.get("info", {})

    scores = []

    # ROE
    roe = info.get("returnOnEquity")
    if roe is not None and not pd.isna(roe):
        scores.append(roe)

    # ROA
    roa = info.get("returnOnAssets")
    if roa is not None and not pd.isna(roa):
        scores.append(roa)

    # Operating margin
    opm = info.get("operatingMargins")
    if opm is not None and not pd.isna(opm):
        scores.append(opm)

    if not scores:
        return None
    return np.mean(scores)


def compute_growth_score(fund: Dict) -> Optional[float]:
    """Compute growth score from revenue and earnings growth."""
    info = fund.get("info", {})

    scores = []
    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None and not pd.isna(rev_growth):
        scores.append(rev_growth)

    earn_growth = info.get("earningsGrowth")
    if earn_growth is not None and not pd.isna(earn_growth):
        scores.append(earn_growth)

    if not scores:
        return None
    return np.mean(scores)


def compute_safety_score(fund: Dict, daily_close: pd.Series,
                         market_close: pd.Series) -> Optional[float]:
    """Compute safety score from beta, volatility, and debt-to-equity."""
    info = fund.get("info", {})
    scores = []

    # Beta (from info)
    beta = info.get("beta")
    if beta is not None and not pd.isna(beta) and beta > 0:
        scores.append(-beta)  # lower beta is better

    # Volatility (annualized from daily returns)
    if len(daily_close) >= 126:
        rets = np.log(daily_close / daily_close.shift(1)).dropna()
        vol = rets.std() * np.sqrt(252)
        if not pd.isna(vol) and vol > 0:
            scores.append(-vol)

    # Debt-to-equity (lower is safer)
    de = info.get("debtToEquity")
    if de is not None and not pd.isna(de):
        scores.append(-de / 100.0)  # scale to roughly same range

    if not scores:
        return None
    return np.mean(scores)


def compute_quality_scores(tickers: List[str], fundamentals: Dict[str, Dict],
                           price_data: Dict[str, pd.DataFrame],
                           market_close: pd.Series) -> Dict[str, Optional[float]]:
    """Compute composite quality score for each stock."""
    prof = {}
    growth = {}
    safety = {}

    for ticker in tickers:
        if ticker not in fundamentals or ticker not in price_data:
            continue
        fund = fundamentals[ticker]
        close = price_data[ticker]["Close"]

        p = compute_profitability_score(fund)
        g = compute_growth_score(fund)
        s = compute_safety_score(fund, close, market_close)

        if p is not None:
            prof[ticker] = p
        if g is not None:
            growth[ticker] = g
        if s is not None:
            safety[ticker] = s

    def z_score(values: Dict[str, float]) -> Dict[str, float]:
        if len(values) < 3:
            return {}
        s = pd.Series(values)
        s = s.clip(lower=s.quantile(0.05), upper=s.quantile(0.95))
        mean, std = s.mean(), s.std()
        if std <= 0:
            return {}
        return {t: (v - mean) / std for t, v in values.items()}

    z_prof = z_score(prof)
    z_growth = z_score(growth)
    z_safety = z_score(safety)

    quality = {}
    for ticker in set(z_prof.keys()) | set(z_growth.keys()) | set(z_safety.keys()):
        components = []
        if ticker in z_prof:
            components.append(z_prof[ticker])
        if ticker in z_growth:
            components.append(z_growth[ticker])
        if ticker in z_safety:
            components.append(z_safety[ticker])
        if len(components) >= 2:  # require at least 2 of 3 components
            quality[ticker] = np.mean(components)

    return quality


# ---------------------------------------------------------------------------
# BACKTEST ENGINE
# ---------------------------------------------------------------------------

def run_backtest(price_data: Dict[str, pd.DataFrame],
                 benchmark_data: Optional[pd.DataFrame],
                 quality_scores_by_date: Dict[pd.Timestamp, Dict[str, float]],
                 start_date: str, end_date: str) -> pd.DataFrame:
    """Run quarterly quality backtest."""

    quarterly_data = {t: compute_monthly_open_close(d) for t, d in price_data.items()}
    close_panel = pd.DataFrame({t: q["Close"] for t, q in quarterly_data.items()})
    open_panel = pd.DataFrame({t: q["Open"] for t, q in quarterly_data.items()})

    # Use quarters where we have quality scores
    quarters = sorted(quality_scores_by_date.keys())
    quarters = [q for q in quarters if start_date <= q.strftime("%Y-%m-%d") <= end_date]

    portfolio_value = INITIAL_CAPITAL
    equity_curve = []

    for i, q_date in enumerate(quarters[:-1]):
        quality = quality_scores_by_date[q_date]
        if not quality:
            continue

        sorted_stocks = sorted(quality.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in sorted_stocks[:TOP_N_STOCKS]]

        entry_date = q_date
        exit_date = quarters[i + 1]

        entry_prices = open_panel.loc[entry_date, selected]
        exit_prices = open_panel.loc[exit_date, selected]
        valid = entry_prices.notna() & exit_prices.notna() & (entry_prices > 0)

        quarterly_return = 0.0
        if valid.sum() > 0:
            stock_rets = (exit_prices[valid] / entry_prices[valid]) - 1.0
            quarterly_return = stock_rets.mean()

        cost = COST_PER_TRADE_PCT * 2
        net_return = quarterly_return - cost
        portfolio_value *= (1 + net_return)

        equity_curve.append({
            "Date": exit_date,
            "PortfolioValue": portfolio_value,
            "GrossReturn": quarterly_return,
            "NetReturn": net_return,
            "Cost": cost,
            "NumStocks": valid.sum(),
            "Selected": ",".join(selected)
        })

    equity_df = pd.DataFrame(equity_curve).set_index("Date")
    return equity_df


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

    vol = rets.std() * np.sqrt(4)  # quarterly
    sharpe = (rets.mean() * 4) / vol if vol > 0 else 0

    metrics = {
        "start": equity.index[0].strftime("%Y-%m-%d"),
        "end": equity.index[-1].strftime("%Y-%m-%d"),
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "max_dd_pct": max_dd * 100,
        "quarterly_vol_annualized": vol * 100,
        "sharpe_annualized": sharpe,
        "avg_quarterly_return_pct": rets.mean() * 100,
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
        bench_vol = bench_rets.std() * np.sqrt(4)
        bench_sharpe = (bench_rets.mean() * 4) / bench_vol if bench_vol > 0 else 0

        metrics.update({
            "bench_total_return_pct": bench_total * 100,
            "bench_cagr_pct": bench_cagr * 100,
            "bench_max_dd_pct": bench_max_dd * 100,
            "bench_sharpe_annualized": bench_sharpe,
            "excess_cagr_pct": (cagr - bench_cagr) * 100,
        })

    return metrics


def compute_benchmark_return(benchmark_data: pd.DataFrame,
                             equity_dates: pd.DatetimeIndex) -> pd.Series:
    """Compute quarterly open-to-open returns for benchmark."""
    quarterly = compute_monthly_open_close(benchmark_data)
    quarterly["Return"] = quarterly["Open"].shift(-1) / quarterly["Open"] - 1
    return quarterly["Return"].reindex(equity_dates).dropna()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Indian Long-Only Quality Strategy Backtest")
    print("=" * 70)
    print(f"Full period: {START_DATE} to {END_DATE}")
    print(f"In-Sample : {IS_START} to {IS_END}")
    print(f"Out-of-Sample : {OOS_START} to {OOS_END}")
    print(f"Top N stocks: {TOP_N_STOCKS}")
    print("=" * 70)

    print("\nDownloading stock price data...")
    price_data = {}
    for idx, ticker in enumerate(NIFTY100_STOCKS, 1):
        print(f"  [{idx}/{len(NIFTY100_STOCKS)}] {ticker}")
        df = download_price_data(ticker, START_DATE, END_DATE)
        if df is not None and not df.empty:
            price_data[ticker] = df
        time.sleep(SLEEP_BETWEEN_DOWNLOADS)

    valid_stocks = list(price_data.keys())
    print(f"\nValid stocks: {len(valid_stocks)} / {len(NIFTY100_STOCKS)}")

    print("\nDownloading benchmark data...")
    benchmark_data = download_price_data(BENCHMARK_TICKER, START_DATE, END_DATE,
                                         cache_dir="data_cache_prices")
    market_close = benchmark_data["Close"] if benchmark_data is not None else None

    print("\nDownloading fundamentals (this may take a while)...")
    fundamentals = {}
    for idx, ticker in enumerate(valid_stocks, 1):
        print(f"  [{idx}/{len(valid_stocks)}] {ticker}")
        fund = download_fundamentals(ticker)
        if fund is not None:
            fundamentals[ticker] = fund
        time.sleep(SLEEP_BETWEEN_DOWNLOADS)

    print(f"\nValid fundamentals: {len(fundamentals)} / {len(valid_stocks)}")

    # Compute quality scores at each quarter-end
    print("\nComputing quality scores at each quarter-end...")
    quarterly_dates = pd.date_range(START_DATE, END_DATE, freq="QE")
    if hasattr(quarterly_dates, "to_timestamp"):
        quarterly_dates = quarterly_dates.to_timestamp()

    quality_scores_by_date = {}
    for q_date in quarterly_dates:
        # Use market data up to this quarter
        if market_close is not None:
            mkt = market_close.loc[:q_date]
        else:
            mkt = None
        scores = compute_quality_scores(valid_stocks, fundamentals, price_data, mkt)
        if scores:
            quality_scores_by_date[q_date] = scores

    print(f"Quality scores computed for {len(quality_scores_by_date)} quarters")

    print("\nRunning backtest...")
    equity_full = run_backtest(price_data, benchmark_data, quality_scores_by_date,
                               START_DATE, END_DATE)

    is_mask = (equity_full.index >= IS_START) & (equity_full.index <= IS_END)
    oos_mask = (equity_full.index >= OOS_START) & (equity_full.index <= OOS_END)

    bench_rets = None
    if benchmark_data is not None:
        bench_rets = compute_benchmark_return(benchmark_data, equity_full.index)

    metrics_full = calculate_metrics(equity_full["PortfolioValue"], bench_rets)
    metrics_is = calculate_metrics(equity_full.loc[is_mask, "PortfolioValue"],
                                   bench_rets.loc[is_mask] if bench_rets is not None else None)
    metrics_oos = calculate_metrics(equity_full.loc[oos_mask, "PortfolioValue"],
                                    bench_rets.loc[oos_mask] if bench_rets is not None else None)

    equity_full.to_csv("quality_equity_curve.csv")
    print("\n[+] Saved quality_equity_curve.csv")

    # Save latest quality ranking
    latest_date = max(quality_scores_by_date.keys())
    latest_scores = quality_scores_by_date[latest_date]
    ranking = pd.DataFrame([
        {"Ticker": t, "QualityScore": s}
        for t, s in sorted(latest_scores.items(), key=lambda x: x[1], reverse=True)
    ])
    ranking.to_csv("quality_latest_ranking.csv", index=False)
    print("[+] Saved quality_latest_ranking.csv")

    # Report
    def format_metrics(m: Dict, label: str) -> List[str]:
        lines = [
            f"{label}",
            "-" * 50,
            f"Total return        : {m['total_return_pct']:.2f}%",
            f"CAGR                : {m['cagr_pct']:.2f}%",
            f"Max drawdown        : {m['max_dd_pct']:.2f}%",
            f"Quarterly vol (ann.): {m['quarterly_vol_annualized']:.2f}%",
            f"Sharpe (ann.)       : {m['sharpe_annualized']:.2f}",
            f"Avg quarterly ret.  : {m['avg_quarterly_return_pct']:.2f}%",
        ]
        if "bench_cagr_pct" in m:
            lines.extend([
                "",
                "BENCHMARK (^CNX100)",
                "-" * 50,
                f"Total return        : {m['bench_total_return_pct']:.2f}%",
                f"CAGR                : {m['bench_cagr_pct']:.2f}%",
                f"Max drawdown        : {m['bench_max_dd_pct']:.2f}%",
                f"Sharpe (ann.)       : {m['bench_sharpe_annualized']:.2f}",
                f"Excess CAGR         : {m['excess_cagr_pct']:.2f}%",
            ])
        lines.append("")
        return lines

    report_lines = [
        "INDIAN LONG-ONLY QUALITY STRATEGY REPORT",
        "=" * 50,
        f"Full period: {metrics_full['start']} to {metrics_full['end']}",
        f"In-Sample: {IS_START} to {IS_END}",
        f"Out-of-Sample: {OOS_START} to {OOS_END}",
        f"Top N stocks: {TOP_N_STOCKS}",
        f"Cost: {COST_PER_TRADE_PCT * 100:.2f}% one-way",
        f"Note: ESG excluded; uses simplified quality proxy from yfinance",
        "",
    ]
    report_lines.extend(format_metrics(metrics_full, "FULL PERIOD PERFORMANCE"))
    report_lines.extend(format_metrics(metrics_is, "IN-SAMPLE PERFORMANCE"))
    report_lines.extend(format_metrics(metrics_oos, "OUT-OF-SAMPLE PERFORMANCE"))

    report = "\n".join(report_lines)
    print("\n" + report)
    with open("quality_summary_report.txt", "w") as f:
        f.write(report)
    print("\n[+] Saved quality_summary_report.txt")


if __name__ == "__main__":
    main()
