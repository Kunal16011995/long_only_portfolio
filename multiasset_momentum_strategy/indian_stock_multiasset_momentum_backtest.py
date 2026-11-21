"""
Indian Long-Only Multi-Asset Momentum Strategy: Stocks + Defensive ETFs
========================================================================
Adaptation of Zambrano & Rizzolo (2022) for Indian equities.

Strategy:
  - Risk-on universe: NIFTY100 stocks
  - Risk-off universe: Gold, Liquid, Gilt ETFs
  - Signal: 3/6/12-month total return momentum
  - Aggregation: Scores approach
  - Selection: Top N stocks; negative-momentum stocks replaced by best risk-off ETF
  - Optional protection layer: reduce equity exposure when many stocks are negative
  - Rebalancing: Monthly

Data source: Yahoo Finance via yfinance
Run: python indian_stock_multiasset_momentum_backtest.py
"""

import os
import time
import warnings
from dataclasses import dataclass
from datetime import date
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

IS_START, IS_END = "2015-01-01", "2019-12-31"
OOS_START, OOS_END = "2020-01-01", END_DATE

INITIAL_CAPITAL = 1_000_000.0
TOP_N_STOCKS = 10                  # top N stocks by momentum score
MIN_HISTORY_DAYS = 252             # ~1 year of trading history required
COST_PER_TRADE_PCT = 0.001         # ~0.1% one-way cost
NEGATIVE_THRESHOLD = "1/3"         # "1/3" or "1/2"
PROTECTION_Q = 1                   # 0=low, 1=medium, 2=high; None to disable
RISK_OFF_REPLACE = True            # replace negative-momentum stocks with best risk-off

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

RISK_OFF_TICKERS = [
    "GOLDBEES.NS",     # Gold
    "LIQUIDBEES.NS",   # Liquid / cash
    "GILT5YBEES.NS",   # 5-year Gilt
]

ALL_TICKERS = NIFTY100_STOCKS + RISK_OFF_TICKERS

# ---------------------------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def compute_monthly_open_close(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to monthly open (first day) and close (last day)."""
    monthly = df.resample("ME").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })
    if hasattr(monthly.index, "to_timestamp"):
        monthly.index = monthly.index.to_timestamp()
    return monthly


def download_data(ticker: str, start: str, end: str,
                  cache_dir: str = "data_cache_stocks") -> Optional[pd.DataFrame]:
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
        print(f"  [!] Failed to download {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# MOMENTUM SIGNALS
# ---------------------------------------------------------------------------

def total_return_momentum(prices: pd.Series, n_months: int) -> Optional[float]:
    if len(prices) < n_months + 1:
        return None
    return prices.iloc[-1] / prices.iloc[-(n_months + 1)] - 1.0


def compute_momentum_signals(daily_close: pd.Series) -> Dict[str, Optional[float]]:
    """Compute 3/6/12-month total return momentum signals."""
    signals = {}
    for n in [3, 6, 12]:
        signals[f"TR_{n}M"] = total_return_momentum(daily_close, n)
    return signals


# ---------------------------------------------------------------------------
# AGGREGATION
# ---------------------------------------------------------------------------

def scores_approach(signals: Dict[str, Dict[str, float]], top_n: int,
                    negative_threshold: str) -> Tuple[List[str], Dict[str, bool], Dict[str, float]]:
    """
    Aggregate momentum signals using scores approach.
    Returns selected top N, negative flags, and total scores.
    """
    tickers = list(signals.keys())
    if not tickers:
        return [], {}, {}

    num_specs = len(next(iter(signals.values())))
    threshold_count = num_specs // 3 if negative_threshold == "1/3" else num_specs // 2

    total_scores = {t: 0.0 for t in tickers}
    negative_counts = {t: 0 for t in tickers}

    for spec_name in signals[tickers[0]].keys():
        spec_values = {t: signals[t][spec_name] for t in tickers
                       if signals[t][spec_name] is not None}
        if not spec_values:
            continue
        ranked = sorted(spec_values.items(), key=lambda x: x[1], reverse=True)
        n_assets = len(ranked)
        for rank, (ticker, _) in enumerate(ranked, start=1):
            total_scores[ticker] += (n_assets - rank + 1)
            if spec_values[ticker] < 0:
                negative_counts[ticker] += 1

    sorted_tickers = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in sorted_tickers[:top_n]]
    negative_flags = {t: negative_counts[t] > threshold_count for t in tickers}

    return selected, negative_flags, total_scores


# ---------------------------------------------------------------------------
# PROTECTION LAYER
# ---------------------------------------------------------------------------

def risk_on_exposure(num_negative: int, num_risk_on: int, q: Optional[int]) -> float:
    if q is None:
        return 1.0
    denom = (num_risk_on - q) / 4.0
    if denom <= 0:
        return 0.0
    return max(1.0 - num_negative / denom, 0.0)


# ---------------------------------------------------------------------------
# BACKTEST ENGINE
# ---------------------------------------------------------------------------

@dataclass
class Rebalance:
    date: pd.Timestamp
    selected_stocks: List[str]
    final_weights: Dict[str, float]
    negative_flags: Dict[str, bool]
    risk_on_exposure: float


def run_backtest(daily_data: Dict[str, pd.DataFrame],
                 start_date: str, end_date: str) -> Tuple[pd.DataFrame, List[Rebalance]]:
    """Run monthly stock + defensive ETF momentum backtest."""

    # Daily close panel for signals
    daily_close_panel = pd.DataFrame({t: d["Close"] for t, d in daily_data.items()})

    # Monthly open/close panels for execution
    monthly_data = {t: compute_monthly_open_close(d) for t, d in daily_data.items()}
    close_panel = pd.DataFrame({t: m["Close"] for t, m in monthly_data.items()})
    open_panel = pd.DataFrame({t: m["Open"] for t, m in monthly_data.items()})

    months = close_panel.index
    months = months[(months >= start_date) & (months <= end_date)]

    portfolio_value = INITIAL_CAPITAL
    equity_curve = []
    rebalances = []

    for i in range(12, len(months) - 1):
        signal_date = months[i]

        available_close = daily_close_panel.loc[:signal_date]

        # Compute signals for all available risk-on stocks
        stock_signals = {}
        for ticker in NIFTY100_STOCKS:
            if ticker not in available_close.columns:
                continue
            series = available_close[ticker].dropna()
            if len(series) < MIN_HISTORY_DAYS:
                continue
            sigs = compute_momentum_signals(series)
            if all(v is None for v in sigs.values()):
                continue
            stock_signals[ticker] = sigs

        if not stock_signals:
            continue

        selected, negative_flags, _ = scores_approach(stock_signals, TOP_N_STOCKS, NEGATIVE_THRESHOLD)

        # Risk-off signals
        risk_off_signals = {}
        for ticker in RISK_OFF_TICKERS:
            if ticker not in daily_close_panel.columns:
                continue
            series = daily_close_panel.loc[:signal_date, ticker].dropna()
            if len(series) < MIN_HISTORY_DAYS:
                continue
            sigs = compute_momentum_signals(series)
            if all(v is None for v in sigs.values()):
                continue
            risk_off_signals[ticker] = sigs

        # Replace negative-momentum selected stocks with best risk-off ETF
        final_tickers = []
        for ticker in selected:
            if RISK_OFF_REPLACE and negative_flags.get(ticker, False):
                if risk_off_signals:
                    best_off, _, _ = scores_approach(risk_off_signals, 1, NEGATIVE_THRESHOLD)
                    if best_off:
                        final_tickers.append(best_off[0])
            else:
                final_tickers.append(ticker)

        seen = set()
        final_tickers = [t for t in final_tickers if not (t in seen or seen.add(t))]

        # Protection layer
        num_negative = sum(1 for t in NIFTY100_STOCKS if negative_flags.get(t, False))
        num_risk_on = len(NIFTY100_STOCKS)
        exposure = risk_on_exposure(num_negative, num_risk_on, PROTECTION_Q)

        # Build weights
        weights = {}
        if final_tickers:
            risk_on_weight = exposure / len(final_tickers)
            for t in final_tickers:
                weights[t] = risk_on_weight

        remaining = 1.0 - sum(weights.values())
        if remaining > 0 and risk_off_signals:
            best_off, _, _ = scores_approach(risk_off_signals, 1, NEGATIVE_THRESHOLD)
            if best_off:
                weights[best_off[0]] = weights.get(best_off[0], 0.0) + remaining

        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {t: w / total_weight for t, w in weights.items()}

        # Execute open-to-open next month
        entry_date = months[i + 1]
        exit_date = months[i + 2] if i + 2 < len(months) else None
        if exit_date is None:
            break

        entry_prices = open_panel.loc[entry_date, weights.keys()]
        exit_prices = open_panel.loc[exit_date, weights.keys()]
        valid = entry_prices.notna() & exit_prices.notna() & (entry_prices > 0)

        monthly_return = 0.0
        if valid.sum() > 0:
            stock_rets = (exit_prices[valid] / entry_prices[valid]) - 1.0
            asset_weights = pd.Series({t: weights[t] for t in stock_rets.index})
            asset_weights = asset_weights / asset_weights.sum()
            monthly_return = (stock_rets * asset_weights).sum()

        cost = COST_PER_TRADE_PCT * 2
        net_return = monthly_return - cost
        portfolio_value *= (1 + net_return)

        equity_curve.append({
            "Date": exit_date,
            "PortfolioValue": portfolio_value,
            "GrossReturn": monthly_return,
            "NetReturn": net_return,
            "Cost": cost,
            "RiskOnExposure": exposure,
            "NumStocks": sum(1 for t in final_tickers if t in NIFTY100_STOCKS),
            "NumETFs": sum(1 for t in final_tickers if t in RISK_OFF_TICKERS),
            "NumNegative": num_negative,
        })

        rebalances.append(Rebalance(
            date=signal_date,
            selected_stocks=selected,
            final_weights=weights,
            negative_flags=negative_flags,
            risk_on_exposure=exposure
        ))

    equity_df = pd.DataFrame(equity_curve).set_index("Date")
    return equity_df, rebalances


# ---------------------------------------------------------------------------
# PERFORMANCE METRICS
# ---------------------------------------------------------------------------

def calculate_metrics(equity: pd.Series) -> Dict:
    rets = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 0.25)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1

    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    vol = rets.std() * np.sqrt(12)
    sharpe = (rets.mean() * 12) / vol if vol > 0 else 0

    rolling_rets = (1 + rets).rolling(window=12).apply(np.prod, raw=True) - 1
    p_positive = (rolling_rets.dropna() > 0).mean() * 100
    p_above_minus5 = (rolling_rets.dropna() > -0.05).mean() * 100

    return {
        "start": equity.index[0].strftime("%Y-%m-%d"),
        "end": equity.index[-1].strftime("%Y-%m-%d"),
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "max_dd_pct": max_dd * 100,
        "monthly_vol_annualized": vol * 100,
        "sharpe_annualized": sharpe,
        "avg_monthly_return_pct": rets.mean() * 100,
        "p_1y_positive": p_positive,
        "p_1y_above_minus5": p_above_minus5,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Indian Stock + Defensive ETF Multi-Asset Momentum")
    print("=" * 70)
    print(f"Full period: {START_DATE} to {END_DATE}")
    print(f"In-Sample : {IS_START} to {IS_END}")
    print(f"Out-of-Sample : {OOS_START} to {OOS_END}")
    print(f"Top N stocks: {TOP_N_STOCKS}")
    print(f"Protection layer q: {PROTECTION_Q}")
    print(f"Negative threshold: {NEGATIVE_THRESHOLD}")
    print("=" * 70)

    print("\nDownloading stock and ETF data...")
    daily_data = {}
    for idx, ticker in enumerate(ALL_TICKERS, 1):
        print(f"  [{idx}/{len(ALL_TICKERS)}] {ticker}")
        df = download_data(ticker, START_DATE, END_DATE)
        if df is not None and not df.empty:
            daily_data[ticker] = df
        time.sleep(SLEEP_BETWEEN_DOWNLOADS)

    valid_stocks = [t for t in NIFTY100_STOCKS if t in daily_data]
    valid_etfs = [t for t in RISK_OFF_TICKERS if t in daily_data]
    print(f"\nValid stocks: {len(valid_stocks)} / {len(NIFTY100_STOCKS)}")
    print(f"Valid risk-off ETFs: {len(valid_etfs)} / {len(RISK_OFF_TICKERS)}")

    print("\nRunning backtest...")
    equity_full, rebalances_full = run_backtest(daily_data, START_DATE, END_DATE)

    is_mask = (equity_full.index >= IS_START) & (equity_full.index <= IS_END)
    oos_mask = (equity_full.index >= OOS_START) & (equity_full.index <= OOS_END)

    metrics_full = calculate_metrics(equity_full["PortfolioValue"])
    metrics_is = calculate_metrics(equity_full.loc[is_mask, "PortfolioValue"])
    metrics_oos = calculate_metrics(equity_full.loc[oos_mask, "PortfolioValue"])

    equity_full.to_csv("stock_multiasset_equity_curve.csv")
    print("\n[+] Saved stock_multiasset_equity_curve.csv")

    rebalance_records = []
    for rb in rebalances_full:
        for ticker, weight in rb.final_weights.items():
            rebalance_records.append({
                "Date": rb.date,
                "Ticker": ticker,
                "Weight": weight,
                "RiskOnExposure": rb.risk_on_exposure,
                "SelectedStocks": ",".join(rb.selected_stocks),
            })
    pd.DataFrame(rebalance_records).to_csv("stock_multiasset_rebalance_log.csv", index=False)
    print("[+] Saved stock_multiasset_rebalance_log.csv")

    def format_metrics(m: Dict, label: str) -> List[str]:
        return [
            f"{label}",
            "-" * 50,
            f"Total return        : {m['total_return_pct']:.2f}%",
            f"CAGR                : {m['cagr_pct']:.2f}%",
            f"Max drawdown        : {m['max_dd_pct']:.2f}%",
            f"Monthly vol (ann.)  : {m['monthly_vol_annualized']:.2f}%",
            f"Sharpe (ann.)       : {m['sharpe_annualized']:.2f}",
            f"Avg monthly return  : {m['avg_monthly_return_pct']:.2f}%",
            f"P(1Y return > 0%)   : {m['p_1y_positive']:.2f}%",
            f"P(1Y return > -5%)  : {m['p_1y_above_minus5']:.2f}%",
            "",
        ]

    report_lines = [
        "INDIAN STOCK + DEFENSIVE ETF MULTI-ASSET MOMENTUM REPORT",
        "=" * 50,
        f"Full period: {metrics_full['start']} to {metrics_full['end']}",
        f"In-Sample: {IS_START} to {IS_END}",
        f"Out-of-Sample: {OOS_START} to {OOS_END}",
        f"Top N stocks: {TOP_N_STOCKS}",
        f"Protection layer q: {PROTECTION_Q}",
        f"Negative threshold: {NEGATIVE_THRESHOLD}",
        f"Cost: {COST_PER_TRADE_PCT * 100:.2f}% one-way",
        "",
    ]
    report_lines.extend(format_metrics(metrics_full, "FULL PERIOD PERFORMANCE"))
    report_lines.extend(format_metrics(metrics_is, "IN-SAMPLE PERFORMANCE"))
    report_lines.extend(format_metrics(metrics_oos, "OUT-OF-SAMPLE PERFORMANCE"))

    report = "\n".join(report_lines)
    print("\n" + report)
    with open("stock_multiasset_summary_report.txt", "w") as f:
        f.write(report)
    print("\n[+] Saved stock_multiasset_summary_report.txt")


if __name__ == "__main__":
    main()
