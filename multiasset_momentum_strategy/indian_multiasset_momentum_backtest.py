"""
Indian Long-Only Multi-Asset Momentum Strategy
================================================
Adaptation of Zambrano & Rizzolo (2022):
"Long-only multi-asset momentum: searching for absolute returns"

Strategy:
  - Universe: Indian ETFs across equities, gold, silver, bonds, cash
  - Signal: 9 momentum specifications (3 measures x 3 lookbacks)
  - Aggregation: Scores approach or Ensemble
  - Selection: Top N risk-on assets; negative-momentum assets replaced by best risk-off
  - Optional protection layer: reduce risk-on exposure when many assets are negative
  - Rebalancing: Monthly

Data source: Yahoo Finance via yfinance
Run: python indian_multiasset_momentum_backtest.py
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
TOP_N = 5                        # top N risk-on assets to hold
COST_PER_TRADE_PCT = 0.001       # ~0.1% one-way cost
AGGREGATION = "scores"           # "scores" or "ensemble"
NEGATIVE_THRESHOLD = "1/3"       # "1/3" or "1/2": fraction of signals negative to flag asset
PROTECTION_Q = None              # 0=low, 1=medium, 2=high protection layer; set None to disable
RISK_OFF_SINGLE = True           # if True, risk-off allocation goes to best single risk-off asset

SLEEP_BETWEEN_DOWNLOADS = 0.3

# ---------------------------------------------------------------------------
# INDIAN ETF UNIVERSE (update tickers as needed)
# ---------------------------------------------------------------------------

RISK_ON_TICKERS = [
    "NIFTYBEES.NS",    # Nifty 50
    "GOLDBEES.NS",     # Gold
    "SILVERBEES.NS",   # Silver
    "MOTILALOFS.NS",   # Motilal Oswal Nasdaq 100 ETF (US tech)
    "BANKBEES.NS",     # Nifty Bank
    "ITBEES.NS",       # Nifty IT
    "MID150BEES.NS",   # Nifty Midcap 150
    "INFRABEES.NS",    # Infrastructure
    "CPSEETF.NS",      # CPSE PSU
    "AUTOBEES.NS",     # Nifty Auto
]

RISK_OFF_TICKERS = [
    "LIQUIDBEES.NS",   # Liquid / cash
    "GILT5YBEES.NS",   # 5-year Gilt
]

ALL_TICKERS = RISK_ON_TICKERS + RISK_OFF_TICKERS

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
                  cache_dir: str = "data_cache") -> Optional[pd.DataFrame]:
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
        if df.empty or len(df) < 100:
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
# MOMENTUM MEASURES
# ---------------------------------------------------------------------------

def total_return_momentum(prices: pd.Series, n_months: int) -> Optional[float]:
    """Total return over n_months."""
    if len(prices) < n_months + 1:
        return None
    return prices.iloc[-1] / prices.iloc[-(n_months + 1)] - 1.0


def price_minus_ma_momentum(prices: pd.Series, n_days: int) -> Optional[float]:
    """Price / SMA - 1."""
    if len(prices) < n_days:
        return None
    ma = prices.iloc[-n_days:].mean()
    if ma <= 0:
        return None
    return prices.iloc[-1] / ma - 1.0


def risk_adjusted_momentum(prices: pd.Series, n_months: int) -> Optional[float]:
    """log(total return) / sum of absolute daily log returns."""
    if len(prices) < n_months + 1:
        return None
    total_ret = np.log(prices.iloc[-1] / prices.iloc[-(n_months + 1)])
    daily_rets = np.log(prices / prices.shift(1)).iloc[-(n_months * 21):]
    path = daily_rets.abs().sum()
    if path <= 0:
        return None
    return total_ret / path


def compute_all_momentum_signals(daily_close: pd.Series) -> Dict[str, Optional[float]]:
    """Compute momentum signals for one asset. Using total-return only for robustness."""
    signals = {}
    # Total return momentum: 3, 6, 12 months
    for n in [3, 6, 12]:
        signals[f"TR_{n}M"] = total_return_momentum(daily_close, n)
    return signals


# ---------------------------------------------------------------------------
# AGGREGATION
# ---------------------------------------------------------------------------

def scores_approach(signals: Dict[str, Dict[str, float]], top_n: int,
                    negative_threshold: str) -> Tuple[List[str], Dict[str, bool]]:
    """
    Aggregate 9 signals using scores approach.
    Returns selected top N risk-on tickers and dict of negative-momentum flags.
    """
    tickers = list(signals.keys())
    if not tickers:
        return [], {}

    num_specs = len(next(iter(signals.values())))
    threshold_count = num_specs // 3 if negative_threshold == "1/3" else num_specs // 2

    # Sum scores across specifications
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
    return selected, negative_flags


def ensemble_approach(signals: Dict[str, Dict[str, float]], top_n: int) -> Tuple[List[str], Dict[str, bool]]:
    """
    Aggregate 9 signals using ensemble approach.
    Returns selected top N risk-on tickers and dict of negative-momentum flags.
    """
    tickers = list(signals.keys())
    if not tickers:
        return [], {}

    weights = {t: 0.0 for t in tickers}
    negative_counts = {t: 0 for t in tickers}

    for spec_name in signals[tickers[0]].keys():
        spec_values = {t: signals[t][spec_name] for t in tickers
                       if signals[t][spec_name] is not None}
        if not spec_values:
            continue
        ranked = sorted(spec_values.items(), key=lambda x: x[1], reverse=True)
        n_assets = len(ranked)
        for rank, (ticker, score) in enumerate(ranked[:top_n], start=1):
            if score > 0:
                weights[ticker] += 1.0 / top_n
        for ticker, score in spec_values.items():
            if score < 0:
                negative_counts[ticker] += 1

    sorted_tickers = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in sorted_tickers[:top_n] if weights[t] > 0]

    num_specs = len(next(iter(signals.values())))
    negative_flags = {t: negative_counts[t] > num_specs // 3 for t in tickers}
    return selected, negative_flags


# ---------------------------------------------------------------------------
# PROTECTION LAYER
# ---------------------------------------------------------------------------

def risk_on_exposure(num_negative: int, num_risk_on: int, q: int) -> float:
    """
    % Risk On Exposure = max(1 - n / ((N - q) / 4), 0)
    """
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
    selected: List[str]
    final_weights: Dict[str, float]
    signals: Dict[str, Dict[str, float]]
    negative_flags: Dict[str, bool]
    risk_on_exposure: float


def run_backtest(daily_data: Dict[str, pd.DataFrame],
                 start_date: str, end_date: str) -> Tuple[pd.DataFrame, List[Rebalance]]:
    """Run monthly multi-asset momentum backtest."""

    # Daily close panel for computing momentum signals
    daily_close_panel = pd.DataFrame({t: d["Close"] for t, d in daily_data.items()})

    # Monthly open/close panels for trade execution
    monthly_data = {t: compute_monthly_open_close(d) for t, d in daily_data.items()}
    close_panel = pd.DataFrame({t: m["Close"] for t, m in monthly_data.items()})
    open_panel = pd.DataFrame({t: m["Open"] for t, m in monthly_data.items()})

    # Monthly dates (end-of-month)
    months = close_panel.index

    # Filter months within backtest period
    months = months[(months >= start_date) & (months <= end_date)]

    portfolio_value = INITIAL_CAPITAL
    equity_curve = []
    rebalances = []

    # Need at least 12 months + 1 for first signal and return
    for i in range(12, len(months) - 1):
        signal_date = months[i]

        # Use daily data up to signal_date
        available_close = daily_close_panel.loc[:signal_date]

        # Compute signals for all risk-on assets
        risk_on_signals = {}
        for ticker in RISK_ON_TICKERS:
            if ticker not in available_close.columns:
                continue
            series = available_close[ticker].dropna()
            if len(series) < 250:  # ~1 year minimum
                continue
            sigs = compute_all_momentum_signals(series)
            if all(v is None for v in sigs.values()):
                continue
            risk_on_signals[ticker] = sigs

        if not risk_on_signals:
            continue

        # Aggregate signals
        if AGGREGATION == "scores":
            selected, negative_flags = scores_approach(risk_on_signals, TOP_N, NEGATIVE_THRESHOLD)
        else:
            selected, negative_flags = ensemble_approach(risk_on_signals, TOP_N)

        # Compute risk-off signals (for replacement)
        risk_off_signals = {}
        for ticker in RISK_OFF_TICKERS:
            if ticker not in daily_close_panel.columns:
                continue
            series = daily_close_panel.loc[:signal_date, ticker].dropna()
            if len(series) < 250:
                continue
            sigs = compute_all_momentum_signals(series)
            if all(v is None for v in sigs.values()):
                continue
            risk_off_signals[ticker] = sigs

        # Replace negative-momentum selected assets with best risk-off asset
        final_tickers = []
        for ticker in selected:
            if negative_flags.get(ticker, False):
                # replace with best risk-off
                if risk_off_signals:
                    # Use same aggregation for risk-off
                    if AGGREGATION == "scores":
                        best_off, _ = scores_approach(risk_off_signals, 1, NEGATIVE_THRESHOLD)
                    else:
                        best_off, _ = ensemble_approach(risk_off_signals, 1)
                    if best_off:
                        final_tickers.append(best_off[0])
            else:
                final_tickers.append(ticker)

        # Remove duplicates while preserving order
        seen = set()
        final_tickers = [t for t in final_tickers if not (t in seen or seen.add(t))]

        # Protection layer
        num_negative = sum(1 for t in RISK_ON_TICKERS if negative_flags.get(t, False))
        exposure = risk_on_exposure(num_negative, len(RISK_ON_TICKERS), PROTECTION_Q)

        # Build weights
        weights = {}
        if final_tickers:
            risk_on_weight = exposure / len(final_tickers)
            for t in final_tickers:
                weights[t] = risk_on_weight

        # Remaining exposure to best risk-off
        remaining = 1.0 - sum(weights.values())
        if remaining > 0 and risk_off_signals:
            if AGGREGATION == "scores":
                best_off, _ = scores_approach(risk_off_signals, 1, NEGATIVE_THRESHOLD)
            else:
                best_off, _ = ensemble_approach(risk_off_signals, 1)
            if best_off:
                weights[best_off[0]] = weights.get(best_off[0], 0.0) + remaining

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {t: w / total_weight for t, w in weights.items()}

        # Compute open-to-open returns for next month
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

        # Cost: assume full turnover each month for simplicity
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
            "NumAssets": len(final_tickers),
            "NumNegative": num_negative,
        })

        rebalances.append(Rebalance(
            date=signal_date,
            selected=selected,
            final_weights=weights,
            signals=risk_on_signals,
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

    # Rolling 1-year positive return probability
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
    print("Indian Long-Only Multi-Asset Momentum Strategy")
    print("=" * 70)
    print(f"Full period: {START_DATE} to {END_DATE}")
    print(f"In-Sample : {IS_START} to {IS_END}")
    print(f"Out-of-Sample : {OOS_START} to {OOS_END}")
    print(f"Aggregation: {AGGREGATION}, Negative threshold: {NEGATIVE_THRESHOLD}")
    print(f"Protection layer q: {PROTECTION_Q}")
    print(f"Top N risk-on: {TOP_N}")
    print("=" * 70)

    print("\nDownloading ETF data...")
    daily_data = {}
    for idx, ticker in enumerate(ALL_TICKERS, 1):
        print(f"  [{idx}/{len(ALL_TICKERS)}] {ticker}")
        df = download_data(ticker, START_DATE, END_DATE)
        if df is not None and not df.empty:
            daily_data[ticker] = df
        time.sleep(SLEEP_BETWEEN_DOWNLOADS)

    print(f"\nValid ETFs with data: {len(daily_data)}")
    for t in ALL_TICKERS:
        if t not in daily_data:
            print(f"  [x] No data: {t}")

    print("\nRunning backtest...")
    equity_full, rebalances_full = run_backtest(daily_data, START_DATE, END_DATE)

    # IS / OOS split
    is_mask = (equity_full.index >= IS_START) & (equity_full.index <= IS_END)
    oos_mask = (equity_full.index >= OOS_START) & (equity_full.index <= OOS_END)

    metrics_full = calculate_metrics(equity_full["PortfolioValue"])
    metrics_is = calculate_metrics(equity_full.loc[is_mask, "PortfolioValue"])
    metrics_oos = calculate_metrics(equity_full.loc[oos_mask, "PortfolioValue"])

    # Save outputs
    equity_full.to_csv("multiasset_equity_curve.csv")
    print("\n[+] Saved multiasset_equity_curve.csv")

    rebalance_records = []
    for rb in rebalances_full:
        for ticker, weight in rb.final_weights.items():
            rebalance_records.append({
                "Date": rb.date,
                "Ticker": ticker,
                "Weight": weight,
                "RiskOnExposure": rb.risk_on_exposure,
                "SelectedRiskOn": ",".join(rb.selected),
                "NegativeMomentum": ",".join([t for t, v in rb.negative_flags.items() if v])
            })
    pd.DataFrame(rebalance_records).to_csv("multiasset_rebalance_log.csv", index=False)
    print("[+] Saved multiasset_rebalance_log.csv")

    # Report
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
        "INDIAN LONG-ONLY MULTI-ASSET MOMENTUM STRATEGY REPORT",
        "=" * 50,
        f"Full period: {metrics_full['start']} to {metrics_full['end']}",
        f"In-Sample: {IS_START} to {IS_END}",
        f"Out-of-Sample: {OOS_START} to {OOS_END}",
        f"Aggregation: {AGGREGATION}",
        f"Negative threshold: {NEGATIVE_THRESHOLD}",
        f"Protection layer q: {PROTECTION_Q}",
        f"Top N risk-on: {TOP_N}",
        f"Cost: {COST_PER_TRADE_PCT * 100:.2f}% one-way",
        "",
    ]
    report_lines.extend(format_metrics(metrics_full, "FULL PERIOD PERFORMANCE"))
    report_lines.extend(format_metrics(metrics_is, "IN-SAMPLE PERFORMANCE"))
    report_lines.extend(format_metrics(metrics_oos, "OUT-OF-SAMPLE PERFORMANCE"))

    report = "\n".join(report_lines)
    print("\n" + report)
    with open("multiasset_summary_report.txt", "w") as f:
        f.write(report)
    print("\n[+] Saved multiasset_summary_report.txt")


if __name__ == "__main__":
    main()
