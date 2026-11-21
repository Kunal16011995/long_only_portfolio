"""
High Risk / High Reward Version

Signal blend: 40% Momentum + 40% Quality + 20% Deep Value
Top 10 stocks, quarterly rebalancing.

Expected (2015-2026):
  CAGR ~28%, Max DD ~-24.5%, Sharpe ~1.52
"""

import os
import sys
import warnings
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.4f}".format

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "signal_blend_strategy"))
from indian_signal_blend_backtest import (
    NIFTY100_TICKERS,
    download_prices, get_fundamentals,
    compute_momentum_score, compute_quality_score, compute_deep_value_score,
    run_backtest as run_signal_backtest,
)

START_DATE = "2010-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")
IS_START = "2015-01-01"
IS_END = "2019-12-31"
OOS_START = "2020-01-01"

SIGNAL_WEIGHTS = {"momentum": 0.40, "quality": 0.40, "deep_value": 0.20}
TOP_N = 10
REBALANCE_FREQ = "QE"


def compute_metrics(equity_df):
    rets = equity_df["equity"].pct_change().dropna()
    total_ret = equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1
    years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan
    peak = equity_df["equity"].cummax()
    dd = (equity_df["equity"] - peak) / peak
    max_dd = dd.min()
    ann_vol = rets.std() * np.sqrt(252)
    ann_ret = rets.mean() * 252
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return {"total_return": total_ret, "cagr": cagr, "max_dd": max_dd, "ann_vol": ann_vol, "sharpe": sharpe}


def main():
    print("=" * 70)
    print("HIGH RISK / HIGH REWARD VERSION")
    print("Signal blend: 40% Momentum + 40% Quality + 20% Deep Value")
    print("Top 10 stocks, quarterly rebalancing")
    print("=" * 70)

    print("\nDownloading data...")
    price_data = {}
    valid_tickers = []
    for ticker in NIFTY100_TICKERS:
        try:
            df = download_prices(ticker, START_DATE, END_DATE)
            if len(df) > 100:
                price_data[ticker] = df
                valid_tickers.append(ticker)
        except Exception:
            pass
    print(f"Valid stocks: {len(valid_tickers)}")

    fundamentals = {t: get_fundamentals(t) for t in valid_tickers}

    mom = compute_momentum_score(price_data)
    qual = compute_quality_score(fundamentals, price_data)
    dv = compute_deep_value_score(price_data, fundamentals)

    all_dates = mom.index
    qual_rank = pd.DataFrame(index=all_dates, columns=qual.index)
    for t in qual.index:
        qual_rank[t] = qual[t]
    qual_rank = qual_rank.rank(axis=1, pct=True)
    dv_rank = dv.reindex(all_dates)
    mom_rank = mom.rank(axis=1, pct=True)

    blended = (
        SIGNAL_WEIGHTS["momentum"] * mom_rank.fillna(0.5)
        + SIGNAL_WEIGHTS["quality"] * qual_rank.fillna(0.5)
        + SIGNAL_WEIGHTS["deep_value"] * dv_rank.fillna(0.5)
    )

    closes = pd.DataFrame({t: df["Close"] for t, df in price_data.items()}).sort_index()
    all_cal_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    rebalance_dates = pd.Series(index=all_cal_dates).resample(REBALANCE_FREQ).last().index
    trading_days = closes.index
    rebalance_dates = [trading_days[trading_days >= d][0] for d in rebalance_dates
                       if any(d <= df.index.max() for df in price_data.values())]

    equity, trades = run_signal_backtest(price_data, blended, rebalance_dates, top_n=TOP_N)

    # Benchmark
    bench = yf.download("^CNX100", start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
    if isinstance(bench.columns, pd.MultiIndex):
        bench.columns = bench.columns.get_level_values(0)
    bench = bench[["Close"]].dropna()
    bench_equity = (bench["Close"] / bench["Close"].iloc[0]) * 1_000_000

    # Align
    common_start = max(equity.index.min(), bench_equity.index.min())
    common_end = min(equity.index.max(), bench_equity.index.max())
    equity = equity.loc[common_start:common_end]
    bench_equity = bench_equity.loc[common_start:common_end]

    full_m = compute_metrics(equity)
    is_m = compute_metrics(equity.loc[IS_START:IS_END])
    oos_m = compute_metrics(equity.loc[OOS_START:END_DATE])
    bench_m = compute_metrics(bench_equity.to_frame("equity"))
    bench_is = compute_metrics(bench_equity.loc[IS_START:IS_END].to_frame("equity"))
    bench_oos = compute_metrics(bench_equity.loc[OOS_START:END_DATE].to_frame("equity"))

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Full period: {common_start.date()} to {common_end.date()}")
    print(f"\nStrategy:")
    print(f"  Total return : {full_m['total_return']*100:8.2f}%")
    print(f"  CAGR         : {full_m['cagr']*100:8.2f}%")
    print(f"  Max DD       : {full_m['max_dd']*100:8.2f}%")
    print(f"  Volatility   : {full_m['ann_vol']*100:8.2f}%")
    print(f"  Sharpe       : {full_m['sharpe']:8.2f}")
    print(f"\nBenchmark (^CNX100):")
    print(f"  Total return : {bench_m['total_return']*100:8.2f}%")
    print(f"  CAGR         : {bench_m['cagr']*100:8.2f}%")
    print(f"  Max DD       : {bench_m['max_dd']*100:8.2f}%")
    print(f"  Sharpe       : {bench_m['sharpe']:8.2f}")
    print(f"\nIS 2015-2019:  Strategy CAGR {is_m['cagr']*100:6.2f}% | Benchmark {bench_is['cagr']*100:6.2f}%")
    print(f"OOS 2020-   :  Strategy CAGR {oos_m['cagr']*100:6.2f}% | Benchmark {bench_oos['cagr']*100:6.2f}%")

    equity.to_csv(os.path.join(os.path.dirname(__file__), "high_risk_high_reward_equity_curve.csv"))
    pd.DataFrame(trades).to_csv(os.path.join(os.path.dirname(__file__), "high_risk_high_reward_trades.csv"), index=False)
    print("\nSaved: high_risk_high_reward_equity_curve.csv, high_risk_high_reward_trades.csv")


if __name__ == "__main__":
    main()
