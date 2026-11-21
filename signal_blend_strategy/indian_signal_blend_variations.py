"""
Indian Long-Only Signal-Blending Strategy — Variations Grid Search

Tests many combinations of signal weights and Top N portfolio sizes.
For each combination it reports full-period, IS, and OOS metrics.

Signals:
  - Momentum
  - Quality
  - Deep Value
"""

import os
import sys
import warnings
from datetime import date, timedelta
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.4f}".format

sys.path.insert(0, os.path.dirname(__file__))
from indian_signal_blend_backtest import (
    START_DATE, END_DATE, IS_START, IS_END, OOS_START,
    CACHE_DIR, COST, INITIAL_CAPITAL,
    DV_IN, DV_OUT, DV_TRAP, DV_LOOKBACK_YEARS, DV_MIN_HISTORY_YEARS,
    NIFTY100_TICKERS,
    download_prices, get_fundamentals,
    compute_momentum_score, compute_quality_score, compute_deep_value_score,
    run_backtest, compute_metrics, compute_benchmark_metrics,
)

# ---------------------------------------------------------------------------
# VARIATION GRID
# ---------------------------------------------------------------------------
# Weights must sum to 1.0. Each tuple is (momentum, quality, deep_value).
WEIGHT_COMBINATIONS = [
    # Equal weight
    (1/3, 1/3, 1/3),
    # Single signal
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    # Two signals
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
    # One dominant
    (0.6, 0.2, 0.2),
    (0.2, 0.6, 0.2),
    (0.2, 0.2, 0.6),
    (0.7, 0.15, 0.15),
    (0.15, 0.7, 0.15),
    (0.15, 0.15, 0.7),
    # Momentum + Quality dominant
    (0.4, 0.4, 0.2),
    (0.4, 0.2, 0.4),
    (0.2, 0.4, 0.4),
    # Momentum heavy
    (0.8, 0.1, 0.1),
    (0.5, 0.25, 0.25),
    (0.25, 0.5, 0.25),
    (0.25, 0.25, 0.5),
]

TOP_N_VALUES = [10, 15, 20]

REBALANCE_FREQ = "QE"


def build_blended_signal(mom, qual_rank, dv_rank, weights):
    """Blend three rank panels into one composite score panel."""
    mom_rank = mom.rank(axis=1, pct=True)
    blended = (
        weights["momentum"] * mom_rank.fillna(0.5)
        + weights["quality"] * qual_rank.fillna(0.5)
        + weights["deep_value"] * dv_rank.fillna(0.5)
    )
    return blended


def run_variation(price_data, blended_signal, rebalance_dates, top_n, is_bench, oos_bench, full_bench):
    equity_full, trades_full = run_backtest(price_data, blended_signal, rebalance_dates, top_n=top_n)

    is_equity = equity_full.loc[IS_START:IS_END]
    oos_equity = equity_full.loc[OOS_START:END_DATE]

    is_metrics = compute_metrics(is_equity)
    oos_metrics = compute_metrics(oos_equity)
    full_metrics = compute_metrics(equity_full)

    return {
        "is": is_metrics,
        "oos": oos_metrics,
        "full": full_metrics,
        "trades": len(trades_full),
    }


def main():
    print("=" * 70)
    print("Signal-Blend Variations Grid Search")
    print("=" * 70)
    print(f"Periods: Full {START_DATE}-{END_DATE} | IS {IS_START}-{IS_END} | OOS {OOS_START}-{END_DATE}")
    print(f"Top N tested: {TOP_N_VALUES}")
    print(f"Weight combinations: {len(WEIGHT_COMBINATIONS)}")
    print(f"Total runs: {len(WEIGHT_COMBINATIONS) * len(TOP_N_VALUES)}")
    print("=" * 70)

    # Download data once
    print("\nDownloading stock data...")
    price_data = {}
    valid_tickers = []
    for i, ticker in enumerate(NIFTY100_TICKERS, 1):
        try:
            df = download_prices(ticker, START_DATE, END_DATE)
            if len(df) > 100:
                price_data[ticker] = df
                valid_tickers.append(ticker)
        except Exception:
            pass
    print(f"Valid stocks with data: {len(valid_tickers)}")

    print("\nDownloading fundamentals...")
    fundamentals = {}
    for ticker in valid_tickers:
        fundamentals[ticker] = get_fundamentals(ticker)
    print(f"Fundamental records: {len(fundamentals)}")

    print("\nComputing signals...")
    mom = compute_momentum_score(price_data)
    qual = compute_quality_score(fundamentals, price_data)
    dv = compute_deep_value_score(price_data, fundamentals)

    # Replicated quality rank panel
    all_dates = mom.index
    qual_rank = pd.DataFrame(index=all_dates, columns=qual.index)
    for t in qual.index:
        qual_rank[t] = qual[t]
    qual_rank = qual_rank.rank(axis=1, pct=True)
    dv_rank = dv.reindex(all_dates)

    # Generate rebalance dates
    closes = pd.DataFrame({t: df["Close"] for t, df in price_data.items()}).sort_index()
    all_cal_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    rebalance_dates = pd.Series(index=all_cal_dates).resample(REBALANCE_FREQ).last().index
    trading_days = closes.index
    rebalance_dates = [trading_days[trading_days >= d][0] for d in rebalance_dates
                       if any(d <= df.index.max() for df in price_data.values())]

    # Benchmark
    print("\nDownloading benchmark data (^CNX100)...")
    bench = yf.download("^CNX100", start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
    if isinstance(bench.columns, pd.MultiIndex):
        bench.columns = bench.columns.get_level_values(0)
    bench = bench[["Close"]].dropna()

    is_bench = compute_benchmark_metrics(bench, IS_START, IS_END)
    oos_bench = compute_benchmark_metrics(bench, OOS_START, END_DATE)
    full_bench = compute_benchmark_metrics(bench, START_DATE, END_DATE)

    # Grid search
    print(f"\nRunning grid search ({len(WEIGHT_COMBINATIONS) * len(TOP_N_VALUES)} combinations)...")
    results = []
    for w_m, w_q, w_d in WEIGHT_COMBINATIONS:
        weights = {"momentum": w_m, "quality": w_q, "deep_value": w_d}
        blended = build_blended_signal(mom, qual_rank, dv_rank, weights)

        for top_n in TOP_N_VALUES:
            res = run_variation(price_data, blended, rebalance_dates, top_n, is_bench, oos_bench, full_bench)
            results.append({
                "momentum_w": w_m,
                "quality_w": w_q,
                "deep_value_w": w_d,
                "top_n": top_n,
                "full_cagr": res["full"].get("cagr", np.nan),
                "full_dd": res["full"].get("max_dd", np.nan),
                "full_sharpe": res["full"].get("sharpe", np.nan),
                "full_total": res["full"].get("total_return", np.nan),
                "is_cagr": res["is"].get("cagr", np.nan),
                "is_dd": res["is"].get("max_dd", np.nan),
                "is_sharpe": res["is"].get("sharpe", np.nan),
                "oos_cagr": res["oos"].get("cagr", np.nan),
                "oos_dd": res["oos"].get("max_dd", np.nan),
                "oos_sharpe": res["oos"].get("sharpe", np.nan),
                "trades": res["trades"],
            })

    results_df = pd.DataFrame(results)
    results_df["blend"] = results_df.apply(
        lambda r: f"M{r['momentum_w']:.0%} Q{r['quality_w']:.0%} V{r['deep_value_w']:.0%}", axis=1
    )

    # Sort by different criteria
    print("\n" + "=" * 70)
    print("TOP 10 BY FULL-PERIOD CAGR")
    print("=" * 70)
    top_cagr = results_df.sort_values("full_cagr", ascending=False).head(10)
    print(top_cagr[["blend", "top_n", "full_total", "full_cagr", "full_dd", "full_sharpe", "oos_cagr", "oos_dd", "trades"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 10 BY FULL-PERIOD SHARPE")
    print("=" * 70)
    top_sharpe = results_df.sort_values("full_sharpe", ascending=False).head(10)
    print(top_sharpe[["blend", "top_n", "full_total", "full_cagr", "full_dd", "full_sharpe", "oos_cagr", "oos_dd", "trades"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 10 BY LOWEST FULL-PERIOD MAX DRAWDOWN")
    print("=" * 70)
    top_dd = results_df.sort_values("full_dd", ascending=False).head(10)
    print(top_dd[["blend", "top_n", "full_total", "full_cagr", "full_dd", "full_sharpe", "oos_cagr", "oos_dd", "trades"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 10 BY OOS CAGR")
    print("=" * 70)
    top_oos = results_df.sort_values("oos_cagr", ascending=False).head(10)
    print(top_oos[["blend", "top_n", "full_total", "full_cagr", "full_dd", "oos_cagr", "oos_dd", "oos_sharpe", "trades"]].to_string(index=False))

    # Save full grid
    out_path = os.path.join(os.path.dirname(__file__), "signal_blend_variations.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved full grid: {out_path}")


if __name__ == "__main__":
    main()
