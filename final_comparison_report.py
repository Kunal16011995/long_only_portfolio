"""
Final Comparison Report: Signal Blending vs Portfolio Blending

Compares the best signal blend (M40% Q40% V20%, Top 10, quarterly)
against the best portfolio blend (M25% Q25% V25% A25%, monthly)
and the benchmark (^CNX100).

Also includes the user's requested portfolio blend (M40% Q30% V20% A10%).
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "portfolio_blend_strategy"))

from indian_signal_blend_backtest import (
    START_DATE as SB_START,
    END_DATE as SB_END,
    NIFTY100_TICKERS,
    download_prices, get_fundamentals,
    compute_momentum_score, compute_quality_score, compute_deep_value_score,
    run_backtest as run_signal_backtest,
)
from indian_portfolio_blend_backtest import (
    START_DATE as PB_START,
    END_DATE as PB_END,
    STRATEGIES,
    load_strategy_curve, align_and_fill,
    run_blend_backtest,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
START_DATE = "2015-01-01"  # Use common IS/OOS period for fair comparison
END_DATE = date.today().strftime("%Y-%m-%d")
IS_START = "2015-01-01"
IS_END = "2019-12-31"
OOS_START = "2020-01-01"

# Signal blend: best from variations grid
SIGNAL_WEIGHTS = {"momentum": 0.40, "quality": 0.40, "deep_value": 0.20}
SIGNAL_TOP_N = 10
SIGNAL_REBALANCE = "QE"

# Portfolio blends to compare
PORTFOLIO_BLENDS = {
    "Balanced (25/25/25/25)": (0.25, 0.25, 0.25, 0.25),
    "Max return (70/20/10/0)": (0.70, 0.20, 0.10, 0.00),
    "Low DD (20/20/20/40)": (0.20, 0.20, 0.20, 0.40),
    "Requested (40/30/20/10)": (0.40, 0.30, 0.20, 0.10),
}
PORTFOLIO_REBALANCE = "ME"  # monthly, based on variation results

COST_BPS = 0.05
COST = COST_BPS / 100.0

# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def compute_metrics(equity_df):
    if equity_df.empty or len(equity_df) < 2:
        return {}
    rets = equity_df["equity"].pct_change().dropna()
    if rets.empty:
        return {}
    total_ret = equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1
    years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan
    peak = equity_df["equity"].cummax()
    dd = (equity_df["equity"] - peak) / peak
    max_dd = dd.min()
    ann_vol = rets.std() * np.sqrt(252)
    ann_ret = rets.mean() * 252
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return {
        "total_return": total_ret,
        "cagr": cagr,
        "max_dd": max_dd,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "avg_daily_ret": rets.mean(),
    }


def compute_annual_returns(equity_df):
    yearly = equity_df["equity"].resample("YE").last().pct_change().dropna()
    return yearly


# ---------------------------------------------------------------------------
# BUILD SIGNAL BLEND CURVE
# ---------------------------------------------------------------------------
def build_signal_blend_curve():
    print("Building best signal-blend curve (M40% Q40% V20%, Top 10, quarterly)...")
    price_data = {}
    valid_tickers = []
    for ticker in NIFTY100_TICKERS:
        try:
            df = download_prices(ticker, SB_START, SB_END)
            if len(df) > 100:
                price_data[ticker] = df
                valid_tickers.append(ticker)
        except Exception:
            pass

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
    all_cal_dates = pd.date_range(SB_START, SB_END, freq="D")
    rebalance_dates = pd.Series(index=all_cal_dates).resample(SIGNAL_REBALANCE).last().index
    trading_days = closes.index
    rebalance_dates = [trading_days[trading_days >= d][0] for d in rebalance_dates
                       if any(d <= df.index.max() for df in price_data.values())]

    equity, _ = run_signal_backtest(price_data, blended, rebalance_dates, top_n=SIGNAL_TOP_N)
    return equity


# ---------------------------------------------------------------------------
# BUILD PORTFOLIO BLEND CURVES
# ---------------------------------------------------------------------------
def fixed_weight_method(weights):
    def f(rets_window):
        return np.array(weights)
    return f


def build_portfolio_blend_curve(name, weights):
    print(f"Building portfolio blend curve: {name} ...")
    curves = []
    for label, config in STRATEGIES.items():
        s = load_strategy_curve(label, config)
        curves.append(s)

    prices = align_and_fill(curves)
    returns = prices.pct_change().dropna()

    all_cal_dates = pd.date_range(returns.index[0], returns.index[-1], freq="D")
    rebalance_dates = pd.Series(index=all_cal_dates).resample(PORTFOLIO_REBALANCE).last().index
    trading_days = returns.index
    rebalance_dates = [trading_days[trading_days >= d][0] for d in rebalance_dates
                       if d <= trading_days[-1]]

    equity = run_blend_backtest(returns, fixed_weight_method(weights), rebalance_dates)
    return equity


# ---------------------------------------------------------------------------
# BENCHMARK
# ---------------------------------------------------------------------------
def load_benchmark():
    print("Loading benchmark (^CNX100)...")
    bench = yf.download("^CNX100", start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
    if isinstance(bench.columns, pd.MultiIndex):
        bench.columns = bench.columns.get_level_values(0)
    bench = bench[["Close"]].dropna()
    bench_equity = (bench["Close"] / bench["Close"].iloc[0]) * 1_000_000
    return bench_equity.to_frame("equity")


# ---------------------------------------------------------------------------
# MAIN REPORT
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("FINAL COMPARISON REPORT")
    print("Signal Blending vs Portfolio Blending")
    print("=" * 70)

    signal_eq = build_signal_blend_curve()
    benchmark_eq = load_benchmark()

    portfolio_eqs = {}
    for name, weights in PORTFOLIO_BLENDS.items():
        portfolio_eqs[name] = build_portfolio_blend_curve(name, weights)

    # Align all curves to common period
    all_curves = {"Signal Blend (M40 Q40 V20)": signal_eq, "Benchmark": benchmark_eq}
    all_curves.update(portfolio_eqs)

    common_start = max(c.index.min() for c in all_curves.values())
    common_end = min(c.index.max() for c in all_curves.values())
    print(f"\nCommon comparison period: {common_start.date()} to {common_end.date()}")

    aligned = {name: c.loc[common_start:common_end] for name, c in all_curves.items()}

    # Metrics
    print("\n" + "=" * 70)
    print("FULL PERIOD METRICS")
    print("=" * 70)
    rows = []
    for name, eq in aligned.items():
        m = compute_metrics(eq)
        rows.append({
            "Strategy": name,
            "Total Return": f"{m['total_return']*100:.2f}%",
            "CAGR": f"{m['cagr']*100:.2f}%",
            "Max DD": f"{m['max_dd']*100:.2f}%",
            "Vol": f"{m['ann_vol']*100:.2f}%",
            "Sharpe": f"{m['sharpe']:.2f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # IS / OOS
    print("\n" + "=" * 70)
    print("IN-SAMPLE (2015-2019) AND OUT-OF-SAMPLE (2020-) METRICS")
    print("=" * 70)
    rows = []
    for name, eq in aligned.items():
        is_m = compute_metrics(eq.loc[IS_START:IS_END])
        oos_m = compute_metrics(eq.loc[OOS_START:END_DATE])
        rows.append({
            "Strategy": name,
            "IS CAGR": f"{is_m.get('cagr', np.nan)*100:.2f}%",
            "IS DD": f"{is_m.get('max_dd', np.nan)*100:.2f}%",
            "IS Sharpe": f"{is_m.get('sharpe', np.nan):.2f}",
            "OOS CAGR": f"{oos_m.get('cagr', np.nan)*100:.2f}%",
            "OOS DD": f"{oos_m.get('max_dd', np.nan)*100:.2f}%",
            "OOS Sharpe": f"{oos_m.get('sharpe', np.nan):.2f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # Annual returns
    print("\n" + "=" * 70)
    print("CALENDAR YEAR RETURNS")
    print("=" * 70)
    annual_rets = pd.DataFrame({name: compute_annual_returns(eq) * 100 for name, eq in aligned.items()})
    annual_rets.index = annual_rets.index.year
    print(annual_rets.round(2).to_string())

    # Correlation matrix of daily returns
    print("\n" + "=" * 70)
    print("DAILY RETURN CORRELATION MATRIX")
    print("=" * 70)
    rets = pd.DataFrame({name: eq["equity"].pct_change() for name, eq in aligned.items()}).dropna()
    print(rets.corr().round(3).to_string())

    # Win rate vs benchmark (annual)
    print("\n" + "=" * 70)
    print("ANNUAL WIN RATE vs BENCHMARK")
    print("=" * 70)
    bench_annual = annual_rets["Benchmark"]
    for name in annual_rets.columns:
        if name == "Benchmark":
            continue
        wins = (annual_rets[name] > bench_annual).sum()
        total = len(annual_rets)
        print(f"{name:30s}: {wins}/{total} years ({wins/total*100:.1f}%)")

    # Save curves
    print("\n" + "=" * 70)
    print("Saving aligned equity curves...")
    for name, eq in aligned.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        eq.to_csv(os.path.join(os.path.dirname(__file__), f"final_compare_{safe_name}.csv"))
    print("Done.")


if __name__ == "__main__":
    main()
