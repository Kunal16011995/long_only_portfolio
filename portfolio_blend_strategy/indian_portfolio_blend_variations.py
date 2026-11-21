"""
Indian Long-Only Portfolio-Blending Strategy — Weight Variations

Tests many fixed-weight combinations of the 4 strategies.
For each combination it reports full-period, IS, and OOS metrics.
"""

import os
import sys
import warnings
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.4f}".format

sys.path.insert(0, os.path.dirname(__file__))
from indian_portfolio_blend_backtest import (
    START_DATE, END_DATE, IS_START, IS_END, OOS_START,
    STRATEGIES, BASE_DIR,
    load_strategy_curve, align_and_fill, run_blend_backtest,
    compute_metrics,
)

# ---------------------------------------------------------------------------
# WEIGHT COMBINATIONS
# Each tuple: (momentum, quality, deep_value, multiasset). Must sum to 1.0.
# ---------------------------------------------------------------------------
WEIGHT_COMBINATIONS = [
    # Equal weight
    (0.25, 0.25, 0.25, 0.25),
    # Momentum heavy
    (0.40, 0.30, 0.20, 0.10),
    (0.50, 0.25, 0.15, 0.10),
    (0.60, 0.20, 0.15, 0.05),
    (0.50, 0.30, 0.15, 0.05),
    (0.45, 0.30, 0.20, 0.05),
    (0.40, 0.40, 0.15, 0.05),
    (0.55, 0.25, 0.15, 0.05),
    (0.70, 0.20, 0.10, 0.00),
    (0.60, 0.30, 0.10, 0.00),
    # Quality heavy
    (0.20, 0.40, 0.25, 0.15),
    (0.20, 0.50, 0.20, 0.10),
    (0.15, 0.45, 0.25, 0.15),
    (0.25, 0.45, 0.20, 0.10),
    (0.10, 0.60, 0.20, 0.10),
    # Deep value heavy
    (0.20, 0.25, 0.40, 0.15),
    (0.15, 0.25, 0.45, 0.15),
    (0.20, 0.30, 0.35, 0.15),
    (0.15, 0.35, 0.35, 0.15),
    # Multi-asset heavy
    (0.20, 0.20, 0.20, 0.40),
    (0.15, 0.25, 0.20, 0.40),
    # No multi-asset
    (0.40, 0.35, 0.25, 0.00),
    (0.35, 0.35, 0.30, 0.00),
    (0.45, 0.30, 0.25, 0.00),
    (0.30, 0.40, 0.30, 0.00),
    (0.50, 0.25, 0.25, 0.00),
    # Balanced with small multiasset
    (0.35, 0.35, 0.25, 0.05),
    (0.30, 0.35, 0.30, 0.05),
    (0.40, 0.25, 0.30, 0.05),
    (0.35, 0.30, 0.30, 0.05),
    # Two equal, two small
    (0.45, 0.45, 0.05, 0.05),
    (0.45, 0.05, 0.45, 0.05),
    (0.05, 0.45, 0.45, 0.05),
]


def fixed_weight_method(weights):
    """Return a weighting function that ignores rets_window and uses fixed weights."""
    def f(rets_window):
        return np.array(weights)
    return f


def main():
    print("=" * 70)
    print("Portfolio-Blend Weight Variations")
    print("=" * 70)
    print(f"Periods: Full {START_DATE}-{END_DATE} | IS {IS_START}-{IS_END} | OOS {OOS_START}-{END_DATE}")
    print(f"Weight combinations: {len(WEIGHT_COMBINATIONS)}")
    print("=" * 70)

    # Load curves once
    print("\nLoading individual strategy equity curves...")
    curves = []
    for label, config in STRATEGIES.items():
        s = load_strategy_curve(label, config)
        print(f"  {config['name']}: {s.index[0].date()} to {s.index[-1].date()}")
        curves.append(s)

    prices = align_and_fill(curves)
    print(f"\nCommon period: {prices.index[0].date()} to {prices.index[-1].date()}")

    returns = prices.pct_change().dropna()

    # Annual rebalance dates (dummy, weights are fixed anyway)
    all_cal_dates = pd.date_range(returns.index[0], returns.index[-1], freq="D")
    rebalance_dates = pd.Series(index=all_cal_dates).resample("YE").last().index
    trading_days = returns.index
    rebalance_dates = [trading_days[trading_days >= d][0] for d in rebalance_dates
                       if d <= trading_days[-1]]

    # Run each combination
    print(f"\nRunning {len(WEIGHT_COMBINATIONS)} weight combinations...")
    results = []
    for w_m, w_q, w_d, w_a in WEIGHT_COMBINATIONS:
        weights = (w_m, w_q, w_d, w_a)
        method = fixed_weight_method(weights)
        equity = run_blend_backtest(returns, method, rebalance_dates)

        is_eq = equity.loc[IS_START:IS_END]
        oos_eq = equity.loc[OOS_START:END_DATE]

        full_m = compute_metrics(equity)
        is_m = compute_metrics(is_eq)
        oos_m = compute_metrics(oos_eq)

        results.append({
            "momentum_w": w_m,
            "quality_w": w_q,
            "deep_value_w": w_d,
            "multiasset_w": w_a,
            "full_total": full_m.get("total_return", np.nan),
            "full_cagr": full_m.get("cagr", np.nan),
            "full_dd": full_m.get("max_dd", np.nan),
            "full_sharpe": full_m.get("sharpe", np.nan),
            "full_vol": full_m.get("ann_vol", np.nan),
            "is_cagr": is_m.get("cagr", np.nan),
            "is_dd": is_m.get("max_dd", np.nan),
            "is_sharpe": is_m.get("sharpe", np.nan),
            "oos_cagr": oos_m.get("cagr", np.nan),
            "oos_dd": oos_m.get("max_dd", np.nan),
            "oos_sharpe": oos_m.get("sharpe", np.nan),
        })

    results_df = pd.DataFrame(results)
    results_df["blend"] = results_df.apply(
        lambda r: f"M{r['momentum_w']:.0%} Q{r['quality_w']:.0%} V{r['deep_value_w']:.0%} A{r['multiasset_w']:.0%}",
        axis=1,
    )

    # Print top 10 by different criteria
    print("\n" + "=" * 70)
    print("TOP 10 BY FULL-PERIOD CAGR")
    print("=" * 70)
    top_cagr = results_df.sort_values("full_cagr", ascending=False).head(10)
    print(top_cagr[["blend", "full_total", "full_cagr", "full_dd", "full_sharpe", "oos_cagr", "oos_dd"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 10 BY FULL-PERIOD SHARPE")
    print("=" * 70)
    top_sharpe = results_df.sort_values("full_sharpe", ascending=False).head(10)
    print(top_sharpe[["blend", "full_total", "full_cagr", "full_dd", "full_sharpe", "oos_cagr", "oos_dd"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 10 BY LOWEST FULL-PERIOD MAX DRAWDOWN")
    print("=" * 70)
    top_dd = results_df.sort_values("full_dd", ascending=False).head(10)
    print(top_dd[["blend", "full_total", "full_cagr", "full_dd", "full_sharpe", "oos_cagr", "oos_dd"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 10 BY OOS CAGR")
    print("=" * 70)
    top_oos = results_df.sort_values("oos_cagr", ascending=False).head(10)
    print(top_oos[["blend", "full_cagr", "full_dd", "oos_cagr", "oos_dd", "oos_sharpe"]].to_string(index=False))

    # Save full grid
    out_path = os.path.join(os.path.dirname(__file__), "portfolio_blend_variations.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved full grid: {out_path}")


if __name__ == "__main__":
    main()
