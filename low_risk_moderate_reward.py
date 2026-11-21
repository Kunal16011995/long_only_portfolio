"""
Low Risk / Moderate-to-High Reward Version

Portfolio blend: 25% Momentum + 25% Quality + 25% Deep Value + 25% Multi-Asset
Monthly rebalancing.

Expected (2015-2026):
  CAGR ~16%, Max DD ~-14%, Sharpe ~1.48
"""

import os
import sys
import warnings
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.4f}".format

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "portfolio_blend_strategy"))
from indian_portfolio_blend_backtest import (
    STRATEGIES, load_strategy_curve, align_and_fill, run_blend_backtest,
)

START_DATE = "2010-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")
IS_START = "2015-01-01"
IS_END = "2019-12-31"
OOS_START = "2020-01-01"

WEIGHTS = (0.25, 0.25, 0.25, 0.25)
REBALANCE_FREQ = "ME"  # monthly


def fixed_weight_method(weights):
    def f(rets_window):
        return np.array(weights)
    return f


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
    print("LOW RISK / MODERATE-TO-HIGH REWARD VERSION")
    print("Portfolio blend: 25% Momentum + 25% Quality + 25% Deep Value + 25% Multi-Asset")
    print("Monthly rebalancing")
    print("=" * 70)

    print("\nLoading individual strategy equity curves...")
    curves = []
    for label, config in STRATEGIES.items():
        s = load_strategy_curve(label, config)
        print(f"  {config['name']}: {s.index[0].date()} to {s.index[-1].date()}")
        curves.append(s)

    prices = align_and_fill(curves)
    returns = prices.pct_change().dropna()
    print(f"\nCommon period: {returns.index[0].date()} to {returns.index[-1].date()}")

    all_cal_dates = pd.date_range(returns.index[0], returns.index[-1], freq="D")
    rebalance_dates = pd.Series(index=all_cal_dates).resample(REBALANCE_FREQ).last().index
    trading_days = returns.index
    rebalance_dates = [trading_days[trading_days >= d][0] for d in rebalance_dates
                       if d <= trading_days[-1]]

    equity = run_blend_backtest(returns, fixed_weight_method(WEIGHTS), rebalance_dates)

    full_m = compute_metrics(equity)
    is_m = compute_metrics(equity.loc[IS_START:IS_END])
    oos_m = compute_metrics(equity.loc[OOS_START:END_DATE])

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Full period: {equity.index[0].date()} to {equity.index[-1].date()}")
    print(f"\nStrategy:")
    print(f"  Total return : {full_m['total_return']*100:8.2f}%")
    print(f"  CAGR         : {full_m['cagr']*100:8.2f}%")
    print(f"  Max DD       : {full_m['max_dd']*100:8.2f}%")
    print(f"  Volatility   : {full_m['ann_vol']*100:8.2f}%")
    print(f"  Sharpe       : {full_m['sharpe']:8.2f}")
    print(f"\nIS 2015-2019:  CAGR {is_m['cagr']*100:6.2f}% | Max DD {is_m['max_dd']*100:6.2f}% | Sharpe {is_m['sharpe']:.2f}")
    print(f"OOS 2020-   :  CAGR {oos_m['cagr']*100:6.2f}% | Max DD {oos_m['max_dd']*100:6.2f}% | Sharpe {oos_m['sharpe']:.2f}")

    equity.to_csv(os.path.join(os.path.dirname(__file__), "low_risk_moderate_reward_equity_curve.csv"))
    print("\nSaved: low_risk_moderate_reward_equity_curve.csv")


if __name__ == "__main__":
    main()
