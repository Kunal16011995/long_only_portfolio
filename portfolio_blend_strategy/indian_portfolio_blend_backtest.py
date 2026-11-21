"""
Indian Long-Only Portfolio-Blending Strategy

Combines 4 independently-run strategies at the portfolio level:
  1. NIFTY 100 Momentum
  2. Quality
  3. Deep Value
  4. Multi-Asset Momentum (stock + ETF version)

Each strategy is allocated a capital share and run independently.
The combined portfolio is rebalanced annually using one of several methods:
  - Equal weight
  - Inverse volatility (risk parity)
  - Momentum of strategies (tilt to best recent performer)

Inputs are the equity-curve CSV files produced by each individual strategy.
"""

import os
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.4f}".format

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
START_DATE = "2010-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")
IS_START = "2015-01-01"
IS_END = "2019-12-31"
OOS_START = "2020-01-01"

# Paths to individual strategy equity curves (relative to this script)
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
STRATEGIES = {
    "momentum": {
        "name": "NIFTY 100 Momentum",
        "path": os.path.join(BASE_DIR, "nifty100_momentum_strategy", "momentum_equity_curve.csv"),
        "value_col": "PortfolioValue",
    },
    "quality": {
        "name": "Quality",
        "path": os.path.join(BASE_DIR, "quality_strategy", "quality_equity_curve.csv"),
        "value_col": "PortfolioValue",
    },
    "deep_value": {
        "name": "Deep Value",
        "path": os.path.join(BASE_DIR, "deep_value_strategy", "deep_value_equity_curve.csv"),
        "value_col": "equity",
    },
    "multiasset": {
        "name": "Multi-Asset Momentum",
        "path": os.path.join(BASE_DIR, "multiasset_momentum_strategy", "stock_multiasset_equity_curve.csv"),
        "value_col": "PortfolioValue",
    },
}

# Alternatively, use ETF-only multiasset by changing the path above to:
# multiasset_momentum_strategy/multiasset_equity_curve.csv

REBALANCE_FREQ = "YE"  # annual rebalancing of blend weights
COST_BPS = 0.05        # one-way cost for blend rebalancing (lower than stock turnover)
COST = COST_BPS / 100.0

# ---------------------------------------------------------------------------
# LOAD STRATEGY RETURNS
# ---------------------------------------------------------------------------
def load_strategy_curve(label, config):
    if not os.path.exists(config["path"]):
        raise FileNotFoundError(f"Missing equity curve: {config['path']}")

    df = pd.read_csv(config["path"], parse_dates=[0], index_col=0)
    df.index.name = "date"
    df = df.sort_index()

    value_col = config["value_col"]
    if value_col not in df.columns:
        # Try common alternatives
        for c in ["equity", "PortfolioValue", "Value", "portfolio_value"]:
            if c in df.columns:
                value_col = c
                break

    series = df[value_col].rename(label)
    return series


def align_and_fill(curves):
    """Align all curves to common daily index and forward-fill."""
    df = pd.concat(curves, axis=1)
    df = df.ffill().dropna(how="any")
    return df


# ---------------------------------------------------------------------------
# BLEND METHODS
# ---------------------------------------------------------------------------
def equal_weights(rets_window):
    n = len(rets_window.columns)
    return np.ones(n) / n


def inverse_vol_weights(rets_window):
    """Risk parity: weight inversely proportional to realised volatility."""
    vols = rets_window.std() * np.sqrt(252)
    vols = vols.replace(0, np.nan)
    inv = 1.0 / vols
    inv = inv.fillna(0)
    if inv.sum() == 0:
        return equal_weights(rets_window)
    return inv / inv.sum()


def momentum_weights(rets_window):
    """Tilt capital toward strategies with best recent return."""
    total_ret = (1 + rets_window).prod() - 1
    pos = total_ret.clip(lower=0)
    if pos.sum() == 0:
        return equal_weights(rets_window)
    return pos / pos.sum()


BLEND_METHODS = {
    "equal": equal_weights,
    "risk_parity": inverse_vol_weights,
    "momentum": momentum_weights,
}


# ---------------------------------------------------------------------------
# PORTFOLIO BLEND BACKTEST
# ---------------------------------------------------------------------------
def run_blend_backtest(returns, method, rebalance_dates, initial_capital=1_000_000):
    """
    returns: DataFrame of daily strategy returns (not prices)
    method: function that takes a window of returns and returns target weights
    """
    rebalance_set = set(rebalance_dates)
    labels = returns.columns.tolist()
    n = len(labels)

    capital_per_strategy = {label: initial_capital / n for label in labels}
    equity_curve = []

    for i, today in enumerate(returns.index):
        # Apply daily returns to each strategy slice
        for label in labels:
            capital_per_strategy[label] *= (1 + returns.loc[today, label])

        # Rebalance weights if today is a rebalance date
        if today in rebalance_set and i > 0:
            total_value = sum(capital_per_strategy.values())

            lookback = 252  # 1 year
            start_idx = max(0, i - lookback)
            rets_window = returns.iloc[start_idx:i]
            target_weights = method(rets_window)

            # Apply transaction costs on weight changes
            current_weights = np.array([capital_per_strategy[l] / total_value for l in labels])
            deltas = target_weights - current_weights

            for j, label in enumerate(labels):
                if deltas[j] > 0:
                    # Need to buy this strategy: move cash from others
                    # Simple approach: rebalance capital directly, charge cost on traded amount
                    target_cap = total_value * target_weights[j]
                    traded = abs(target_cap - capital_per_strategy[label])
                    capital_per_strategy[label] = target_cap * (1 - COST * traded / target_cap if target_cap > 0 else 1)
                elif deltas[j] < 0:
                    target_cap = total_value * target_weights[j]
                    traded = abs(target_cap - capital_per_strategy[label])
                    capital_per_strategy[label] = target_cap * (1 - COST * traded / capital_per_strategy[label] if capital_per_strategy[label] > 0 else 1)

        total_value = sum(capital_per_strategy.values())
        equity_curve.append({"date": today, "equity": total_value})

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    return equity_df


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
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Indian Long-Only Portfolio-Blending Strategy")
    print("=" * 70)
    print(f"Full period: {START_DATE} to {END_DATE}")
    print(f"In-Sample : {IS_START} to {IS_END}")
    print(f"Out-of-Sample : {OOS_START} to {END_DATE}")
    print(f"Strategies: {', '.join(STRATEGIES[s]['name'] for s in STRATEGIES)}")
    print(f"Blend methods: Equal weight, Risk parity, Momentum-of-strategies")
    print("=" * 70)

    # Load individual curves
    print("\nLoading individual strategy equity curves...")
    curves = []
    for label, config in STRATEGIES.items():
        try:
            s = load_strategy_curve(label, config)
            print(f"  {config['name']}: {s.index[0].date()} to {s.index[-1].date()}")
            curves.append(s)
        except Exception as e:
            print(f"  ERROR loading {config['name']}: {e}")
            raise

    prices = align_and_fill(curves)
    print(f"\nCommon period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Trading days: {len(prices)}")

    # Daily returns
    returns = prices.pct_change().dropna()

    # Generate annual rebalance dates
    all_cal_dates = pd.date_range(returns.index[0], returns.index[-1], freq="D")
    rebalance_dates = pd.Series(index=all_cal_dates).resample(REBALANCE_FREQ).last().index
    trading_days = returns.index
    rebalance_dates = [trading_days[trading_days >= d][0] for d in rebalance_dates
                       if d <= trading_days[-1]]

    # Run each blend method
    print("\nRunning portfolio blends...")
    results = {}
    for method_name, method_func in BLEND_METHODS.items():
        print(f"  {method_name} ...")
        equity = run_blend_backtest(returns, method_func, rebalance_dates)
        results[method_name] = equity

    # Also keep individual strategy curves for comparison
    individual_metrics = {}
    for label in returns.columns:
        individual_equity = (1 + returns[label]).cumprod() * 1_000_000
        individual_metrics[label] = compute_metrics(individual_equity.to_frame("equity"))

    # Report
    print("\n" + "=" * 70)
    print("PORTFOLIO BLEND RESULTS")
    print("=" * 70)

    def print_metrics(name, equity):
        m = compute_metrics(equity)
        if not m:
            print(f"\n{name}: no data")
            return
        print(f"\n{name}")
        print("-" * 50)
        print(f"Total return        : {m['total_return']*100:>8.2f}%")
        print(f"CAGR                : {m['cagr']*100:>8.2f}%")
        print(f"Max drawdown        : {m['max_dd']*100:>8.2f}%")
        print(f"Ann. volatility     : {m['ann_vol']*100:>8.2f}%")
        print(f"Sharpe (ann.)       : {m['sharpe']:>8.2f}")

    for method_name, equity in results.items():
        print_metrics(f"BLEND: {method_name.upper()}", equity)

    print("\n" + "-" * 70)
    print("INDIVIDUAL STRATEGIES (for comparison)")
    print("-" * 70)
    for label, m in individual_metrics.items():
        print_metrics(STRATEGIES[label]["name"].upper(), (1 + returns[label]).cumprod().to_frame("equity") * 1_000_000)

    # IS / OOS split for best blend (risk parity usually)
    print("\n" + "=" * 70)
    print("IS / OOS BREAKDOWN")
    print("=" * 70)
    for method_name, equity in results.items():
        is_eq = equity.loc[IS_START:IS_END]
        oos_eq = equity.loc[OOS_START:END_DATE]
        is_m = compute_metrics(is_eq)
        oos_m = compute_metrics(oos_eq)
        print(f"\n{method_name.upper()}")
        print(f"  IS  2015-2019: CAGR {is_m.get('cagr', np.nan)*100:6.2f}%, DD {is_m.get('max_dd', np.nan)*100:6.2f}%, Sharpe {is_m.get('sharpe', np.nan):5.2f}")
        print(f"  OOS 2020-   : CAGR {oos_m.get('cagr', np.nan)*100:6.2f}%, DD {oos_m.get('max_dd', np.nan)*100:6.2f}%, Sharpe {oos_m.get('sharpe', np.nan):5.2f}")

    # Save curves
    for method_name, equity in results.items():
        equity.to_csv(os.path.join(os.path.dirname(__file__), f"portfolio_blend_{method_name}_equity_curve.csv"))
    print("\nSaved blended equity curves.")


if __name__ == "__main__":
    main()
