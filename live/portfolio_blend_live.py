"""
Portfolio blend — live daily runner.

Strategy: 25% Momentum + 25% Quality + 25% Deep Value + 25% Multi-Asset
Monthly rebalancing of capital allocation.

Note: This tracks allocation across 4 strategy buckets using their backtested
or live equity curves. It does NOT generate individual stock/ETF entry-exit
alerts for the underlying strategies. For stock-level alerts, run each
strategy's own live module.
"""

import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "portfolio_blend_strategy"))
from indian_portfolio_blend_backtest import STRATEGIES, load_strategy_curve, align_and_fill

from . import config, state_manager, telegram_bot

STRATEGY_NAME = "PortfolioBlend"


def is_rebalance_day(today):
    """Check if today is the last trading day of the month."""
    next_day = today + timedelta(days=1)
    return next_day.month != today.month


def fetch_strategy_returns():
    """Load latest daily returns for each of the 4 strategies from saved equity curves."""
    curves = []
    labels = []
    for label, cfg in STRATEGIES.items():
        s = load_strategy_curve(label, cfg)
        curves.append(s)
        labels.append(label)

    prices = align_and_fill(curves)
    rets = prices.pct_change().dropna()
    rets.columns = labels
    return rets


def rebalance(state, target_weights, today):
    """Reallocate capital across 4 strategy buckets."""
    total_value = sum(state["positions"].values()) + state.get("cash", 0)

    # Current weights
    current_weights = {k: v / total_value for k, v in state["positions"].items()}

    new_positions = {}
    for label, target_w in target_weights.items():
        target_value = total_value * target_w
        current_value = state["positions"].get(label, 0)
        traded = abs(target_value - current_value)
        cost = traded * config.PORTFOLIO_BLEND_COST_BPS / 100
        new_positions[label] = target_value - cost

        action = "BUY" if target_value > current_value else "SELL"
        telegram_bot.send_message(
            f"<b>{STRATEGY_NAME} — Allocation Change</b>\n"
            f"{action} {label}\n"
            f"Old: ₹{current_value:,.0f} ({current_weights.get(label, 0)*100:.1f}%)\n"
            f"New: ₹{target_value:,.0f} ({target_w*100:.1f}%)"
        )

    state["positions"] = new_positions
    state["cash"] = 0
    state["last_rebalance_date"] = today.strftime("%Y-%m-%d")
    return state


def run_portfolio_blend(today=None):
    """Main daily function."""
    if today is None:
        today = date.today()

    print(f"\n[{STRATEGY_NAME}] Running for {today}")

    state = state_manager.load_state(STRATEGY_NAME, config.PORTFOLIO_BLEND_CAPITAL)

    # Ensure positions dict has all 4 labels
    for label in STRATEGIES.keys():
        if label not in state["positions"]:
            state["positions"][label] = 0.0

    print("  Fetching strategy returns...")
    rets = fetch_strategy_returns()
    today_str = today.strftime("%Y-%m-%d")

    if today_str in rets.index:
        daily_ret = rets.loc[today_str]
        for label in state["positions"]:
            state["positions"][label] *= (1 + daily_ret.get(label, 0))
    else:
        print(f"  No return data for {today_str}. Skipping daily return application.")

    # Rebalance if month-end
    if is_rebalance_day(today):
        print("  Rebalance day detected.")
        state = rebalance(state, config.PORTFOLIO_BLEND_WEIGHTS, today)

    # Compute value
    current_value = sum(state["positions"].values()) + state.get("cash", 0)
    day_pnl = current_value - state["last_value"]
    day_pnl_pct = (current_value / state["last_value"] - 1) * 100 if state["last_value"] > 0 else 0
    total_pnl = current_value - config.PORTFOLIO_BLEND_CAPITAL
    total_pnl_pct = (current_value / config.PORTFOLIO_BLEND_CAPITAL - 1) * 100

    telegram_bot.send_pnl_report(
        STRATEGY_NAME,
        today.strftime("%Y-%m-%d"),
        config.PORTFOLIO_BLEND_CAPITAL,
        current_value,
        day_pnl,
        day_pnl_pct,
        total_pnl,
        total_pnl_pct,
    )

    state["last_value"] = current_value
    state_manager.save_state(STRATEGY_NAME, state)
    print(f"  Current value: Rs {current_value:,.0f}")


if __name__ == "__main__":
    run_portfolio_blend()
