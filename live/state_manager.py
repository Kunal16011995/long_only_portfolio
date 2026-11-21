"""
State manager for live paper trading.

Persists cash and positions per strategy in JSON files under live/state/.
"""

import json
import os
from datetime import date

from . import config

STATE_DIR = os.path.join(os.path.dirname(__file__), "state")
os.makedirs(STATE_DIR, exist_ok=True)


def state_file(strategy_name):
    return os.path.join(STATE_DIR, f"{strategy_name}_state.json")


def load_state(strategy_name, default_cash):
    """Load state for a strategy. If missing, initialize with cash and empty positions."""
    path = state_file(strategy_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {
        "cash": float(default_cash),
        "positions": {},  # ticker -> shares
        "last_value": float(default_cash),
        "last_rebalance_date": None,
    }


def save_state(strategy_name, state):
    """Save state for a strategy."""
    if config.DRY_RUN:
        print(f"[DRY-RUN] Would save state for {strategy_name}")
        return
    path = state_file(strategy_name)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def initialize_state(strategy_name, cash):
    """Force-initialize a strategy state with cash and no positions."""
    state = {
        "cash": float(cash),
        "positions": {},
        "last_value": float(cash),
        "last_rebalance_date": None,
    }
    save_state(strategy_name, state)
    return state


def compute_portfolio_value(state, prices):
    """Compute total portfolio value given current prices."""
    value = state["cash"]
    for ticker, shares in state["positions"].items():
        price = prices.get(ticker)
        if price is not None and price > 0:
            value += shares * price
    return value


def reset_state_for_backtest(strategy_name, cash):
    """Reset state for a clean backtest or paper start."""
    return initialize_state(strategy_name, cash)
