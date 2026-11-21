"""
High-risk signal blend — live daily runner.

Strategy: 40% Momentum + 40% Quality + 20% Deep Value
Top 10 NIFTY 100 stocks, quarterly rebalancing.
"""

import os
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "signal_blend_strategy"))
from indian_signal_blend_backtest import (
    NIFTY100_TICKERS,
    download_prices, get_fundamentals,
    compute_momentum_score, compute_quality_score, compute_deep_value_score,
)

from . import config, state_manager, telegram_bot

STRATEGY_NAME = "HighRisk"


def is_rebalance_day(today):
    """Check if today is the last trading day of a quarter."""
    # Last day of Mar, Jun, Sep, Dec
    if today.month not in [3, 6, 9, 12]:
        return False
    # If next calendar day is in the next month, today is month-end
    next_day = today + timedelta(days=1)
    return next_day.month != today.month


def get_target_tickers(price_data, fundamentals, as_of_date):
    """Compute blended signal and return top N target tickers."""
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
        config.HIGH_RISK_SIGNAL_WEIGHTS["momentum"] * mom_rank.fillna(0.5)
        + config.HIGH_RISK_SIGNAL_WEIGHTS["quality"] * qual_rank.fillna(0.5)
        + config.HIGH_RISK_SIGNAL_WEIGHTS["deep_value"] * dv_rank.fillna(0.5)
    )

    # Use signal as of previous close
    signal_date = blended.index.asof(as_of_date - timedelta(days=1))
    if signal_date is None or pd.isna(signal_date):
        return []
    score_row = blended.loc[signal_date].dropna()
    if len(score_row) < config.HIGH_RISK_TOP_N:
        return score_row.sort_values(ascending=False).index.tolist()
    return score_row.sort_values(ascending=False).head(config.HIGH_RISK_TOP_N).index.tolist()


def fetch_latest_prices(tickers):
    """Fetch latest close prices for a list of tickers."""
    end = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=5)
    prices = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                prices[ticker] = float(df["Close"].iloc[-1])
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")
    return prices


def rebalance(state, target_tickers, prices):
    """Full reconstitution: sell all existing, buy target equal-weight."""
    # Sell all current positions
    for ticker in list(state["positions"].keys()):
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        shares = state["positions"][ticker]
        proceeds = shares * price * (1 - config.HIGH_RISK_COST_BPS / 100)
        state["cash"] += proceeds
        telegram_bot.send_trade_alert(STRATEGY_NAME, "SELL", ticker, shares, price, shares * price)
        del state["positions"][ticker]

    # Buy target positions
    target_tickers = [t for t in target_tickers if prices.get(t, 0) > 0]
    if not target_tickers:
        return state

    weight = 1.0 / len(target_tickers)
    new_positions = {}
    for ticker in target_tickers:
        price = prices[ticker]
        alloc = state["cash"] * weight
        shares = alloc * (1 - config.HIGH_RISK_COST_BPS / 100) / price
        state["cash"] -= alloc
        state["positions"][ticker] = shares
        new_positions[ticker] = {"shares": shares, "price": price}
        telegram_bot.send_trade_alert(STRATEGY_NAME, "BUY", ticker, shares, price, alloc)

    telegram_bot.send_rebalance_alert(STRATEGY_NAME, date.today().strftime("%Y-%m-%d"), new_positions)
    return state


def run_high_risk(today=None):
    """Main daily function."""
    if today is None:
        today = date.today()

    print(f"\n[{STRATEGY_NAME}] Running for {today}")

    state = state_manager.load_state(STRATEGY_NAME, config.HIGH_RISK_CAPITAL)

    # Need price history for signal computation
    end = today + timedelta(days=1)
    start = today - timedelta(days=365 * 3)  # 3 years of history for signals

    print("  Fetching price history...")
    price_data = {}
    valid_tickers = []
    for ticker in NIFTY100_TICKERS:
        try:
            df = download_prices(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            if len(df) > 100:
                price_data[ticker] = df
                valid_tickers.append(ticker)
        except Exception:
            pass

    print(f"  Valid stocks: {len(valid_tickers)}")
    print("  Fetching fundamentals...")
    fundamentals = {t: get_fundamentals(t) for t in valid_tickers}

    # Latest prices for PnL and execution
    all_tickers = list(state["positions"].keys()) + valid_tickers
    prices = fetch_latest_prices(list(set(all_tickers)))

    # Compute current portfolio value
    current_value = state_manager.compute_portfolio_value(state, prices)
    day_pnl = current_value - state["last_value"]
    day_pnl_pct = (current_value / state["last_value"] - 1) * 100 if state["last_value"] > 0 else 0
    total_pnl = current_value - config.HIGH_RISK_CAPITAL
    total_pnl_pct = (current_value / config.HIGH_RISK_CAPITAL - 1) * 100

    # Rebalance if quarter-end
    if is_rebalance_day(today):
        print("  Rebalance day detected.")
        target_tickers = get_target_tickers(price_data, fundamentals, today)
        print(f"  Target tickers: {target_tickers}")
        state = rebalance(state, target_tickers, prices)
        current_value = state_manager.compute_portfolio_value(state, prices)
        state["last_rebalance_date"] = today.strftime("%Y-%m-%d")

    # Send PnL report
    telegram_bot.send_pnl_report(
        STRATEGY_NAME,
        today.strftime("%Y-%m-%d"),
        config.HIGH_RISK_CAPITAL,
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
    run_high_risk()
