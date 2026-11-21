"""
Main daily live runner.

Run this once per day after market close to:
  - Update PnL for both strategies
  - Send day-end PnL to Telegram
  - Execute rebalancing and send entry/exit alerts on rebalance days

Usage:
  python -m live.live_runner

For VPS scheduling, run via cron at 6:30 PM IST (after NSE close):
  30 18 * * * cd /path/to/project && /usr/bin/python3 -m live.live_runner >> live/logs/runner.log 2>&1
"""

import os
import sys
import traceback
from datetime import date

# Add parent directory to path so we can import live.*
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from live import config, telegram_bot
from live.high_risk_live import run_high_risk
from live.portfolio_blend_live import run_portfolio_blend


def initialize():
    """Initialize state files if they don't exist."""
    from live import state_manager
    state_manager.load_state("HighRisk", config.HIGH_RISK_CAPITAL)
    state_manager.load_state("PortfolioBlend", config.PORTFOLIO_BLEND_CAPITAL)


def main():
    print(f"Live runner started: {date.today()}")

    if config.TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("WARNING: Telegram token not configured. Messages will print to console only.")

    try:
        run_high_risk()
    except Exception as e:
        error_msg = f"HighRisk error: {e}\n{traceback.format_exc()}"
        print(error_msg)
        telegram_bot.send_error_alert(error_msg)

    try:
        run_portfolio_blend()
    except Exception as e:
        error_msg = f"PortfolioBlend error: {e}\n{traceback.format_exc()}"
        print(error_msg)
        telegram_bot.send_error_alert(error_msg)

    print("Live runner finished.")


if __name__ == "__main__":
    # Optional: pass --init to reset state files
    if "--init" in sys.argv:
        initialize()
        print("State files initialized.")
    else:
        main()
