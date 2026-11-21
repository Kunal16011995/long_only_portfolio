"""
Live trading system configuration.

Fill in your Telegram bot token and chat ID before running.
"""

# ---------------------------------------------------------------------------
# CAPITAL ALLOCATION (INR)
# ---------------------------------------------------------------------------
HIGH_RISK_CAPITAL = 100_000      # 1 lakh INR
PORTFOLIO_BLEND_CAPITAL = 200_000  # 2 lakh INR

# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------
# Create a bot via @BotFather and get the token.
# Get your chat ID by messaging @userinfobot or @getidsbot.
# Recommended: set these as environment variables so the token is not stored
# in this file. Example:
#   export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
#   export TELEGRAM_CHAT_ID="123456789"
import os
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

# ---------------------------------------------------------------------------
# STRATEGY PARAMETERS
# ---------------------------------------------------------------------------
# High risk: signal blend M40 Q40 V20, Top 10, quarterly
HIGH_RISK_SIGNAL_WEIGHTS = {"momentum": 0.40, "quality": 0.40, "deep_value": 0.20}
HIGH_RISK_TOP_N = 10
HIGH_RISK_REBALANCE_FREQ = "QE"  # quarter-end
HIGH_RISK_COST_BPS = 0.10

# Portfolio blend: 25/25/25/25 across 4 strategies, monthly
PORTFOLIO_BLEND_WEIGHTS = {
    "momentum": 0.25,
    "quality": 0.25,
    "deep_value": 0.25,
    "multiasset": 0.25,
}
PORTFOLIO_BLEND_REBALANCE_FREQ = "ME"  # month-end
PORTFOLIO_BLEND_COST_BPS = 0.05

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
# yfinance is used for end-of-day prices. Note: yfinance is free and may have
# delays. For production live trading, consider a real-time data provider.
CACHE_DIR = "data_cache"

# ---------------------------------------------------------------------------
# MODE
# ---------------------------------------------------------------------------
# Set to True to send Telegram messages, False to print to console only.
LIVE_TELEGRAM = False

# Set to True to actually write state changes, False for dry-run.
DRY_RUN = True
