"""
Telegram notification helper.
"""

import requests
from . import config


def send_message(text, disable_notification=False):
    """Send a Telegram message. Falls back to console print if not configured."""
    if not config.LIVE_TELEGRAM or config.TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("[TELEGRAM NOT SENT - configure token]")
        print(text)
        return

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_notification": disable_notification,
    }
    try:
        response = requests.post(url, data=payload, timeout=20)
        response.raise_for_status()
    except Exception as e:
        print(f"[Telegram error: {e}]")
        print(text)


def send_pnl_report(strategy_name, date_str, capital, value, day_pnl, day_pnl_pct, total_pnl, total_pnl_pct):
    """Send end-of-day PnL report."""
    sign = "+" if day_pnl >= 0 else ""
    text = (
        f"<b>{strategy_name} - Day End PnL</b>\n"
        f"Date: {date_str}\n"
        f"Capital: Rs {capital:,.0f}\n"
        f"Current Value: Rs {value:,.0f}\n"
        f"Day PnL: {sign}Rs {day_pnl:,.0f} ({day_pnl_pct:+.2f}%)\n"
        f"Total PnL: Rs {total_pnl:,.0f} ({total_pnl_pct:+.2f}%)"
    )
    send_message(text)


def send_trade_alert(strategy_name, action, ticker, shares, price, value):
    """Send entry/exit trade alert."""
    action_text = "BUY" if action.upper() == "BUY" else "SELL"
    text = (
        f"<b>{strategy_name} - Trade Alert</b>\n"
        f"{action_text} {ticker}\n"
        f"Shares: {shares:.2f}\n"
        f"Price: Rs {price:.2f}\n"
        f"Value: Rs {value:,.0f}"
    )
    send_message(text)


def send_rebalance_alert(strategy_name, date_str, new_positions):
    """Send a summary of new target positions after rebalance."""
    text = f"<b>{strategy_name} - Rebalance {date_str}</b>\n\n<b>Target positions:</b>\n"
    for ticker, info in new_positions.items():
        text += f"{ticker}: {info['shares']:.2f} shares @ Rs {info['price']:.2f}\n"
    send_message(text)


def send_error_alert(error_msg):
    """Send error notification."""
    text = f"<b>Live Runner Error</b>\n<pre>{error_msg}</pre>"
    send_message(text)
