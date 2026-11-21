"""
Test Telegram notifications.

Run after setting environment variables:
  export TELEGRAM_BOT_TOKEN="..."
  export TELEGRAM_CHAT_ID="..."

Then:
  python -m live.test_telegram
"""

from . import config, telegram_bot


def main():
    print(f"Token configured: {config.TELEGRAM_BOT_TOKEN != 'YOUR_BOT_TOKEN_HERE'}")
    print(f"Chat ID configured: {config.TELEGRAM_CHAT_ID != 'YOUR_CHAT_ID_HERE'}")

    if config.TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or config.TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        print("ERROR: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables first.")
        return

    telegram_bot.send_message(
        "<b>Test alert</b>\n"
        "Your live trading bot is configured correctly.\n"
        "You will receive day-end PnL and trade alerts here."
    )
    print("Test message sent. Check Telegram.")


if __name__ == "__main__":
    main()
