# Live Trading System

Paper/live monitoring system for two strategies:

1. **High Risk / High Reward** — Signal blend (40% Momentum + 40% Quality + 20% Deep Value), Top 10 stocks, quarterly rebalancing. Virtual capital: **₹1,00,000**.
2. **Low Risk / Moderate Reward** — Portfolio blend (25% Momentum + 25% Quality + 25% Deep Value + 25% Multi-Asset), monthly rebalancing. Virtual capital: **₹2,00,000**.

Prices are fetched from **Yahoo Finance** via `yfinance`.

---

## Files

| File | Purpose |
|---|---|
| `config.py` | Capital, Telegram token, strategy parameters |
| `telegram_bot.py` | Telegram messaging helpers |
| `state_manager.py` | Persists cash/positions in `live/state/` |
| `high_risk_live.py` | Daily logic for high-risk strategy |
| `portfolio_blend_live.py` | Daily logic for portfolio blend |
| `live_runner.py` | Main daily runner |

---

## Setup

### 1. Install dependencies

```bash
pip install yfinance pandas numpy requests
```

### 2. Configure Telegram

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy the bot token.
2. Get your chat ID by messaging [@userinfobot](https://t.me/userinfobot).
3. Set environment variables (recommended, keeps token out of code):

```bash
export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
export TELEGRAM_CHAT_ID="123456789"
```

Or edit `live/config.py` directly:

```python
TELEGRAM_BOT_TOKEN = "123456:ABC-DEF..."
TELEGRAM_CHAT_ID = "123456789"
```

Then set:

```python
LIVE_TELEGRAM = True
DRY_RUN = False   # set True for testing without writing state
```

### 3. Initialize state files

```bash
cd /path/to/NSE/Long\ Term
python -m live.live_runner --init
```

This creates:
- `live/state/HighRisk_state.json`
- `live/state/PortfolioBlend_state.json`

### 4. Test manually (dry-run recommended first)

```bash
python -m live.live_runner
```

With `DRY_RUN = True` and `LIVE_TELEGRAM = False`, it will print to console without saving state or sending Telegram messages.

### 5. Schedule on VPS (Linux cron)

Run daily at 6:30 PM IST (after NSE close):

```bash
crontab -e
```

Add:

```cron
30 18 * * * cd /path/to/NSE/Long\ Term && /usr/bin/python3 -m live.live_runner >> /path/to/NSE/Long\ Term/live/logs/runner.log 2>&1
```

Create log directory first:

```bash
mkdir -p live/logs
```

For monthly schedule verification:

```bash
# Run once manually
python -m live.live_runner
```

---

## What the bot sends

### Day-end PnL (every day)

```
HighRisk — Day End PnL
Date: 2025-11-21
Capital: Rs 100,000
Current Value: Rs 145,230
Day PnL: +Rs 1,250 (+0.87%)
Total PnL: Rs 45,230 (+45.23%)
```

### Trade alerts (on rebalance days only)

```
HighRisk — Trade Alert
SELL RELIANCE.NS
Shares: 10.50
Price: Rs 2,450.00
Value: Rs 25,725
```

### Allocation alerts (portfolio blend, monthly)

```
PortfolioBlend — Allocation Change
BUY momentum
Old: Rs 48,000 (24.0%)
New: Rs 50,000 (25.0%)
```

### Error alerts

If any strategy fails, you get a Telegram error message.

---

## Important caveats

- **Yahoo Finance is free and can be delayed.** It is fine for end-of-day signals but not for intraday execution.
- **This is a paper/virtual system.** No real broker API is connected.
- **Fundamental data is not point-in-time.** Live signals use current yfinance fundamentals, so expect some look-ahead bias versus true historical results.
- **Survivorship bias:** universe is current NIFTY 100 constituents.

---

## Customization

### Change capital

Edit `live/config.py`:

```python
HIGH_RISK_CAPITAL = 100_000
PORTFOLIO_BLEND_CAPITAL = 200_000
```

Then re-initialize:

```bash
python -m live.live_runner --init
```

### Change signal weights (high risk)

Edit `live/config.py`:

```python
HIGH_RISK_SIGNAL_WEIGHTS = {"momentum": 0.40, "quality": 0.40, "deep_value": 0.20}
```

### Change portfolio blend weights

Edit `live/config.py`:

```python
PORTFOLIO_BLEND_WEIGHTS = {
    "momentum": 0.25,
    "quality": 0.25,
    "deep_value": 0.25,
    "multiasset": 0.25,
}
```

---

## Troubleshooting

### Telegram messages not received

- Check token and chat ID.
- Start a chat with your bot first.
- Try sending a test message:

```python
from live.telegram_bot import send_message
send_message("Test message")
```

### yfinance download fails

- Check internet connectivity.
- Yahoo Finance may rate-limit. Add delays or use `yfinance` cache.
- For Indian stocks, tickers must end with `.NS`.

### State looks wrong

Delete `live/state/*.json` and re-run `--init`.
