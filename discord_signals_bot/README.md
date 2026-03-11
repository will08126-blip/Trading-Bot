# Discord Signals Bot

A standalone Discord bot that analyzes BTC, ETH, SOL, and XRP on Binance Futures and sends **leveraged trade signals** to two Discord channels — one for entries, one for exits.

You get the signal, you place the trade yourself on whatever exchange you want.

---

## How It Works

```
Every 60 seconds:
  ├─ Fetches 1m / 5m / 15m / 4h candles from Binance (free public API)
  ├─ Runs 4 strategies: Trend Pullback, Breakout Retest, Liquidity Sweep, Volatility Expansion
  ├─ Votes on confidence score (0–100) for long + short on each pair
  ├─ If score ≥ 40 → posts entry signal to #entry-signals
  └─ Checks your active trades for TP1 / TP2 / Stop Loss → posts to #exit-signals
```

### Leverage Tiers

| Score     | Tier   | Leverage  | Portfolio % |
|-----------|--------|-----------|-------------|
| 40 – 59   | MEDIUM | 5x – 10x  | 5% – 15%    |
| 60 – 79   | STRONG | 10x – 25x | 15% – 40%   |
| 80 – 100  | ELITE  | 25x – 100x| 40% – 100%  |

### Trade Confirmation Flow

1. Bot posts entry signal in `#entry-signals` with full price levels
2. You react ✅ to confirm you entered → bot starts tracking it
3. When price hits **TP1**: exit channel posts "take 50% profit, move stop to breakeven"
4. When price hits **TP2**: exit channel posts "close remaining position"
5. If stop is hit: exit channel posts stop-loss alert

---

## Setup

### 1. Create a Discord Bot

1. Go to [https://discord.com/developers/applications](https://discord.com/developers/applications)
2. Click **New Application**, give it a name
3. Go to **Bot** → click **Add Bot**
4. Under **Privileged Gateway Intents**, enable:
   - **Message Content Intent**
   - **Server Members Intent**
5. Copy the **Bot Token**
6. Go to **OAuth2 → URL Generator**:
   - Scopes: `bot`, `applications.commands`
   - Bot Permissions: `Send Messages`, `Read Messages/View Channels`, `Read Message History`, `Add Reactions`, `Use Slash Commands`
7. Use the generated URL to invite the bot to your server

### 2. Get Channel and User IDs

Enable **Developer Mode** in Discord (Settings → Advanced → Developer Mode):
- Right-click your `#entry-signals` channel → **Copy ID**
- Right-click your `#exit-signals` channel → **Copy ID**
- Right-click your own username → **Copy ID**

### 3. Configure Environment

```bash
cd discord_signals_bot
cp .env.example .env
# Edit .env with your token, channel IDs, and user ID
```

### 4. Install Dependencies

```bash
pip install -r discord_signals_bot/requirements.txt
```

### 5. Run the Bot

```bash
# From the Trading-Bot root directory
python -m discord_signals_bot.main
```

---

## Slash Commands

| Command | Description |
|---------|-------------|
| `/status` | Show active trades and bot stats |
| `/trades` | List your currently active trades |
| `/close <symbol> <long\|short>` | Manually mark a trade as closed |

---

## Configuration

Edit `discord_signals_bot/config/signals_config.yaml` to adjust:

- **Symbols** — which pairs to scan
- **Scoring thresholds** — minimum score to post a signal
- **Leverage tiers** — adjust leverage and portfolio % per tier
- **Exit R-multiples** — TP1/TP2 distances from entry (default: 1.5R and 3.0R)
- **Signal cooldown** — min time between signals for same pair+direction

---

## Data Source

All market data comes from **Binance USDⓈ-M Futures** public API — no API key required.
The signals are for analysis only. You place trades manually on any exchange you prefer.
