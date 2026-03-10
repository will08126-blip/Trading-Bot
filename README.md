# Crypto Futures Trading Bot

A production-style, fully automated hybrid scalp/swing crypto futures trading system for BTC and ETH on Coinbase Advanced / Coinbase Derivatives infrastructure.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Trading Worker Process                        │
│                                                                     │
│  Market Data → Features → Strategies → Voting → Risk → Execution   │
│       ↑              ↓          ↓         ↓       ↓        ↓       │
│  Exchange         Regime     4 Signal  Score    Size    Orders      │
│  Adapter         Classifier  Families  0-100   & Lev.   & Fills    │
│                      ↓          ↓         ↓                        │
│                   ML Layer   Portfolio   State                      │
│                              Manager    Persist                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Dashboard / API Service                        │
│                                                                     │
│   FastAPI REST API → SQLite DB                                      │
│   JWT Auth → Dashboard UI                                           │
│   Discord Notifications → Claude API Reports                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer Summary

| Layer | Description |
|---|---|
| Exchange Adapter | Abstract interface + Coinbase implementation. Swap out exchange without touching business logic. |
| Market Data | Fetches, caches, and streams OHLCV candles for all symbols/timeframes |
| Feature Calculation | ATR, EMAs, RSI, MACD, BB, ADX, swing highs/lows, structure levels |
| Strategy Engine | 4 strategy families: Trend Pullback, Breakout Retest, Liquidity Sweep Reversal, Volatility Expansion |
| Voting Engine | Weighted aggregation of signals + context filters → score 0–100 |
| Risk Engine | Deterministic risk controls: drawdown, daily loss, stop distance, slippage, position sizing |
| Execution Engine | Order placement, fill monitoring, SL/TP placement, state persistence |
| Portfolio / State | Equity tracking, PnL, drawdown, positions, cooldowns |
| ML Adaptive Layer | Regime classifier, trade quality scorer, strategy performance monitor |
| LLM Reporting | Claude API for summaries and anomaly explanations only (no trading decisions) |
| Kill Switch | Hard safety stop: drawdown, daily loss, API failure, connectivity loss |
| Dashboard / API | FastAPI + JWT auth, REST endpoints, equity curve, metrics |
| Notifications | Discord webhook alerts for all key events |
| Backtesting | Full historical simulation with fees, slippage, and latency modeling |
| Paper Trading | Real market data + simulated orders using same logic as live |

---

## Setup

### 1. Requirements

- Python 3.11+
- SQLite (bundled with Python)

### 2. Clone and Install

```bash
git clone <repo>
cd Trading-Bot
pip install -r requirements.txt
```

### 3. Environment Variables

```bash
cp .env.example .env
# Edit .env with your actual values
```

Key variables:
```
COINBASE_API_KEY=...           # Coinbase Advanced Trade API key
COINBASE_API_SECRET=...        # Coinbase API secret
DISCORD_WEBHOOK_URL=...        # Discord webhook for alerts
ANTHROPIC_API_KEY=...          # Claude API for reports (optional)
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=...         # Change before deploying
JWT_SECRET_KEY=...             # Random string, 32+ chars
BOT_MODE=paper                 # paper | live | backtest
```

### 4. Trading Configuration

Edit `config/trading_config.yaml` to adjust trading behavior. All important parameters are exposed here:
- Risk limits (drawdown, daily loss, position size, leverage)
- Score thresholds (when to trade / not trade)
- Strategy weights and enable/disable flags
- Session preferences
- Cooldown rules
- ML and LLM settings

---

## Running

### Paper Trading Mode (Recommended Starting Point)

```bash
python -m app.main --mode paper
```

Uses real market data from Coinbase but simulates all order execution. No real money at risk.

### Live Trading Mode

```bash
python -m app.main --mode live
```

⚠️ **WARNING**: This places real orders on a real exchange. Ensure you have:
1. Tested thoroughly in paper mode
2. Set conservative risk limits in config
3. Verified your API keys have correct permissions
4. Understood all risks of automated trading

### Dashboard / API Server

```bash
python -m app.main --mode api
```

API available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

Default login: `admin` / whatever `DASHBOARD_PASSWORD` is set to in `.env`.

### Backtesting

```bash
python -m app.main --mode backtest --symbol BTC-PERP --start 2023-01-01 --end 2024-01-01
```

### ML Model Retraining

```bash
python -m app.main --mode retrain
```

Requires at least 500 closed trades in the database. Falls back to heuristics until enough data is collected.

### Daily Report

```bash
python -m app.main --mode report-daily
```

Requires `ANTHROPIC_API_KEY` to be set. Generates a human-readable daily summary using Claude.

### Weekly Report

```bash
python -m app.main --mode report-weekly
```

---

## Deploying

### Docker (Recommended)

```bash
# Build and run both services
docker-compose up -d

# Paper trading worker only
docker-compose up -d worker

# Dashboard/API only
docker-compose up -d api
```

### Render

The dashboard/API service can be deployed to [Render](https://render.com):

1. Create a new **Web Service** pointing to this repo
2. Set **Start Command**: `python -m app.main --mode api`
3. Add all environment variables from `.env.example`
4. Set a **persistent disk** mounted at `/app/data` for SQLite storage

The trading worker should run on a separate always-on VPS or cloud service (not Render's free tier which sleeps).

### Separate Worker Deployment

The trading worker is designed to run as a standalone process:

```bash
# On your VPS
BOT_MODE=paper python -m app.main --mode paper

# Or with systemd
# See docs/systemd-service.example
```

The worker connects to the same SQLite DB. If deploying remotely with separate DB instances, ensure both services share a database (or use a networked DB solution for scale).

---

## How the Exchange Adapter Works

The system uses a clean adapter interface (`app/exchange/base.py`) that defines all exchange interactions as abstract methods. The Coinbase implementation (`app/exchange/coinbase.py`) implements this interface.

**To add a new exchange:**
1. Create `app/exchange/my_exchange.py` that subclasses `BaseExchangeAdapter`
2. Implement all abstract methods
3. Add it to `app/exchange/factory.py`
4. No other code needs to change

**Coinbase Integration Notes:**
- Uses Coinbase Advanced Trade REST API
- TODO annotations in `coinbase.py` mark areas requiring final account-level verification
- Futures/perpetuals product IDs may differ based on your account type
- Verify leverage limits and margin requirements against your specific account tier

---

## How Claude API Integration Works

Claude is used **only** for analysis and reporting. It has **no trading authority**.

What Claude can do:
- Generate daily/weekly performance summaries
- Explain anomalous events
- Provide narrative status descriptions for the dashboard

What Claude **cannot** do:
- Place or cancel orders
- Change risk parameters
- Override stop losses or leverage settings
- Modify strategy behavior

The LLM layer gracefully degrades if the API is unavailable — reports simply won't generate, but trading continues normally. Set `LLM.ENABLED=false` in config to disable entirely.

---

## How to Tune Weights and Thresholds

All tunable parameters are in `config/trading_config.yaml`.

**Key areas to tune:**

### Score Thresholds
```yaml
voting:
  score_thresholds:
    no_trade_max: 40.0    # Raise to require higher quality
    medium_min: 40.0
    strong_min: 60.0
    elite_min: 80.0
```

### Strategy Weights
Adjust how much each strategy contributes to the final score:
```yaml
voting:
  strategy_weights:
    trend_pullback: 1.0         # Highest weight = most trusted
    breakout_retest: 1.0
    liquidity_sweep_reversal: 0.85
    volatility_expansion: 0.90
```

### Risk Limits
```yaml
risk:
  risk_per_trade_pct: 1.0       # Start conservative, increase only after proven performance
  max_account_drawdown_pct: 15.0
  max_daily_loss_pct: 10.0
  leverage_tiers:
    medium_leverage: 2.0        # Keep low until strategy is well-tested
    strong_leverage: 5.0
    elite_leverage: 10.0
```

### Enable/Disable Strategies
```yaml
strategy_enable:
  trend_pullback: true
  breakout_retest: true
  liquidity_sweep_reversal: false  # Disable if it's underperforming
  volatility_expansion: true
```

---

## Caveats and Safety Warnings

1. **This is experimental software.** Automated trading carries significant financial risk. Test thoroughly in paper mode before going live.

2. **Exchange integration requires verification.** The Coinbase adapter skeleton is provided, but exact endpoint behavior, authentication details, and product availability must be verified against your live account. Look for `TODO:` annotations in `app/exchange/coinbase.py`.

3. **Leverage amplifies losses.** The default configuration uses modest leverage. Never increase leverage without understanding the liquidation mechanics of your specific exchange account type.

4. **Futures products vary.** BTC-PERP and ETH-PERP product IDs and specifications depend on your Coinbase account tier (retail vs. Coinbase International Exchange vs. Coinbase Derivatives). Verify product IDs before going live.

5. **The kill switch is your safety net.** If something goes wrong, the kill switch will halt new trades. Existing positions may still be open and require manual management depending on your configuration.

6. **Never disable safety controls for any reason.** The risk engine and kill switch are not adjustable by the ML or LLM layers and must not be disabled in production.

7. **Paper trading is not a guarantee of live performance.** Slippage, liquidity, and API behavior in live conditions may differ from simulation.

---

## Testing

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/unit/test_voting.py -v
pytest tests/unit/test_risk.py -v
pytest tests/unit/test_strategies.py -v

# Run with coverage
pytest --cov=app tests/
```

---

## Project Structure

```
Trading-Bot/
├── app/
│   ├── api/            # FastAPI endpoints
│   ├── auth/           # JWT auth
│   ├── backtesting/    # Historical simulation engine
│   ├── config/         # Settings and trading config
│   ├── db/             # Database engine and session
│   ├── exchange/       # Exchange adapters (Coinbase, Paper)
│   ├── execution/      # Order placement and monitoring
│   ├── market_data/    # Candle management and features
│   ├── ml/             # Regime classifier, quality scorer, retrainer
│   ├── models/         # SQLAlchemy ORM models
│   ├── monitoring/     # Session classifier, kill switch, logging
│   ├── notifications/  # Discord alerts
│   ├── paper/          # Paper trading mode
│   ├── portfolio/      # State and equity tracking
│   ├── reporting/      # Claude API report generation
│   ├── risk/           # Risk engine (deterministic)
│   ├── signals/        # Signal schemas
│   ├── strategies/     # 4 strategy family implementations
│   ├── voting/         # Weighted voting engine
│   ├── main.py         # CLI entry point
│   └── trading_worker.py  # Main trading loop
├── config/
│   └── trading_config.yaml
├── tests/
│   ├── unit/           # Unit tests
│   └── fixtures/       # Test data generators
├── data/               # SQLite DB, ML models, historical cache
├── logs/               # Application logs
├── docs/               # Additional documentation
├── Dockerfile.api
├── Dockerfile.worker
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```
