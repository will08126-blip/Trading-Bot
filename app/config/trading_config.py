"""
Trading configuration loaded from YAML config file.
These are tunable trading parameters (not secrets).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class SymbolConfig(BaseModel):
    name: str
    enabled: bool = True
    max_leverage: float = 10.0
    min_volume_24h_usd: float = 50_000_000.0
    max_spread_pct: float = 0.05
    min_order_book_depth_usd: float = 100_000.0

class TimeframeConfig(BaseModel):
    enabled: List[str] = ["1m", "5m", "15m", "4h"]
    primary: str = "5m"
    context: str = "4h"
    setup: str = "15m"
    execution: str = "1m"

class ScoreThresholdConfig(BaseModel):
    no_trade_max: float = 40.0
    medium_min: float = 40.0
    strong_min: float = 60.0
    elite_min: float = 80.0

class LeverageTierConfig(BaseModel):
    # score -> leverage multiplier cap
    medium_leverage: float = 2.0
    strong_leverage: float = 5.0
    elite_leverage: float = 10.0
    max_leverage_cap: float = 20.0  # hard cap regardless of score

class RiskConfig(BaseModel):
    # Drawdown / daily loss
    max_account_drawdown_pct: float = 15.0   # kill switch trigger
    max_daily_loss_pct: float = 10.0          # daily loss kill
    # Position limits
    max_capital_deployed_pct: float = 50.0
    max_simultaneous_positions: int = 4
    max_positions_per_symbol: int = 2
    # Sizing
    risk_per_trade_pct: float = 1.0           # % of bot capital risked per trade
    max_risk_per_trade_pct: float = 2.0       # absolute cap
    min_stop_distance_pct: float = 0.15
    max_stop_distance_pct: float = 3.0
    # Slippage
    max_slippage_pct: float = 0.1
    # Notional cap
    max_notional_per_trade_usd: float = 50_000.0
    # Leverage tiers
    leverage_tiers: LeverageTierConfig = Field(default_factory=LeverageTierConfig)
    # Reduced risk mode
    reduced_risk_multiplier: float = 0.5      # risk multiplier when in reduced risk mode

class CooldownConfig(BaseModel):
    consecutive_losses_short: int = 3        # losses before short pause
    pause_short_hours: float = 1.0
    consecutive_losses_long: int = 5
    pause_long_hours: float = 4.0
    per_strategy_loss_streak: int = 4        # before strategy cooldown
    per_strategy_pause_hours: float = 2.0
    per_symbol_loss_streak: int = 4
    per_symbol_pause_hours: float = 2.0

class StopTakeProfitConfig(BaseModel):
    # Stop
    atr_multiplier_stop: float = 1.5
    structure_weight: float = 0.5            # blend of structure vs ATR
    # Take profit
    tp_mode: str = "r_multiple"              # r_multiple | structure | volatility_adjusted
    r_multiple: float = 2.0                  # default R:R
    tp_atr_multiplier: float = 3.0           # for volatility_adjusted mode
    # Partial exits (disabled by default in V1)
    partial_exits_enabled: bool = False

class ExecutionConfig(BaseModel):
    order_type_preference: str = "limit_first"  # market | limit | limit_first
    limit_order_timeout_seconds: int = 30
    max_fill_attempts: int = 3
    slippage_estimate_basis_points: float = 5.0
    max_allowed_slippage_pct: float = 0.1

class SessionConfig(BaseModel):
    # Session quality weights for scoring
    london_open_weight: float = 1.2
    ny_open_weight: float = 1.3
    asia_weight: float = 0.85
    overlap_weight: float = 1.4             # London/NY overlap
    off_hours_weight: float = 0.7
    # Whether to hard-block off-hours (False = scoring only)
    hard_block_off_hours: bool = False
    # Minimum session score to trade
    min_session_score: float = 0.6

class StrategyWeightConfig(BaseModel):
    trend_pullback: float = 1.0
    breakout_retest: float = 1.0
    liquidity_sweep_reversal: float = 0.85
    volatility_expansion: float = 0.90

class StrategyEnableConfig(BaseModel):
    trend_pullback: bool = True
    breakout_retest: bool = True
    liquidity_sweep_reversal: bool = True
    volatility_expansion: bool = True

class RegimeStrategyMapConfig(BaseModel):
    # Which strategies are allowed in each regime
    trending: List[str] = ["trend_pullback", "breakout_retest"]
    ranging: List[str] = ["liquidity_sweep_reversal", "breakout_retest"]
    volatility_expansion: List[str] = ["volatility_expansion", "breakout_retest"]
    compression: List[str] = ["volatility_expansion"]
    unknown: List[str] = ["trend_pullback", "breakout_retest", "volatility_expansion"]

class VotingConfig(BaseModel):
    score_thresholds: ScoreThresholdConfig = Field(default_factory=ScoreThresholdConfig)
    strategy_weights: StrategyWeightConfig = Field(default_factory=StrategyWeightConfig)
    # Context factor weights in final score
    trend_alignment_weight: float = 0.20
    momentum_weight: float = 0.15
    volatility_regime_weight: float = 0.10
    liquidity_weight: float = 0.15
    session_weight: float = 0.10
    strategy_signal_weight: float = 0.30
    # Penalty weights
    spread_penalty_weight: float = 0.10
    slippage_penalty_weight: float = 0.10
    recent_performance_weight: float = 0.10

class MLConfig(BaseModel):
    regime_model_type: str = "random_forest"   # random_forest | gradient_boosting | logistic
    quality_model_type: str = "gradient_boosting"
    retrain_schedule: str = "weekly"            # weekly | daily | manual
    min_training_samples: int = 500
    feature_lookback_candles: int = 100
    model_confidence_threshold: float = 0.55
    fallback_to_heuristics: bool = True         # if no model available

class LLMConfig(BaseModel):
    enabled: bool = True
    daily_report_enabled: bool = True
    weekly_report_enabled: bool = True
    anomaly_summary_enabled: bool = True
    max_tokens_per_report: int = 1500
    cache_report_hours: float = 12.0            # avoid regenerating too often
    graceful_degrade_on_failure: bool = True

class BacktestConfig(BaseModel):
    default_start_date: str = "2022-01-01"
    default_end_date: str = ""                  # empty = today
    fee_pct: float = 0.05                       # taker fee %
    slippage_pct: float = 0.05                  # assumed slippage %
    assumed_latency_ms: int = 200

class UniverseConfig(BaseModel):
    symbols: List[str] = ["BTC-PERP", "ETH-PERP"]
    max_symbols_active: int = 2
    # Future: enable dynamic ranking by liquidity
    dynamic_universe_enabled: bool = False

# ---------------------------------------------------------------------------
# Root trading config
# ---------------------------------------------------------------------------

class TradingConfig(BaseModel):
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    timeframes: TimeframeConfig = Field(default_factory=TimeframeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    cooldown: CooldownConfig = Field(default_factory=CooldownConfig)
    stop_tp: StopTakeProfitConfig = Field(default_factory=StopTakeProfitConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    voting: VotingConfig = Field(default_factory=VotingConfig)
    strategy_enable: StrategyEnableConfig = Field(default_factory=StrategyEnableConfig)
    regime_strategy_map: RegimeStrategyMapConfig = Field(default_factory=RegimeStrategyMapConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    symbol_overrides: Dict[str, SymbolConfig] = Field(default_factory=dict)
    # Kill switch
    kill_switch_enabled: bool = True
    # Logging
    verbose_signal_logging: bool = True
    verbose_voting_logging: bool = True


_config_instance: Optional[TradingConfig] = None


def load_trading_config(config_path: Optional[str] = None) -> TradingConfig:
    global _config_instance
    if _config_instance is not None:
        return _config_instance

    if config_path is None:
        config_path = os.environ.get("TRADING_CONFIG_PATH", "config/trading_config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        _config_instance = TradingConfig(**data)
    else:
        # Use defaults
        _config_instance = TradingConfig()

    return _config_instance


def get_trading_config() -> TradingConfig:
    return load_trading_config()
