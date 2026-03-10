"""
Tests for configuration system.
"""
import pytest
import os
from app.config.trading_config import TradingConfig, load_trading_config


class TestTradingConfig:
    def test_default_config_loads(self):
        cfg = TradingConfig()
        assert cfg.risk.max_account_drawdown_pct == 15.0
        assert cfg.risk.max_daily_loss_pct == 10.0
        assert cfg.risk.max_simultaneous_positions == 4
        assert cfg.risk.min_stop_distance_pct == 0.15
        assert cfg.risk.max_stop_distance_pct == 3.0

    def test_score_thresholds_valid(self):
        cfg = TradingConfig()
        t = cfg.voting.score_thresholds
        assert t.no_trade_max <= t.strong_min
        assert t.strong_min <= t.elite_min

    def test_leverage_tiers_ordered(self):
        cfg = TradingConfig()
        lt = cfg.risk.leverage_tiers
        assert lt.medium_leverage <= lt.strong_leverage <= lt.elite_leverage
        assert lt.elite_leverage <= lt.max_leverage_cap

    def test_symbols_configured(self):
        cfg = TradingConfig()
        assert len(cfg.universe.symbols) >= 1

    def test_timeframes_configured(self):
        cfg = TradingConfig()
        assert len(cfg.timeframes.enabled) >= 1

    def test_strategy_weights_positive(self):
        cfg = TradingConfig()
        w = cfg.voting.strategy_weights
        assert w.trend_pullback > 0
        assert w.breakout_retest > 0
        assert w.liquidity_sweep_reversal > 0
        assert w.volatility_expansion > 0

    def test_cooldown_config_valid(self):
        cfg = TradingConfig()
        cd = cfg.cooldown
        assert cd.consecutive_losses_short < cd.consecutive_losses_long
        assert cd.pause_short_hours < cd.pause_long_hours

    def test_risk_per_trade_bounded(self):
        cfg = TradingConfig()
        assert 0 < cfg.risk.risk_per_trade_pct <= cfg.risk.max_risk_per_trade_pct

    def test_yaml_config_load(self, tmp_path):
        """Test loading from a YAML file."""
        import yaml
        config_data = {
            "risk": {"max_account_drawdown_pct": 20.0},
            "universe": {"symbols": ["BTC-PERP"]},
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Reset cached config to load fresh
        import app.config.trading_config as tc_module
        tc_module._config_instance = None

        cfg = tc_module.load_trading_config(str(config_file))
        assert cfg.risk.max_account_drawdown_pct == 20.0
        assert "BTC-PERP" in cfg.universe.symbols

        # Reset
        tc_module._config_instance = None
