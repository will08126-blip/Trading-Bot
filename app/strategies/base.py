"""
Base strategy interface. All strategy families must implement evaluate().
"""
from __future__ import annotations

import abc
from typing import Dict, Optional

import pandas as pd

from app.signals.schema import StrategySignal


class BaseStrategy(abc.ABC):
    """
    Abstract base for all trading strategy families.
    Each strategy evaluates market data and returns a StrategySignal or None.
    """

    name: str = "base"

    @abc.abstractmethod
    async def evaluate(
        self,
        symbol: str,
        candles: Dict[str, pd.DataFrame],  # timeframe -> DataFrame
        current_price: float,
        regime: str,
    ) -> Optional[StrategySignal]:
        """
        Evaluate the strategy given multi-timeframe candle data.

        Args:
            symbol: trading symbol
            candles: dict mapping timeframe (e.g. '5m') to enriched OHLCV DataFrame
            current_price: latest market price
            regime: current regime classification string

        Returns:
            StrategySignal if a signal is found, None otherwise.
        """
        ...

    def is_enabled_in_regime(self, regime: str, allowed_strategies: list[str]) -> bool:
        """Check if this strategy is allowed in the current regime."""
        return self.name in allowed_strategies
