"""Abstract base class for all strategy families."""
from __future__ import annotations

import abc
from typing import Dict, Optional

import pandas as pd

from .schema import StrategySignal


class BaseStrategy(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    async def evaluate(
        self,
        symbol: str,
        candles: Dict[str, pd.DataFrame],
        current_price: float,
        regime: str,
    ) -> Optional[StrategySignal]:
        """Return a StrategySignal if conditions are met, else None."""
        ...
