"""
Market data service: orchestrates data fetching for all symbols/timeframes.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import structlog

from app.config.trading_config import get_trading_config
from app.exchange.base import BaseExchangeAdapter, Ticker
from app.market_data.candles import CandleManager

logger = structlog.get_logger(__name__)


class MarketDataService:
    """
    Central market data hub.
    - Manages candle refreshes for all configured symbols/timeframes
    - Provides current ticker data
    - Provides unified access to OHLCV DataFrames
    """

    def __init__(self, adapter: BaseExchangeAdapter, candle_manager: CandleManager) -> None:
        self._adapter = adapter
        self._candle_mgr = candle_manager
        self._cfg = get_trading_config()
        self._tickers: Dict[str, Ticker] = {}
        self._running = False
        self._refresh_tasks: List[asyncio.Task] = []

    async def initialize(self) -> None:
        """Pre-load candle data for all symbols and timeframes."""
        symbols = self._cfg.universe.symbols
        timeframes = self._cfg.timeframes.enabled
        logger.info("market_data_init", symbols=symbols, timeframes=timeframes)

        tasks = []
        for symbol in symbols:
            for tf in timeframes:
                tasks.append(self._load_candles(symbol, tf))
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("market_data_init_complete")

    async def _load_candles(self, symbol: str, timeframe: str, limit: int = 300) -> None:
        try:
            await self._candle_mgr.get_candles(symbol, timeframe, limit=limit)
            logger.debug("candles_loaded", symbol=symbol, timeframe=timeframe)
        except Exception as e:
            logger.error("candle_load_error", symbol=symbol, timeframe=timeframe, error=str(e))

    async def start_streaming(self) -> None:
        """Start background candle refresh loops."""
        self._running = True
        for symbol in self._cfg.universe.symbols:
            for tf in self._cfg.timeframes.enabled:
                task = asyncio.create_task(
                    self._candle_refresh_loop(symbol, tf),
                    name=f"candle_refresh_{symbol}_{tf}",
                )
                self._refresh_tasks.append(task)
        logger.info("market_data_streaming_started")

    async def stop_streaming(self) -> None:
        self._running = False
        for task in self._refresh_tasks:
            task.cancel()
        self._refresh_tasks.clear()

    async def _candle_refresh_loop(self, symbol: str, timeframe: str) -> None:
        """Periodically refresh candles for a symbol/timeframe pair."""
        from app.market_data.candles import TIMEFRAME_SECONDS
        interval = TIMEFRAME_SECONDS.get(timeframe, 60)
        while self._running:
            try:
                await self._candle_mgr.update_latest(symbol, timeframe)
                # Also refresh ticker
                await self._refresh_ticker(symbol)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("candle_refresh_loop_error", symbol=symbol, tf=timeframe, error=str(e))
            await asyncio.sleep(min(interval, 60))  # refresh at most every 60s

    async def _refresh_ticker(self, symbol: str) -> None:
        try:
            ticker = await self._adapter.get_ticker(symbol)
            self._tickers[symbol] = ticker
        except Exception as e:
            logger.warning("ticker_refresh_error", symbol=symbol, error=str(e))

    # ------------------------------------------------------------------
    # Public access
    # ------------------------------------------------------------------

    def get_candles(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Synchronously return cached candles."""
        return self._candle_mgr.get_cached(symbol, timeframe)

    async def get_candles_async(self, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        return await self._candle_mgr.get_candles(symbol, timeframe, limit=limit)

    def get_ticker(self, symbol: str) -> Optional[Ticker]:
        return self._tickers.get(symbol)

    async def get_ticker_async(self, symbol: str) -> Optional[Ticker]:
        await self._refresh_ticker(symbol)
        return self._tickers.get(symbol)

    def get_spread_pct(self, symbol: str) -> float:
        """Return current bid/ask spread as a percentage."""
        ticker = self._tickers.get(symbol)
        if not ticker or ticker.bid <= 0:
            return 999.0
        return (ticker.ask - ticker.bid) / ticker.bid * 100

    def get_mid_price(self, symbol: str) -> Optional[float]:
        ticker = self._tickers.get(symbol)
        if not ticker:
            return None
        return (ticker.bid + ticker.ask) / 2
