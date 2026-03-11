"""
Binance Futures public REST client.
Fetches OHLCV candles from the Binance USDⓈ-M Futures API.
No API key required for public endpoints.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

# Binance USDⓈ-M Futures base URL
BASE_URL = "https://fapi.binance.com"

# Map our interval strings to Binance interval strings (they match)
VALID_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}


class BinanceClient:
    """
    Async Binance Futures candle fetcher.
    Uses a shared aiohttp session for connection pooling.
    """

    def __init__(self, session: Optional[aiohttp.ClientSession] = None) -> None:
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> "BinanceClient":
        if self._owns_session:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_) -> None:
        if self._owns_session and self._session:
            await self._session.close()

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
        retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV candles for symbol/interval.

        Args:
            symbol: e.g. 'BTCUSDT'
            interval: e.g. '5m', '15m', '4h'
            limit: number of candles (max 1500)
            retries: number of retry attempts on failure

        Returns:
            DataFrame with columns: open_time, open, high, low, close, volume
            Index is the open_time as a DatetimeIndex.
        """
        if interval not in VALID_INTERVALS:
            logger.error("invalid_interval", interval=interval)
            return None

        url = f"{BASE_URL}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}

        for attempt in range(retries):
            try:
                async with self._session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 429:
                        # Rate limited — back off
                        retry_after = int(resp.headers.get("Retry-After", 5))
                        logger.warning("binance_rate_limited", symbol=symbol, wait=retry_after)
                        await asyncio.sleep(retry_after)
                        continue
                    resp.raise_for_status()
                    raw = await resp.json()
                return self._parse_klines(raw)

            except asyncio.TimeoutError:
                logger.warning("binance_timeout", symbol=symbol, interval=interval, attempt=attempt)
            except aiohttp.ClientError as exc:
                logger.warning("binance_request_error", symbol=symbol, interval=interval,
                               error=str(exc), attempt=attempt)
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)  # exponential back-off: 1s, 2s, 4s

        logger.error("binance_fetch_failed", symbol=symbol, interval=interval)
        return None

    @staticmethod
    def _parse_klines(raw: List) -> pd.DataFrame:
        """
        Convert raw Binance kline list to OHLCV DataFrame.
        Binance kline format:
          [open_time, open, high, low, close, volume, close_time, ...]
        """
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
        df = df.set_index("open_time")
        df = df.dropna()
        return df

    async def get_all_timeframes(
        self,
        symbol: str,
        timeframes: List[str],
        limits: Dict[str, int],
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetch candles for multiple timeframes concurrently.

        Returns:
            Dict mapping timeframe -> DataFrame (or None on failure)
        """
        tasks = {
            tf: self.get_klines(symbol, tf, limits.get(tf, 200))
            for tf in timeframes
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        output: Dict[str, Optional[pd.DataFrame]] = {}
        for tf, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning("timeframe_fetch_error", symbol=symbol, tf=tf, error=str(result))
                output[tf] = None
            else:
                output[tf] = result
        return output

    async def get_ticker_24h(self, symbol: str) -> Optional[Dict]:
        """Fetch 24-hour ticker statistics (used for volume/liquidity scoring)."""
        url = f"{BASE_URL}/fapi/v1/ticker/24hr"
        try:
            async with self._session.get(
                url, params={"symbol": symbol}, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as exc:
            logger.warning("ticker_fetch_error", symbol=symbol, error=str(exc))
            return None
