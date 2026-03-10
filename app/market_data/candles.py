"""
Candle manager: fetch, cache, and provide OHLCV data.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import structlog
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.exchange.base import BaseExchangeAdapter, Candle
from app.models.candle_cache import CandleCache

logger = structlog.get_logger(__name__)

TIMEFRAME_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "1d": 86400,
}


class CandleManager:
    """
    Manages candle data fetching, caching, and access.
    Provides DataFrames ready for indicator calculation.
    """

    def __init__(self, adapter: BaseExchangeAdapter, db: AsyncSession) -> None:
        self._adapter = adapter
        self._db = db
        self._memory_cache: Dict[str, pd.DataFrame] = {}
        # key: "symbol:timeframe"

    def _cache_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}:{timeframe}"

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 300,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Return candles as a DataFrame with columns:
        [open_time, open, high, low, close, volume, close_time]
        Indexed by open_time ascending.
        """
        key = self._cache_key(symbol, timeframe)

        # Try memory cache first
        if use_cache and key in self._memory_cache:
            df = self._memory_cache[key]
            if len(df) >= limit:
                return df.tail(limit).copy()

        # Fetch from DB cache
        df = await self._load_from_db(symbol, timeframe, limit)
        if len(df) >= limit * 0.8:
            # Fill remaining gap from exchange
            last_ts = df["open_time"].max() if len(df) > 0 else None
            if last_ts and (datetime.now(timezone.utc) - last_ts).total_seconds() > TIMEFRAME_SECONDS.get(timeframe, 60):
                fresh = await self._fetch_from_exchange(symbol, timeframe,
                                                        start=last_ts, limit=50)
                df = self._merge_candles(df, fresh)
                await self._save_to_db(symbol, timeframe, fresh)
        else:
            # Fetch full batch from exchange
            df = await self._fetch_from_exchange(symbol, timeframe, limit=limit)
            await self._save_to_db(symbol, timeframe,
                                   [self._row_to_candle(row, symbol, timeframe) for _, row in df.iterrows()])

        self._memory_cache[key] = df
        return df.tail(limit).copy()

    async def update_latest(self, symbol: str, timeframe: str) -> Optional[pd.Series]:
        """Fetch and update the latest candle. Returns the new/updated row."""
        try:
            candles = await self._adapter.fetch_candles(symbol, timeframe, limit=2)
            if not candles:
                return None
            df = _candles_to_df(candles)
            key = self._cache_key(symbol, timeframe)
            if key in self._memory_cache:
                self._memory_cache[key] = self._merge_candles(self._memory_cache[key], df)
            await self._save_to_db(symbol, timeframe, candles)
            return df.iloc[-1] if len(df) > 0 else None
        except Exception as e:
            logger.error("candle_update_error", symbol=symbol, timeframe=timeframe, error=str(e))
            return None

    def get_cached(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Synchronously return from memory cache only."""
        return self._memory_cache.get(self._cache_key(symbol, timeframe))

    async def _fetch_from_exchange(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        limit: int = 300,
    ) -> pd.DataFrame:
        candles = await self._adapter.fetch_candles(symbol, timeframe, start=start, limit=limit)
        logger.debug("fetched_candles", symbol=symbol, timeframe=timeframe, count=len(candles))
        return _candles_to_df(candles)

    async def _load_from_db(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        result = await self._db.execute(
            select(CandleCache)
            .where(CandleCache.symbol == symbol, CandleCache.timeframe == timeframe)
            .order_by(CandleCache.open_time.desc())
            .limit(limit)
        )
        rows = result.scalars().all()
        if not rows:
            return pd.DataFrame()
        records = [
            {
                "open_time": r.open_time,
                "close_time": r.close_time,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in reversed(rows)  # reverse to chronological
        ]
        df = pd.DataFrame(records)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df.set_index("open_time", inplace=True)
        return df

    async def _save_to_db(self, symbol: str, timeframe: str, candles: List[Candle]) -> None:
        for c in candles:
            if not c.is_closed:
                continue
            existing = await self._db.execute(
                select(CandleCache).where(
                    CandleCache.symbol == symbol,
                    CandleCache.timeframe == timeframe,
                    CandleCache.open_time == c.open_time,
                )
            )
            if existing.scalar_one_or_none():
                continue
            self._db.add(CandleCache(
                symbol=symbol,
                timeframe=timeframe,
                open_time=c.open_time,
                close_time=c.close_time,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume,
                is_closed=c.is_closed,
            ))
        try:
            await self._db.commit()
        except Exception as e:
            await self._db.rollback()
            logger.warning("candle_save_error", error=str(e))

    def _merge_candles(self, df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return new_df
        if new_df.empty:
            return df
        combined = pd.concat([df, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        return combined

    def _row_to_candle(self, row: pd.Series, symbol: str, timeframe: str) -> Candle:
        from app.exchange.base import Candle
        return Candle(
            symbol=symbol,
            timeframe=timeframe,
            open_time=row.name if hasattr(row, "name") else row.get("open_time"),
            close_time=row.get("close_time", row.name),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            is_closed=True,
        )


def _candles_to_df(candles: List[Candle]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()
    records = [
        {
            "open_time": c.open_time,
            "close_time": c.close_time,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ]
    df = pd.DataFrame(records)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)
    return df
